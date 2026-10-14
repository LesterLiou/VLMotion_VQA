#!/usr/bin/env bash

set -euo pipefail

# 用法:
#   ./docker/run.sh [--root] [容器名稱] [ROS_DOMAIN_ID]
#   ./docker/run.sh [--root] [ROS_DOMAIN_ID]
# 說明:
#   - 若第一個非 --root 的參數是數字 (0-232)，視為 ROS_DOMAIN_ID（容器名稱使用預設）
#   - 若第一個是名稱，第二個是數字，則分別視為 容器名稱 與 ROS_DOMAIN_ID
# 預設容器名稱: vlpoint-inference

EXEC_USER=""
DOMAIN_ID=""
if [[ "${1-}" == "--root" ]]; then
  EXEC_USER="root"
  shift || true
fi

# 解析參數：支援 --root、容器名稱與數字型 ROS_DOMAIN_ID
CONTAINER_NAME="vlpoint-inference"

is_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

if [[ ${#@} -gt 0 ]]; then
  if is_integer "${1-}"; then
    DOMAIN_ID="$1"
    shift || true
  elif [[ -n "${1-}" ]]; then
    CONTAINER_NAME="$1"
    shift || true
    if [[ ${#@} -gt 0 ]] && is_integer "${1-}"; then
      DOMAIN_ID="$1"
      shift || true
    fi
  fi
fi

RUNTIME_DOMAIN_ID="${DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}"
if ! is_integer "${RUNTIME_DOMAIN_ID}" || (( ${#RUNTIME_DOMAIN_ID} > 3 )) || (( 10#${RUNTIME_DOMAIN_ID} > 232 )); then
  echo "ROS_DOMAIN_ID 必須為 0–232 的整數" >&2
  exit 1
fi
# Normalize leading zeros before passing the value to environment.sh.
RUNTIME_DOMAIN_ID=$((10#${RUNTIME_DOMAIN_ID}))

if ! command -v docker >/dev/null 2>&1; then
  echo "找不到 docker 指令，請先安裝 Docker" >&2
  exit 1
fi

# 先確保服務已啟動（若尚未建立/啟動會自動 up -d）
echo "確保 docker compose 服務已啟動（up -d）..."
compose_file="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/docker-compose.yml"
docker compose -f "$compose_file" up -d >/dev/null

# 是否存在此容器（包含已停止）
if ! docker ps -a --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"; then
  echo "找不到容器：${CONTAINER_NAME}（已嘗試 docker compose up -d）" >&2
  exit 1
fi

# 若容器存在但未運行，先啟動
if ! docker ps --format '{{.Names}}' | grep -wq "${CONTAINER_NAME}"; then
  echo "容器未在運行，正在啟動：${CONTAINER_NAME}"
  docker start "${CONTAINER_NAME}" >/dev/null
fi

# 傳遞目前 terminal 的 X11 display。SSH forwarding 通常是 localhost:10.0，
# 不可固定成 :0 或 :1。
DISPLAY_ARGS=()
if [[ -n "${DISPLAY-}" ]]; then
  echo "設定 X11 display: ${DISPLAY}"
  DISPLAY_ARGS=(-e "DISPLAY=${DISPLAY}" -e "XAUTHORITY=/root/.Xauthority")
  docker exec -u root "${CONTAINER_NAME}" bash -c \
    'touch /root/.Xauthority && chmod 600 /root/.Xauthority'

  # SSH X11 forwarding 使用 MIT-MAGIC-COOKIE。把目前 display 的 cookie 合併進
  # container，ffff family 可同時處理 host/container hostname 不同的情況。
  if command -v xauth >/dev/null 2>&1; then
    XAUTH_ENTRY="$(xauth nlist "${DISPLAY}" 2>/dev/null | head -n 1 || true)"
    if [[ -n "${XAUTH_ENTRY}" ]]; then
      if docker exec -u root "${CONTAINER_NAME}" \
          bash -c 'command -v xauth >/dev/null 2>&1'; then
        printf '%s\n' "${XAUTH_ENTRY}" \
          | sed 's/^..../ffff/' \
          | docker exec -i -u root "${CONTAINER_NAME}" xauth nmerge -
      else
        echo "警告：container 沒有 xauth；請重新 build image。暫時略過 X11 cookie 匯入" >&2
      fi
    else
      echo "警告：找不到 ${DISPLAY} 的 xauth cookie，SSH GUI 可能無法連線" >&2
    fi
  else
    echo "警告：主機沒有 xauth，SSH GUI 可能無法連線" >&2
  fi

  # 本機實體 display 使用 Unix socket 時，只授權 local root，不開放所有 local users。
  if [[ "${DISPLAY}" == :* ]] && command -v xhost >/dev/null 2>&1; then
    xhost +SI:localuser:root >/dev/null 2>&1 || true
  fi
else
  echo "未設定 DISPLAY；將以 headless 模式進入 container"
fi

echo "進入容器: ${CONTAINER_NAME}"
echo "會在容器內自動 source /workspace/environment.sh（ROS_DOMAIN_ID=${RUNTIME_DOMAIN_ID}）"
if [[ -n "${EXEC_USER}" ]]; then
  exec docker exec "${DISPLAY_ARGS[@]}" -u "${EXEC_USER}" -it "${CONTAINER_NAME}" bash -lc \
    "if [ -f /workspace/environment.sh ]; then source /workspace/environment.sh ${RUNTIME_DOMAIN_ID}; else echo '提示: /workspace/environment.sh 不存在，略過'; fi; exec bash -i"
else
  exec docker exec "${DISPLAY_ARGS[@]}" -it "${CONTAINER_NAME}" bash -lc \
    "if [ -f /workspace/environment.sh ]; then source /workspace/environment.sh ${RUNTIME_DOMAIN_ID}; else echo '提示: /workspace/environment.sh 不存在，略過'; fi; exec bash -i"
fi
