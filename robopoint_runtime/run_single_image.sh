#!/usr/bin/env bash
# Run from the host or from inside vlpoint-inference. Paths must be in the repo.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f /.dockerenv ]]; then
  cd "$repo_root"
  exec python3 "$repo_root/robopoint_runtime/single_image_vqa.py" "$@"
fi
mapped_args=()
for arg in "$@"; do
  case "$arg" in
    "$repo_root"/*) mapped_args+=("/workspace/${arg#"$repo_root"/}") ;;
    *) mapped_args+=("$arg") ;;
  esac
done
docker_io=(-i)
if [[ -t 0 && -t 1 ]]; then docker_io+=(-t); fi
exec docker exec "${docker_io[@]}" -e "VLMOTION_HOST_ROOT=$repo_root" -w /workspace "${VLMOTION_CONTAINER:-vlpoint-inference}" \
  python3 /workspace/robopoint_runtime/single_image_vqa.py "${mapped_args[@]}"
