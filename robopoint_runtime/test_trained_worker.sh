#!/usr/bin/env bash
# Run inside vlpoint-inference after sourcing ROS 2 and the workspace.
set -euo pipefail
runtime_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
checkpoint="${ROBOPOINT_CHECKPOINT:-$runtime_dir/runs/robopoint_vqa_only_317677}"
if [[ ! -f "$checkpoint/config.json" ]]; then
  echo "Missing checkpoint config: $checkpoint/config.json" >&2
  exit 1
fi
exec ros2 launch vlpoint worker.launch.py \
  model_path:="$checkpoint" \
  quantization:=4bit \
  mm_use_sam3_conditioning:=false \
  limit_model_concurrency:=1 \
  enable_latency_timing:=true "$@"
