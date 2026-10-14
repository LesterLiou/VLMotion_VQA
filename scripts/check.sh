#!/usr/bin/env bash
# Default: CPU tests. --ros also exercises imports and inference modes in ROS.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
mode="${1:-cpu}"
if [[ "$mode" != cpu && "$mode" != --ros && "$mode" != --static ]]; then
  echo 'Usage: bash scripts/check.sh [--static|--ros]' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1
python3 scripts/check_project.py
[[ "$mode" == --static ]] && exit 0
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONPATH="$repo_root/ros2_ws/src/future_risk_vqa:$repo_root/ros2_ws/src/risk_vqa:$repo_root/ros2_ws/src/risk_vqa_sim:$repo_root/sam3_benchmark/future_risk_benchmark${PYTHONPATH:+:$PYTHONPATH}"
args=(--import-mode=importlib -p no:cacheprovider -q
  ros2_ws/src/future_risk_vqa/test ros2_ws/src/risk_vqa/test ros2_ws/src/risk_vqa_sim/test)
if [[ "$mode" == --ros ]]; then
  python3 -c 'import rclpy, cv_bridge'  # Fail explicitly when ROS dependencies are missing.
else
  args+=(--ignore=ros2_ws/src/future_risk_vqa/test/test_future_vqa_modes.py)
  echo 'CPU mode: ROS node inference-mode tests excluded; run --ros in a provisioned ROS environment.'
fi
if [[ -f sam3_benchmark/future_risk_benchmark/test_metrics.py ]]; then
  args+=(sam3_benchmark/future_risk_benchmark/test_metrics.py)
else
  echo 'Benchmark tests unavailable in this checkout; publish the benchmark submodule changes first.'
fi
python3 -m pytest "${args[@]}"
