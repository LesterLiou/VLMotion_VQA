#!/usr/bin/env bash
set -eo pipefail
x9_tools_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$x9_tools_dir/connect_x9_cameras.sh"
if ! ros2 pkg prefix compressed_image_transport >/dev/null 2>&1; then
    echo 'Missing compressed image decoder. On THIS viewing computer, install:' >&2
    echo 'sudo apt install ros-humble-compressed-image-transport' >&2
    exit 1
fi
exec ros2 run rviz2 rviz2 -d "$x9_tools_dir/x9_cameras.rviz"
