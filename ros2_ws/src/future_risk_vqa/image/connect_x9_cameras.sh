# Run on every ROS 2 computer: source ./connect_x9_cameras.sh
# Applies to this shell only. Peers are discovered automatically by DDS multicast.
if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo 'ROS 2 Humble is required at /opt/ros/humble.' >&2
    return 1 2>/dev/null || exit 1
fi
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
unset CYCLONEDDS_URI ROS_DISCOVERY_SERVER
unset FASTDDS_DEFAULT_PROFILES_FILE FASTRTPS_DEFAULT_PROFILES_FILE
unset ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS
echo 'ROS 2 domain 0 | Fast DDS automatic multicast discovery | no fixed peer IP'
echo 'Next: ros2 daemon stop; ros2 topic list -t --no-daemon --spin-time 10'
