#!/bin/bash

echo "Setting up ROS 2 environment..."
cd ~/ros2_ws
source install/setup.bash

echo "Sending 'start' command to the Brain..."
ros2 topic pub /start_trick std_msgs/msg/String "{data: 'start'}" --once

echo "Start command sent."
