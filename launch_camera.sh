#!/bin/bash


cd ~/ros2_ws
source install/setup.bash

echo "Starting RealSense Camera..."
ros2 launch realsense2_camera rs_launch.py


