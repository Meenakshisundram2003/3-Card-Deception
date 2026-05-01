#!/bin/bash

# Check if user provided the 3 card labels
if [ "$#" -ne 3 ]; then
    echo "Usage:   ./run_brain.sh <LEFT_CARD> <MIDDLE_CARD> <RIGHT_CARD>"
    echo "Example: ./run_brain.sh '4C' '10H' 'JS'"
    exit 1
fi

echo "Setting up ROS 2 environment..."
cd ~/ros2_ws
source install/setup.bash

echo "Activating Python virtual environment..."
source .venv/bin/activate

export PYTHONPATH=$HOME/ros2_ws/.venv/lib/python3.10/site-packages:$PYTHONPATH

echo "Starting Brain (Logic Node) with cards:"
echo "  LEFT:   $1"
echo "  MIDDLE: $2"
echo "  RIGHT:  $3"

# Pass the script arguments directly to the ros2 run command
ros2 run card_detector trick_logic_node_rl --ros-args \
   -p left_card:="$1" \
   -p middle_card:="$2" \
   -p right_card:="$3"
