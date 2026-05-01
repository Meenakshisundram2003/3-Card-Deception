#!/bin/bash

# Check if user provided the 4 card labels
if [ "$#" -ne 4 ]; then
    echo "Usage:   ./run_brain.sh <CARD_0> <CARD_1> <CARD_2> <CARD_3>"
    echo "Example: ./run_brain.sh '4C' '10H' 'JS' 'AD'"
    exit 1
fi

echo "Setting up ROS 2 environment..."
cd ~/ros2_ws
source install/setup.bash

echo "Activating Python virtual environment..."
source .venv/bin/activate

export PYTHONPATH=$HOME/ros2_ws/.venv/lib/python3.10/site-packages:$PYTHONPATH

echo "Starting Brain (Logic Node) with cards:"
echo "  CARD 0: $1"
echo "  CARD 1: $2"
echo "  CARD 2: $3"
echo "  CARD 3: $4"

# Pass the script arguments directly to the ros2 run command
ros2 run card_detector trick_logic_node_rl_4 --ros-args \
   -p card_0:="$1" \
   -p card_1:="$2" \
   -p card_2:="$3" \
   -p card_3:="$4"
