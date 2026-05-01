#!/bin/bash

echo "Setting up ROS 2 environment..."
# 1. Source your workspace
source ~/ros2_ws/install/setup.bash

echo "Activating Python virtual environment..."
# 2. Activate your virtual environment
source ~/ros2_ws/.venv/bin/activate

echo "Exporting OpenAPI key"
# 3. Setting API key
export OPENAI_API_KEY="your_openai_api_key_here"

echo "Starting 4-Card flipper action server..."
# 4. Run the new 4-card flipper node
python3 -m card_detector.flipper_node_4
