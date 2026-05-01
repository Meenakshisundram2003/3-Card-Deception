#!/bin/bash

# --- CONFIGURATION ---
# (You can edit these default values right here)
TOPIC=/camera/camera/color/image_rect_raw
CONFIDENCE=0.60

# ---------------------

echo "Setting up ROS 2 environment..."
# 1. Source your workspace
source ~/ros2_ws/install/setup.bash

echo "Activating Python virtual environment..."
# 2. Activate your virtual environment
source ~/ros2_ws/.venv/bin/activate

echo "Starting detector node..."
# 4. Run the node with your parameters
python3 -m card_detector.detector_node --ros-args \
    -p image_topic:=$TOPIC \
    -p confidence_threshold:=$CONFIDENCE \
    -p min_stable_frames:=3 \
    -p log_on_change_only:=true \
    -p enable_tts:=true \
    -p tts_engine:=gtts
