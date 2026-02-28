# Decoding Deception: Autonomous Strategy Discovery via Hierarchical RL

This repository contains the ongoing research and codebase for solving high-entropy physical states (modeled via the 3-Card and 4-Card trick) using a Partially Observable Markov Decision Process (POMDP) and Hierarchical Reinforcement Learning (HRL). 

This work is currently being developed for deployment on the Stretch 3 mobile manipulator.

## 🧠 System Architecture

To solve the credit assignment problem inherent in standard RL, this architecture decouples the task into two Proximal Policy Optimization (PPO) networks, translating to three distinct ROS 2 modules:

1. **Brain_Logic (The Cognitive Engine):**
   * Houses the decoupled active perception ("Scout") and state estimation ("Detective") PPO networks.
   * Processes observed visual states, mathematically updates the board's belief state, and determines the optimal index to investigate next.

2. **Eyes_Perception (The Visual System):**
   * A custom YOLOv8 pipeline fine-tuned on images from the Stretch 3's wrist camera.
   * Dynamically identifies card values and suits to feed into the Brain's belief state.

3. **Hands_Manipulation (The Physical Execution):**
   * Translates the integer actions published by the Brain into Cartesian arm trajectories for the Stretch 3 to physically flip the cards.

## 📂 Repository Structure

```text
├── Brain_Logic/
│   ├── models/
│   │   ├── brain_1_3card.zip              # Trained Scout weights (3-card)
│   │   ├── brain_1_4card.zip              # Trained Scout weights (4-card)
│   │   ├── brain_2_3card.zip              # Trained Detective weights (3-card)
│   │   └── brain_2_4Card.zip              # Trained Detective weights (4-card)
│   ├── train_flip_decision_3card.py       # 3-Card Gym Env & PPO Training
│   ├── train_flip_decision_4card.py       # 4-Card Gym Env & PPO Training
│   └── trick_logic_node_rl.py             # Main ROS 2 logic node
│
├── Eyes_Perception/
│   ├── model/
│   │   └── best.pt                        # YOLOv8 Custom Weights
│   ├── Card_Detector.ipynb                # Vision training/testing notebook
│   └── detector_node.py                   # ROS 2 Vision publisher
│
├── Hands_Manipulation/
│   └── flipper_node.py                    # ROS 2 Cartesian execution node
│
└── README.md
