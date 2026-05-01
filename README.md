# Decoding the Card Trick: Autonomous Strategy Discovery via Reinforcement Learning

This repository contains the software stack for solving high-entropy physical states—specifically modeled via the 3-Card and 4-Card Deception benchmarks—using a Partially Observable Markov Decision Process (POMDP) and Reinforcement Learning (RL). 

This heuristic-free active perception framework was developed for and deployed on the Hello Robot Stretch 3 mobile manipulator.

## System Architecture

To solve the credit assignment problem inherent in standard RL, this architecture decouples the task into two Proximal Policy Optimization (PPO) networks, translating to three distinct asynchronous ROS 2 modules:

1. **Brain_Logic (The Cognitive Engine):**
   * Houses the decoupled active perception ("Scout") and state estimation ("Detective") PPO networks.
   * Processes observed visual states, mathematically updates the board's belief state, and determines the optimal index to investigate next to break ambiguity.

2. **Eyes_Perception (The Visual System):**
   * A custom YOLOv8 pipeline fine-tuned on images from the Stretch 3's wrist camera (Intel RealSense D405).
   * Dynamically identifies card values and suits to feed into the Brain's continuous observation state.

3. **Hands_Manipulation (The Physical Execution):**
   * Translates the discrete actions published by the Brain into Cartesian arm and joint trajectories for the Stretch 3 to physically execute the sensing strategy.
   * Integrates LLM (GPT-4o) and gTTS audio pipelines for a dynamic, interactive robotic persona.

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
│   ├── trick_logic_node_rl.py             # Main ROS 2 logic node (3-Card)
│   └── trick_logic_node_rl_4.py           # Main ROS 2 logic node (4-Card)
│
├── Eyes_Perception/
│   ├── model/
│   │   └── best.pt                        # YOLOv8 Custom Weights
│   ├── Card_Detector.ipynb                # Vision training/testing notebook
│   └── detector_node.py                   # ROS 2 Vision publisher
│
├── Hands_Manipulation/
│   ├── flipper_node.py                    # ROS 2 Cartesian execution node (3-Card)
│   └── flipper_node_4.py                  # ROS 2 Cartesian execution node (4-Card)
│
└── README.md
