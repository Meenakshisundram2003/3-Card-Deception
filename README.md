# Decoding the Card Trick: Autonomous Strategy Discovery via Reinforcement Learning

This repository contains the software stack for solving high-entropy physical states—specifically modeled via the 3-Card and 4-Card Deception benchmarks—using a Partially Observable Markov Decision Process (POMDP) and Reinforcement Learning (RL). 

This heuristic-free active perception framework was developed for and deployed on the Hello Robot Stretch 3 mobile manipulator.

## 🎥 System Demonstrations

Watch the Stretch 3 mobile manipulator execute the heuristic-free active perception strategy in real-time.
* [**Watch the physical 3-Card and 4-Card strategy execution here**](https://www.linkedin.com/posts/meenakshisundramg_robotics-reinforcementlearning-computervision-activity-7455909353750126592-cluS?utm_source=social_share_send&utm_medium=member_desktop_web&rcm=ACoAADygdiwBZW0WD-thNoOUJRY9uuJ7dbKhFpE)

## 🧠 System Architecture

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
|
├── docs/
│   ├── Thesis_Presentation-Final.pdf      # Presentation slides 
|
├── launch_camera.sh                       # Script: Starts RealSense Camera
├── run_detector.sh                        # Script: Starts YOLO Vision Node
├── run_flipper.sh                         # Script: Starts Cartesian Execution Node (3-Card)
├── run_flipper_4.sh                       # Script: Starts Cartesian Execution Node (4-Card)
├── run_brain.sh                           # Script: Starts Logic Brain Node (3-Card)
├── run_brain_4.sh                         # Script: Starts Logic Brain Node (4-Card)
├── start_magic.sh                         # Script: Publishes Start Command
├── .gitignore                             # Ignores .venv, __pycache__, and .DS_Store
├── LICENSE                                # MIT License file
└── README.md
```
## 🚀 Getting Started

### Dependencies
Ensure you have ROS 2 (Humble recommended) installed. You can install all the required Python packages for the RL agents, YOLOv8 vision pipeline, and LLM interaction directly via `pip`:

pip install stable-baselines3 ultralytics gymnasium torch gTTS openai numpy pandas

*Note: Low-level hardware execution also requires the Hello Robot `stretch_body` API to be configured on your machine.*

### Workspace Setup
Clone this repository into your ROS 2 workspace:

cd ~/ros2_ws/src  
git clone https://github.com/Meenakshisundram2003/3-Card-Deception.git  
cd ~/ros2_ws  
colcon build --packages-select Brain_Logic Eyes_Perception Hands_Manipulation  
source install/setup.bash  


Before running the scripts, ensure they are executable:

chmod +x launch_camera.sh run_detector.sh run_flipper.sh run_flipper_4.sh run_brain.sh run_brain_4.sh start_magic.sh  


### Execution Flow
To run the full decoupled pipeline, execute the following shell scripts in separate terminals:

**Terminal 1: Launch the Hardware Camera**

./launch_camera.sh


**Terminal 2: Start the YOLO Vision Pipeline**

./run_detector.sh


**Terminal 3: Start the Physical Execution & Persona**
*(You will be prompted to securely enter your OpenAI API key for the LLM interaction)*

#### For 3-Card Deception:
./run_flipper.sh

#### For 4-Card Cyclic Shift:
./run_flipper_4.sh


**Terminal 4: Initialize the Logic Brain**
*Pass the initial known state of the board as arguments.*

#### For 3-Card Deception (Left, Middle, Right):
./run_brain.sh 'AS' 'KH' 'QD'

#### For 4-Card Cyclic Shift (Index 0, 1, 2, 3):
./run_brain_4.sh 'AS' 'KH' 'QD' 'JC'


**Terminal 5: Trigger the Magic Sequence**
*Once all nodes are active and the human participant is ready, publish the start command.*

./start_magic.sh


## 📖 Read the Research

For a deep dive into the POMDP formulation, the "multiverse" reward structure, and the sim-to-real domain adaptation, you can read the full academic documentation here:
* [Master's Thesis: Decoding the Card Trick](https://login.ezproxy1.lib.asu.edu/login?url=https://www.proquest.com/dissertations-theses/decoding-card-trick-autonomous-strategy-discovery/docview/3335835673/se-2?accountid=4485)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📝 Citation
If you utilize this heuristic-free POMDP framework or the decoupled PPO architecture in your research, please cite:

Subramanian, M. G. (2026). Decoding the Card Trick: Autonomous Strategy Discovery via Reinforcement Learning (Order No. 32582765). Available from Dissertations & Theses @ Arizona State University; ProQuest Dissertations & Theses Global. (3335835673). https://login.ezproxy1.lib.asu.edu/login?url=https://www.proquest.com/dissertations-theses/decoding-card-trick-autonomous-strategy-discovery/docview/3335835673/se-2

