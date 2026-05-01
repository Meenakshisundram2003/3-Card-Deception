#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from magic_interfaces.action import TriggerFlip
from rclpy.action import ActionClient
import os
import numpy as np
from stable_baselines3 import PPO

class TrickLogicNodeRL(Node):
    def __init__(self):
        super().__init__('trick_logic_node_rl')
       
        # --- CONFIGURATION (4 CARDS) ---
        self.declare_parameter('card_0', 'UNSET')
        self.declare_parameter('card_1', 'UNSET')
        self.declare_parameter('card_2', 'UNSET')
        self.declare_parameter('card_3', 'UNSET')

        c0 = self.get_parameter('card_0').get_parameter_value().string_value
        c1 = self.get_parameter('card_1').get_parameter_value().string_value
        c2 = self.get_parameter('card_2').get_parameter_value().string_value
        c3 = self.get_parameter('card_3').get_parameter_value().string_value

        # --- LOAD THE HIERARCHICAL BRAINS ---
        self.path_scout = "/home/hello-robot/ros2_ws/src/models/brain_1_4card.zip"
        self.path_detective = "/home/hello-robot/ros2_ws/src/models/brain_2_4Card.zip"

        # 1. Load Brain 1 (The Scout)
        if not os.path.exists(self.path_scout):
            self.get_logger().error(f"BRAIN 1 MISSING at {self.path_scout}")
            self.brain_scout = None
        else:
            self.brain_scout = PPO.load(self.path_scout)
            self.get_logger().info("✅ Brain 1 (Scout) Loaded.")

        # 2. Load Brain 2 (The Detective)
        if not os.path.exists(self.path_detective):
            self.get_logger().error(f"BRAIN 2 MISSING at {self.path_detective}")
            self.brain_detective = None
        else:
            self.brain_detective = PPO.load(self.path_detective)
            self.get_logger().info("✅ Brain 2 (Detective) Loaded.")

        # Map Card Names -> Index (0, 1, 2, 3)
        self.CARD_TO_INDEX = {c0: 0, c1: 1, c2: 2, c3: 3}
        self.INDEX_TO_CARD = {v: k for k, v in self.CARD_TO_INDEX.items()}

        # 4-Card Action Map
        self.pair_map = {
            0: [0, 1], 1: [1, 2], 2: [2, 3],
            3: [3, 0], 4: [0, 2], 5: [1, 3]
        }

        # --- ROS SETUP ---
        self.start_sub = self.create_subscription(String, '/start_trick', self.start_callback, 10)
        self.detection_sub = self.create_subscription(String, '/card_detections', self.detection_callback, 10)
        self._action_client = ActionClient(self, TriggerFlip, '/trigger_flip')
        self.speech_pub = self.create_publisher(String, '/speech', 10)

        # --- STATE MACHINE VARIABLES ---
        self.waiting_for_detection = False
        self.optimal_sensor_indices = [] # Holds the [Index_A, Index_B] plan
        self.current_flip_step = 0       # Tracks loop progress (0 or 1)
        self.observed_values = []        # Holds the math values of the cards we see

        self.get_logger().info("--- 4-CARD SYSTEM READY. Waiting for /start_trick ---")

    def start_callback(self, msg):
        self.get_logger().info("\n" + "="*40)
        self.get_logger().info("PHASE 1: PERCEPTION PLANNING (Brain 1)")
        
        # Reset state for a new trick
        self.observed_values = []
        self.current_flip_step = 0
       
        if self.brain_scout:
            action, _ = self.brain_scout.predict(np.array([0.0]), deterministic=True)
            self.optimal_sensor_indices = self.pair_map[int(action)]
            self.get_logger().info(f"🧠 Brain 1 Decision: The optimal strategy is to flip indices {self.optimal_sensor_indices}.")
        else:
            self.get_logger().warn("Brain 1 not loaded! Defaulting to [0, 2].")
            self.optimal_sensor_indices = [0, 2]

        self.execute_next_flip()

    def execute_next_flip(self):
        target_index = self.optimal_sensor_indices[self.current_flip_step]
        self.get_logger().info(f"🤖 Robot Action: Sending trigger command for INDEX {target_index}...")

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Flipper server not available.')
            return
        self.waiting_for_detection = True  
        goal_msg = TriggerFlip.Goal()
        # CRITICAL: We pass the specific target index to the flipper_node!
        goal_msg.card_index = int(target_index) 
        
        self._future = self._action_client.send_goal_async(goal_msg)
        self._future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected.')
            return
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info("✅ Single Flip Complete. WAITING FOR OBSERVATION...")
            # Unmute the camera listener
            # self.waiting_for_detection = True
        else:
            self.get_logger().error("Flip Action Failed!")

    def get_spoken_name(self, card_code):
        if not card_code or len(card_code) < 2: return card_code
        suits = {'C': 'Clubs', 'D': 'Diamonds', 'H': 'Hearts', 'S': 'Spades'}
        ranks = {'A': 'Ace', 'J': 'Jack', 'Q': 'Queen', 'K': 'King', 'T': '10'}
        if card_code.startswith('10'):
            rank = '10'
            suit = card_code[2:]
        else:
            rank = card_code[:-1]
            suit = card_code[-1]
        return f"{ranks.get(rank, rank)} of {suits.get(suit, suit)}"

    def detection_callback(self, msg):
        if not self.waiting_for_detection: return
           
        detected_label = msg.data.strip()
        obs_value = self.CARD_TO_INDEX.get(detected_label)
       
        if obs_value is None:
            return

        # Ensure we don't accidentally log the exact same card twice
        if obs_value not in self.observed_values:
            self.observed_values.append(obs_value)
            self.get_logger().info(f"👁️ Sensor Reading {self.current_flip_step + 1}/2: {detected_label} (Val {obs_value})")
            
            # IMMEDIATELY mute the camera so it doesn't keep reading the same card
            self.waiting_for_detection = False

            # State Machine Check: Do we need a second card?
            if self.current_flip_step == 0:
                self.current_flip_step += 1
                self.execute_next_flip() # Trigger the loop for the second flip
            else:
                self.run_detective_logic() # Break the loop and run the final math

    def run_detective_logic(self):
        self.get_logger().info("\n" + "="*40)
        self.get_logger().info("PHASE 3: LOGICAL DEDUCTION (Brain 2)")
        
        if self.brain_detective is None: 
            return

        # Pass the 2 observations as a numpy array into Brain 2
        obs_array = np.array(self.observed_values, dtype=np.float32)
        action, _ = self.brain_detective.predict(obs_array, deterministic=True)
       
        predicted_user_choice_index = int(action)
        predicted_card_code = self.INDEX_TO_CARD.get(predicted_user_choice_index, "Unknown")
        spoken_card_name = self.get_spoken_name(predicted_card_code)
       
        self.get_logger().info(f"🧠 Brain 2 Inference: Pattern matches User Choice {predicted_user_choice_index}")
        self.get_logger().info(f"🎉 FINAL PREDICTION: {predicted_card_code}")
        self.get_logger().info("="*40 + "\n")
       
        speech_msg = String()
        speech_msg.data = f"My neural network predicts you picked the {spoken_card_name}"
        self.speech_pub.publish(speech_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrickLogicNodeRL()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()