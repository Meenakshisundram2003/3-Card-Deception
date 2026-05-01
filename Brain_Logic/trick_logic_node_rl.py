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
       
        # --- CONFIGURATION ---
        self.declare_parameter('left_card', 'UNSET')
        self.declare_parameter('middle_card', 'UNSET')
        self.declare_parameter('right_card', 'UNSET')

        self.left = self.get_parameter('left_card').get_parameter_value().string_value
        self.middle = self.get_parameter('middle_card').get_parameter_value().string_value
        self.right = self.get_parameter('right_card').get_parameter_value().string_value

        # --- LOAD THE HIERARCHICAL BRAINS ---
        # Update these paths to match where you put the zip files
        self.path_scout = "/home/hello-robot/ros2_ws/src/brain_1_3card.zip"
        self.path_detective = "/home/hello-robot/ros2_ws/src/brain_2_3card.zip"

        # 1. Load Brain 1 (The Scout - Planner)
        if not os.path.exists(self.path_scout):
            self.get_logger().error(f"BRAIN 1 MISSING at {self.path_scout}")
            self.brain_scout = None
        else:
            self.brain_scout = PPO.load(self.path_scout)
            self.get_logger().info("Brain 1 (Scout) Loaded.")

        # 2. Load Brain 2 (The Detective - Logic)
        if not os.path.exists(self.path_detective):
            self.get_logger().error(f"BRAIN 2 MISSING at {self.path_detective}")
            self.brain_detective = None
        else:
            self.brain_detective = PPO.load(self.path_detective)
            self.get_logger().info("Brain 2 (Detective) Loaded.")

        # Map Card Names -> Index (0, 1, 2)
        self.CARD_TO_INDEX = {
            self.left: 0,
            self.middle: 1,
            self.right: 2
        }
       
        # Map Index -> Card Names (for speaking the result)
        self.INDEX_TO_CARD = {v: k for k, v in self.CARD_TO_INDEX.items()}

        # --- ROS SETUP ---
        self.start_sub = self.create_subscription(String, '/start_trick', self.start_callback, 10)
        self.detection_sub = self.create_subscription(String, '/card_detections', self.detection_callback, 10)
        self._action_client = ActionClient(self, TriggerFlip, '/trigger_flip')
       
        self.speech_pub = self.create_publisher(String, '/speech', 10)

        self.waiting_for_detection = False
        self.optimal_sensor_index = None # Store Brain 1's decision


    def start_callback(self, msg):
       
        if self.brain_scout:
            # Brain 1 takes a dummy input [0.0] and outputs the best index to flip
            action, _ = self.brain_scout.predict(np.array([0.0]), deterministic=True)
            self.optimal_sensor_index = int(action)
           
        else:
            self.get_logger().warn("Brain 1 not loaded! Defaulting to Index 0.")
            self.optimal_sensor_index = 0

        # Trigger the physical robot action
        self.send_flip_goal()

    def send_flip_goal(self):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Flipper server not available.')
            return
        goal_msg = TriggerFlip.Goal()
        # If your action supports specifying WHICH card, add: goal_msg.card_index = self.optimal_sensor_index
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
            self.waiting_for_detection = True
        else:
            self.get_logger().error("Flip Action Failed!")

    def get_spoken_name(self, card_code):
        if not card_code or len(card_code) < 2: return card_code
        suits = {'C': 'Clubs', 'D': 'Diamonds', 'H': 'Hearts', 'S': 'Spades'}
        ranks = {'A': 'Ace', 'J': 'Jack', 'Q': 'Queen', 'K': 'King'}
        return f"{ranks.get(card_code[:-1], card_code[:-1])} of {suits.get(card_code[-1], card_code[-1])}"

    def detection_callback(self, msg):
        if not self.waiting_for_detection: return
        if self.brain_detective is None: return
           
        detected_label = msg.data.strip()
       
        # 1. Translate "Camera Image" -> "Math Input"
        obs_value = self.CARD_TO_INDEX.get(detected_label)
       
        if obs_value is None:
            self.get_logger().warn(f"I see {detected_label}, but it's not part of the trick!")
            return


        # 2. Ask Brain 2 (The Detective)
        # Input: The value of the single card we see [obs_value]
        # Output: The index of the card the User Picked
        action, _ = self.brain_detective.predict(np.array([obs_value]), deterministic=True)
       
        predicted_user_choice_index = int(action)
        predicted_card_code = self.INDEX_TO_CARD.get(predicted_user_choice_index, "Unknown")
        spoken_card_name = self.get_spoken_name(predicted_card_code)
       
        self.get_logger().info(f"FINAL PREDICTION: {predicted_card_code}")
        self.get_logger().info("="*40 + "\n")
       
        # 3. Speak Result
        self.waiting_for_detection = False
        speech_msg = String()
        speech_msg.data = f"You picked the {spoken_card_name}"
        self.speech_pub.publish(speech_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TrickLogicNodeRL()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()