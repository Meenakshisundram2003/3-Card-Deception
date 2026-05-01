#!/usr/bin/env python3

# This file is: ~/ros2_ws/src/card_detector/card_detector/flipper_node.py

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from std_msgs.msg import String
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import stretch_body.robot
import time
import threading
import os
import shutil
import tempfile
import queue
import speech_recognition as sr
import pygame
from openai import OpenAI
from gtts import gTTS

# We must import the "contract" from the 'magic_interfaces' package
from magic_interfaces.action import TriggerFlip

def _sanitize_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, '_')
    return s.strip().replace(' ', '_')[:128] or 'tts'

# --- HELPER FUNCTIONS FOR LLM & SPEECH ---

def get_llm_response(client, messages, model="gpt-4o", max_tokens=60):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        # Append assistant's reply to messages
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "I am having trouble connecting to my brain."

def llm_response(messages):
    # CHANGED: Securely load key from Environment Variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("CRITICAL ERROR: 'OPENAI_API_KEY' environment variable is not set.")
        return "I cannot speak right now because my API key is missing."

    client = OpenAI(api_key=api_key)
    return get_llm_response(client=client, messages=messages)

def play_mp3(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def speak(text):
    if not text:
        return
    try:
        # Saving to a temp file prevents permission/overwrite issues
        filename = os.path.join(tempfile.gettempdir(), "welcome.mp3")
        myobj = gTTS(text=text, lang='en', slow=False)
        myobj.save(filename)
        play_mp3(filename)
    except Exception as e:
        print(f"TTS Error: {e}")

def recognize_from_microphone(timeout, phrase_time_limit):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening ... ")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            print("Processing")
            text = recognizer.recognize_google(audio)
            return text
        
        except sr.WaitTimeoutError:
            print("No speech detected")
            return 'No speech detected'
        except sr.UnknownValueError:
            print("Couldn't understand audio")
            return "Couldn't understand audio"
        except sr.RequestError as e:
            print(f"Error with speech recognition service: {e}")
        
    return None


# --- MAIN NODE CLASS ---

class Speaker:
    """
    Internal threaded speaker class for non-blocking TTS (ROS-friendly).
    """
    def __init__(self, logger) -> None:
        self.log = logger
        self._queue = queue.Queue()
        self._stop = threading.Event()
        
        self._gtts_lang = 'en'
        self._gtts_cache = os.path.join(tempfile.gettempdir(), 'card_tts_cache')
        os.makedirs(self._gtts_cache, exist_ok=True)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        if not text: return
        self._queue.put(text)

    def shutdown(self) -> None:
        self._stop.set()
        try: self._queue.put(None)
        except Exception: pass
        try: self._thread.join(timeout=2.0)
        except Exception: pass

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get()
                if item is None or self._stop.is_set(): break
                text = str(item)
                self._speak_gtts(text)
            except Exception: pass

    def _speak_gtts(self, text: str) -> None:
        try:
            fname = os.path.join(self._gtts_cache, _sanitize_filename(text) + '.mp3')
            if not os.path.exists(fname):
                tts = gTTS(text=text, lang=self._gtts_lang)
                tts.save(fname)
            candidates = [('ffplay', ['ffplay', '-autoexit', '-nodisp', '-loglevel', 'quiet', fname]),
                          ('mpg123', ['mpg123', '-q', fname])]
            for exe, cmd in candidates:
                if shutil.which(exe):
                    os.spawnvp(os.P_WAIT, cmd[0], cmd)
                    return
        except Exception: pass


class FlipperActionServer(Node):
    def __init__(self):
        super().__init__('flipper_action_server')
        self.get_logger().info('Flipper Action Server starting...')
        self.speech_sub = self.create_subscription(String, '/speech', self.speech_callback, 10)
        self.speaker = Speaker(self.get_logger())

        try:
            self.robot = stretch_body.robot.Robot()
            self.robot.startup()
            self.get_logger().info('Stretch robot has started up.')
        except Exception as e:
            self.get_logger().fatal(f"Failed to initialize Stretch robot: {e}")
            raise e 

        self._action_server = ActionServer(
            self,
            TriggerFlip,
            'trigger_flip', 
            goal_callback=self.goal_callback,
            execute_callback=self.execute_flip_sequence,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())

        self.get_logger().info('Ready to receive flip commands at /trigger_flip')

    def goal_callback(self, goal_request):
        self.get_logger().info('Received flip goal request. Accepting.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request. Stopping robot.')
        self.robot.stop()
        return CancelResponse.ACCEPT
    
    def speech_callback(self,msg):
        text=msg.data
        self.get_logger().info(f"I heard on /speech: '{text}'")
        speak(text)

    def execute_flip_sequence(self, goal_handle):
        feedback_msg = TriggerFlip.Feedback()
        try:
            # --- START OF FLIPPER SCRIPT ---
            feedback_msg.status = "Resetting arm position..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.lift.move_to(1.0)
            self.robot.push_command()
            time.sleep(4)

            # --- IMPROVED LLM INTERACTION START ---
            
            # 1. SETUP: Define the Persona
            # We use a single 'system' message to define the character and brevity constraints.
            system_instruction = (
                "You are Stretch, a mysterious magician robot. You perform a card trick where the guest picks a card, swaps the other two, and you magically identify their card. "
                "You must be charismatic and engaging, but also concise. Always speak in a friendly tone. "
                "Keep your responses under 20 words, and never reveal the secret of the trick. Your goal is to entertain and mystify the audience."
            )
            
            messages = [{"role": "system", "content": system_instruction}]
            
            # 2. STEP 1: Greet
            messages.append({"role": "user", "content": "Greet the audience and ask for their name."})
            
            response_text = llm_response(messages) # "Welcome to my card trick! What's your name?"
            speak(response_text)
            
            # 3. STEP 2: Listen for Name
            user_response = recognize_from_microphone(5, 10)
            if user_response:
                # We feed the real name back to the LLM
                messages.append({"role": "user", "content": f"My name is {user_response}"})
            else:
                messages.append({"role": "user", "content": "I didn't say anything"})

            # 4. STEP 3: Acknowledge Name (The LLM will naturally welcome them based on the previous turn)
            response_text = llm_response(messages) # "Nice to meet you, [Name]! Let's have some fun with cards!"
            speak(response_text)

            # Move arm while talking (Keep your existing physical moves)
            self.robot.arm.move_to(0)
            self.robot.push_command()
            time.sleep(2)

            self.robot.end_of_arm.move_to('wrist_pitch', -0.9)
            self.robot.push_command()
            time.sleep(2)

            # 5. STEP 4: Give the Instructions
            # We instruct the LLM to give the specific rules of the trick.
            instruction_prompt = (
                "Explain the card trick to the audience. Tell them to pick a card, memorize it, and replace it. Then, instruct them to swap the positions of the other two cards to confuse you. Keep it engaging and concise."
                "Remember, you are a charismatic magician! Use simple language and keep it under 30 words."
            )
            messages.append({"role": "user", "content": instruction_prompt})
            
            response_text = llm_response(messages) # "Pick a card, remember it, and put it back. Now swap the other two cards to confuse me. Ready?"
            speak(response_text)
            
            time.sleep(2) 

            # --- PHYSICAL MANIPULATION ---
            self.robot.end_of_arm.move_to('wrist_yaw', 3.14)
            self.robot.push_command()
            time.sleep(20) # Long sleep for user interaction
            
            self.robot.end_of_arm.move_to('wrist_yaw', 0)
            self.robot.push_command()
            time.sleep(2)
            self.robot.arm.move_to(0.2)
            self.robot.push_command()
            time.sleep(2)
            self.robot.end_of_arm.move_to('stretch_gripper', 25)
            self.robot.push_command()
            time.sleep(2)
            self.robot.end_of_arm.move_to('wrist_pitch', 0)
            self.robot.push_command()
            time.sleep(2)
            
            feedback_msg.status = "Moving to grip position..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.lift.move_to(.57)
            self.robot.push_command()
            time.sleep(5)
            
            feedback_msg.status = "Gripping..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.end_of_arm.move_to('stretch_gripper', -15)
            self.robot.push_command()
            time.sleep(2)
            self.robot.lift.move_to(.8)
            self.robot.push_command()
            time.sleep(4)
            
            feedback_msg.status = "Flipping..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.end_of_arm.move_to('wrist_roll', 3.14)
            self.robot.push_command()
            time.sleep(2)
            self.robot.lift.move_to(.62)
            self.robot.push_command()
            time.sleep(3)
            
            feedback_msg.status = "Releasing..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.end_of_arm.move_to('stretch_gripper', 40)
            self.robot.push_command()
            time.sleep(2)
            self.robot.lift.move_to(.9)
            self.robot.push_command()
            time.sleep(3)
            
            feedback_msg.status = "Resetting wrist..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.end_of_arm.move_to('wrist_roll', 0)
            self.robot.push_command()
            time.sleep(2)
            self.robot.end_of_arm.move_to('wrist_pitch', -0.9)
            self.robot.push_command()
            time.sleep(2)
            self.robot.arm.move_to(0.3)
            self.robot.push_command()
            time.sleep(2)
            self.robot.lift.move_to(.8)
            self.robot.push_command()
            time.sleep(3)
            # --- END OF SCRIPT ---

        except Exception as e:
            self.get_logger().error(f"Flip sequence failed: {e}")
            goal_handle.abort()
            result = TriggerFlip.Result()
            result.success = False
            return result

        self.get_logger().info('Flip sequence complete.')
        goal_handle.succeed()
        result = TriggerFlip.Result()
        result.success = True
        return result

    def destroy_node(self):
        self.get_logger().info('Shutting down flipper node and robot...')
        self._action_server.destroy()
        if hasattr(self, 'speaker'):
            self.speaker.shutdown()
        if hasattr(self, 'robot'):
            self.robot.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = FlipperActionServer()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
    except Exception as e:
        if 'node' in locals():
            node.get_logger().fatal(f"Node failed to spin: {e}")
        else:
            print(f"Failed to initialize FlipperActionServer node: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()