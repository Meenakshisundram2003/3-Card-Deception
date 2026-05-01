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
        response = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "I am having trouble connecting to my brain."

def llm_response(messages):
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
    if not text: return
    try:
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
            return recognizer.recognize_google(audio)
        except Exception:
            return None

# --- MAIN NODE CLASS ---
class Speaker:
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
                self._speak_gtts(str(item))
            except Exception: pass

    def _speak_gtts(self, text: str) -> None:
        try:
            fname = os.path.join(self._gtts_cache, _sanitize_filename(text) + '.mp3')
            if not os.path.exists(fname):
                tts = gTTS(text=text, lang=self._gtts_lang)
                tts.save(fname)
            candidates = [('ffplay', ['ffplay', '-autoexit', '-nodisp', '-loglevel', 'quiet', fname]), ('mpg123', ['mpg123', '-q', fname])]
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
            self, TriggerFlip, 'trigger_flip', 
            goal_callback=self.goal_callback,
            execute_callback=self.execute_flip_sequence,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup())
        self.flip_count=0
        self.get_logger().info('Ready to receive dynamic flip commands!')

    def goal_callback(self, goal_request):
        self.get_logger().info('Received flip goal request. Accepting.')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info('Received cancel request. Stopping robot.')
        self.robot.stop()
        return CancelResponse.ACCEPT
    
    def speech_callback(self,msg):
        text = msg.data
        self.get_logger().info(f"I heard on /speech: '{text}'")
        speak(text)

    def execute_flip_sequence(self, goal_handle):
        feedback_msg = TriggerFlip.Feedback()
        try:
            # 1. READ THE COMMAND FROM THE BRAIN
            target_index = goal_handle.request.card_index
            self.get_logger().info(f"Received command to flip card at Index {target_index}")

            # 2. MAP INDEX TO HORIZONTAL DISTANCE 
            # ---> TUNE THESE NUMBERS ON THE REAL ROBOT <---
            if target_index == 0:
                move_distance = -0.2    # Far Left Card
            elif target_index == 1:
                move_distance = -0.1    # Middle Left Card
            elif target_index == 2:
                move_distance = 0.1   # Middle Right Card
            elif target_index == 3:
                move_distance = 0.2   # Far Right Card
            else:
                move_distance = 0.0

            # --- START OF PHYSICAL SCRIPT ---
            feedback_msg.status = "Resetting arm position..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.lift.move_to(1.0)
            self.robot.push_command()
            time.sleep(4)

            # --- LLM INTERACTION START (COMMENTED OUT FOR TESTING) ---
            '''
            system_instruction = "You are 'Stretch', a robotic magician..."
            messages = [{"role": "system", "content": system_instruction}]
            messages.append({"role": "user", "content": "Greet the guest..."})
            speak(llm_response(messages))
            
            user_response = recognize_from_microphone(5, 10)
            if user_response: messages.append({"role": "user", "content": f"My name is {user_response}"})
            else: messages.append({"role": "user", "content": "I didn't say anything"})
            speak(llm_response(messages))
            '''
            self.robot.arm.move_to(0)
            self.robot.push_command()
            time.sleep(2)
            self.robot.end_of_arm.move_to('wrist_pitch', -0.9)
            self.robot.push_command()
            time.sleep(2)

            if self.flip_count == 0:
                speak("Hello! I am Stretch, your robotic magician. In front of you are four cards. Pick a card, see it and place it back.")
                time.sleep(2)

            '''
            messages.append({"role": "user", "content": "Explain the rules..."})
            speak(llm_response(messages))
            time.sleep(2) 
            '''
            if self.flip_count == 0:
                speak("Now do a cyclic right shift of the remaining three cards. I will try to guess which card you picked.")
                self.robot.end_of_arm.move_to('wrist_yaw', 3.14)
                self.robot.push_command()
                time.sleep(20) # Long sleep for user interaction
            
            # --- END OF COMMENTED OUT SECTION ---

            self.robot.end_of_arm.move_to('wrist_yaw', 0)
            self.robot.push_command()
            time.sleep(2)
            self.robot.arm.move_to(0.2)
            self.robot.push_command()
            time.sleep(2)
            self.robot.end_of_arm.move_to('stretch_gripper', 20)
            self.robot.push_command()
            time.sleep(2)
            self.robot.end_of_arm.move_to('wrist_pitch', 0)
            self.robot.push_command()
            time.sleep(2)
            
            
            # 3. TRANSLATE TO THE CORRECT CARD
            feedback_msg.status = f"Translating base to Index {target_index}..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.base.translate_by(move_distance)
            self.robot.push_command()
            time.sleep(5)

            feedback_msg.status = "Moving to grip height..."
            goal_handle.publish_feedback(feedback_msg)
            self.robot.lift.move_to(.57)
            self.robot.push_command()
            time.sleep(5)
            
            # 4. EXECUTE THE SINGLE FLIP
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

            # 5. RETURN CAMERA TO CENTER
            feedback_msg.status = "Moving back to center..."
            goal_handle.publish_feedback(feedback_msg)
            # We move by the exact negative of what we just moved
            
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
            time.sleep(4)

            self.robot.base.translate_by(-move_distance)
            self.robot.push_command()
            time.sleep(5)

            self.flip_count += 1

            if self.flip_count >= 2:
                self.flip_count = 0
            # --- END OF SINGLE FLIP SCRIPT ---

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
        if hasattr(self, 'speaker'): self.speaker.shutdown()
        if hasattr(self, 'robot'): self.robot.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = FlipperActionServer()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        try: executor.spin()
        except KeyboardInterrupt: pass
        finally: node.destroy_node()
    except Exception as e: print(f"Failed to initialize node: {e}")
    finally: rclpy.shutdown()

if __name__ == '__main__':
    main()