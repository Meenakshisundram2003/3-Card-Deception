import os
import re
import shutil
import tempfile
import threading
import queue
from typing import Optional, Any, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

# pyttsx3 import has been removed.

try:
    from gtts import gTTS  # needs internet for first-time synthesis per phrase
except Exception:
    gTTS = None


def _sanitize_filename(s: str) -> str:
    bad = '<>:"/\\|?*'
    for ch in bad:
        s = s.replace(ch, '_')
    return s.strip().replace(' ', '_')[:128] or 'tts'


# ---------------- Card label -> spoken helpers ----------------
_RANK_WORDS = {
    'a': 'Ace', 'j': 'Jack', 'q': 'Queen', 'k': 'King', 't': '10'
}
_SUIT_WORDS = {
    'c': 'Clubs', 'd': 'Diamonds', 'h': 'Hearts', 's': 'Spades'
}
_WORD_RANKS = {
    'ace': 'Ace', 'jack': 'Jack', 'queen': 'Queen', 'king': 'King',
    'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
}
_WORD_SUITS = {
    'club': 'Clubs', 'clubs': 'Clubs',
    'diamond': 'Diamonds', 'diamonds': 'Diamonds',
    'heart': 'Hearts', 'hearts': 'Hearts',
    'spade': 'Spades', 'spades': 'Spades'
}

def to_spoken_card(label: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Convert '4C', 'AH', '10S', 'four of clubs', 'c4', etc. -> ('4 of Clubs', '4', 'Clubs')
    """
    if not label:
        return label, None, None
    raw = label.strip()
    s = re.sub(r'[\s_\-]+', '', raw).lower()



    # Short forms like "4c", "ah", "10h", or suit-first "c4"
    m = re.match(r'^(10|[2-9]|[ajqkt])([cdhs])$', s)
    if not m:
        m = re.match(r'^([cdhs])(10|[2-9]|[ajqkt])$', s)
        if m:
            s = m.group(2) + m.group(1)
            m = re.match(r'^(10|[2-9]|[ajqkt])([cdhs])$', s)
    if m:
        r = m.group(1).lower()
        u = m.group(2).lower()
        r_name = _RANK_WORDS.get(r, r)
        suit_name = _SUIT_WORDS[u]
        return f"{r_name} of {suit_name}", r_name, suit_name

    # Long forms like "fourofclubs", "tenhearts"
    m = re.match(r'^(.*)of(.*)$', s)
    cand_r, cand_u = None, None
    if m:
        left, right = m.group(1), m.group(2)
        cand_r = _WORD_RANKS.get(left, left)
        if re.fullmatch(r'(10|[2-9])', left):
            cand_r = left
        cand_u = _WORD_SUITS.get(right, right)
    else:
        rank_match = re.search(r'(10|[2-9]|ace|jack|queen|king)', s)
        suit_match = re.search(r'(clubs?|diamonds?|hearts?|spades?)', s)
        if rank_match and suit_match:
            cand_r = rank_match.group(1)
            cand_u = suit_match.group(1)

    if cand_r and cand_u:
        r_name = _WORD_RANKS.get(cand_r.lower(), cand_r.capitalize())
        if re.fullmatch(r'(10|[2-9])', cand_r.lower()):
            r_name = cand_r
        suit_name = _WORD_SUITS.get(cand_u.lower(), cand_u.capitalize())
        return f"{r_name} of {suit_name}", r_name, suit_name

    return raw, None, None


class Speaker:

    def __init__(self, logger: Any,
                 engine: str = 'gtts',  # Default to gtts
                 rate: int = 180,       # No-op, but keep for API compatibility
                 volume: float = 1.0,   # No-op, but keep for API compatibility
                 voice_id: str = '',    # No-op, but keep for API compatibility
                 gtts_lang: str = 'en') -> None:
        self.log = logger
        self.engine_name = 'gtts' if gTTS is not None else 'none'
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._stop = threading.Event()

        # pyttsx3 setup has been removed

        # gTTS setup
        self._gtts_lang = gtts_lang or 'en'
        self._gtts_cache = os.path.join(tempfile.gettempdir(), 'card_tts_cache')
        os.makedirs(self._gtts_cache, exist_ok=True)

        # Start background thread
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        if self.engine_name == 'none':
            self._warn_once("gTTS not importable; speech will be disabled.")

    def say(self, text: str) -> None:
        if not text:
            return
        if self._queue.qsize() > 10:
            try:
                self._queue.get_nowait()
            except Exception:
                pass
        self._queue.put(text)

    def shutdown(self) -> None:
        self._stop.set()
        try:
            self._queue.put(None)
        except Exception:
            pass
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        # pyttsx3 stop() logic removed

    def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get()
                if item is None or self._stop.is_set():
                    break
                text = str(item)
                # pyttsx3 logic removed
                if self.engine_name == 'gtts' and gTTS is not None:
                    self._speak_gtts(text)
            except Exception as e:
                self._warn_once(f"TTS error: {e}")

    # _speak_pyttsx3 function has been removed

    def _speak_gtts(self, text: str) -> None:
        try:
            fname = os.path.join(self._gtts_cache, _sanitize_filename(text) + '.mp3')
            if not os.path.exists(fname):
                try:
                    tts = gTTS(text=text, lang=self._gtts_lang)
                    tts.save(fname)
                except Exception as e:
                    self._warn_once(f"gTTS synthesis failed: {e}")
                    return
            self._play_mp3(fname)
        except Exception as e:
            self._warn_once(f"gTTS playback failed: {e}")

    def _play_mp3(self, path: str) -> None:
        candidates = [
            ('ffplay', ['ffplay', '-autoexit', '-nodisp', '-loglevel', 'quiet', path]),
            ('mpg123', ['mpg123', '-q', path]),
            ('afplay', ['afplay', path]),           # macOS
            ('cvlc', ['cvlc', '--play-and-exit', '--quiet', path]),
            ('vlc', ['vlc', '--play-and-exit', '--quiet', path]),
            ('mplayer', ['mplayer', '-really-quiet', path]),
        ]
        for exe, cmd in candidates:
            if shutil.which(exe):
                try:
                    os.spawnvp(os.P_WAIT, cmd[0], cmd)
                    return
                except Exception:
                    continue
        self._warn_once("No audio player found for gTTS MP3s (ffplay/mpg123/afplay/cvlc/vlc/mplayer).")

    _warned = set()
    def _warn_once(self, msg: str) -> None:
        if msg not in self._warned:
            try:
                self.log.warning(msg)
            except Exception:
                pass
            self._warned.add(msg)


class CardDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__('card_detector_node')
        self.get_logger().info('Card Detector Node starting...')

        # -------- Print-on-change and stability parameters --------
        self.declare_parameter('log_on_change_only', True)
        self._log_on_change_only = bool(self.get_parameter('log_on_change_only').value)
        self._last_logged_label = None

        self.declare_parameter('min_stable_frames', 3)
        self._min_stable_frames = int(self.get_parameter('min_stable_frames').value)
        self._prev_label_for_stability = None
        self._same_count = 0

        import inspect, sys
        self.get_logger().info(
            f"CODE_PATH={inspect.getfile(sys.modules[__name__])} | "
            f"log_on_change_only={self._log_on_change_only} | "
            f"min_stable_frames={self._min_stable_frames}"
        )

        # ---------------- Parameters ----------------
        self.declare_parameter('image_topic', '/camera/camera/color/image_rect_raw')
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.60)
        self.declare_parameter('device', 'auto')
        self.declare_parameter('publish_debug', True)

        # ----- TTS parameters -----
        self.declare_parameter('enable_tts', False)
        self.declare_parameter('tts_engine', 'gtts') # Default changed to gtts
        self.declare_parameter('tts_on_change_only', True)
        # Default now says spoken form; you can override on CLI safely
        self.declare_parameter('tts_phrase_template')
        self.declare_parameter('tts_rate', 180)
        self.declare_parameter('tts_volume', 1.0)
        self.declare_parameter('tts_voice_id', '')
        self.declare_parameter('gtts_language', 'en')

        image_topic: str = self.get_parameter('image_topic').get_parameter_value().string_value
        model_path_param: str = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold: float = float(self.get_parameter('confidence_threshold').value)
        device_str: str = self.get_parameter('device').get_parameter_value().string_value
        self.publish_debug: bool = bool(self.get_parameter('publish_debug').value)

        # Resolve model path
        if model_path_param and os.path.exists(model_path_param):
            model_path = model_path_param
        else:
            share_dir = get_package_share_directory('card_detector')
            model_path = os.path.join(share_dir, 'models', 'best.pt')

        if not os.path.exists(model_path):
            msg = (f"Model file not found at '{model_path}'. "
                   f"Place your model at '.../card_detector/models/best.pt' "
                   f"or set the 'model_path' parameter.")
            self.get_logger().fatal(msg)
            raise FileNotFoundError(msg)

        # Load model
        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

        self._predict_kwargs = {}
        if device_str and device_str.lower() != 'auto':
            self._predict_kwargs['device'] = device_str

        self.bridge = CvBridge()

        # ---------------- Subscribers & Publishers ----------------
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, qos_profile_sensor_data
        )
        self.detection_pub = self.create_publisher(String, '/card_detections', 10)
        self.debug_image_pub = self.create_publisher(Image, '/card_detections/debug_image', 10)

        self._busy = False

        # ------------- TTS init -------------
        self._speaker: Optional[Speaker] = None
        self._last_spoken_label: Optional[str] = None
        try:
            if bool(self.get_parameter('enable_tts').value):
                # We only care about the gtts language parameter now
                glang = self.get_parameter('gtts_language').get_parameter_value().string_value or 'en'

                self._speaker = Speaker(
                    logger=self.get_logger(),
                    engine='gtts', # Hard-code to gtts
                    gtts_lang=glang
                )
                self._tts_on_change_only = bool(self.get_parameter('tts_on_change_only').value)
                self._tts_template = (self.get_parameter('tts_phrase_template')
                                      .get_parameter_value().string_value) or '{spoken}'
            else:
                self._tts_on_change_only = True
                self._tts_template = '{spoken}'
        except Exception as e:
            self.get_logger().warning(f"TTS initialization skipped due to error: {e}")
            self._speaker = None
            self._tts_on_change_only = True
            self._tts_template = '{spoken}'

        self.get_logger().info(
            f"Subscribed to: {image_topic} | conf>={self.conf_threshold:.2f} | "
            f"device={device_str} | debug={self.publish_debug} | "
            f"tts={'on (gTTS only)' if self._speaker else 'off'}"
        )

    def image_callback(self, msg: Image) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False, **self._predict_kwargs)

            best_label: str = ''
            best_conf: float = 0.0

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                names = getattr(results[0], 'names', None) or getattr(self.model, 'names', {})
                for conf_tensor, cls_tensor in zip(boxes.conf, boxes.cls):
                    conf = float(conf_tensor.item())
                    if conf >= best_conf:
                        best_conf = conf
                        cid = int(cls_tensor.item())
                        if isinstance(names, dict):
                            best_label = names.get(cid, str(cid))
                        elif isinstance(names, (list, tuple)) and 0 <= cid < len(names):
                            best_label = names[cid]
                        else:
                            best_label = str(cid)

            if best_conf >= self.conf_threshold and best_label:
                out = String()
                out.data = best_label.strip()
                self.detection_pub.publish(out)

                if best_label == self._prev_label_for_stability:
                    self._same_count += 1
                else:
                    self._same_count = 1
                    self._prev_label_for_stability = best_label

                if self._same_count >= self._min_stable_frames:
                    if (not self._log_on_change_only) or (best_label != self._last_logged_label):
                        self.get_logger().info(f"Detected: {best_label} (conf={best_conf:.2f})")
                        self._last_logged_label = best_label

                    if self._speaker is not None:
                        should_say = True
                        if self.get_parameter('tts_on_change_only').value and best_label == self._last_spoken_label:
                            should_say = False
                        if should_say:
                            spoken, rank_name, suit_name = to_spoken_card(best_label)
                            tmpl = (self._tts_template or '{spoken}')
                            phrase = tmpl.format(
                                label=best_label,
                                spoken=spoken or best_label,
                                rank=rank_name or '',
                                suit=suit_name or ''
                            )
                            #self._speaker.say(phrase)
                            self._last_spoken_label = best_label
            else:
                self._prev_label_for_stability = None
                self._same_count = 0

            if self.publish_debug and len(results) > 0:
                annotated = results[0].plot()
                dbg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                dbg.header = msg.header
                self.debug_image_pub.publish(dbg)

        except Exception as e:
            self.get_logger().error(f"Inference error: {e}")
        finally:
            self._busy = False

    def destroy_node(self) -> bool:
        try:
            if self._speaker is not None:
                self._speaker.shutdown()
        except Exception:
            pass
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = CardDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()