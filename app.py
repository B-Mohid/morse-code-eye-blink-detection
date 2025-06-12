import cv2
import time
import dlib
import numpy as np
from scipy.spatial import distance as dist
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import base64
import os
import sys
from collections import deque
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import json
from datetime import datetime, timedelta
import traceback
import psutil
import gc
from functools import wraps
import signal

# Configure comprehensive logging for production readiness
# Logs will go to 'morse_app.log' file and also to stdout
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('morse_app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__)
app.config.update({
    # SECRET_KEY is crucial for security. Use a strong, unique key in production.
    # It's recommended to load this from an environment variable.
    'SECRET_KEY': os.environ.get('SECRET_KEY', 'a_strong_secret_key_for_production_replace_this'),
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16MB max file size for uploads
    'SEND_FILE_MAX_AGE_DEFAULT': 31536000,  # 1 year cache for static files
})

# SocketIO with production settings
socketio = SocketIO(
    app,
    # For production, replace "*" with specific origins (e.g., "https://your-domain.com")
    cors_allowed_origins="*",
    # 'async_mode' can be 'threading', 'eventlet', or 'gevent'.
    # 'threading' is good for simple setups, but 'eventlet' or 'gevent'
    # are recommended for higher concurrency and performance in production.
    async_mode='threading',
    logger=False,  # Disable SocketIO internal logging for production to reduce noise
    engineio_logger=False, # Disable Engine.IO internal logging
    ping_timeout=60,       # Increased ping timeout (seconds)
    ping_interval=25       # Increased ping interval (seconds)
)

class ApplicationError(Exception):
    """Custom application error class for specific operational failures."""
    pass

class ConfigManager:
    """
    Manages application configuration, allowing dynamic updates and persistence.
    Includes default values and basic validation.
    """

    DEFAULT_CONFIG = {
        'eye_ar_thresh': 0.22,              # Eye Aspect Ratio threshold for blink detection
        'open_eye_time_thresh': 0.35,       # Time (s) for eyes to be open to consider a pause/space
        'consec_frames': 3,                 # Number of consecutive frames below EAR threshold for a blink
        'frame_skip_rate': 2,               # Process every Nth frame to reduce load
        'max_queue_size': 10,               # Maximum number of frames in the processing queue
        'fps_update_interval': 1.0,         # Interval (s) to update FPS metric
        'auto_calibration_frames': 30,      # Number of frames for initial auto-calibration of EAR
        'max_blink_duration': 1.0,          # Maximum duration (s) of a blink (to filter out long closures)
        'min_blink_duration': 0.08,         # Minimum duration (s) of a blink (to filter out noise)
        'performance_monitoring': True      # Enable/disable system performance monitoring
    }

    def __init__(self):
        self.config = self.DEFAULT_CONFIG.copy()
        self._load_config() # Load configuration from file on initialization

    def _load_config(self):
        """
        Loads configuration from a 'config.json' file if it exists.
        Updates default configuration with values from the file.
        """
        try:
            if os.path.exists('config.json'):
                with open('config.json', 'r') as f:
                    user_config = json.load(f)
                    # Update only known configuration keys
                    for key, value in user_config.items():
                        if key in self.DEFAULT_CONFIG:
                            self.config[key] = value
                logger.info("Configuration loaded from config.json")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding config.json, file might be corrupted: {e}")
        except Exception as e:
            logger.warning(f"Could not load config.json: {e}. Using default configuration.")

    def get(self, key, default=None):
        """Retrieves a configuration value by key."""
        return self.config.get(key, default)

    def set(self, key, value):
        """
        Sets a configuration value. Includes basic type validation
        to ensure consistency with default types.
        """
        if key in self.DEFAULT_CONFIG:
            expected_type = type(self.DEFAULT_CONFIG[key])
            if not isinstance(value, expected_type) and not (isinstance(value, (int, float)) and isinstance(expected_type, (int, float))):
                logger.warning(f"Attempted to set config key '{key}' with incorrect type. Expected {expected_type}, got {type(value)}.")
                return False # Indicate failure due to type mismatch

            self.config[key] = value
            logger.info(f"Config updated: {key} = {value}")
            return True
        else:
            logger.warning(f"Attempted to set unknown config key: {key}")
            return False # Indicate failure due to unknown key

    def save_config(self):
        """Saves the current configuration to 'config.json'."""
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=2) # Pretty print JSON
            logger.info("Configuration saved to config.json")
        except Exception as e:
            logger.error(f"Could not save config to config.json: {e}")

class PerformanceMonitor:
    """
    Monitors system performance metrics (FPS, CPU, Memory) and processing times.
    Provides thread-safe access to metrics.
    """

    def __init__(self):
        self.metrics = {
            'fps': 0,
            'cpu_percent': 0,
            'memory_percent': 0,
            'frames_processed': 0,
            'frames_dropped': 0,
            'processing_time_avg_ms': 0, # Average frame processing time in milliseconds
            'last_update': time.time()
        }
        self.processing_times = deque(maxlen=100) # Store last 100 processing times for average
        self.lock = threading.Lock() # Lock for thread-safe access to metrics

    def update_fps(self, fps):
        """Updates the frames per second metric."""
        with self.lock:
            self.metrics['fps'] = fps

    def record_processing_time(self, processing_time):
        """
        Records a single frame's processing time and updates the average.
        @param processing_time: Time taken to process the frame in seconds.
        """
        with self.lock:
            self.processing_times.append(processing_time)
            if self.processing_times:
                self.metrics['processing_time_avg_ms'] = (sum(self.processing_times) / len(self.processing_times)) * 1000 # Convert to ms

    def update_system_metrics(self):
        """Updates system resource metrics (CPU and Memory usage)."""
        try:
            with self.lock:
                self.metrics['cpu_percent'] = psutil.cpu_percent()
                self.metrics['memory_percent'] = psutil.virtual_memory().percent
                self.metrics['last_update'] = time.time()
        except Exception as e:
            logger.warning(f"Could not update system metrics: {e}")

    def get_metrics(self):
        """Returns a copy of the current performance metrics."""
        with self.lock:
            return self.metrics.copy()

class MorseCodeProcessor:
    """
    Handles encoding text to Morse code, decoding Morse code to text,
    and interpreting decoded text for meaningful outputs.
    """

    def __init__(self):
        # Standard Morse code dictionary mapping Morse sequences to English characters
        self.MORSE_CODE_DICT = {
            '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
            '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
            '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
            '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
            '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
            '--..': 'Z', '-----': '0', '.----': '1', '..---': '2', '...--': '3',
            '....-': '4', '.....': '5', '-....': '6', '--...': '7', '---..': '8',
            '----.': '9', '/': ' ', '--..--': ',', '.-.-.-': '.', '..--..': '?',
            '-..-.': '/', '-....-': '-', '-.--.': '(', '-.--.-': ')', '.--.-.': '@',
            '---...': ':', '-.-.-.': ';', '-...-': '=', '.-.-.': '+', '.-..-.': '"',
            '...-.-': '!', '.-.-..': '&', '.-..-': '_', '.-..-': '$', '.-...': '#',
            '..-.--': '%' # Added a few common symbols, adjust as needed
        }

        # Reverse dictionary for encoding text to Morse code
        self.TEXT_TO_MORSE_DICT = {v: k for k, v in self.MORSE_CODE_DICT.items()}
        self.TEXT_TO_MORSE_DICT[' '] = '/' # Explicitly define space as '/' for encoding

        # Enhanced interpretation patterns for decoded text with associated messages and priority
        self.INTERPRETATION_PATTERNS = {
            'SOS': {'message': 'Emergency signal detected! 🚨', 'priority': 'critical'},
            'MAYDAY': {'message': 'Emergency mayday signal! 🚨', 'priority': 'critical'},
            'HELP': {'message': 'Request for assistance 🆘', 'priority': 'high'},
            'HELLO': {'message': 'Greeting received 👋', 'priority': 'normal'},
            'HI': {'message': 'Greeting received 👋', 'priority': 'normal'},
            'OK': {'message': 'Acknowledgment received ✓', 'priority': 'normal'},
            'YES': {'message': 'Affirmative received ✓', 'priority': 'normal'},
            'NO': {'message': 'Negative received ✗', 'priority': 'normal'},
            'STOP': {'message': 'Stop signal received 🛑', 'priority': 'high'},
            'GO': {'message': 'Go signal received 🟢', 'priority': 'normal'},
            'WAIT': {'message': 'Wait signal received ⏳', 'priority': 'normal'},
            'THANK': {'message': 'Thank you received 🙏', 'priority': 'normal'},
            'THANKS': {'message': 'Thank you received 🙏', 'priority': 'normal'},
            'SORRY': {'message': 'Apology received 💙', 'priority': 'normal'},
            'BYE': {'message': 'Goodbye received 👋', 'priority': 'normal'},
            'GOODBYE': {'message': 'Goodbye received 👋', 'priority': 'normal'}
        }

        # Statistics tracking for decoding operations
        self.stats = {
            'total_decoded': 0,
            'successful_decodes': 0,
            'error_count': 0,
            'start_time': time.time()
        }

    def decode_morse(self, morse_code):
        """
        Decodes a Morse code string into human-readable text.
        Handles individual character decoding and word separation.
        @param morse_code: String of Morse code (e.g., ".... . .-.. .-.. --- / .-- --- .-. .-.. -..")
        @return: Decoded text string.
        """
        if not morse_code or not isinstance(morse_code, str) or not morse_code.strip():
            return ""

        try:
            self.stats['total_decoded'] += 1
            morse_code = morse_code.strip()

            # Split by single spaces to get individual Morse characters or word separators
            chunks = morse_code.split(' ')
            decoded_message_parts = []

            for chunk in chunks:
                if not chunk: # Skip empty chunks from multiple spaces
                    continue

                if chunk in self.MORSE_CODE_DICT:
                    decoded_message_parts.append(self.MORSE_CODE_DICT[chunk])
                else:
                    # Log unknown chunks for debugging, but don't break the decoding
                    logger.debug(f"Unknown Morse chunk encountered during decoding: '{chunk}'")
                    # Optionally, append a placeholder for unknown sequences
                    # decoded_message_parts.append('[UNK]')

            result = ''.join(decoded_message_parts).strip()

            if result:
                self.stats['successful_decodes'] += 1

            return result

        except Exception as e:
            self.stats['error_count'] += 1
            logger.error(f"Error decoding Morse code '{morse_code}': {e}", exc_info=True)
            return "DECODE_ERROR"

    def encode_text_to_morse(self, text):
        """
        Encodes a given text string into its Morse code equivalent.
        Handles character and word separation.
        @param text: Input text string (e.g., "Hello World")
        @return: Morse code string (e.g., ".... . .-.. .-.. --- / .-- --- .-. .-.. -..")
        """
        if not text or not isinstance(text, str):
            return ""

        try:
            morse_output = []
            text = text.upper().strip() # Convert to uppercase for dictionary lookup

            for char in text:
                if char in self.TEXT_TO_MORSE_DICT:
                    morse_output.append(self.TEXT_TO_MORSE_DICT[char])
                elif char == ' ':
                    morse_output.append('/') # Word space as per frontend instruction
                else:
                    logger.debug(f"Unknown character '{char}' for Morse encoding. Skipping.")
                    # Optionally, you might want to signal an error or use a placeholder
                    # morse_output.append('[UNK]')

            return ' '.join(morse_output)

        except Exception as e:
            logger.error(f"Error encoding text to Morse '{text}': {e}", exc_info=True)
            return ""

    def interpret_text(self, text):
        """
        Provides a meaningful interpretation of the decoded text based on predefined patterns.
        @param text: Decoded text string.
        @return: Dictionary containing interpretation message, priority, and context.
        """
        if not text or text == "DECODE_ERROR":
            return {"message": "Awaiting input...", "priority": "normal", "context": "waiting"}

        text_upper = text.upper().strip()

        # Check for exact pattern matches first for highest confidence
        for pattern, info in self.INTERPRETATION_PATTERNS.items():
            if pattern in text_upper: # Using 'in' for substring match, could be exact match for higher precision
                return {
                    "message": info['message'],
                    "priority": info['priority'],
                    "context": "recognized_pattern",
                    "pattern": pattern
                }

        # Check for emergency-related terms (broader search)
        emergency_terms = ['EMERGENCY', 'URGENT', 'CRITICAL', 'DANGER', 'FIRE', 'ACCIDENT', 'HELP']
        if any(term in text_upper for term in emergency_terms):
            return {
                "message": f"Emergency message detected: '{text}' 🚨",
                "priority": "critical",
                "context": "emergency"
            }
        
        # General communication detection
        if len(text_upper) > 0:
            return {
                "message": f"Message received: '{text}'",
                "priority": "normal",
                "context": "general"
            }
        
        # Fallback if no specific interpretation
        return {"message": "Awaiting input...", "priority": "normal", "context": "waiting"}


    def get_statistics(self):
        """Returns current processing statistics for Morse code operations."""
        uptime = time.time() - self.stats['start_time']
        success_rate = 0
        if self.stats['total_decoded'] > 0:
            success_rate = (self.stats['successful_decodes'] / self.stats['total_decoded']) * 100

        return {
            'total_decoded': self.stats['total_decoded'],
            'successful_decodes': self.stats['successful_decodes'],
            'error_count': self.stats['error_count'],
            'success_rate': round(success_rate, 2),
            'uptime_hours': round(uptime / 3600, 2)
        }

    def reset_statistics(self):
        """Resets all Morse code processing statistics."""
        self.stats = {
            'total_decoded': 0,
            'successful_decodes': 0,
            'error_count': 0,
            'start_time': time.time()
        }

class EyeBlinkDetector:
    """
    Detects eye blinks from video frames, calculates Eye Aspect Ratio (EAR),
    and classifies blinks into Morse code dots and dashes.
    Includes auto-calibration and robust landmark detection.
    """

    def __init__(self, config_manager):
        self.config = config_manager # Reference to the global ConfigManager
        
        self.face_cascade = None
        self.predictor = None
        self._initialize_cv_components() # Initialize OpenCV and dlib components

        # Detection parameters, loaded from config and potentially auto-calibrated
        self.eye_ar_thresh = self.config.get('eye_ar_thresh')
        self.open_eye_time_thresh = self.config.get('open_eye_time_thresh')
        self.consec_frames = self.config.get('consec_frames')
        self.max_blink_duration = self.config.get('max_blink_duration')
        self.min_blink_duration = self.config.get('min_blink_duration')

        # Calibration system variables
        self.ear_history = deque(maxlen=self.config.get('auto_calibration_frames'))
        self.blink_durations = deque(maxlen=20) # History of recent blink durations for adaptive thresholding
        self.calibration_complete = False
        self.calibration_start_time = time.time()
        
        # State tracking for blink logic
        self.counter = 0            # Frame counter for consecutive frames below EAR threshold
        self.total_blinks = 0       # Total blinks detected in a session
        self.blink_start_time = None # Timestamp when an eye closure began
        self.open_eye_start_time = None # Timestamp when eyes were last open (for space detection)
        self.last_char = ""         # Stores the last Morse character generated ('.' or '-', or ' ')

        # Performance optimization: frame skipping
        self.frame_skip_counter = 0
        self.process_every_n_frames = self.config.get('frame_skip_rate')

        # Error recovery: track consecutive errors in detection
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10 # Max errors before logging a critical message

        # Statistics for detection accuracy and performance
        self.detection_stats = {
            'faces_detected': 0,
            'faces_failed': 0,
            'blinks_detected': 0,
            'false_positives': 0, # Could be improved with more sophisticated blink validation
            'processing_errors': 0
        }
        logger.info("EyeBlinkDetector initialized.")

    def _initialize_cv_components(self):
        """
        Initializes OpenCV's Haar Cascade for face detection and dlib's facial landmark predictor.
        Includes robust path checking for the dlib model file.
        """
        try:
            # Initialize face detector using a Haar Cascade classifier
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                # Fallback to a common system path if not found in cv2 data
                cascade_path = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
                if not os.path.exists(cascade_path):
                    raise ApplicationError(f"Haar cascade file not found: {cv2.data.haarcascades} and {cascade_path}")

            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise ApplicationError(f"Failed to load Haar cascade classifier from {cascade_path}")
            logger.info(f"Haar cascade loaded from: {cascade_path}")

            # Initialize dlib shape predictor for facial landmarks
            # Try multiple common paths for the dlib model file
            predictor_paths = [
                "shape_predictor_68_face_landmarks.dat", # Current directory
                os.path.join(os.path.dirname(__file__), "models", "shape_predictor_68_face_landmarks.dat"), # In a 'models' subfolder
                "/opt/models/shape_predictor_68_face_landmarks.dat", # Common for Docker/Linux deployments
                os.path.join(os.path.expanduser("~"), ".dlib_models", "shape_predictor_68_face_landmarks.dat") # User home directory
            ]

            predictor_path = None
            for path in predictor_paths:
                if os.path.exists(path):
                    predictor_path = path
                    break

            if not predictor_path:
                raise ApplicationError(
                    "Dlib 'shape_predictor_68_face_landmarks.dat' not found. "
                    "Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2, "
                    "extract it, and place it in the application's root directory or a 'models' subfolder."
                )

            self.predictor = dlib.shape_predictor(predictor_path)
            logger.info(f"Dlib predictor loaded from: {predictor_path}")

        except Exception as e:
            logger.critical(f"Failed to initialize CV components: {e}", exc_info=True)
            raise ApplicationError(f"Critical: CV initialization failed. Please ensure OpenCV and dlib models are correctly installed and accessible. Error: {e}")

    def calculate_ear(self, eye_landmarks):
        """
        Calculates the Eye Aspect Ratio (EAR) given 6 eye landmarks.
        EAR is a metric used to quantify eye closure.
        @param eye_landmarks: A list of 6 (x, y) tuples representing the eye landmarks.
        @return: The calculated EAR value. Returns a default if an error occurs.
        """
        try:
            if len(eye_landmarks) != 6:
                logger.warning("Invalid number of eye landmarks provided for EAR calculation. Expected 6 points.")
                return 0.3 # A sensible default if input is malformed

            # Compute the euclidean distances between the two sets of vertical eye landmarks
            A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])

            # Compute the euclidean distance between the horizontal eye landmark
            C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])

            if C == 0:  # Prevent division by zero if horizontal distance is zero
                return 0.3 # Indicate a flat line, likely fully closed or bad detection

            # Calculate the Eye Aspect Ratio
            ear = (A + B) / (2.0 * C)

            # Sanity check for extremely unrealistic EAR values
            if not 0.0 <= ear <= 0.5: # Realistic EAR values typically range from 0.0 to 0.4
                logger.debug(f"Calculated EAR out of typical range: {ear}. Returning default.")
                return 0.3

            return ear
        except Exception as e:
            logger.debug(f"Error calculating EAR: {e}", exc_info=True)
            return 0.3 # Return a default value on calculation error

    def auto_calibrate(self, ear_value):
        """
        Performs automatic calibration of the Eye Aspect Ratio (EAR) threshold.
        Collects EAR samples and statistically determines a robust threshold for blink detection.
        This helps adapt to different lighting conditions and user physiognomy.
        @param ear_value: The current EAR value from an open eye.
        """
        if self.calibration_complete:
            return

        self.ear_history.append(ear_value)

        # Ensure enough samples are collected for reliable calibration
        if len(self.ear_history) < self.config.get('auto_calibration_frames'):
            # Log progress if calibration is taking time
            if len(self.ear_history) % 10 == 0:
                logger.info(f"Calibrating: {len(self.ear_history)}/{self.config.get('auto_calibration_frames')} samples collected.")
            return

        try:
            ear_array = np.array(list(self.ear_history))

            # Remove outliers using the Interquartile Range (IQR) method
            Q1 = np.percentile(ear_array, 25)
            Q3 = np.percentile(ear_array, 75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            filtered_ears = ear_array[(ear_array >= lower_bound) & (ear_array <= upper_bound)]

            if len(filtered_ears) < self.config.get('auto_calibration_frames') * 0.7:
                # Not enough clean data after filtering outliers, continue collecting
                logger.warning("Not enough valid EAR samples after outlier filtering for calibration. Continuing collection.")
                self.ear_history.clear() # Clear and restart if data quality is too low
                return

            mean_ear = np.mean(filtered_ears)
            std_ear = np.std(filtered_ears)

            # Set the EAR threshold. This formula places the threshold
            # significantly below the mean open-eye EAR, using standard deviation
            # to account for variability. `max/min` ensure it stays within a reasonable range.
            new_threshold = max(0.15, min(0.35, mean_ear - 2.5 * std_ear))
            
            # Additional validation: ensure threshold is not too close to the mean
            if new_threshold > mean_ear * 0.8: # If threshold is > 80% of mean open-eye EAR, it's too high
                new_threshold = mean_ear * 0.7 # Adjust to a safer lower value

            self.eye_ar_thresh = new_threshold
            self.calibration_complete = True
            calibration_time = time.time() - self.calibration_start_time
            logger.info(f"Auto-calibration complete in {calibration_time:.1f}s.")
            logger.info(f"EAR threshold set to: {self.eye_ar_thresh:.3f} (based on {len(filtered_ears)} samples, mean: {mean_ear:.3f}, std: {std_ear:.3f}).")
            # Update the config manager so the new threshold is saved and broadcasted
            self.config.set('eye_ar_thresh', round(self.eye_ar_thresh, 3))

        except Exception as e:
            logger.error(f"Error during auto-calibration: {e}", exc_info=True)
            # Reset calibration to retry if a significant error occurs
            self.ear_history.clear()
            self.calibration_complete = False
            self.calibration_start_time = time.time()


    def detect_face_and_eyes(self, frame):
        """
        Detects faces and extracts eye landmarks from a given video frame.
        Applies frame skipping for performance optimization and includes error handling.
        @param frame: The OpenCV image frame (numpy array).
        @return: (left_eye_coords, right_eye_coords, face_rect, detection_success)
                 Returns None/False if detection fails or frame is skipped.
        """
        try:
            # Apply frame skipping to reduce processing load
            self.frame_skip_counter += 1
            if self.frame_skip_counter % self.process_every_n_frames != 0:
                return None, None, None, False

            if frame is None or frame.size == 0:
                logger.warning("Received an empty or invalid frame for detection.")
                return None, None, None, False

            # Optimize frame size for faster processing by scaling down if too large
            height, width = frame.shape[:2]
            scale_factor = 1.0
            if width > 640: # If width is greater than 640 pixels
                scale_factor = 640 / width
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray) # Enhance contrast for better detection

            # Detect faces using the Haar cascade classifier
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,     # How much the image size is reduced at each image scale
                minNeighbors=5,      # How many neighbors each candidate rectangle should have to retain it
                minSize=(60, 60),    # Minimum possible object size. Objects smaller than that are ignored.
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            if len(faces) == 0: # Corrected from `len(faces == 0)`
                self.detection_stats['faces_failed'] += 1
                return None, None, None, False

            # Select the largest detected face (assuming only one person is being tracked)
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            # Basic validation of face size to avoid very small or malformed detections
            if w < 60 or h < 60:
                logger.debug(f"Detected face too small (w={w}, h={h}). Skipping.")
                self.detection_stats['faces_failed'] += 1
                return None, None, None, False

            self.detection_stats['faces_detected'] += 1

            # Use dlib's shape predictor to get 68 facial landmarks
            rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = self.predictor(gray, rect)
            
            # Extract left and right eye coordinates (landmarks 36-41 for left, 42-47 for right)
            left_eye = []
            right_eye = []
            
            for i in range(36, 42): # Left eye
                point = landmarks.part(i)
                # Scale landmarks back to original frame size for display consistency
                left_eye.append((int(point.x / scale_factor), int(point.y / scale_factor)))
            
            for i in range(42, 48): # Right eye
                point = landmarks.part(i)
                # Scale landmarks back to original frame size
                right_eye.append((int(point.x / scale_factor), int(point.y / scale_factor)))
            
            # Scale face rectangle back to original frame size
            face_rect_original_scale = (
                int(x / scale_factor),
                int(y / scale_factor),
                int(w / scale_factor),
                int(h / scale_factor)
            )
            
            self.consecutive_errors = 0 # Reset error counter on successful detection
            
            return left_eye, right_eye, face_rect_original_scale, True
            
        except Exception as e:
            self.consecutive_errors += 1
            self.detection_stats['processing_errors'] += 1
            
            if self.consecutive_errors < self.max_consecutive_errors:
                logger.debug(f"Face/eye detection error (#{self.consecutive_errors}): {e}", exc_info=True)
            else:
                logger.error(f"Too many consecutive face/eye detection errors ({self.consecutive_errors}). Potential camera/input issue. Last error: {e}", exc_info=True)
                # Consider emitting a critical error to the client or restarting detector here
            
            return None, None, None, False
    
    def process_blink(self, left_eye, right_eye, capture_active):
        """
        Processes eye landmarks to detect blinks and classify them as Morse code
        dots or dashes. Also handles detection of pauses for spaces.
        @param left_eye: List of coordinates for the left eye.
        @param right_eye: List of coordinates for the right eye.
        @param capture_active: Boolean indicating if capture is currently active.
        @return: (morse_char_added, blink_detected_this_frame)
                 morse_char_added: "dot", "dash", "space", or None
                 blink_detected_this_frame: Boolean if a blink was just registered.
        """
        morse_char_added = None
        blink_detected_this_frame = False

        if not capture_active:
            return None, False

        if not left_eye or not right_eye:
            # If eyes are not detected, reset blink counter and return
            self.counter = 0
            self.blink_start_time = None
            return None, False

        try:
            # Calculate average Eye Aspect Ratio (EAR) for both eyes
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Auto-calibrate if still in calibration phase
            if not self.calibration_complete:
                self.auto_calibrate(avg_ear)
                return None, False # Do not process blinks until calibration is complete
            
            current_time = time.time()
            
            # Logic for eye closure (blink) detection
            if avg_ear < self.eye_ar_thresh:
                # If eyes just started closing, record start time
                if self.blink_start_time is None:
                    self.blink_start_time = current_time
                self.counter += 1 # Increment consecutive frames counter

                # Check for "space" (pause between characters/words) if eyes were open for long enough
                if self.open_eye_start_time is not None:
                    open_duration = current_time - self.open_eye_start_time
                    # If open duration exceeds threshold AND last character was not already a space,
                    # register a space. This prevents multiple spaces for one long pause.
                    if (open_duration > self.open_eye_time_thresh and
                        self.last_char not in [" ", ""]): # Added condition to avoid multiple spaces
                        morse_char_added = "space"
                        self.last_char = " "
                        self.open_eye_start_time = None # Reset open eye timer after adding space

            # Logic for eye opening (end of blink) detection
            else:
                # If eyes were closed for enough consecutive frames and a blink started
                if self.counter >= self.consec_frames and self.blink_start_time is not None:
                    blink_duration = current_time - self.blink_start_time

                    # Validate blink duration to filter noise (too short) or non-blinks (too long closures)
                    if self.min_blink_duration <= blink_duration <= self.max_blink_duration:
                        self.blink_durations.append(blink_duration) # Add to history for adaptive threshold

                        # Adaptive threshold for dot/dash classification
                        # Use median of recent blink durations for robustness against outliers
                        if len(self.blink_durations) >= 5: # Need at least 5 samples for a meaningful median
                            recent_durations = list(self.blink_durations) # Get all recent durations
                            median_duration = np.median(recent_durations)
                            # Threshold for distinguishing dot from dash (e.g., 2 times the median dot duration)
                            # Using a fixed ratio (e.g., 1.5-2.0) of median blink duration can be effective.
                            dot_dash_threshold = min(0.30, max(0.12, median_duration * 1.5)) # Ensure sensible range

                            # Update config if this adaptive threshold is significantly different
                            if abs(dot_dash_threshold - self.config.get('open_eye_time_thresh')) > 0.05:
                                self.config.set('open_eye_time_thresh', round(dot_dash_threshold, 2))
                                self.open_eye_time_thresh = dot_dash_threshold # Update local variable as well
                        else:
                            dot_dash_threshold = 0.25 # Default threshold if not enough data for adaptive

                        # Classify blink as dot or dash
                        if blink_duration < dot_dash_threshold:
                            morse_char_added = "dot"
                            self.last_char = "."
                        else:
                            morse_char_added = "dash"
                            self.last_char = "-"

                        self.total_blinks += 1
                        self.detection_stats['blinks_detected'] += 1
                        blink_detected_this_frame = True # Mark that a blink has just finished

                    else:
                        # Blink duration out of acceptable range, consider it noise or a non-blink event
                        logger.debug(f"Invalid blink duration: {blink_duration:.3f}s (expected {self.min_blink_duration}-{self.max_blink_duration}s). Ignoring.")
                        self.detection_stats['false_positives'] += 1 # Or filter as "ignored"
                
                # Reset blink-related counters since eyes are now open
                self.counter = 0
                self.blink_start_time = None
                
                # Record time when eyes became open (for subsequent space detection)
                if self.open_eye_start_time is None:
                    self.open_eye_start_time = current_time
            
            return morse_char_added, blink_detected_this_frame
            
        except Exception as e:
            logger.error(f"Error processing blink: {e}", exc_info=True)
            self.detection_stats['processing_errors'] += 1
            return None, False
    
    def get_calibration_status(self):
        """Returns the current status of the auto-calibration process."""
        if self.calibration_complete:
            return {
                'status': 'complete',
                'threshold': round(self.eye_ar_thresh, 3),
                'samples': len(self.ear_history),
                'message': 'Calibration complete - ready for detection'
            }
        else:
            progress = (len(self.ear_history) / max(1, self.config.get('auto_calibration_frames')))
            return {
                'status': 'in_progress',
                'progress': round(progress * 100, 1),
                'samples': len(self.ear_history),
                'needed': self.config.get('auto_calibration_frames'),
                'message': f'Calibrating... {len(self.ear_history)} samples collected'
            }
    
    def get_statistics(self):
        """Returns detailed statistics about eye blink detection."""
        total_attempts = self.detection_stats['faces_detected'] + self.detection_stats['faces_failed']
        face_detection_rate = 0
        if total_attempts > 0:
            face_detection_rate = (self.detection_stats['faces_detected'] / total_attempts) * 100
        
        return {
            'total_blinks': self.total_blinks,
            'faces_detected': self.detection_stats['faces_detected'],
            'face_detection_rate': round(face_detection_rate, 1),
            'processing_errors': self.detection_stats['processing_errors'],
            'calibration_complete': self.calibration_complete,
            'current_ear_threshold': round(self.eye_ar_thresh, 3),
            'current_open_eye_time_threshold': round(self.open_eye_time_thresh, 3)
        }
    
    def reset_statistics(self):
        """Resets all detection-related statistics and calibration state."""
        logger.info("Resetting EyeBlinkDetector statistics and calibration state.")
        self.detection_stats = {key: 0 for key in self.detection_stats}
        self.total_blinks = 0
        self.ear_history.clear()
        self.blink_durations.clear()
        self.calibration_complete = False
        self.calibration_start_time = time.time()
        self.counter = 0
        self.blink_start_time = None
        self.open_eye_start_time = None
        self.last_char = ""
        # Reset thresholds to default until new calibration completes
        self.eye_ar_thresh = self.config.get('eye_ar_thresh')
        self.open_eye_time_thresh = self.config.get('open_eye_time_thresh')


class FrameProcessor:
    """
    Manages the processing of incoming video frames for eye blink detection
    and subsequent Morse code conversion. Runs in a separate thread.
    """

    def __init__(self, morse_processor, blink_detector, config_manager, performance_monitor):
        self.morse_processor = morse_processor
        self.blink_detector = blink_detector
        self.config = config_manager
        self.performance_monitor = performance_monitor
        
        # Thread-safe queue for incoming frames
        self.frame_queue = queue.Queue(maxsize=self.config.get('max_queue_size'))
        
        # State management variables
        self.capture_active = False # Flag to control frame processing
        self.morse_code = ""        # Accumulates Morse code for current character/word
        self.decoded_text = ""      # Accumulates full decoded text
        self.meaningful_output = {} # Interpretation of the decoded text
        self.processing_active = True # Flag to control the processing thread lifecycle
        
        # Performance tracking for the processing thread itself
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0 # FPS of frames processed by backend
        self.frames_processed = 0
        self.frames_dropped = 0
        
        # Setup and start the dedicated frame processing thread
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
        # Health monitoring: track last time a frame was successfully added
        self.last_frame_time = time.time()
        self.health_check_interval = 5.0 # Seconds after which to log a warning if no frames are received
        
        logger.info("FrameProcessor initialized and processing thread started.")

    def add_frame(self, frame):
        """
        Adds a raw video frame to the processing queue.
        If the queue is full, the oldest frame is dropped to prevent buildup and latency.
        @param frame: The OpenCV image frame (numpy array) to add.
        @return: True if frame was added, False if dropped.
        """
        try:
            self.last_frame_time = time.time() # Update last received frame time
            
            if self.frame_queue.full():
                # Drop the oldest frame to make space for the new one (prioritize fresh data)
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                    logger.debug("Frame queue full, dropped oldest frame.")
                except queue.Empty:
                    # Should theoretically not happen if .full() was true
                    logger.warning("Queue unexpectedly empty while trying to drop old frame.")
                    pass
            
            self.frame_queue.put_nowait(frame) # Add new frame to queue
            return True
            
        except Exception as e:
            logger.error(f"Error adding frame to queue: {e}", exc_info=True)
            return False
    
    def _process_frames(self):
        """
        The main loop for the frame processing thread.
        Continuously pulls frames from the queue, processes them, and emits results.
        Includes robust error handling to keep the thread alive.
        """
        logger.info("Frame processing thread started.")
        
        while self.processing_active:
            try:
                # Attempt to get a frame from the queue with a short timeout
                # This allows the thread to periodically check self.processing_active
                frame = self.frame_queue.get(timeout=0.1)
                
                # If a frame is retrieved, process it
                start_time = time.time()
                result = self._process_single_frame(frame)
                processing_time = time.time() - start_time
                
                self.performance_monitor.record_processing_time(processing_time)
                self.frames_processed += 1
                
                if result:
                    # Emit results to the connected client via SocketIO
                    socketio.emit('visual_feedback', result, namespace='/')
                
                self._update_fps() # Update backend processing FPS
                
                # Periodically force garbage collection to manage memory, especially with image data
                if self.frames_processed % 100 == 0:
                    gc.collect()
                
            except queue.Empty:
                # No frames in queue, continue loop (expected behavior)
                pass
            except Exception as e:
                # Catch any unhandled exceptions in the processing logic
                logger.error(f"Unhandled error in frame processing loop: {traceback.format_exc()}")
                # Add a short delay to prevent a tight error loop that consumes CPU
                time.sleep(0.5)
        
        logger.info("Frame processing thread stopped.")

    def _process_single_frame(self, frame):
        """
        Core logic to process a single video frame: detect eyes, process blinks,
        update Morse code and decoded text, and prepare data for the client.
        @param frame: The OpenCV image frame (numpy array).
        @return: Dictionary of data to be sent to the client, or None on error.
        """
        blink_detected_this_frame = False
        morse_char_added = None
        left_eye_coords = []
        right_eye_coords = []
        
        try:
            # 1. Detect face and eyes using the blink detector
            left_eye, right_eye, face_rect, detection_success = self.blink_detector.detect_face_and_eyes(frame)

            # 2. Process blinks and get new Morse characters/signals
            morse_char_added, blink_detected_this_frame = self.blink_detector.process_blink(left_eye, right_eye, self.capture_active)

            # 3. Update internal Morse code and decoded text buffers based on detected character
            if morse_char_added == "dot":
                self.morse_code += "."
            elif morse_char_added == "dash":
                self.morse_code += "-"
            elif morse_char_added == "space":
                # When a space is detected, decode the accumulated morse_code into text
                if self.morse_code: # Only decode if there's an actual character to decode
                    decoded_char_or_word = self.morse_processor.decode_morse(self.morse_code)
                    self.decoded_text += decoded_char_or_word
                    self.morse_code = "" # Reset Morse buffer after decoding a character/word
                self.decoded_text += " " # Add space for word separation in the decoded text

            # Prepare current live decoded character (if morse_code buffer is not empty)
            current_decoded_char = ""
            if self.morse_code:
                current_decoded_char = self.morse_processor.decode_morse(self.morse_code)

            # 4. Interpret the overall current decoded text for meaningful output
            full_text_for_interpretation = (self.decoded_text.strip() + " " + current_decoded_char.strip()).strip()
            self.meaningful_output = self.morse_processor.interpret_text(full_text_for_interpretation)

            # Prepare eye coordinates for frontend overlay drawing (convert tuples to lists for JSON serialization)
            if left_eye:
                left_eye_coords = [list(pt) for pt in left_eye]
            if right_eye:
                right_eye_coords = [list(pt) for pt in right_eye]

            # 5. Prepare data to send back to the client
            data_to_client = {
                'morse_code': self.morse_code, # Current accumulating Morse string
                'decoded_text': self.decoded_text.strip(), # Full decoded text (stripped for display)
                'meaningful_output': self.meaningful_output,
                'fps': round(self.current_fps, 1), # Backend processing FPS
                'calibration_status': self.blink_detector.get_calibration_status(),
                'detection_stats': self.blink_detector.get_statistics(),
                'morse_stats': self.morse_processor.get_statistics(),
                'eyes_detected': detection_success,
                'left_eye_coords': left_eye_coords,
                'right_eye_coords': right_eye_coords,
                'blink_detected': blink_detected_this_frame, # Flag if a blink occurred this cycle
                'morse_char_added': morse_char_added # What new Morse character was added (dot/dash/space)
            }
            return data_to_client

        except Exception as e:
            logger.error(f"Error in _process_single_frame: {traceback.format_exc()}")
            self.blink_detector.detection_stats['processing_errors'] += 1
            return None # Return None to indicate a failure in processing this frame

    def _update_fps(self):
        """Calculates and updates the frames per second being processed by this backend thread."""
        self.fps_counter += 1
        # Update FPS value at a configured interval
        if time.time() - self.fps_start_time >= self.config.get('fps_update_interval'):
            self.current_fps = self.fps_counter / (time.time() - self.fps_start_time)
            self.performance_monitor.update_fps(self.current_fps) # Update global monitor
            self.fps_counter = 0
            self.fps_start_time = time.time()

    def start_capture(self):
        """Activates the eye blink capture and resets all related state."""
        self.capture_active = True
        logger.info("Eye blink capture activated.")
        # Reset all buffers and statistics for a new session
        self.morse_code = ""
        self.decoded_text = ""
        self.meaningful_output = {}
        self.blink_detector.reset_statistics()
        self.morse_processor.reset_statistics()
        self.frames_processed = 0
        self.frames_dropped = 0
        self.frame_queue.queue.clear() # Clear any pending frames
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.last_frame_time = time.time() # Reset last frame time

    def stop_capture(self):
        """Deactivates the eye blink capture."""
        self.capture_active = False
        logger.info("Eye blink capture deactivated.")

    def get_status(self):
        """Returns the current operational status of the frame processor."""
        return {
            'capture_active': self.capture_active,
            'morse_code_buffer': self.morse_code,
            'decoded_text_buffer': self.decoded_text,
            'meaningful_output': self.meaningful_output,
            'frames_in_queue': self.frame_queue.qsize(),
            'current_backend_fps': round(self.current_fps, 1),
            'frames_processed_total': self.frames_processed,
            'frames_dropped_total': self.frames_dropped,
            'last_frame_received_ago_s': round(time.time() - self.last_frame_time, 2)
        }
    
    def shutdown(self):
        """Gracefully shuts down the frame processor thread."""
        logger.info("Initiating FrameProcessor shutdown...")
        self.processing_active = False # Signal the thread to stop
        # Wait for the thread to finish, with a timeout
        self.processing_thread.join(timeout=5)
        if self.processing_thread.is_alive():
            logger.warning("Frame processing thread did not terminate gracefully within 5 seconds.")
        else:
            logger.info("FrameProcessor thread successfully shut down.")
        # Clear the queue to release any remaining frame data
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break


# Global instances of our core application components
config_manager = ConfigManager()
performance_monitor = PerformanceMonitor()
morse_processor = MorseCodeProcessor()
# The EyeBlinkDetector needs the config_manager to get thresholds
eye_blink_detector = EyeBlinkDetector(config_manager)
# The FrameProcessor orchestrates interactions between the other components
frame_processor = FrameProcessor(morse_processor, eye_blink_detector, config_manager, performance_monitor)

# Thread pool for non-blocking operations if needed (e.g., database writes for feedback)
# Currently not explicitly used for feedback submission, but kept for future expansion
executor = ThreadPoolExecutor(max_workers=5)

# --- Flask Routes and SocketIO Event Handlers ---

@app.route('/')
def index():
    """Serves the main HTML page of the application."""
    logger.info("Serving index.html")
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    """Handles new client connections to the SocketIO server."""
    logger.info(f"Client connected: {request.sid}")
    # Emit initial status and configuration to the newly connected client
    emit('server_status', {'message': 'Connected to server', 'status': 'online'}, namespace='/')
    emit('config_update', config_manager.config, namespace='/')
    emit('calibration_status', eye_blink_detector.get_calibration_status(), namespace='/')
    emit('system_metrics', performance_monitor.get_metrics(), namespace='/')

@socketio.on('disconnect')
def handle_disconnect():
    """Handles client disconnections."""
    logger.info(f"Client disconnected: {request.sid}")
    # In a multi-client scenario, you might only stop capture if ALL clients disconnect.
    # For now, the capture state persists unless explicitly stopped by a client.

@socketio.on('start_capture')
def handle_start_capture():
    """Handles request from client to start eye blink capture."""
    logger.info(f"Start capture request received from client {request.sid}.")
    frame_processor.start_capture()
    # Emit confirmation back to the client
    socketio.emit('capture_status', {'active': True, 'message': 'Capture started'}, namespace='/')

@socketio.on('stop_capture')
def handle_stop_capture():
    """Handles request from client to stop eye blink capture."""
    logger.info(f"Stop capture request received from client {request.sid}.")
    # Get final Morse and decoded text before stopping for display on frontend
    final_morse = frame_processor.morse_code
    # Append any currently buffered Morse code to the final decoded text
    final_decoded_text = frame_processor.decoded_text + morse_processor.decode_morse(frame_processor.morse_code)
    
    frame_processor.stop_capture()
    # Emit confirmation and final results back to the client
    socketio.emit('capture_status', {'active': False, 'message': 'Capture stopped', 'final_morse': final_morse, 'final_decoded_text': final_decoded_text}, namespace='/')

@socketio.on('process_frame')
def handle_process_frame(data):
    """
    Handles incoming video frames from the frontend.
    Decodes base64 image data and adds the frame to the processing queue.
    This function is designed to be non-blocking.
    """
    if not frame_processor.capture_active:
        return # Do not process frames if capture is not active to save resources

    img_data_b64 = data.get('image_data')
    if not img_data_b64:
        logger.warning(f"Received empty image_data from client {request.sid}.")
        return

    try:
        # Remove the data URL header (e.g., "data:image/jpeg;base64,")
        header, encoded_data = img_data_b64.split(",", 1)
        # Decode base64 string to bytes
        np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        # Decode image from bytes using OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode image from base64 data. Image might be corrupted.")

        # Add the frame to the frame processor's queue
        success = frame_processor.add_frame(frame)
        if not success:
            logger.warning(f"Frame queue is full, frame dropped for client {request.sid}.")

    except ValueError as ve:
        logger.error(f"Data processing error for client {request.sid}: {ve}", exc_info=True)
        emit('error', {'message': 'Image decoding error', 'details': str(ve)}, namespace='/')
    except Exception as e:
        logger.error(f"Unexpected error processing incoming frame from client {request.sid}: {traceback.format_exc()}")
        emit('error', {'message': 'Server error processing frame', 'details': str(e)}, namespace='/')

@socketio.on('get_status')
def handle_get_status():
    """Provides comprehensive application status and metrics to the client."""
    status = frame_processor.get_status()
    status['system_metrics'] = performance_monitor.get_metrics()
    status['calibration_status'] = eye_blink_detector.get_calibration_status()
    status['detection_stats'] = eye_blink_detector.get_statistics()
    status['morse_stats'] = morse_processor.get_statistics()
    emit('app_status', status, namespace='/')

@socketio.on('update_ear_thresh')
def handle_update_ear_thresh(data):
    """Handles request to update the Eye Aspect Ratio threshold."""
    try:
        value = float(data.get('value'))
        if config_manager.set('eye_ar_thresh', value):
            config_manager.save_config()
            emit('config_update', config_manager.config, namespace='/')
            logger.info(f"EAR threshold updated to: {value} by client {request.sid}.")
        else:
            emit('error', {'message': 'Invalid EAR threshold value provided.'}, namespace='/')
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid value for EAR threshold received from client {request.sid}: {e}")
        emit('error', {'message': 'Invalid data format for EAR threshold.'}, namespace='/')
    except Exception as e:
        logger.error(f"Error updating EAR threshold for client {request.sid}: {traceback.format_exc()}")
        emit('error', {'message': 'Server error updating EAR threshold.'}, namespace='/')

@socketio.on('update_open_eye_time_thresh')
def handle_update_open_eye_time_thresh(data):
    """Handles request to update the open eye time threshold."""
    try:
        value = float(data.get('value'))
        if config_manager.set('open_eye_time_thresh', value):
            config_manager.save_config()
            emit('config_update', config_manager.config, namespace='/')
            logger.info(f"Open eye time threshold updated to: {value} by client {request.sid}.")
        else:
            emit('error', {'message': 'Invalid Open Eye Time threshold value provided.'}, namespace='/')
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid value for Open Eye Time threshold received from client {request.sid}: {e}")
        emit('error', {'message': 'Invalid data format for Open Eye Time threshold.'}, namespace='/')
    except Exception as e:
        logger.error(f"Error updating Open Eye Time threshold for client {request.sid}: {traceback.format_exc()}")
        emit('error', {'message': 'Server error updating Open Eye Time threshold.'}, namespace='/')

@socketio.on('convert_text_to_morse_request')
def handle_encode_text_request(data):
    """Handles client request to encode text to Morse code."""
    text = data.get('text', '')
    if isinstance(text, str) and text:
        morse = morse_processor.encode_text_to_morse(text)
        emit('conversion_result', {'type': 'text_to_morse', 'output': morse}, namespace='/')
        logger.info(f"Encoded '{text[:30]}...' to Morse.")
    else:
        emit('conversion_result', {'type': 'text_to_morse', 'output': 'Invalid text input.'}, namespace='/')
        logger.warning(f"Received invalid text for Morse encoding from client {request.sid}.")

@socketio.on('convert_morse_to_text_request')
def handle_decode_morse_request(data):
    """Handles client request to decode Morse code to text."""
    morse = data.get('morse', '')
    if isinstance(morse, str) and morse:
        text = morse_processor.decode_morse(morse)
        emit('conversion_result', {'type': 'morse_to_text', 'output': text}, namespace='/')
        logger.info(f"Decoded '{morse[:30]}...' from Morse.")
    else:
        emit('conversion_result', {'type': 'morse_to_text', 'output': 'Invalid Morse code input.'}, namespace='/')
        logger.warning(f"Received invalid Morse for decoding from client {request.sid}.")

def emit_system_metrics():
    """
    Background task to periodically emit system performance metrics to all connected clients.
    """
    while True:
        try:
            if config_manager.get('performance_monitoring'):
                performance_monitor.update_system_metrics()
                metrics = performance_monitor.get_metrics()
                socketio.emit('system_metrics', metrics, namespace='/')
            socketio.sleep(5) # Emit every 5 seconds
        except Exception as e:
            logger.error(f"Error in emit_system_metrics background task: {e}", exc_info=True)
            socketio.sleep(10) # Wait longer on error to prevent busy loop

def graceful_shutdown(signum, frame):
    """
    Signal handler for graceful application shutdown (e.g., Ctrl+C, system termination).
    Ensures that long-running threads and resources are properly closed.
    """
    logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
    frame_processor.shutdown() # Shut down the frame processing thread
    config_manager.save_config() # Save current config
    socketio.stop() # Stop the SocketIO server
    logger.info("Application exiting.")
    sys.exit(0)

# Register signal handlers for graceful shutdown (SIGINT for Ctrl+C, SIGTERM for process termination)
signal.signal(signal.SIGINT, graceful_shutdown)
signal.signal(signal.SIGTERM, graceful_shutdown)

# Main entry point for the Flask application
if __name__ == '__main__':
    logger.info("Starting Morse Code Application...")
    
    # Start the system metrics emission in a separate background thread
    socketio.start_background_task(target=emit_system_metrics)

    # Run the Flask app with SocketIO.
    # For production, consider using a WSGI server like Gunicorn with Eventlet/Gevent workers
    # (e.g., `gunicorn -k eventlet -w 1 'app:socketio' --bind 0.0.0.0:5000`).
    # debug=False is crucial for production.
    # allow_unsafe_werkzeug=False should be used in production for security.
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=False)
    except Exception as e:
        logger.critical(f"Failed to start Flask/SocketIO server: {e}", exc_info=True)
        frame_processor.shutdown() # Ensure processor is shut down even on startup failure
    finally:
        logger.info("Application main process exiting.")
        # Ensure frame processor is shut down if the main loop exits unexpectedly
        frame_processor.shutdown()
