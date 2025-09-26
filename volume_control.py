import cv2
import mediapipe as mp
import math
import time

# Volume control imports
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    VOLUME_AVAILABLE = True
except ImportError:
    print("Warning: pycaw not installed. Volume control will be disabled.")
    print("Install with: py -3.10 -m pip install pycaw")
    VOLUME_AVAILABLE = False

class VolumeController:
    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Volume control parameters
        self.last_volume_change = 0
        self.volume_change_interval = 0.1  # Minimum time between volume changes (100ms)
        self.min_distance = 20   # Minimum distance between thumb and index finger
        self.max_distance = 200  # Maximum distance for volume control range
        
        # Initialize volume control
        self.volume = None
        if VOLUME_AVAILABLE:
            try:
                devices = AudioUtilities.GetSpeakers()
                interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self.volume = cast(interface, POINTER(IAudioEndpointVolume))
                self.volume_range = self.volume.GetVolumeRange()
                self.min_vol = self.volume_range[0]
                self.max_vol = self.volume_range[1]
                current_vol = self.volume.GetMasterVolumeLevelScalar()
                print(f"Volume control initialized! Current volume: {int(current_vol * 100)}%")
                print(f"Volume range: {self.min_vol} to {self.max_vol}")
            except Exception as e:
                print(f"Warning: Could not initialize volume control: {e}")
                self.volume = None
        else:
            print("Volume control not available - pycaw not installed")
        
        # Landmark indices for finger detection
        self.THUMB_TIP = 4
        self.THUMB_IP = 3
        self.INDEX_FINGER_TIP = 8
        self.INDEX_FINGER_PIP = 6
        self.MIDDLE_FINGER_TIP = 12
        self.MIDDLE_FINGER_PIP = 10
        self.RING_FINGER_TIP = 16
        self.RING_FINGER_PIP = 14
        self.PINKY_TIP = 20
        self.PINKY_PIP = 18
        
    def calculate_distance(self, point1, point2, frame_width, frame_height):
        """Calculate pixel distance between two normalized points"""
        x1, y1 = int(point1.x * frame_width), int(point1.y * frame_height)
        x2, y2 = int(point2.x * frame_width), int(point2.y * frame_height)
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2), (x1, y1), (x2, y2)
    
    def control_volume_by_pinch(self, distance):
        """Control system volume based on pinch distance"""
        if not self.volume:
            return None
        
        current_time = time.time()
        
        # Only change volume if enough time has passed since last change
        if current_time - self.last_volume_change < self.volume_change_interval:
            return None
        
        try:
            # Map pinch distance to volume level (0.0 to 1.0)
            # Closer pinch = lower volume, wider pinch = higher volume
            clamped_distance = max(self.min_distance, min(self.max_distance, distance))
            volume_level = (clamped_distance - self.min_distance) / (self.max_distance - self.min_distance)
            
            # Convert to decibel scale
            volume_db = self.min_vol + (volume_level * (self.max_vol - self.min_vol))
            
            # Set the volume
            self.volume.SetMasterVolumeLevel(volume_db, None)
            self.last_volume_change = current_time
            
            # Get current volume percentage for display
            current_scalar = self.volume.GetMasterVolumeLevelScalar()
            volume_percent = int(current_scalar * 100)
            
            return f"Volume: {volume_percent}%"
        
        except Exception as e:
            print(f"Error controlling volume: {e}")
        
        return None

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(1)  # Change to 0 if you want to use default camera
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Trying default camera (index 0)...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera")
            return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize volume controller
    controller = VolumeController()
    
    # Create window
    cv2.namedWindow("Hand Volume Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand Volume Control", 640, 480)
    
    print("\n=== Hand Volume Control Started ===")
    print("👌 Pinch your thumb and index finger together")
    print("📐 Closer pinch = Lower volume")
    print("� Wider pinch = Higher volume")
    print("⌨️  Press ESC to exit")
    print("🖼️  You can resize the window by dragging corners")
    print("=====================================\n")
    
    # Get current volume to display
    if controller.volume:
        try:
            current_vol = controller.volume.GetMasterVolumeLevelScalar()
            print(f"Starting volume: {int(current_vol * 100)}%\n")
        except:
            pass
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally to act like a mirror
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = controller.hands.process(rgb_frame)
        
        # Handle hand detection and volume control
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                controller.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, controller.mp_hands.HAND_CONNECTIONS,
                    controller.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    controller.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Get thumb and index finger positions
                thumb_tip = hand_landmarks.landmark[controller.THUMB_TIP]
                index_tip = hand_landmarks.landmark[controller.INDEX_FINGER_TIP]
                
                # Calculate distance between thumb and index finger
                distance, thumb_pos, index_pos = controller.calculate_distance(
                    thumb_tip, index_tip, frame_width, frame_height)
                
                # Control volume based on pinch distance
                volume_action = controller.control_volume_by_pinch(distance)
                
                # Draw connection line between thumb and index finger
                cv2.line(frame, thumb_pos, index_pos, (255, 0, 0), 3)
                cv2.circle(frame, thumb_pos, 8, (255, 0, 0), -1)
                cv2.circle(frame, index_pos, 8, (255, 0, 0), -1)
                
                # Display pinch distance
                cv2.putText(frame, f"Distance: {int(distance)}px", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                # Display volume action if any
                if volume_action:
                    print(volume_action)
                    cv2.putText(frame, volume_action, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.8, (255, 0, 255), 2)
                
                # Display current volume
                if controller.volume:
                    try:
                        current_vol = controller.volume.GetMasterVolumeLevelScalar()
                        vol_text = f"Volume: {int(current_vol * 100)}%"
                        cv2.putText(frame, vol_text, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (255, 255, 255), 2)
                        
                        # Draw volume bar
                        bar_width = 300
                        bar_height = 20
                        bar_x = 20
                        bar_y = 180
                        
                        # Background bar
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                                     (100, 100, 100), -1)
                        
                        # Volume level bar
                        vol_width = int(bar_width * current_vol)
                        color = (0, 255, 0) if current_vol > 0.5 else (0, 255, 255) if current_vol > 0.2 else (0, 0, 255)
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + vol_width, bar_y + bar_height), 
                                     color, -1)
                        
                        # Volume bar outline
                        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                                     (255, 255, 255), 2)
                        
                    except Exception as e:
                        cv2.putText(frame, "Volume: Error", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.8, (0, 0, 255), 2)
                
                # Show gesture instructions
                cv2.putText(frame, f"Pinch Range: {controller.min_distance}-{controller.max_distance}px", 
                           (20, frame_height - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Closer Pinch = Lower Volume", (20, frame_height - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, "Wider Pinch = Higher Volume", (20, frame_height - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "ESC = Exit", (20, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
            cv2.putText(frame, "Show your hand and pinch to control volume", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow("Hand Volume Control", frame)
        
        # Check for ESC key press (key code 27)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nVolume control ended. Goodbye!")

if __name__ == "__main__":
    main()