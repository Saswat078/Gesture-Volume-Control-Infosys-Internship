import cv2
import mediapipe as mp

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use default camera
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils
    
    print("\n=== Milestone 1: Webcam + Hand Detection ===")
    print("Press ESC to exit\n")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Flip frame horizontally to act like a mirror
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = hands.process(rgb_frame)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        
        # Display frame
        cv2.imshow("Milestone 1 - Hand Detection", frame)
        
        # ESC key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nMilestone 1 completed successfully!")

if __name__ == "__main__":
    main()
