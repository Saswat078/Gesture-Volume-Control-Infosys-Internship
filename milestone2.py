import cv2
import mediapipe as mp
import math

# Hand Detection and Distance Calculation using Mediapipe
class HandDistance:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.thumb_tip = 4
        self.index_tip = 8

    def get_distance(self, p1, p2, w, h):
        # Convert normalized values to pixel positions
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        d = math.hypot(x2 - x1, y2 - y1)
        return d, (x1, y1), (x2, y2)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found or not accessible.")
        return

    handTracker = HandDistance()
    print("\nMilestone 2: Gesture Recognition & Distance Measurement")
    print("=> Show your hand and pinch your thumb and index finger.")
    print("=> Distance will be displayed on the screen.\nPress ESC to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = handTracker.hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                handTracker.mpDraw.draw_landmarks(
                    frame, handLms, handTracker.mpHands.HAND_CONNECTIONS,
                    handTracker.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    handTracker.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Get landmark points
                thumb = handLms.landmark[handTracker.thumb_tip]
                index = handLms.landmark[handTracker.index_tip]

                # Find distance between thumb and index
                dist, p1, p2 = handTracker.get_distance(thumb, index, w, h)

                # Draw points and line
                cv2.circle(frame, p1, 8, (0, 0, 255), -1)
                cv2.circle(frame, p2, 8, (0, 0, 255), -1)
                cv2.line(frame, p1, p2, (255, 255, 0), 3)

                # Show distance value
                cv2.putText(frame, f"Distance: {int(dist)} px", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        else:
            cv2.putText(frame, "No hand detected", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Distance (Milestone 2)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram closed successfully.")

if __name__ == "__main__":
    main()
