import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]

            cv2.circle(frame, thumb_tip, 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, index_tip, 10, (255, 0, 0), cv2.FILLED)
            cv2.line(frame, thumb_tip, index_tip, (0, 255, 0), 3)

            distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
            cv2.putText(frame, f"Distance: {int(distance)} px", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Milestone 2: Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
