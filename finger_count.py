

import cv2
import mediapipe as mp
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


class VolumeByFingers:
    def __init__(self):
        # Hand tracking
        self.hands_module = mp.solutions.hands
        self.hand_detector = self.hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.drawer = mp.solutions.drawing_utils

        # Windows volume control
        speaker = AudioUtilities.GetSpeakers()
        interface = speaker.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self.endpoint = cast(interface, POINTER(IAudioEndpointVolume))
        self.min_vol, self.max_vol, _ = self.endpoint.GetVolumeRange()

        # Finger tip landmark indices
        self.finger_tips = [4, 8, 12, 16, 20]

    def detect_fingers(self, landmarks):
        """Count how many fingers are raised"""
        count = []

        # Thumb (x comparison)
        if landmarks.landmark[self.finger_tips[0]].x < landmarks.landmark[self.finger_tips[0] - 1].x:
            count.append(1)
        else:
            count.append(0)

        # Other 4 fingers (y comparison)
        for idx in range(1, 5):
            if landmarks.landmark[self.finger_tips[idx]].y < landmarks.landmark[self.finger_tips[idx] - 2].y:
                count.append(1)
            else:
                count.append(0)

        return sum(count)

    def adjust_volume(self, num_fingers):
        """Map finger count to system volume"""
        vol_level = np.interp(num_fingers, [0, 5], [0.0, 1.0])
        self.endpoint.SetMasterVolumeLevelScalar(vol_level, None)
        return int(vol_level * 100)


def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not found")
        return

    controller = VolumeByFingers()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = controller.hand_detector.process(rgb)

        fingers = 0
        volume = 0

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            controller.drawer.draw_landmarks(frame, hand, controller.hands_module.HAND_CONNECTIONS)

            fingers = controller.detect_fingers(hand)
            volume = controller.adjust_volume(fingers)

        # Display text only
        cv2.putText(frame, f"Fingers: {fingers}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
        cv2.putText(frame, f"Volume: {volume}%", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        cv2.imshow("Finger Volume Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
