import cv2
import mediapipe as mp
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Volume Controller
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            thumb, index = landmarks[4], landmarks[8]
            distance = math.hypot(index[0] - thumb[0], index[1] - thumb[1])

            # Map distance to volume
            vol = np.interp(distance, [20, 200], [min_vol, max_vol])
            vol_percent = np.interp(distance, [20, 200], [0, 100])
            volume.SetMasterVolumeLevel(vol, None)

            cv2.rectangle(frame, (50, 150), (85, 400), (0, 255, 0), 2)
            bar = np.interp(distance, [20, 200], [400, 150])
            cv2.rectangle(frame, (50, int(bar)), (85, 400), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f'{int(vol_percent)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Milestone 3: Volume Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
