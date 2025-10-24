import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import pandas as pd
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math
import altair as alt


# ---------------- Streamlit UI Setup ----------------
st.set_page_config(layout="wide", page_title="Gesture Volume Controller")

# 🎨 CHANGE 1: Bright Gradient Theme (Purple–Blue)
st.markdown("""
    <style>
        body {
            background-color: #121826;
            color: #e0e6ff;
        }
        .stApp {
            background-color: #121826;
            color: #e0e6ff;
        }
        h1, h2, h3 {
            color: #7df9ff; /* neon cyan titles */
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        .stMetric {
            background-color: #1e2433;
            border-radius: 15px;
            padding: 10px;
            color: #ff6ec7; /* pink metric values */
        }
        hr {
            border: 1px solid #7df9ff;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🎵 Smart Hand Gesture Volume Controller")
st.write("Control your system volume smoothly using your hand — redesigned in a new light theme!")

col1, col2 = st.columns([3, 2])

with col1:
    st.header("📹 Camera Feed")
    frame_placeholder = st.empty()

with col2:
    st.header("⚙️ Live Control Panel")
    gesture_placeholder = st.empty()
    distance_placeholder = st.empty()
    volume_display_placeholder = st.empty()
    st.write("---")
    mapping_chart_placeholder = st.empty()


# ---------------- Hand Tracking Setup ----------------
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, 
                               min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw_utils = mp.solutions.drawing_utils
FINGER_TIPS = [4, 8, 12, 16, 20]

# ---------------- Volume Control Setup ----------------
try:
    speakers = AudioUtilities.GetSpeakers()
    audio_interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_controller = cast(audio_interface, POINTER(IAudioEndpointVolume))
    VOL_MIN, VOL_MAX, _ = volume_controller.GetVolumeRange()
    volume_control_enabled = True
except Exception as e:
    st.error(f"Could not initialize volume control. Error: {e}")
    volume_control_enabled = False


# ---------------- Webcam Setup ----------------
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    st.error("Could not open webcam.")
else:
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mapping_df = pd.DataFrame({
        'Normalized Distance': range(101),
        'Volume %': range(101)
    })

    if 'stop' not in st.session_state:
        st.session_state.stop = False

    def stop_camera():
        st.session_state.stop = True

    st.button("🛑 Stop Webcam", on_click=stop_camera)

    while not st.session_state.stop:
        ret, frame = cam.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(frame_rgb)

        landmarks = []
        gesture_name = "No Hand Detected"
        distance_mm = 0
        vol_percent = 0
        pinch_distance_pixels = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append((cx, cy))

                mp_draw_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if len(landmarks) >= 21:
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    pinch_distance_pixels = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
                    
                    wrist, middle_mcp = landmarks[0], landmarks[9]
                    ref_dist_pixels = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])
                    
                    KNOWN_DISTANCE_MM = 85.0
                    if ref_dist_pixels > 0:
                        mm_per_pixel = KNOWN_DISTANCE_MM / ref_dist_pixels
                        distance_mm = pinch_distance_pixels * mm_per_pixel

                    fingers_up = [1 if landmarks[FINGER_TIPS[0]][0] < landmarks[FINGER_TIPS[0] - 1][0] else 0]
                    for i in range(1, 5):
                        fingers_up.append(1 if landmarks[FINGER_TIPS[i]][1] < landmarks[FINGER_TIPS[i] - 2][1] else 0)
                    
                    num_fingers = fingers_up.count(1)

                    if pinch_distance_pixels < 30:
                        gesture_name = "Pinch"
                    elif num_fingers == 5:
                        gesture_name = "Open Hand"
                    elif num_fingers == 0:
                        gesture_name = "Fist"
                    else:
                        gesture_name = f"{num_fingers} Fingers Up"

                    # 🎨 CHANGE 2: New Purple Line Color
                    pinch_color = (180, 0, 255)

                    if pinch_distance_pixels > 20:
                        volume_level = np.interp(pinch_distance_pixels, [20, 200], [VOL_MIN, VOL_MAX])
                        vol_percent = int(np.interp(pinch_distance_pixels, [20, 200], [0, 100]))
                        
                        if volume_control_enabled:
                            volume_controller.SetMasterVolumeLevel(volume_level, None)

                    cv2.circle(frame, thumb_tip, 10, pinch_color, cv2.FILLED)
                    cv2.circle(frame, index_tip, 10, pinch_color, cv2.FILLED)
                    cv2.line(frame, thumb_tip, index_tip, pinch_color, 3)

        # 🎨 CHANGE 3: Volume Bar – Purple Gradient with Soft Glow
        vol_bar_pos = np.interp(vol_percent, [0, 100], [400, 150])
        bar_color = (150, 0, 255)
        cv2.rectangle(frame, (50, 150), (85, 400), (220, 220, 220), 2)
        cv2.rectangle(frame, (50, int(vol_bar_pos)), (85, 400), bar_color, cv2.FILLED)
        cv2.putText(frame, f'{vol_percent}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 0, 255), 3)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        gesture_placeholder.metric("Gesture", gesture_name)
        distance_placeholder.metric("Pinch Distance", f"{distance_mm:.2f} mm")
        volume_display_placeholder.metric("Volume", f"{vol_percent} %")

        # 🎨 CHANGE 4: Chart Colors (Lavender Line + Pink Dot)
        normalized_dist = np.interp(pinch_distance_pixels, [20, 200], [0, 100])
        current_pos_df = pd.DataFrame({'Normalized Distance': [normalized_dist], 'Volume %': [vol_percent]})

        line_chart = alt.Chart(mapping_df).mark_line(color='#b19cd9').encode(
            x='Normalized Distance',
            y='Volume %'
        )

        point_chart = alt.Chart(current_pos_df).mark_circle(size=150, color='#ff66b2').encode(
            x='Normalized Distance',
            y='Volume %'
        )

        final_chart = (line_chart + point_chart).properties(
            title='Distance vs Volume Mapping'
        ).interactive()

        mapping_chart_placeholder.altair_chart(final_chart, use_container_width=True)

    cam.release()
    cv2.destroyAllWindows()
    st.success("✅ Webcam feed stopped successfully.")
