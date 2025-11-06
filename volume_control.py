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
import time
import pythoncom

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Gesture Volume Dashboard", layout="wide")

# -------------------- LOGIN PAGE --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.markdown("""
        <style>
            body, .stApp {
                background: linear-gradient(120deg, #3B82F6, #9333EA);
                color: white;
                text-align: center;
                font-family: 'Poppins', sans-serif;
            }
            .login-box {
                background-color: rgba(255,255,255,0.15);
                padding: 40px 50px;
                border-radius: 20px;
                width: 400px;
                margin: 12% auto;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='login-box'>
            <h1>🎵 Gesture Volume Controller</h1>
            <p>Login to access the dashboard</p>
        </div>
    """, unsafe_allow_html=True)

    username = st.text_input("👤 Username", placeholder="Enter any username")
    password = st.text_input("🔒 Password", placeholder="Enter any password", type="password")

    if st.button("Login 🚀", use_container_width=True):
        st.session_state.logged_in = True
        st.success("Login successful! Redirecting...")
        st.rerun()
    st.stop()

# -------------------- SIDEBAR SETTINGS --------------------
st.sidebar.title("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["🌞 Light Mode", "🌙 Dark Mode"])
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.run_camera = False
    st.rerun()

# -------------------- THEME SETTINGS --------------------
if theme == "🌙 Dark Mode":
    bg_color = "#0F172A"
    text_color = "#F1F5F9"
    card_bg = "#1E293B"
    accent = "#3B82F6"
    border = "#475569"
    gradient_bg = "linear-gradient(90deg, #1E3A8A, #3B82F6, #1E40AF)"
else:
    bg_color = "#F9FAFB"
    text_color = "#111827"
    card_bg = "#E5E7EB"
    accent = "#2563EB"
    border = "#CBD5E1"
    gradient_bg = "linear-gradient(90deg, #60A5FA, #3B82F6, #2563EB)"

# -------------------- STYLING --------------------
st.markdown(f"""
    <style>
        body, .stApp {{
            background-color: {bg_color};
            color: {text_color};
            font-family: 'Poppins', sans-serif;
        }}
        .banner {{
            background: {gradient_bg};
            color: white;
            text-align: center;
            padding: 25px 10px;
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.2);
        }}
        .stButton>button {{
            width: 100%; max-width: 240px; height: 52px;
            font-size: 17px; font-weight: 700;
            border-radius: 14px; border: none;
            color: white !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .start-btn {{ background: linear-gradient(90deg, #3B82F6, #1D4ED8); }}
        .pause-btn {{ background: linear-gradient(90deg, #F59E0B, #D97706); }}
        .stop-btn {{ background: linear-gradient(90deg, #DC2626, #B91C1C); }}
        .metric-box {{
            background-color: {card_bg};
            border: 2px solid {border};
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 12px;
            text-align: center;
        }}
        .metric-value {{ color: {accent}; font-size: 22px; font-weight: 800; }}
        .status-bar {{
            background: {accent}; color: white; text-align: center;
            font-weight: 700; padding: 15px; border-radius: 10px;
        }}
        .finger-bar {{
            background: rgba(203, 213, 225, 0.6);
            border-radius: 12px; height: 25px; overflow: hidden;
        }}
        .finger-bar-fill {{
            background: linear-gradient(90deg, #60A5FA, #2563EB);
            height: 100%; border-radius: 12px;
            transition: width 0.3s ease;
        }}
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown(f"""
    <div class='banner'>
        <h1>🎵 Gesture-Based Smart Volume Controller</h1>
        <p>Use real-time AI hand tracking to control system volume seamlessly.</p>
    </div>
""", unsafe_allow_html=True)

# -------------------- SESSION STATES --------------------
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "pause_camera" not in st.session_state:
    st.session_state.pause_camera = False

# -------------------- BUTTONS --------------------
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn1:
    if st.button("▶️ Start Webcam", use_container_width=True):
        st.session_state.run_camera = True
        st.session_state.pause_camera = False
        st.toast("🎥 Webcam Started")

with col_btn2:
    if st.button("⏸️ Pause Webcam", use_container_width=True):
        st.session_state.pause_camera = not st.session_state.pause_camera
        st.toast("⏸️ Webcam Paused" if st.session_state.pause_camera else "▶️ Webcam Resumed")

with col_btn3:
    if st.button("⛔ Stop Webcam", use_container_width=True):
        st.session_state.run_camera = False
        st.toast("🛑 Webcam Stopped")

st.write("---")

# -------------------- DASHBOARD --------------------
col1, col2 = st.columns([3, 2])
with col1:
    st.header("📹 Live Camera Feed")
    frame_placeholder = st.empty()

with col2:
    st.header("📊 Real-Time Gesture Dashboard")
    gesture_status = st.empty()
    gesture_display = st.empty()
    distance_display = st.empty()
    volume_display = st.empty()
    finger_bar_placeholder = st.empty()

st.write("---")
st.subheader("📈 Distance & Volume Trend")
mapping_chart_placeholder = st.empty()

# -------------------- HAND + AUDIO MODULE --------------------
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

try:
    pythoncom.CoInitialize()
    speakers = AudioUtilities.GetSpeakers()
    interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_ctrl = cast(interface, POINTER(IAudioEndpointVolume))
    VOL_MIN, VOL_MAX, _ = volume_ctrl.GetVolumeRange()
    volume_enabled = True
except Exception:
    st.warning("⚠️ Audio control not available. Running in visual-only mode.")
    VOL_MIN, VOL_MAX = -65.25, 0.0
    volume_enabled = False

# -------------------- CAMERA DETECTION --------------------
def auto_detect_camera():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        time.sleep(1)
        ret, frame = cap.read()
        if ret:
            st.success(f"✅ Camera {i} initialized successfully.")
            cap.release()
            return i
        cap.release()
    st.error("❌ No working camera found.")
    return None

# -------------------- CAMERA LOOP --------------------
def run_camera():
    cam_index = auto_detect_camera()
    if cam_index is None:
        return

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(1.5)

    smooth_distance, smooth_volume = [], []

    while st.session_state.run_camera:
        if st.session_state.pause_camera:
            time.sleep(0.3)
            continue

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to grab frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(frame_rgb)

        gesture = "No Hand Detected"
        gesture_icon = "❌"
        distance_mm, vol_percent = 0, 0
        num_fingers = 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip, index_tip = landmarks[4], landmarks[8]
                cv2.line(frame, thumb_tip, index_tip, (0, 255, 0), 3)

                distance_px = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
                distance_mm = np.interp(distance_px, [20, 200], [0, 100])
                vol_percent = int(np.interp(distance_px, [20, 200], [0, 100]))

                FINGER_TIPS = [4, 8, 12, 16, 20]
                fingers_up = [1 if landmarks[FINGER_TIPS[0]][0] < landmarks[FINGER_TIPS[0]-1][0] else 0]
                for i in range(1, 5):
                    fingers_up.append(1 if landmarks[FINGER_TIPS[i]][1] < landmarks[FINGER_TIPS[i]-2][1] else 0)
                num_fingers = fingers_up.count(1)

                if distance_px < 30:
                    gesture, gesture_icon = "Pinch (Mute)", "🤏"
                elif num_fingers == 5:
                    gesture, gesture_icon = "Open Hand (Max Volume)", "🖐️"
                elif num_fingers == 0:
                    gesture, gesture_icon = "Fist (Min Volume)", "✊"
                else:
                    gesture, gesture_icon = "Adjusting Volume", "✋"

                if volume_enabled:
                    volume_level = np.interp(vol_percent, [0, 100], [VOL_MIN, VOL_MAX])
                    volume_ctrl.SetMasterVolumeLevel(volume_level, None)

        smooth_distance.append(distance_mm)
        smooth_volume.append(vol_percent)
        if len(smooth_distance) > 50:
            smooth_distance.pop(0)
            smooth_volume.pop(0)

        # --- UI Updates ---
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        gesture_status.markdown(f"<div class='status-bar'>{gesture_icon} {gesture}</div>", unsafe_allow_html=True)
        gesture_display.markdown(f"<div class='metric-box'>🖐️ Gesture<br><span class='metric-value'>{gesture}</span></div>", unsafe_allow_html=True)
        distance_display.markdown(f"<div class='metric-box'>📏 Distance<br><span class='metric-value'>{distance_mm:.2f} mm</span></div>", unsafe_allow_html=True)
        volume_display.markdown(f"<div class='metric-box'>🔊 Volume<br><span class='metric-value'>{vol_percent}%</span></div>", unsafe_allow_html=True)

        bar_width = (num_fingers / 5) * 100
        finger_bar_placeholder.markdown(
            f"<div class='metric-box'>✋ Fingers Up: {num_fingers}/5<div class='finger-bar'><div class='finger-bar-fill' style='width:{bar_width}%'></div></div></div>",
            unsafe_allow_html=True
        )

        # --- Stable Line Chart (non-flickering) ---
        df = pd.DataFrame({
            "Frame": list(range(len(smooth_distance))),
            "Distance (mm)": smooth_distance,
            "Volume (%)": smooth_volume
        })

        base = alt.Chart(df.melt("Frame")).encode(
            x=alt.X("Frame", axis=alt.Axis(title="Time (frames)")),
            y=alt.Y("value:Q", axis=alt.Axis(title="Value (%)"), scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("variable:N", scale=alt.Scale(domain=["Distance (mm)", "Volume (%)"], range=["#3B82F6", "#EF4444"]))
        )

        chart = base.mark_line(interpolate="monotone", strokeWidth=3).properties(height=300)
        mapping_chart_placeholder.altair_chart(chart, use_container_width=True)

        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

# -------------------- RUN CAMERA --------------------
if st.session_state.run_camera:
    run_camera()
else:
    st.info("👋 Click **Start Webcam** to begin hand gesture detection.")
