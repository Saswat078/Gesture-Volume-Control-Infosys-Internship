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
                font-family: 'Poppins', sans-serif;
                color: white;
                text-align: center;
            }
            .login-box {
                background-color: rgba(255,255,255,0.15);
                padding: 40px 50px;
                border-radius: 20px;
                width: 400px;
                margin: 10% auto;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            }
            h1 {
                color: white;
                font-weight: 900;
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
    st.session_state.pause_camera = False
    st.rerun()

# -------------------- THEME CONFIG --------------------
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
        .banner h1 {{
            font-size: 2.6rem;
            font-weight: 900;
        }}
        .stButton>button {{
            width: 100%;
            max-width: 250px;
            height: 52px;
            font-size: 17px;
            font-weight: 700;
            border-radius: 14px;
            border: none;
            color: white !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .start-btn {{ background: linear-gradient(90deg, #3B82F6, #1D4ED8); }}
        .pause-btn {{ background: linear-gradient(90deg, #F59E0B, #B45309); }}
        .stop-btn {{ background: linear-gradient(90deg, #DC2626, #991B1B); }}
        .metric-box {{
            background-color: {card_bg};
            border: 2px solid {border};
            border-radius: 16px;
            padding: 18px 20px;
            margin-bottom: 12px;
            text-align: center;
            font-weight: 700;
        }}
        .metric-value {{
            color: {accent};
            font-size: 22px;
            font-weight: 800;
        }}
        .status-bar {{
            background: {accent};
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            font-weight: 700;
            color: white;
            font-size: 20px;
        }}
    </style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("""
    <div class='banner'>
        <h1>🎵 Gesture-Based Smart Volume Controller</h1>
        <p>Use AI-driven hand tracking to control system volume in real time.</p>
    </div>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "run_camera" not in st.session_state:
    st.session_state.run_camera = False
if "pause_camera" not in st.session_state:
    st.session_state.pause_camera = False
if "chart_data" not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(columns=["Frame", "Distance", "Volume"])
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "last_update" not in st.session_state:
    st.session_state.last_update = 0

# -------------------- BUTTONS --------------------
col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    if st.button("▶️ Start Webcam", use_container_width=True):
        st.session_state.run_camera = True
        st.session_state.pause_camera = False
        st.toast("🎥 Webcam Started Successfully")

with col_btn2:
    if st.button("⏸️ Pause", use_container_width=True):
        if st.session_state.run_camera:
            st.session_state.pause_camera = True
            st.toast("⏸️ Webcam Paused")

with col_btn3:
    if st.button("⛔ Stop Webcam", use_container_width=True):
        st.session_state.run_camera = False
        st.session_state.pause_camera = False
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

st.write("---")
st.subheader("📈 Distance & Volume Trend ")
graph_placeholder = st.empty()

# -------------------- HAND + AUDIO SETUP --------------------
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw_utils = mp.solutions.drawing_utils
FINGER_TIPS = [4, 8, 12, 16, 20]

try:
    pythoncom.CoInitialize()
    speakers = AudioUtilities.GetSpeakers()
    audio_interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_controller = cast(audio_interface, POINTER(IAudioEndpointVolume))
    VOL_MIN, VOL_MAX, _ = volume_controller.GetVolumeRange()
    volume_enabled = True
except Exception:
    VOL_MIN, VOL_MAX = -65.25, 0.0
    volume_enabled = False

# -------------------- CAMERA FUNCTION --------------------
def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
        return

    while st.session_state.run_camera:
        if st.session_state.pause_camera:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(frame_rgb)

        gesture_name, gesture_icon = "No Hand", "❌"
        distance_mm, vol_percent = 0, 0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                mp_draw_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip, index_tip = landmarks[4], landmarks[8]
                pinch_distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
                ref_dist = math.hypot(landmarks[9][0] - landmarks[0][0], landmarks[9][1] - landmarks[0][1])
                if ref_dist > 0:
                    distance_mm = (pinch_distance / ref_dist) * 85

                # Draw a smooth green line between thumb and index
                cv2.line(frame, thumb_tip, index_tip, (0, 255, 0), 4)
                cv2.circle(frame, thumb_tip, 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, index_tip, 10, (0, 255, 0), cv2.FILLED)

                vol_percent = int(np.interp(pinch_distance, [20, 200], [0, 100]))
                if volume_enabled:
                    vol_level = np.interp(vol_percent, [0, 100], [VOL_MIN, VOL_MAX])
                    volume_controller.SetMasterVolumeLevel(vol_level, None)

                if pinch_distance < 30:
                    gesture_name, gesture_icon = "Pinch (Mute)", "🤏"
                elif vol_percent > 90:
                    gesture_name, gesture_icon = "Open Hand (Max)", "🖐️"
                elif vol_percent < 10:
                    gesture_name, gesture_icon = "Fist (Min)", "✊"
                else:
                    gesture_name, gesture_icon = "Adjusting", "✋"

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        gesture_status.markdown(f"<div class='status-bar'>{gesture_icon} {gesture_name}</div>", unsafe_allow_html=True)
        gesture_display.markdown(f"<div class='metric-box'>🖐️ Gesture<br><span class='metric-value'>{gesture_name}</span></div>", unsafe_allow_html=True)
        distance_display.markdown(f"<div class='metric-box'>📏 Distance<br><span class='metric-value'>{distance_mm:.2f} mm</span></div>", unsafe_allow_html=True)
        volume_display.markdown(f"<div class='metric-box'>🔊 Volume<br><span class='metric-value'>{vol_percent}%</span></div>", unsafe_allow_html=True)

        # Smooth Graph (update every 0.5s to prevent flicker)
        now = time.time()
        if now - st.session_state.last_update > 0.5:
            st.session_state.frame_count += 1
            new_row = pd.DataFrame([[st.session_state.frame_count, distance_mm, vol_percent]],
                                   columns=["Frame", "Distance", "Volume"])
            st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_row]).tail(60)

            df = st.session_state.chart_data
            base = alt.Chart(df).encode(x='Frame')

            line1 = base.mark_line(color='#3B82F6', strokeWidth=3).encode(y='Distance')
            line2 = base.mark_line(color='#EF4444', strokeWidth=3, strokeDash=[4, 4]).encode(y='Volume')
            combined_chart = (line1 + line2).properties(height=350, title="📊 Distance & Volume Trend (Smooth)")
            graph_placeholder.altair_chart(combined_chart, use_container_width=True)

            st.session_state.last_update = now

    cap.release()
    cv2.destroyAllWindows()

# -------------------- RUN CAMERA --------------------
if st.session_state.run_camera:
    run_camera()
else:
    st.info("👋 Click **Start Webcam** to begin hand gesture detection.")
