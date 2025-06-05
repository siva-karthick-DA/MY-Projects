import streamlit as st
import cv2
import numpy as np
import time
import sqlite3
from PIL import Image
from datetime import datetime
import os
from imutils import face_utils
import dlib
import pyttsx3


# ---------- DATABASE SETUP ----------
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT, password TEXT)')

def add_user(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    return c.fetchone()

create_usertable()

# ---------- SESSION STATE ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

# ---------- STYLING ----------
st.markdown("""
    <style>
        .main {
            background-color:black;
        }
        .block-container {
            padding: 2rem 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            border: none;
            margin: 5px;
        }
        .stButton>button:hover {
            background-color:red;
            color:white;
        }
        .stSidebar {
            background-color: #e3e6ec;
        }

    </style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR LOGIN/SIGNUP ----------
st.sidebar.title("🔐 Login / Signup")
auth_option = st.sidebar.radio("Select Option", ["Login", "Signup"])

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if auth_option == "Signup":
    if st.sidebar.button("Create Account"):
        add_user(username, password)
        st.sidebar.success("✅ Account created! Please login.")
elif auth_option == "Login":
    if st.sidebar.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.sidebar.success(f"👋 Welcome, {username}!")
        else:
            st.sidebar.error("❌ Invalid credentials")

# ---------- EAR COMPUTATION ----------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ---------- BEEP SOUND FUNCTION ----------
def play_beep():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say("Fatigue detected. Please take a break.")
    engine.runAndWait()

def play_alert():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say("Disease detected. Please Check the Doctor")
    engine.runAndWait()

def play_norm():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say("Good Health. Please continue work.")
    engine.runAndWait()

# ---------- MAIN UI ----------
if not st.session_state.logged_in:
    st.title("👁️ Real-time Eye Blink Detection System")
    st.markdown("This application detects eye blinks using your webcam in real time. Please log in or sign up from the sidebar to get started.")
    st.image("static/bg.png", use_column_width=True)
else:
    st.title("🎥 Live Blink Detection")

    EAR_THRESH = st.sidebar.slider("EAR Threshold", 0.15, 0.3, 0.21, 0.01)
    EAR_CONSEC_FRAMES = st.sidebar.slider("Consecutive Frames", 2, 10, 3)

    col1, col2 = st.columns([1, 1])
    if col1.button("▶️ Start Camera"):
        st.session_state.camera_active = True
    if col2.button("⏹️ Stop Camera"):
        st.session_state.camera_active = False

    FRAME_WINDOW = st.empty()

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        COUNTER = 0
        TOTAL = 0
        blinks_in_minute = 0
        start_time = time.time()

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EAR_CONSEC_FRAMES:
                        TOTAL += 1
                        blinks_in_minute += 1
                    COUNTER = 0

                cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 10:
                if blinks_in_minute > 3:
                    st.markdown("""
                    <div style='
                    background-color: #fff3cd;
                    padding: 1rem;
                    border-left: 5px solid #ffa500;
                    color: #212529;
                    font-weight: bold;
                    border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                    margin-top: 1rem;
                    '>
                    ⚠️ Fatigue Detected: High blink rate observed in the last minute.
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("dismiss"):
                        st.experimental_rerun()
                    play_beep()
                    start_time = time.time()
                    blinks_in_minute = 0
                elif blinks_in_minute < 3:
                    st.markdown("""
                    <div style='
                    background-color: #fff3cd;
                    padding: 1rem;
                    border-left: 5px solid #ffa500;
                    color: #212529;
                    font-weight: bold;
                    border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                    margin-top: 1rem;
                    '>
                    ⚠️ Eye Problem Detected: Low blink rate observed in the last minute.
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("dismiss"):
                        st.experimental_rerun()
                    play_alert()
                    start_time = time.time()
                    blinks_in_minute = 0
                else:
                    st.markdown("""
                    <div style='
                    background-color: #fff3cd;
                    padding: 1rem;
                    border-left: 5px solid #ffa500;
                    color: #212529;
                    font-weight: bold;
                    border-radius: 8px;
                    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
                    margin-top: 1rem;
                    '>
                    ⚠️ Good Health Enjoy your life.
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("dismiss"):
                        st.experimental_rerun()
                    play_norm()
                    start_time = time.time()
                    blinks_in_minute = 0

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()
        FRAME_WINDOW.empty()
        st.success("📷 Camera stopped")
