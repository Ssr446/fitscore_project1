import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import sklearn
import tempfile
import os
import pandas as pd
from numpy.linalg import norm

# --- Load Models and Prototypes ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Error: model.pkl not found. Please run the training notebook first.")
    st.stop()

try:
    prototypes = pd.read_csv('prototype_poses_normalized.csv', index_col='class')
except FileNotFoundError:
    st.error("Error: prototype_poses_normalized.csv not found. Please run the updated training notebook.")
    st.stop()

# --- Helper Functions ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_form_feedback(report):
    avg_q, consistency, min_q = report['avg_q_score'], report['consistency'], report['min_q_score']
    if avg_q >= 85 and consistency >= 90 and min_q >= 70:
        verdict = "Excellent Form! 👍"
        feedback = "Your movements are highly consistent and closely match the ideal form. Keep up the great work!"
    elif avg_q >= 70 and consistency >= 80 and min_q >= 55:
        verdict = "Good Form! 🙂"
        feedback = "You have a solid foundation. Focus on maintaining stability throughout the entire movement to improve further."
    else:
        verdict = "Needs Improvement 💡"
        feedback = f"Your form appears inconsistent. Try slowing down and focus on control. Your lowest score of {min_q:.0f} suggests a point where form broke down."
    return verdict, feedback

# --- Initialize MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Streamlit App UI ---
st.set_page_config(page_title="FitScore AI", layout="wide")
st.title('FitScore: AI Exercise Analysis 🏋️')
st.markdown("---")

# --- Initialize Session State ---
if 'analysis_report' not in st.session_state:
    st.session_state.analysis_report = None
if 'run_processing' not in st.session_state:
    st.session_state.run_processing = False

uploaded_file = st.file_uploader("Upload a video for analysis...", type=["mp4", "mov", "avi"])

if uploaded_file and not st.session_state.run_processing:
    st.session_state.run_processing = True
    st.session_state.analysis_report = None
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    st.session_state.video_path = video_path

# --- Main Logic controlled by session state ---
if st.session_state.run_processing:
    q_scores, angles_for_reps = [], []
    cap = None
    try:
        cap = cv2.VideoCapture(st.session_state.video_path)
        stframe = st.empty()
        predicted_class, q_score = "waiting...", 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                feature_row_raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                prediction = model.predict([feature_row_raw])
                predicted_class = prediction[0]

                if predicted_class in prototypes.index:
                    user_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    hip_center = (user_landmarks[23] + user_landmarks[24]) / 2
                    shoulder_width = norm(user_landmarks[11] - user_landmarks[12])

                    if shoulder_width > 1e-5:
                        user_pose_norm = ((user_landmarks - hip_center) / shoulder_width).flatten()
                        prototype_pose_norm = prototypes.loc[predicted_class].values
                        cosine_similarity = np.dot(user_pose_norm, prototype_pose_norm) / (norm(user_pose_norm) * norm(prototype_pose_norm))
                        q_score = max(0, cosine_similarity * 100)
                    else: q_score = 0
                else: q_score = 0

                q_scores.append(q_score)
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angles_for_reps.append(calculate_angle(l_shoulder, l_elbow, l_wrist))
            except Exception as e: pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.rectangle(image, (0, 0), (320, 120), (20, 20, 20), -1)
            cv2.putText(image, 'EXERCISE', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.putText(image, predicted_class.replace("_", " ").title(), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Q-SCORE', (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(int(q_score)), (180, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
            
            stframe.image(image, channels="BGR", use_container_width=True)
            
    finally:
        if cap is not None: cap.release()
        if os.path.exists(st.session_state.get('video_path', '')):
            try: os.remove(st.session_state.video_path)
            except (PermissionError, FileNotFoundError): pass

    if q_scores:
        rep_count, stage = 0, "down"
        for angle in angles_for_reps:
            if angle > 160: stage = "down"
            if angle < 40 and stage == "down": stage = "up"; rep_count += 1
        
        report_data = {
            "rep_count": rep_count, "avg_q_score": np.mean(q_scores),
            "consistency": 100 - np.std(q_scores), "min_q_score": np.min(q_scores),
            "q_scores_over_time": q_scores
        }
        verdict, feedback = get_form_feedback(report_data)
        report_data["verdict"], report_data["feedback"] = verdict, feedback
        print("\n--- APP DATA FOR PLOTTING ---")
        print("angles_for_reps =", angles_for_reps)
        print("q_scores =", q_scores)
        st.session_state.analysis_report = report_data
    
    st.session_state.run_processing = False
    st.rerun()

# --- Display Report Logic ---
if st.session_state.analysis_report:
    report = st.session_state.analysis_report
    st.header("📋 Final Analysis Report")
    st.subheader(report['verdict'])
    st.info(report['feedback'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric(label="Total Reps", value=f"{report['rep_count']}")
    with col2: st.metric(label="Average Q-Score", value=f"{report['avg_q_score']:.1f}")
    with col3: st.metric(label="Consistency", value=f"{report['consistency']:.1f}%")
    with col4: st.metric(label="Lowest Score Point", value=f"{report['min_q_score']:.1f}")
        
    st.subheader("Q-Score Over Time")
    st.line_chart(report['q_scores_over_time'])