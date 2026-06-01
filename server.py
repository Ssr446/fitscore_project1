import cv2
import mediapipe as mp
import numpy as np
import pickle
import tempfile
import os
import pandas as pd
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uuid

# --- Load Models and Prototypes ---
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Warning: model.pkl not found. Make sure it exists.")
    model = None

try:
    prototypes = pd.read_csv('prototype_poses_normalized.csv', index_col='class')
except FileNotFoundError:
    print("Warning: prototype_poses_normalized.csv not found.")
    prototypes = None

# --- Helper Functions ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360 - angle
    return angle

def get_form_feedback(report):
    if 'avg_q_score' not in report or report['avg_q_score'] is None or pd.isna(report['avg_q_score']):
        return "Not Enough Data", "Could not analyze form correctly."
        
    avg_q, consistency, min_q = report['avg_q_score'], report['consistency'], report['min_q_score']
    
    if np.isnan(avg_q):
        return "Unknown Form", "Could not analyze."
        
    if avg_q >= 85 and consistency >= 90 and min_q >= 70:
        verdict = "Excellent Form! 👍"
        feedback = "Your movements are highly consistent and closely match the ideal form."
    elif avg_q >= 70 and consistency >= 80 and min_q >= 55:
        verdict = "Good Form! 🙂"
        feedback = "You have a solid foundation. Focus on maintaining stability to improve further."
    else:
        verdict = "Needs Improvement 💡"
        feedback = f"Your form appears inconsistent. Your lowest score of {min_q:.0f} suggests a point where form broke down."
    return verdict, feedback

# --- Initialize FastAPI ---
app = FastAPI(title="FitScore API")

# Add CORS so our frontend can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "output_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount the static files (frontend) and output videos
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    if not model or prototypes is None:
        raise HTTPException(status_code=500, detail="Models not loaded properly on the server.")
        
    # Save uploaded video to a temporary file
    temp_input_fd, temp_input_path = tempfile.mkstemp(suffix=".mp4")
    with os.fdopen(temp_input_fd, "wb") as f:
        f.write(await file.read())

    # ID for the output files
    run_id = str(uuid.uuid4())
    output_filename = f"{run_id}.webm"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(temp_input_path)
    
    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # Using VP8 codec for WebM container (great for browser playback)
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    q_scores = []
    angles_for_reps = []
    predicted_class = "waiting..."
    q_score = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                feature_row_raw = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                
                # Check feature count
                expected_features = model.n_features_in_
                if len(feature_row_raw) == expected_features:
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
                        else:
                            q_score = 0
                    else:
                        q_score = 0
                
                q_scores.append(float(q_score))
                
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angles_for_reps.append(calculate_angle(l_shoulder, l_elbow, l_wrist))
        except Exception as e:
            print(f"Error during frame processing: {e}")
            pass

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        cv2.rectangle(image, (0, 0), (320, 120), (20, 20, 20), -1)
        cv2.putText(image, 'EXERCISE', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
        cv2.putText(image, predicted_class.replace("_", " ").title(), (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Q-SCORE', (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
        cv2.putText(image, str(int(q_score)), (180, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(image)

    cap.release()
    out.release()
    os.remove(temp_input_path)

    # Process metrics
    rep_count, stage = 0, "down"
    for angle in angles_for_reps:
        if angle > 160: stage = "down"
        if angle < 40 and stage == "down": 
            stage = "up"
            rep_count += 1

    if not q_scores:
        avg_q_score, consistency, min_q_score = 0, 0, 0
    else:
        avg_q_score = float(np.mean(q_scores))
        consistency = float(100 - np.std(q_scores))
        min_q_score = float(np.min(q_scores))

    report_data = {
        "rep_count": rep_count,
        "avg_q_score": avg_q_score,
        "consistency": consistency,
        "min_q_score": min_q_score,
        "q_scores_over_time": q_scores,
        "video_url": f"/outputs/{output_filename}"
    }
    
    verdict, feedback = get_form_feedback(report_data)
    report_data["verdict"] = verdict
    report_data["feedback"] = feedback

    return JSONResponse(content=report_data)

@app.get("/")
def redirect_to_frontend():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/frontend/index.html")
