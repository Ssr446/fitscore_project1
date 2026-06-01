# FitScore: AI Exercise Analysis 🏋️

FitScore is an AI-powered fitness analysis tool that uses computer vision to evaluate exercise form in real-time. It analyzes video input to detect body landmarks, compares the user's form against idealized prototype poses, and calculates a "Q-Score" to provide actionable feedback.

## Features
- **Real-Time Pose Detection:** Uses MediaPipe for accurate 3D pose landmarks.
- **Form Evaluation:** Computes Q-Scores based on cosine similarity with normalized prototype poses.
- **Rep Counting & Angle Tracking:** Automatically counts repetitions and tracks joint angles.
- **Dual Interfaces:** Choose between a Streamlit web app or a FastAPI backend with a custom HTML/JS frontend.

## Project Structure
- `app.py`: Streamlit application.
- `server.py`: FastAPI backend application.
- `frontend/`: Contains HTML, CSS, and JS files for the custom web interface.
- `training_notebook.ipynb`: Jupyter notebook for model training and prototype generation.
- `model.pkl` & `prototype_poses_normalized.csv`: Pre-trained model and prototype data.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fitscore_project1.git
   cd fitscore_project1
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You have two options for running the application:

### Option 1: Streamlit App
Provides a quick and easy-to-use web interface.
```bash
streamlit run app.py
```

### Option 2: FastAPI + Custom Frontend
Runs a robust REST API backend with a separate static frontend.
```bash
uvicorn server:app --reload
```
Once the server is running, open your browser and navigate to `http://localhost:8000/` to access the interface.

## Deployment

### Streamlit App Deployment (Easiest)
1. Push this repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app", select your repository, and set the main file path to `app.py`.
4. Click "Deploy".

### FastAPI Deployment (Advanced)
You can deploy the FastAPI server using services like **Render**, **Railway**, or **Heroku**. You will need to create a `Procfile` or `Dockerfile` depending on the platform's requirements.

## License
MIT License
