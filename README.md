# Drowsiness Detection System 👀😴🚗




https://github.com/user-attachments/assets/519e1b30-6c8d-433a-8883-21a9f2c1d19c


A real-time computer vision system that detects signs of driver drowsiness and distraction using facial landmarks analysis.

## Features ✨
- 👁 **Eye Closure Detection** using Eye Aspect Ratio (EAR)
- 😮 **Yawn Detection** using Mouth Aspect Ratio (MAR)
- 🧭 **Head Pose Estimation** to detect distracted driving
- 📊 **Visual Indicators** for all detected drowsiness signals
- ⚙️ Configurable thresholds for sensitivity adjustment

## Installation 🛠️

### Prerequisites
- Python 3.6+
- Webcam

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection


2. Install Dependencies:
   ```bash
   pip install -r requirements.txt

 3. Download the facial landmark predictor:
- Download shape_predictor_68_face_landmarks.dat
- Extract and place it in dilb_shape_predictor/ directory

 

    ## Usage 🚀

  ```bash
   python drowsiness_detection.py

Controls:
- Press q to quit the application





## Configuration ⚙️
- EYE_AR_THRESH = 0.25         # Eye aspect ratio threshold
- MOUTH_AR_THRESH = 0.79       # Mouth aspect ratio threshold
- EYE_AR_CONSEC_FRAMES = 3     # Consecutive frames for drowsiness alert
- frame_width = 1024           # Camera resolution
- frame_height = 576




## How It Works 🔍
**Face Detection** : Uses dlib's HOG-based face detector

**Facial Landmarks** : 68-point landmark detection

**Eye Aspect Ratio** : Calculates ratio of eye width to height

**Mouth Aspect Ratio** : Calculates ratio of mouth opening

**Head Pose** : Estimates head orientation using solvePnP


### Project Structure 📂
   ```bash
├── drowsiness_detection.py       # Main script
├── EAR.py                        # Eye Aspect Ratio calculations
├── MAR.py                        # Mouth Aspect Ratio calculations
├── Headpose.py                   # Head pose estimation
├── dilb_shape_predictor/
│   └── shape_predictor_68_face_landmarks.dat
├── requirements.txt              # Dependencies
└── README.md








