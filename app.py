import streamlit as st
import cv2
import numpy as np
import time
from imutils import face_utils
import dlib

from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from Headpose import getHeadTiltAndCoords

# Load models once and cache them
@st.cache_resource
def load_models():
    predictor_path = r'C:\Users\Rushik\OneDrive\Desktop\Drowsiness_Detection\dilb_shape_predictor\shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor

detector, predictor = load_models()

# Streamlit UI
st.title("Driver Drowsiness Detection")

run = st.checkbox('Start Camera', key='start_camera')

FRAME_WINDOW = st.image([])

# Parameters
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0

if run:
    cap = cv2.VideoCapture(0)
    time.sleep(1.0)

    frame_width = 1024
    frame_height = 576

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = (49, 68)

    # Initialize keep_running checkbox outside the loop
    keep_running = st.checkbox('Keep running', value=True, key='keep_running')

    while keep_running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture frame from camera.")
            break

        frame = cv2.resize(frame, (frame_width, frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        rects = detector(gray, 0)

        if len(rects) > 0:
            cv2.putText(frame, f"{len(rects)} face(s) found", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        for rect in rects:
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "Eyes Closed!", (500, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0

            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.putText(frame, f"MAR: {mar:.2f}", (650, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "Yawning!", (800, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            landmark_indices = {33: 0, 8: 1, 36: 2, 45: 3, 48: 4, 54: 5}
            image_points = np.zeros((6, 2), dtype="double")
            for (i, (x, y)) in enumerate(shape):
                if i in landmark_indices:
                    image_points[landmark_indices[i]] = (x, y)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                else:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            head_tilt_degree, start_point, end_point, end_point_alt = getHeadTiltAndCoords(size, image_points, frame_height)
            cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
            cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

            if head_tilt_degree:
                cv2.putText(frame, 'Head Tilt Degree: ' + str(round(head_tilt_degree[0], 2)), (170, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Convert BGR to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        FRAME_WINDOW.image(frame)

        # Update keep_running from session state to detect checkbox toggle
        keep_running = st.session_state.keep_running

    cap.release()
else:
    st.text("Click 'Start Camera' to begin detection")
