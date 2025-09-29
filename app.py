# app.py
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from flask import Flask, request, send_file
import io

# Initialize Flask App and MediaPipe
app = Flask(__name__)
mp_face_mesh = mp.solutions.face_mesh

def detect_faces_in_image(image_bytes: bytes) -> bytes:
    """Takes image bytes, detects faces, draws boxes, and returns image bytes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    output_frame = img.copy()

    with mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = output_frame.shape
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                x_min, y_min = int(min(x_coords)), int(min(y_coords))
                x_max, y_max = int(max(x_coords)), int(max(y_coords))

                cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    success, encoded_image = cv2.imencode('.png', output_frame)
    if success:
        return encoded_image.tobytes()
    else:
        # Fallback to return the original image if encoding fails
        return image_bytes

# API endpoint that receives the image
@app.route('/detect', methods=['POST'])
def upload_and_detect():
    if 'file' not in request.files:
        return "Please provide a file.", 400
    file = request.files['file']
    if file:
        image_bytes = file.read()
        processed_image_bytes = detect_faces_in_image(image_bytes)
        
        return send_file(
            io.BytesIO(processed_image_bytes),
            mimetype='image/png'
        )

    return "Error processing file.", 500
