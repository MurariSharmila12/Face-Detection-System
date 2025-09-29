"""
High-Performance Real-Time Face Detection with Bounding Box.

This script uses the MediaPipe Face Mesh model for highly accurate face detection
and then calculates a bounding box from the landmarks to draw a green rectangle,
similar to traditional face detectors.

Dependencies:
    - opencv-python
    - mediapipe

Installation:
    pip install opencv-python mediapipe

Usage:
    Run the script from your terminal: python face_mesh.py
    Press 'q' to quit the application.
"""
import cv2
import mediapipe as mp
import numpy as np

# Define type aliases for clarity
Mat = np.ndarray

# --- Configuration Constants ---
WEBCAM_INDEX: int = 1  # Make sure this is the correct number for your webcam!
WINDOW_NAME: str = "Real-Time Face Detection"
BOX_COLOR = (0, 255, 0)  # Green color for the rectangle (BGR format)
BOX_THICKNESS = 2

# --- MediaPipe Initialization ---
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def process_frame(frame: Mat, face_mesh: mp_face_mesh.FaceMesh) -> Mat:
    """
    Processes a single video frame to detect a face and draw a bounding box.

    Args:
        frame (Mat): The input video frame from OpenCV (in BGR format).
        face_mesh (mp.solutions.face_mesh.FaceMesh): The MediaPipe Face Mesh model instance.

    Returns:
        Mat: The processed frame with the bounding box drawn on it.
    """
    frame = cv2.flip(frame, 1)
    output_frame = frame.copy()  # Work on a copy to keep the original clean

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    results = face_mesh.process(rgb_frame)
    rgb_frame.flags.writeable = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # --- NEW: Bounding Box Calculation ---
            h, w, _ = output_frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # Find the min and max coordinates from all 478 landmarks
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                if x < x_min: x_min = x
                if y < y_min: y_min = y
                if x > x_max: x_max = x
                if y > y_max: y_max = y

            # Draw the green rectangle on the output frame
            cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max),
                          BOX_COLOR, BOX_THICKNESS)
            # --- End of New Code ---

            # --- OLD MESH DRAWING (Now commented out) ---
            # You can uncomment these lines if you ever want the detailed mesh again.
            # mp_drawing.draw_landmarks(
            #     image=output_frame,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_TESSELATION,
            #     connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0)))

    return output_frame


def main() -> None:
    """
    Main function to run the face detection application.
    """
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with index {WEBCAM_INDEX}.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_face_mesh.FaceMesh(
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            processed_frame = process_frame(frame, face_mesh)
            cv2.imshow(WINDOW_NAME, processed_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


# # streamlit_app.py
# import cv2
# import mediapipe as mp
# import streamlit as st
# from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# # --- MediaPipe Initialization ---
# mp_drawing = mp.solutions.drawing_utils
# mp_face_mesh = mp.solutions.face_mesh

# # Define the styles for the face mesh
# MESH_TESSELATION_STYLE = mp_drawing.DrawingSpec(
#     color=(0, 255, 0), thickness=1, circle_radius=1)
# MESH_CONTOURS_STYLE = mp_drawing.DrawingSpec(
#     color=(255, 255, 255), thickness=2, circle_radius=1)

# class FaceMeshTransformer(VideoTransformerBase):
#     """
#     A class to process video frames and apply the Face Mesh model.
#     """
#     def __init__(self):
#         # Initialize the Face Mesh model once
#         self.face_mesh = mp_face_mesh.FaceMesh(
#             max_num_faces=5,
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5)

#     def transform(self, frame):
#         # The transform method receives a video frame and returns a processed one.
#         img = frame.to_ndarray(format="bgr24")
        
#         # Convert the BGR image to RGB.
#         rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Process the image and find face landmarks.
#         results = self.face_mesh.process(rgb_frame)

#         # Draw the face mesh annotations on the image.
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image=img,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_TESSELATION,
#                     connection_drawing_spec=MESH_TESSELATION_STYLE)
#                 mp_drawing.draw_landmarks(
#                     image=img,
#                     landmark_list=face_landmarks,
#                     connections=mp_face_mesh.FACEMESH_CONTOURS,
#                     connection_drawing_spec=MESH_CONTOURS_STYLE)
        
#         return img

# # --- Streamlit App UI ---
# st.set_page_config(page_title="Real-Time 3D Face Mesh", page_icon="😊")

# st.title("Real-Time 3D Face Mesh")
# st.write("This app uses your webcam to detect and overlay a 3D face mesh in real-time.")
# st.write("Click 'START' to begin, and make sure to grant camera permissions in your browser.")

# # The main component that handles the webcam stream
# webrtc_streamer(
#     key="face-mesh",
#     video_transformer_factory=FaceMeshTransformer,
#     rtc_configuration={  # This is needed for deployment
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     },
#     media_stream_constraints={"video": True, "audio": False}
# )

# st.sidebar.header("About")
# st.sidebar.info("This app is powered by MediaPipe and Streamlit. It demonstrates real-time facial landmark detection directly in the browser.")