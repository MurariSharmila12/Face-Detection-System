import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import io

# Set page configuration
st.set_page_config(
    page_title="Face Detection System",
    page_icon="🎭",
    layout="centered"
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

def detect_faces_in_image(image_bytes: bytes) -> bytes:
    """Takes image bytes, detects faces, draws boxes, and returns image bytes."""
    # Convert bytes to a NumPy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Create a copy to draw on
    output_frame = img.copy()

    # Initialize FaceMesh with parameters optimized for multiple faces
    with mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=20, 
            refine_landmarks=False,
            min_detection_confidence=0.1, # Set to the lowest practical confidence
            min_tracking_confidence=0.5) as face_mesh:

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the frame to find face landmarks
        results = face_mesh.process(rgb_frame)

        # Draw rectangles around detected faces
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = output_frame.shape
                # Get the bounding box coordinates
                x_coords = [landmark.x * w for landmark in face_landmarks.landmark]
                y_coords = [landmark.y * h for landmark in face_landmarks.landmark]
                
                x_min, y_min = int(min(x_coords)), int(min(y_coords))
                x_max, y_max = int(max(x_coords)), int(max(y_coords))

                # Draw the rectangle on the output frame
                cv2.rectangle(output_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Encode the processed image back to bytes
    success, encoded_image = cv2.imencode('.png', output_frame)
    if success:
        return encoded_image.tobytes()
    else:
        # Fallback to return the original image if encoding fails
        return image_bytes

# --- Streamlit User Interface ---

st.title("🎭 Face Detection System")
st.write("Upload an image and the application will detect the faces in it.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file bytes
    image_bytes = uploaded_file.getvalue()
    
    # Display the original image using the correct parameter
    st.image(image_bytes, caption='Original Image', use_column_width='auto')
    
    # A button to trigger the detection
    if st.button('Detect Faces'):
        with st.spinner('Processing...'):
            # Process the image to detect faces
            processed_image_bytes = detect_faces_in_image(image_bytes)
            
            # Display the processed image using the correct parameter
            st.image(processed_image_bytes, caption='Processed Image', use_column_width='auto')
