import cv2
import numpy as np

# --- Configuration ---
# Path to the YuNet model
MODEL_PATH = "face_detection_yunet_2023mar.onnx"
# Confidence threshold to filter out weak detections
CONFIDENCE_THRESHOLD = 0.9
# Non-Maximum Suppression threshold
NMS_THRESHOLD = 0.3
# Input image size for the model
MODEL_INPUT_SIZE = (320, 320)

# --- Visualization Colors ---
# Bounding box color (Blue, Green, Red)
BBOX_COLOR = (255, 0, 0)
# Landmark color
LANDMARK_COLOR = (0, 0, 255)
# Text color
TEXT_COLOR = (0, 255, 0)

def visualize(image, detections):
    """Draws the detection results on the image."""
    output_image = image.copy()
    
    if detections is None:
        return output_image

    for det in detections:
        # Bounding box: det[0:4] -> (x, y, w, h)
        bbox = det[0:4].astype(np.int32)
        cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), BBOX_COLOR, 2)

        # Confidence score: det[14]
        confidence = det[14]
        confidence_text = f"{confidence:.2f}"
        cv2.putText(output_image, confidence_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
        
        # Landmarks: det[4:14] -> 5 points (x, y)
        # right eye, left eye, nose tip, right mouth corner, left mouth corner
        landmarks = det[4:14].reshape((5, 2)).astype(np.int32)
        for point in landmarks:
            cv2.circle(output_image, tuple(point), 2, LANDMARK_COLOR, 2)
            
    return output_image

def main():
    # Initialize the face detector using the YuNet model
    try:
        detector = cv2.FaceDetectorYN.create(
            model=MODEL_PATH,
            config="",
            input_size=MODEL_INPUT_SIZE,
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            top_k=5000
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please make sure '{MODEL_PATH}' is in the correct directory.")
        return

    # Start video capture from the default webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
            
        # Flip the frame horizontally for a mirror-like view
        # frame = cv2.flip(frame, 1)
        cap = cv2.VideoCapture(2)
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Set the detector's input size to match the frame size for better accuracy
        detector.setInputSize((frame_width, frame_height))

        # Perform face detection
        # The result is a tuple, where the second element contains the detections
        _, faces = detector.detect(frame)

        # Draw the results on the frame
        result_frame = visualize(frame, faces)
        
        # Display the output
        cv2.imshow('Advanced Face Detection (YuNet)', result_frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()