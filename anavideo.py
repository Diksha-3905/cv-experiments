import argparse
import cv2
import mediapipe as mp

def process_frame(frame, face_detection):
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = face_detection.process(frame_rgb)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box

            x1 = int(bbox.xmin * W)
            y1 = int(bbox.ymin * H)
            w = int(bbox.width * W)
            h = int(bbox.height * H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            roi = frame[y1:y2, x1:x2]
            if roi.size != 0:
                frame[y1:y2, x1:x2] = cv2.blur(roi, (30, 30))
    return frame

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='video', choices=['image', 'video'])
parser.add_argument("--filepath", default='0', help="Path to video file or use 0 for webcam")
args = parser.parse_args()

# Setup face detection
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    # Video Mode
    if args.mode == 'video':
        # Use webcam if '0' is passed
        video_path = 0 if args.filepath == '0' else args.filepath
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Unable to open video source.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame, face_detection)

            cv2.imshow('Blurred Faces - Press Q to Quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
