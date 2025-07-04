import argparse
import cv2
import mediapipe as mp

def process_img(img , face_detection):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)


    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w , h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            #blur faces
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30)) 
    return img

args = argparse.ArgumentParser()

args.add_argument("--mode", default='image')
args.add_argument("--filepath",default=r'C:\\Users\\waghd\\Downloads\\face.jpeg')

args = parser.parse_args()

H, W, _ = img.shape
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ["image"]:
        img = cv2.imread(args.filepath)
        
        H, W, _ = img.shape
        img = process_img(img, face_detection)

    
    



    

   # cv2.imshow('img', img)

   # cv2.waitKey(0)
        cv2.imwrite(r'C:\\Users\\waghd\\Downloads\\output.jpeg', img) 