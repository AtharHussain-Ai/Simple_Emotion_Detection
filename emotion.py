import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def apply_terminator_overlay(image):
    overlay = image.copy()
    red_overlay = np.zeros_like(image, dtype=np.uint8)
    red_overlay[:, :, 2] = 180  
    cv2.addWeighted(red_overlay, 0.4, overlay, 0.6, 0, overlay)
 
    grid_overlay = np.zeros_like(image, dtype=np.uint8)
    for i in range(0, grid_overlay.shape[1], 50):
        cv2.line(grid_overlay, (i, 0), (i, grid_overlay.shape[0]), (255, 255, 255), 1)
    for i in range(0, grid_overlay.shape[0], 50):
        cv2.line(grid_overlay, (0, i), (grid_overlay.shape[1], i), (255, 255, 255), 1)
    
    cv2.addWeighted(grid_overlay, 0.2, overlay, 1.0, 0, overlay)
    
    return overlay

def terminator_face_hand_emotion():
    cap = cv2.VideoCapture(0)

    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection, \
         mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty frame.")
                continue
            frame = cv2.flip(frame, 1)
            overlay_frame = apply_terminator_overlay(frame)
            
            rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)

            face_results = face_detection.process(rgb_frame)
            detected_emotion = "Unknown"

            if face_results.detections:
                for detection in face_results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    box = (int(bbox.xmin * w), int(bbox.ymin * h),
                           int(bbox.width * w), int(bbox.height * h))
                    face_roi = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                    
    
                    if face_roi.size > 0: 
                        try:
                            emotions = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    
                            if emotions:
                                detected_emotion = emotions[0]['dominant_emotion']
                        except Exception as e:
                            print("Emotion detection error:", e)

                
                    cv2.rectangle(overlay_frame, (box[0], box[1]),
                                  (box[0] + box[2], box[1] + box[3]),
                                  (255, 255, 255), 2)

   
            hand_results = hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        overlay_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            center_x, center_y = overlay_frame.shape[1] // 2, overlay_frame.shape[0] // 2
            cv2.drawMarker(overlay_frame, (center_x, center_y), (255, 255, 255), 
                           cv2.MARKER_CROSS, 20, 2)

            emotion_text = f'Emotion: {detected_emotion}'
            text_size = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = overlay_frame.shape[1] - text_size[0] - 10
            text_y = 30 
            cv2.putText(overlay_frame, emotion_text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Terminator Face, Hand & Emotion Recognition', overlay_frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

terminator_face_hand_emotion()
