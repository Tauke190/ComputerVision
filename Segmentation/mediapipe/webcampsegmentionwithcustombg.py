import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Open webcam
cap = cv2.VideoCapture(0)

# Open background video
bg_video = cv2.VideoCapture('../assets/bg_video.mp4')  # Change this to your background video path

print(bg_video)

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Read frame from background video
        ret, bg_frame = bg_video.read()
        if not ret:
            bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            ret, bg_frame = bg_video.read()
        bg_frame = cv2.resize(bg_frame, (image.shape[1], image.shape[0]))

        # Flip and process webcam image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = selfie_segmentation.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create mask condition
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        # Combine segmented person with video background
        output_image = np.where(condition, image, bg_frame)

        cv2.imshow('MediaPipe Selfie Segmentation with Video Background', output_image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
bg_video.release()
cv2.destroyAllWindows()
