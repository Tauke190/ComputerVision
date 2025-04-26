import cv2
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentor = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

cap = cv2.VideoCapture("input_video.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = segmentor.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mask = (results.segmentation_mask > 0.5).astype('uint8') * 255

    # Show or save output
    cv2.imshow("Hand Mask", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
