import cv2
import mediapipe as mp
import numpy as np
import os

mp_selfie_segmentation = mp.solutions.selfie_segmentation
input_video_path = '../assets/hand.mp4'
background_video_path = '../assets/vrbg.mp4'
output_video_path = 'output_segmented.mp4'

input_cap = cv2.VideoCapture(input_video_path)
bg_cap = cv2.VideoCapture(background_video_path)

if not input_cap.isOpened():
    print("❌ Failed to open input video.")
    exit()
if not bg_cap.isOpened():
    print("❌ Failed to open background video.")
    exit()

frame_width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_cap.get(cv2.CAP_PROP_FPS)

if fps == 0 or frame_width == 0 or frame_height == 0:
    print("❌ Invalid video properties.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while input_cap.isOpened():
        success, frame = input_cap.read()
        if not success:
            break

        ret, bg_frame = bg_cap.read()
        if not ret or bg_frame is None:
            bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bg_frame = bg_cap.read()
        if bg_frame is None:
            bg_frame = np.zeros(frame.shape, dtype=np.uint8)
        else:
            bg_frame = cv2.resize(bg_frame, (frame.shape[1], frame.shape[0]))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_frame)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1

        output_frame = np.where(condition, frame, bg_frame)
        out.write(output_frame)

        cv2.imshow("Segmented Output", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

input_cap.release()
bg_cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Saved to: {os.path.abspath(output_video_path)}")
