import cv2
import mediapipe as mp
import numpy as np
import os

# Input/output paths
input_video_path = '../assets/hand.mp4'
output_video_path = 'output_foreground.mp4'

# Initialize video capture
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("❌ Failed to open input video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Set up output writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie_segmentation:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_frame)

        # Create mask
        mask = results.segmentation_mask
        condition = mask > 0.1
        condition = np.stack((condition,) * 3, axis=-1)

        # Keep foreground, black out background
        output_frame = np.where(condition, frame, 0)

        out.write(output_frame)

        # Optional display for debug
        # cv2.imshow("Segmented Foreground", output_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Foreground video saved at: {os.path.abspath(output_video_path)}")
