import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import numpy as np

# Load DeepLabV3 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet101(pretrained=True).to(device).eval()

# Transformation
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((360, 640)),  # Resize to match video frame size
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def get_person_mask(frame):
    image = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()

    print(mask)
    return mask == 15  # Class 15 is "person" in COCO

input_video = cv2.VideoCapture("input_video.mp4")
bg_video = cv2.VideoCapture("background_video.mp4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (640, 360))


while input_video.isOpened():
    ret1, frame1 = input_video.read()
    ret2, frame2 = bg_video.read()

    if not ret1 or not ret2:
        break

    frame1 = cv2.resize(frame1, (640, 360))
    frame2 = cv2.resize(frame2, (640, 360))

    person_mask = get_person_mask(frame1)

    person_mask_3c = np.stack([person_mask]*3, axis=-1)
    composite = np.where(person_mask_3c, frame1, frame2)

    out.write(composite)
    cv2.imshow('Background Replaced', composite)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

input_video.release()
bg_video.release()
out.release()
cv2.destroyAllWindows()