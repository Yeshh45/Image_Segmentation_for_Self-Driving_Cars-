import torch
import cv2
import numpy as np
from model import UNet

model = UNet()
model.load_state_dict(torch.load("unet_model.pth"))
model.eval()

def predict(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (256, 256)) / 255.0
    img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        mask = (output.squeeze().numpy() > 0.5).astype(np.uint8) * 255

    cv2.imwrite("output.png", mask)
    print("Segmentation saved as output.png")

predict("test_image.png")
