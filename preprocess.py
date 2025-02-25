import cv2
import os
import numpy as np
from glob import glob

IMAGE_SIZE = (256, 256)

def load_images_and_masks(image_path, mask_path):
    images = sorted(glob(os.path.join(image_path, "*.png")))
    masks = sorted(glob(os.path.join(mask_path, "*.png")))

    X, Y = [], []
    for img, mask in zip(images, masks):
        img = cv2.imread(img)
        img = cv2.resize(img, IMAGE_SIZE)
        img = img / 255.0  # Normalize
        
        mask = cv2.imread(mask, 0)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = mask / 255.0  # Normalize

        X.append(img)
        Y.append(mask)

    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, Y = load_images_and_masks("data/images", "data/masks")
    print(f"Loaded {len(X)} images and {len(Y)} masks")
