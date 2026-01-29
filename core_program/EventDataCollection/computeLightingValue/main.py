import cv2
import numpy as np
import pandas as pd

# File paths from uploaded images
paths = {
    "img_dark": "F:\\video\\illustatation\\low.jpg",
    "img_bright": "F:\\video\\illustatation\\high.jpg",
    "img_medium": "F:\\video\\illustatation\\mediate.jpg",
}


def compute_metrics(path):
    img = cv2.imread(path)
    if img is None:
        return None

    # 1. Grayscale brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_mean = gray.mean()

    # 2. HSV V-channel brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_mean = hsv[:, :, 2].mean()

    # 3. Lab L-channel brightness
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l_mean = lab[:, :, 0].mean()

    return gray_mean, v_mean, l_mean


# Collect results
results = []
for name, path in paths.items():
    metrics = compute_metrics(path)
    if metrics:
        results.append([name, *metrics])

# Create a DataFrame
df = pd.DataFrame(results, columns=["Image", "Gray Mean", "HSV-V Mean", "Lab-L Mean"])
print(df)
