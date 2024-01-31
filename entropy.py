import cv2
import numpy as np

def calculate_entropy(region):
    hist = cv2.calcHist([region], [0], None, [256], [0,256])
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def frame_entropy(frame, grid_size):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    entropy_values = []

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            region = gray[y:y+grid_size, x:x+grid_size]
            entropy_values.append(calculate_entropy(region))

    return np.mean(entropy_values)

# Example usage
frame = cv2.imread('path_to_frame.jpg')
avg_entropy = frame_entropy(frame, 32)  # Example grid size of 32x32
print(f"Average Entropy: {avg_entropy}")
