import cv2
import numpy as np

def calculate_entropy(region):
    hist = cv2.calcHist([region], [0], None, [256], [0, 256])
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

# Load video
cap = cv2.VideoCapture('/home/rahim/Documents/datasets/traffic/sh/1_Relaxing_highway_traffic.mp4')  # Update with the path to your video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate average entropy for the frame
    avg_entropy = frame_entropy(frame, 32)  # Example grid size of 32x32

    # Display the entropy on the frame
    entropy_text = f"Average Entropy: {avg_entropy:.2f}"
    cv2.putText(frame, entropy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
