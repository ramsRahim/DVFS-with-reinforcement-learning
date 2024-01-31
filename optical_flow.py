import cv2
import numpy as np
import torch
from pathlib import Path
import sys

# Load YOLOv5 model 
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # For example, yolov5s model

# Load class names and define vehicle classes
with open('category_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
vehicle_class_names = {'car', 'motorcycle', 'bus', 'truck' , 'bicycle'}  # Add other vehicle types if needed
vehicle_classes = [i for i, name in enumerate(class_names) if name in vehicle_class_names]

# Load video
cap = cv2.VideoCapture('/home/rahim/Documents/datasets/traffic/sh/1_Relaxing_highway_traffic.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)

# Parameters for Shi-Tomasi corner detection (good features to track)
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Function to calculate speed (pixels/frame to pixels/second)
def calculate_speed(old_points, new_points, fps):
    # Calculate displacement
    displacement = np.linalg.norm(new_points - old_points, axis=1)

    # Speed in pixels/sec (assuming consecutive frames)
    speed = displacement * fps  
    return np.mean(speed)  # Returning average speed of points

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

     # Perform object detection using YOLOv5
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()   # Detections in xyxy format

    # Filter out detections that are not vehicles
    vehicle_detections = [det for det in detections if int(det[5]) in vehicle_classes]
    vehicle_count = len(vehicle_detections)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw bounding boxes and speed on the frame
    for det in vehicle_detections:
        x1, y1, x2, y2, _, _ = map(int, det)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Calculate speed
        speed = calculate_speed(np.float32(good_old), np.float32(good_new), fps)  # Placeholder function
        cv2.putText(frame, f"Speed: {speed:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # # Draw the tracks
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     # Extract coordinates and ensure they are integers
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     a, b, c, d = int(a), int(b), int(c), int(d)

    #     # Draw line and circle
    #     mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
    #     frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), -1)

    count_text = f"Vehicle Count: {vehicle_count}"
    cv2.putText(frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    img = cv2.add(frame, mask)

    # Show the frame
    cv2.imshow('Frame', img)

    # Update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
