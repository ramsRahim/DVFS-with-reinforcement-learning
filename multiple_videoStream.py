import cv2
import torch
import numpy as np
import time


with open('category_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
vehicle_class_names = {'car', 'motorcycle', 'bus', 'truck' , 'bicycle'}  # Add other vehicle types if needed
vehicle_classes = [i for i, name in enumerate(class_names) if name in vehicle_class_names]

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Paths to your video streams
video_paths = ['/home/rahim/Documents/datasets/traffic/sh/1_Relaxing_highway_traffic.mp4',
                '/home/rahim/Documents/datasets/traffic/sh/2_night_traffic_shanghai_guangfuxilu_202308102030_720.mp4', 
                '/home/rahim/Documents/datasets/traffic/sh/3_traffic_shanghai_jinshajianglu_202308050815_720.mp4'] 
               #'/home/rahim/Documents/datasets/traffic/sh/4_traffic_shanghai_jinshajianglu_202308050815_720.mp4']

# Initialize video captures
caps = [cv2.VideoCapture(path) for path in video_paths]
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
old_frames = [ [] for _ in range(len(caps))]
old_grays = [ [] for _ in range(len(caps))]
p0 = [ [] for _ in range(len(caps))]
for i, cap in enumerate(caps):
    _, old_frames[i] = cap.read()
    old_grays[i] = cv2.cvtColor(old_frames[i], cv2.COLOR_BGR2GRAY)
    p0[i] = cv2.goodFeaturesToTrack(old_grays[i], mask=None, **feature_params)
# Create a mask image for drawing purposes
masks = [np.zeros_like(old_frame) for old_frame in old_frames]


frame_count =  [0 for _ in range(len(caps))]
start_time = time.time()

# Initialize batching structures
frame_batches = [[] for _ in caps]  # Hold batches of frames for each stream
batch_size = 3  # Number of frames to accumulate before processing

# Process each frame in the videos
while True:
    for i, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue  # Skip if frame is not read successfully
        
        # Accumulate frames for the current stream
        frame_batches[i].append(frame)
        if len(frame_batches[i]) == batch_size:
            # Process the batch
            for batch_frame in frame_batches[i]:
                frame_count[i] += 1
                # Preprocess the frame

                # Perform inference
                results = model(batch_frame)
                detections = results.xyxy[0].cpu().numpy()

                vehicle_detections = [det for det in detections if int(det[5]) in vehicle_classes]
                vehicle_count = len(vehicle_detections)

                frame_gray = cv2.cvtColor(batch_frame, cv2.COLOR_BGR2GRAY)

                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_grays[i], frame_gray, p0[i], None, **lk_params)

                # Select good points
                good_new = p1[st == 1]
                good_old = p0[i][st == 1]
                fps = 30
                # Calculate FPS every second or every few frames
                if frame_count[i] % 1 == 0:  # Adjust the interval as needed
                    end_time = time.time()
                    fps = frame_count[i] / (end_time - start_time)
                    cv2.putText(batch_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Reset frame count and start time for next interval
                    frame_count[i] = 0
                    start_time = time.time()

                for det in vehicle_detections:
                    x1, y1, x2, y2, _, _ = map(int, det)
                    cv2.rectangle(batch_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Calculate speed
                    speed = calculate_speed(np.float32(good_old), np.float32(good_new),fps)  # Placeholder function
                    cv2.putText(batch_frame, f"Speed: {speed:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


                count_text = f"Vehicle Count: {vehicle_count}"
                cv2.putText(batch_frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                window_name = f"Stream {i+1}"
                cv2.imshow(window_name, batch_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Break the loop if 'q' is pressed
        break

# Release all video captures
for cap in caps:
    cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
