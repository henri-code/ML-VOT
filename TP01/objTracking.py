### import ###
import cv2
from KalmanFilter import KalmanFilter 
import TP1_data.Detector as Detector

import numpy as np
### end import ###

### variables ###
dt = 0.1
u_x = 1
u_y = 1
std_acc = 1
x_std_meas = 0.1
y_std_meas = 0.1
### end variables ###

# Create video capture object
video_capture = cv2.VideoCapture('TP1_data/randomball.avi')

kf = KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

# Create an empty list to store trajectory points
trajectory_points = []

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Object detection
    centroid = Detector.detect(frame)  

    if centroid is not None:
        # Track the object using Kalman filter
        kf.predict()
        kf.update(np.array([centroid[0][0], centroid[0][1]]))

        # Get the estimated state from Kalman filter
        state = kf.get_state()
        estimated_position = (int(state[0]), int(state[1]))

        # Add the estimated position to the trajectory
        trajectory_points.append(estimated_position)

        # Visualize tracking results
        # Draw detected circle (green color)
        cv2.circle(frame, (int(centroid[0][0][0]), int(centroid[0][1][0])), 5, (0, 255, 0), -1)

        # Draw a blue rectangle as the predicted object position
        cv2.rectangle(frame, (int(estimated_position[0]) - 10, int(estimated_position[1]) - 10),
                      (int(estimated_position[0]) + 10, int(estimated_position[1]) + 10), (255, 0, 0), 2)

        # Draw a red rectangle as the estimated object position
        cv2.rectangle(frame, (int(centroid[0][0][0]) - 10, int(centroid[0][1][0]) - 10),
                      (int(centroid[0][0][0]) + 10, int(centroid[0][1][0]) + 10), (0, 0, 255), 2)

        # Display the trajectory (tracking path) in an image
        for i in range(1, len(trajectory_points)):
            cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (255, 255, 0), 2)

    # Show the frame with tracking information
    cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
