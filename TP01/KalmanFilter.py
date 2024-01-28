### import ###
import numpy as np


class KalmanFilter:
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        self.dt = dt
        self.u = np.array([[u_x], [u_y]]) 
        self.std_acc = std_acc
        self.x_std_meas = x_std_meas
        self.y_std_meas = y_std_meas

        # State matrix: [x, y, vx, vy] where vx and vy are velocities in x and y directions
        self.state = np.zeros((4, 1))

        # State transition matrix: describes how the state evolves from time k-1 to k
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Control matrix: relates control input to state transition
        self.B = np.array([[0.5 * self.dt ** 2, 0],
                           [0, 0.5 * self.dt ** 2],
                           [self.dt, 0],
                           [0, self.dt]])
        
        # Measurement matrix: relates measurements to the state space
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        # Process noise covariance matrix
        self.Q = np.array([[0.25 * self.dt ** 4, 0, 0.5 * self.dt ** 3, 0],
                           [0, 0.25 * self.dt ** 4, 0, 0.5 * self.dt ** 3],
                           [0.5 * self.dt ** 3, 0, self.dt ** 2, 0],
                           [0, 0.5 * self.dt ** 3, 0, self.dt ** 2]]) * self.std_acc ** 2

        # Measurement noise covariance matrix
        self.R = np.array([[self.x_std_meas ** 2, 0],
                           [0, self.y_std_meas ** 2]])

        # Covariance matrix: initial uncertainty
        self.P = np.eye(self.A.shape[0])
    
    def predict(self):
        # Predict the next state based on motion model
        # print(f"self.A = {self.A}")
        # print(f"self.state = {self.state}")
        # print(f"self.B = {self.B}")
        # print(f"self.u = {self.u}")
        
        predict_state = np.dot(self.A, self.state) + np.dot(self.B, self.u)

        # print(f"predict_state = {predict_state}")

        # Update the covariance matrix based on the process noise
        predict_P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

        return predict_state, predict_P
    
    def update(self, measurement):
        predict_state, predict_P = self.predict()

        # Kalman gain calculation
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        
        # Update the state estimate based on measurement
        self.state = predict_state + np.dot(K, (measurement - np.dot(self.H, predict_state)))
        
        # Update the covariance matrix
        self.P = predict_P - np.dot(np.dot(K, self.H), predict_P)

    def get_state(self):
        return self.state