import numpy as np


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, state_transition_model, observation_model, process_noise_cov, measurement_noise_cov):
        self.state = initial_state
        self.covariance = initial_covariance
        self.state_transition_model = state_transition_model
        self.observation_model = observation_model
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.delta_t = 1

    def update(self, measurement):

        # prediction
        self.state = np.dot(self.state_transition_model, self.state.T).T
        self.covariance = np.linalg.multi_dot([self.state_transition_model, self.covariance, self.state_transition_model.T]) + self.process_noise_cov

        # correction
        innovation = measurement.T - np.dot(self.observation_model, self.state.T)
        innovation_cov = np.linalg.multi_dot([self.observation_model, self.covariance, self.observation_model.T]) + self.measurement_noise_cov
        kalman_gain = np.linalg.multi_dot([self.covariance, self.observation_model.T, np.linalg.inv(innovation_cov)])

        self.state = (self.state.T + np.dot(kalman_gain, innovation)).T
        self.covariance = np.dot((np.identity(2) - np.dot(kalman_gain, self.observation_model)), self.covariance)

        return self.state
