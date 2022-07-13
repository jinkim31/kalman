import numpy as np


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise_cov, measurement_noise_cov):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise_cov = process_noise_cov
        self.measurement_noise_cov = measurement_noise_cov
        self.delta_t = 1

    def update(self, measurement):
        state_transition_model = np.array(
            [[1, self.delta_t],
             [0, 1]])
        observation_model = np.array(
            [[1, 0],
             [0, 1]])

        # prediction
        self.state = np.dot(state_transition_model, self.state.T).T
        self.covariance = np.linalg.multi_dot([state_transition_model, self.covariance, state_transition_model.T]) + self.process_noise_cov

        # correction
        innovation = measurement.T - np.dot(observation_model, self.state.T)
        innovation_cov = np.linalg.multi_dot([observation_model, self.covariance, observation_model.T]) + self.measurement_noise_cov
        kalman_gain = np.linalg.multi_dot([self.covariance, observation_model.T, np.linalg.inv(innovation_cov)])

        self.state = (self.state.T + np.dot(kalman_gain, innovation)).T
        self.covariance = np.dot((np.identity(2) - np.dot(kalman_gain, observation_model)), self.covariance)

        return self.state
