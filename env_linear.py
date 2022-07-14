import numpy as np
import matplotlib.pyplot as plt


class Env:
    def __init__(self, initial_state, state_transition_model, observation_model, process_noise_cov, measurement_noise_cov):
        self.state_transition_model = state_transition_model
        self.observation_model = observation_model
        self.measurement_noise_cov = measurement_noise_cov
        self.process_noise_cov = process_noise_cov
        self.states = [initial_state]
        self.measurements = [np.zeros([1, self.observation_model.shape[0]])]

    def step(self, control):
        state = np.dot(self.state_transition_model, self.states[-1].T) + control.T + np.array(
            [np.random.multivariate_normal(np.zeros(self.state_transition_model.shape[0]), self.process_noise_cov)]).T
        state = state.T

        measurement = np.dot(self.observation_model, state.T) + np.array(
            [np.random.multivariate_normal(np.zeros(self.observation_model.shape[0]), self.measurement_noise_cov)]).T
        measurement = measurement.T

        self.states.append(state)
        self.measurements.append(measurement)

        return state, measurement
