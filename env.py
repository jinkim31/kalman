import numpy as np
import matplotlib.pyplot as plt


class Env:
    def __init__(self, delta_t, initial_state, measurement_noise_cov, process_noise_cov):
        self.delta_t = delta_t
        self.measurement_noise_cov = measurement_noise_cov
        self.process_noise_cov = process_noise_cov
        self.states = [initial_state]
        self.measurements = [np.array([[0, 0]])]

    def step(self, control):
        state_transition_model = np.array(
            [[1, self.delta_t],
             [0, 1]])

        state = np.dot(state_transition_model, self.states[-1].T) + control.T + np.array(
            [np.random.multivariate_normal([0, 0], self.process_noise_cov)]).T
        state = state.T

        observation_model = np.array(
            [[1, 0],
             [0, 1]])

        measurement = np.dot(observation_model, state.T) + np.array(
            [np.random.multivariate_normal([0, 0], self.measurement_noise_cov)]).T
        measurement = measurement.T

        self.states.append(state)
        self.measurements.append(measurement)

        return state, measurement

    def plot(self):
        print(self.measurements)
        timeline = np.linspace(0, len(self.states) - 1, len(self.states))
        plt.plot(timeline, [state[0][0] for state in self.states], label='pos truth')
        plt.plot(timeline, [state[0][1] for state in self.states], label='vel truth')
        plt.plot(timeline, [m[0][0] for m in self.measurements], label='pos measurement')
        plt.plot(timeline, [m[0][1] for m in self.measurements], label='vel measurement')
        plt.legend()
        plt.show()
