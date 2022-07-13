import numpy as np
import env
import kf
import matplotlib.pyplot as plt

delta_t = 1.0
acc = 0.1

initial_state = np.array(
    [[0, 1.0]])
initial_covariance = np.array(
    [[10, 0],
     [0, 10]])
measurement_noise_cov = np.array(
    [[0.5, 0],
     [0, 0.5]])
process_noise_cov = np.array(
    [[0.25 * delta_t ** 4, 0.5 * delta_t ** 3],
     [0.5 * delta_t ** 3, delta_t ** 2]]
) * acc ** 2

states = []
measurements = []
estimates = []

env = env.Env(1.0, initial_state, measurement_noise_cov, process_noise_cov)
kf = kf.KalmanFilter(initial_state, initial_covariance, process_noise_cov, measurement_noise_cov)

for _ in range(100):
    state, measurement = env.step(np.array([[0, 0]]))
    estimate = kf.update(measurement)

    states.append(state)
    measurements.append(measurement)
    estimates.append(estimate)

timeline = np.linspace(0, len(states) - 1, len(states))
plt.plot(timeline, [s[0][0] for s in states], label='pos truth')
plt.plot(timeline, [s[0][1] for s in states], label='vel truth')
plt.plot(timeline, [m[0][0] for m in measurements], label='pos measurement')
plt.plot(timeline, [m[0][1] for m in measurements], label='vel measurement')
plt.plot(timeline, [e[0][0] for e in estimates], label='pos estimation')
plt.plot(timeline, [e[0][1] for e in estimates], label='vel estimation')
plt.legend()
plt.show()


#env.plot()
