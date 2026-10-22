
import numpy as np
import matplotlib.pyplot as plt

from part_2.observer import NonlinearPassiveObserver

dt = 0.1
T = 600.0
time = np.arange(0.0, T, dt)

observer = NonlinearPassiveObserver()

print("Test dt:", dt)
print("Test K3:")
print(observer.K3)

M = observer.M.copy()
D = observer.D.copy()

eta = np.zeros(3)
nu = np.zeros(3)

observer.reset(np.zeros(6))

#Constant environmental bias in NED coordinates
b_true = np.array([1000.0, 500.0, 5000.0])

bias_estimates = []
position_errors = []
velocity_errors = []

true_positions = []
estimated_positions = []
true_velocities = []
estimated_velocities = []

for t in time:
    #No commanded control force
    tau = np.zeros(3)

    #True vessel dynamics with environmental bias
    J = observer.rotation_matrix(eta[2])

    nu_dot = np.linalg.solve(
        M,
        -D @ nu + J.T @ b_true + tau
    )
    eta_dot = J @ nu

    nu = nu + dt * nu_dot
    eta = eta + dt * eta_dot
    eta[2] = (eta[2] + np.pi) % (2 * np.pi) - np.pi

    #Add wave-frequency motion to measured pose
    wave = np.array([
        0.3 * np.sin(0.8 * t),
        0.2 * np.sin(0.8 * t + 0.5),
        0.01 * np.sin(0.8 * t + 1.0)
    ])

    eta_measured = np.zeros(6)
    eta_measured[[0, 1, 5]] = eta + wave

    tau_6d = np.zeros(6)
    
    y = eta_measured[[0, 1, 5]]
    y_hat = observer.eta_hat + observer.Cw @ observer.xi_hat
    innovation = y - y_hat
    innovation[2] = (innovation[2] + np.pi) % (2 * np.pi) - np.pi

    measurement_correction = observer.K2 @ innovation
    velocity_contribution = observer.rotation_matrix(
    observer.eta_hat[2]
    ) @ observer.nu_hat

    if abs(t - 300.0) < dt / 2:
        print("Velocity contribution:", velocity_contribution)
        print("Measurement correction:", measurement_correction)

    result = observer.step(t, dt, eta_measured, tau_6d)

    eta_hat = result.eta[[0, 1, 5]]
    nu_hat = result.nu[[0, 1, 5]]
    b_hat = result.bias[[0, 1, 5]]

    bias_estimates.append(b_hat.copy())
    position_errors.append(eta_hat - eta)
    velocity_errors.append(nu_hat - nu)
    true_positions.append(eta.copy())
    estimated_positions.append(eta_hat.copy())
    true_velocities.append(nu.copy())
    estimated_velocities.append(nu_hat.copy())

bias_estimates = np.array(bias_estimates)
position_errors = np.array(position_errors)
velocity_errors = np.array(velocity_errors)
true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)
true_velocities = np.array(true_velocities)
estimated_velocities = np.array(estimated_velocities)

#Compare true and estimated environmental bias
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = [
    "North force [N]",
    "East force [N]",
    "Yaw moment [Nm]"
]

for j, ax in enumerate(axes):
    ax.plot(
        time,
        np.full_like(time, b_true[j]),
        label="True bias"
    )
    ax.plot(
        time,
        bias_estimates[:, j],
        label="NPO estimated bias"
    )
    ax.set_ylabel(labels[j])
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
fig.suptitle("NPO – Environmental Bias Estimation")
plt.tight_layout()
plt.show()

print("Final true bias:", b_true)
print("Final estimated bias:", bias_estimates[-1])
print(
    "Final bias error:",
    bias_estimates[-1] - b_true
)
print(
    "Position RMSE:",
    np.sqrt(np.mean(position_errors**2, axis=0))
)
print(
    "Velocity RMSE:",
    np.sqrt(np.mean(velocity_errors**2, axis=0))
)

# Compare true and estimated LF position
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = ["North [m]", "East [m]", "Heading [rad]"]

for j, ax in enumerate(axes):
    ax.plot(time, true_positions[:, j], label="True")
    ax.plot(time, estimated_positions[:, j], label="NPO estimate")
    ax.set_ylabel(labels[j])
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
fig.suptitle("NPO - Position and Heading Estimation")
plt.tight_layout()


# Compare true and estimated body-fixed velocity
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = ["Surge [m/s]", "Sway [m/s]", "Yaw rate [rad/s]"]

for j, ax in enumerate(axes):
    ax.plot(time, true_velocities[:, j], label="True")
    ax.plot(time, estimated_velocities[:, j], label="NPO estimate")
    ax.set_ylabel(labels[j])
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
fig.suptitle("NPO - Velocity Estimation")
plt.tight_layout()

plt.show()