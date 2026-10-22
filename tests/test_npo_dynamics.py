
import numpy as np
import matplotlib.pyplot as plt

from part_2.observer import NonlinearPassiveObserver

dt = 0.05
T = 120.0
time = np.arange(0.0, T, dt)

observer = NonlinearPassiveObserver()

#Use the same nominal vessel model as the observer
M = observer.M.copy()
D = observer.D.copy()

eta = np.zeros(3)
nu = np.zeros(3)

observer.reset(np.zeros(6))

eta_true = []
eta_est = []
nu_true = []
nu_est = []

for t in time:
    #Known applied force and moment
    tau = np.array([
        2000.0 * np.sin(0.05 * t),
        1000.0 * np.cos(0.04 * t),
        20000.0 * np.sin(0.03 * t)
    ])

    #Body-to-NED rotation
    J = observer.rotation_matrix(eta[2])

    #Simulate simplified vessel dynamics
    nu_dot = np.linalg.solve(M, tau - D @ nu)
    eta_dot = J @ nu

    nu = nu + dt * nu_dot
    eta = eta + dt * eta_dot
    eta[2] = (eta[2] + np.pi) % (2 * np.pi) - np.pi

    #Add synthetic wave-frequency measurements
    wave = np.array([
        0.3 * np.sin(0.8 * t),
        0.2 * np.sin(0.8 * t + 0.5),
        0.01 * np.sin(0.8 * t + 1.0)
    ])

    eta_measured = np.zeros(6)
    eta_measured[[0, 1, 5]] = eta + wave

    tau_6d = np.zeros(6)
    tau_6d[[0, 1, 5]] = tau

    result = observer.step(t, dt, eta_measured, tau_6d)

    eta_true.append(eta.copy())
    eta_est.append(result.eta[[0, 1, 5]].copy())
    nu_true.append(nu.copy())
    nu_est.append(result.nu[[0, 1, 5]].copy())

eta_true = np.array(eta_true)
eta_est = np.array(eta_est)
nu_true = np.array(nu_true)
nu_est = np.array(nu_est)

#Plot velocity estimates
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
labels = ["Surge u [m/s]", "Sway v [m/s]", "Yaw rate r [rad/s]"]

for j, ax in enumerate(axes):
    ax.plot(time, nu_true[:, j], label="True velocity")
    ax.plot(time, nu_est[:, j], label="NPO estimate")
    ax.set_ylabel(labels[j])
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
fig.suptitle("NPO – Velocity Estimation with Vessel Dynamics")
plt.tight_layout()
plt.show()

print("Velocity RMSE:",
      np.sqrt(np.mean((nu_est - nu_true) ** 2, axis=0)))
print("Position RMSE:",
      np.sqrt(np.mean((eta_est - eta_true) ** 2, axis=0)))
