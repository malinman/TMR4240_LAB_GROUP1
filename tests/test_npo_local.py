
import numpy as np
import matplotlib.pyplot as plt

from part_2.observer import NonlinearPassiveObserver

#Simulation settings
dt = 0.05
T = 120.0
time = np.arange(0, T, dt)

observer = NonlinearPassiveObserver()
eta0 = np.zeros(6)
eta0[[0, 1, 5]] = [0.0, 1.0, 0.0]
observer.reset(eta0)

#Synthetic LF motion and WF oscillations
omega_w = 0.8

eta_true = np.zeros((len(time), 3))
eta_measured = np.zeros((len(time), 3))
eta_estimated = np.zeros((len(time), 3))
nu_estimated = np.zeros((len(time), 3))
bias_estimated = np.zeros((len(time), 3))

for i, t in enumerate(time):
    #Known low-frequency motion
    eta_lf = np.array([
        2.0 * np.sin(0.03 * t),
        1.0 * np.cos(0.025 * t),
        0.05 * np.sin(0.02 * t)
    ])

    #Wave-frequency motion
    eta_wf = np.array([
        0.3 * np.sin(omega_w * t),
        0.2 * np.sin(omega_w * t + 0.5),
        0.01 * np.sin(omega_w * t + 1.0)
    ])

    measurement = eta_lf + eta_wf

    eta_6d = np.zeros(6)
    eta_6d[[0, 1, 5]] = measurement

    #Zero input force in this isolated test
    tau_6d = np.zeros(6)

    result = observer.step(t, dt, eta_6d, tau_6d)

    eta_true[i] = eta_lf
    eta_measured[i] = measurement
    eta_estimated[i] = result.eta[[0, 1, 5]]
    nu_estimated[i] = result.nu[[0, 1, 5]]
    bias_estimated[i] = result.bias[[0, 1, 5]]

#Plot LF position estimation
labels = ["Surge x [m]", "Sway y [m]", "Yaw [rad]"]

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

for j, ax in enumerate(axes):
    ax.plot(time, eta_true[:, j], label="True LF")
    ax.plot(time, eta_measured[:, j], alpha=0.45,
            label="Measured LF + WF")
    ax.plot(time, eta_estimated[:, j], label="NPO estimate")
    ax.set_ylabel(labels[j])
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
fig.suptitle("NPO – Isolated LF Position Estimation")
plt.tight_layout()
plt.show()


#Plot estimated body-fixed velocities
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

velocity_labels = [
    "Surge velocity u [m/s]",
    "Sway velocity v [m/s]",
    "Yaw rate r [rad/s]"
]

for j, ax in enumerate(axes):
    ax.plot(time, nu_estimated[:, j],
            label="NPO estimated velocity")
    ax.set_ylabel(velocity_labels[j])
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel("Time [s]")
fig.suptitle("NPO – Estimated Body-Fixed Velocities")
plt.tight_layout()
plt.show()


# Basic numerical checks
print("All estimates finite:",
      np.isfinite(eta_estimated).all()
      and np.isfinite(nu_estimated).all()
      and np.isfinite(bias_estimated).all())

print("LF position RMSE:",
      np.sqrt(np.mean(
          (eta_estimated - eta_true) ** 2,
          axis=0
      )))

print("Maximum estimated speed:",
      np.max(np.abs(nu_estimated), axis=0))

print("Maximum estimated bias:",
      np.max(np.abs(bias_estimated), axis=0))
