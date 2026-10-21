

import numpy as np

from part_2.config import default_thrusters_part_2
from part_2.thrust_allocation import ThrustAllocator
from models.thruster_dynamics import ThrusterSet


# Create the thrusters and allocator
thrusters = default_thrusters_part_2()
allocator = ThrustAllocator(thrusters)

# Enable thruster dynamics
thruster_model = ThrusterSet(thrusters, dynamics=True)

# Simulation settings
dt = 0.1
T = 30.0

# Run the simulation
for k in range(int(T / dt) + 1):
    t = k * dt

    # Change the desired force direction after 10 seconds
    if t < 10.0:
        # 20 kN forward
        tau_d = np.array([20000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    else:
        # 20 kN backward
        tau_d = np.array([-20000.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Allocate thrust using the current actuator states
    u_now = thruster_model.get_thrusts()
    alpha_now = thruster_model.get_angles()

    u_cmd, alpha_cmd = allocator.allocate(
        t, dt, tau_d, u_now, alpha_now
    )

    # Apply actuator dynamics
    u_actual, alpha_actual, tau_actual = thruster_model.step(
        u_cmd, alpha_cmd, dt
    )

    # Desired and actual 3-DOF wrench
    tau_desired = tau_d[[0, 1, 5]]
    residual = tau_desired - tau_actual

    # Print results every 5 seconds and around the direction change
    if k % 50 == 0 or 9.9 <= t <= 10.5:

        print("-----------------------------")
        print("Time:", round(t, 1), "s")

        print("Commanded thrust [kN]:",
              np.round(u_cmd / 1000.0, 2))

        print("Actual thrust [kN]:",
              np.round(u_actual / 1000.0, 2))

        print("Commanded azimuth [deg]:",
              np.round(np.rad2deg(alpha_cmd), 2))

        print("Actual azimuth [deg]:",
              np.round(np.rad2deg(alpha_actual), 2))

        print("Desired wrench [kN, kN, kNm]:",
              np.round(tau_desired / 1000.0, 2))

        print("Actual wrench [kN, kN, kNm]:",
              np.round(tau_actual / 1000.0, 2))

        print("Residual [kN, kN, kNm]:",
              np.round(residual / 1000.0, 2))

