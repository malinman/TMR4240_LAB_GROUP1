
import numpy as np

from part_2.config import default_thrusters_part_2
from part_2.thrust_allocation import ThrustAllocator
from models.thruster_dynamics import ThrusterSet


# Create thrusters with dynamics enabled
thrusters = default_thrusters_part_2()
allocator = ThrustAllocator(thrusters)
thruster_set = ThrusterSet(thrusters, dynamics=True)

# Desired wrench: [Fx, Fy, Fz, Mx, My, Mz]
tau_d = np.array([20000.0, 10000.0, 0.0, 0.0, 0.0, 100000.0])

# Simulation settings
dt = 0.1
simulation_time = 30.0
number_of_steps = int(simulation_time / dt)

# Run the thruster dynamics
for step in range(number_of_steps):

    t = step * dt

    # Calculate thrust and azimuth commands
    u_cmd, alpha_cmd = allocator.allocate(
        t,
        dt,
        tau_d,
        u_now=thruster_set.get_thrusts(),
        alpha_now=thruster_set.get_angles()
    )

    # Apply commands through the thruster dynamics
    u_actual, alpha_actual, tau_actual = thruster_set.step(
        u_cmd,
        alpha_cmd,
        dt
    )

    # Print results every 5 seconds
    if step % 50 == 0 or step == number_of_steps - 1:

        print("-----------------------------")
        print("Time:", round(t + dt, 1), "s")
        print("Commanded thrust [kN]:", np.round(u_cmd / 1000.0, 2))
        print("Actual thrust [kN]:", np.round(u_actual / 1000.0, 2))
        print("Commanded azimuth [deg]:", np.round(np.degrees(alpha_cmd), 2))
        print("Actual azimuth [deg]:", np.round(np.degrees(alpha_actual), 2))
        print("Desired wrench [kN, kN, kNm]:", tau_d[[0, 1, 5]] / 1000.0)
        print("Actual wrench [kN, kN, kNm]:", np.round(tau_actual / 1000.0, 2))
        print(
            "Residual [kN, kN, kNm]:",
            np.round((tau_d[[0, 1, 5]] - tau_actual) / 1000.0, 2)
        )
