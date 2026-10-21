import numpy as np

from part_2.config import default_thrusters_part_2
from part_2.thrust_allocation import ThrustAllocator
from models.thruster_dynamics import ThrusterSet


# Test forces [surge, sway] in kN
test_forces = [
    [200, 200],
    [200, 0],
    [0, 400],
    [-200, 200],
    [-200, -200],
    [200, -200]
]

# Create thrusters and allocator
thrusters = default_thrusters_part_2()
allocator = ThrustAllocator(thrusters)

for Fx, Fy in test_forces:

    # Desired wrench [Fx, Fy, Fz, Mx, My, Mz]
    tau_d = np.array([
        Fx * 1000.0,
        Fy * 1000.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    # Use ideal thrusters without rate limitations
    thruster_set = ThrusterSet(thrusters, dynamics=False)

    try:
        # Calculate thrust and azimuth commands
        # Current azimuth angles [degrees]
        alpha_now = np.radians([90.0, 0.0, 0.0])

        u_cmd, alpha_cmd = allocator.allocate(
            0.0,
            0.1,
            tau_d,
            alpha_now=alpha_now
        )

        # Calculate actual forces and yaw moment
        _, _, tau_actual = thruster_set.step(
            u_cmd,
            alpha_cmd,
            0.1
        )

        Fx_actual = tau_actual[0] / 1000.0
        Fy_actual = tau_actual[1] / 1000.0
        Mz_actual = tau_actual[2] / 1000.0

        # Calculate desired and actual force directions
        angle_desired = np.degrees(np.arctan2(Fy, Fx))
        angle_actual = np.degrees(
            np.arctan2(Fy_actual, Fx_actual)
        )

        # Calculate direction error in degrees
        angle_error = angle_actual - angle_desired
        angle_error = (angle_error + 180) % 360 - 180

        # Check maximum thrust limits
        thrust_limits = np.array([
            thruster.u_max for thruster in thrusters
        ])

        thrust_ok = np.all(
            np.abs(u_cmd) <= thrust_limits + 1.0
        )

        # Print test results
        print("-----------------------------")
        print("Desired force [kN]:", Fx, Fy)
        print(
            "Actual force [kN]:",
            round(Fx_actual, 2),
            round(Fy_actual, 2)
        )
        print("Desired direction:", round(angle_desired, 2), "deg")
        print("Actual direction:", round(angle_actual, 2), "deg")
        print("Direction error:", round(angle_error, 2), "deg")
        print("Actual yaw moment:", round(Mz_actual, 2), "kNm")
        print("Thrust limits satisfied:", thrust_ok)

        # Print azimuth commands and signed thrust
        print("Azimuth commands [deg]:", np.round(np.degrees(alpha_cmd), 2))
        print("Thrust commands [kN]:", np.round(u_cmd / 1000.0, 2))

    except RuntimeError as error:
        print("-----------------------------")
        print("Desired force [kN]:", Fx, Fy)
        print("Test failed:", error)