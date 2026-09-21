from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig

#Normalizes an angle to the range [-pi, pi], making it easier to compare angles and determine the shortest rotation.
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi





class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)

        # Desired 3-DOF wrench: [Fx, Fy, Mz]
        tau = tau_d[[0, 1, 5]]

        # Extended configuration matrix
        B_e = np.array([
            [0.0,   1.0,   0.0,   1.0,   0.0],
            [1.0,   0.0,   1.0,   0.0,   1.0],
            [12.0, -3.0, -13.0,   3.0, -13.0]
        ])

        # Calculating the allocation variables using the pseudo-inverse
        z = np.linalg.pinv(B_e) @ tau

        # Extracting the individual force components
        u_T = z[0]

        Fx1 = z[1]
        Fy1 = z[2]

        Fx2 = z[3]
        Fy2 = z[4]

        # Check if the tunnel thruster exceeds its maximum thrust
        tunnel_max = self.thrusters[0].u_max

        if abs(u_T) > tunnel_max:

            # Saturate the tunnel thruster at its maximum allowed thrust
            u_T = np.clip(u_T, -tunnel_max, tunnel_max)

            # Wrench produced by the saturated tunnel thruster
            B_tunnel = np.array([0.0, 1.0, 12.0])
            tau_tunnel = B_tunnel * u_T

            # Wrench that must be produced by the azimuth thrusters
            tau_remaining = tau - tau_tunnel
            # Configuration matrix for the two azimuth thrusters
            B_azimuth = np.array([
            [1.0,   0.0,   1.0,   0.0],
            [0.0,   1.0,   0.0,   1.0],
            [-3.0, -13.0,   3.0, -13.0]
            ])

            # Reallocate the remaining wrench to the azimuth thrusters
            z_azimuth = np.linalg.pinv(B_azimuth) @ tau_remaining

            # Update the azimuth force components
            Fx1 = z_azimuth[0]
            Fy1 = z_azimuth[1]

            Fx2 = z_azimuth[2]
            Fy2 = z_azimuth[3]

        # Converting force components to thrust magnitude and angle

        u_1 = np.sqrt(Fx1**2 + Fy1**2) # magnitude
        alpha_1 = np.arctan2(Fy1, Fx1) # angle

        u_2 = np.sqrt(Fx2**2 + Fy2**2) 
        alpha_2 = np.arctan2(Fy2, Fx2)

        # If an azimuth thruster produces no thrust, keep its current angle
        if alpha_now is not None:
            if np.isclose(u_1, 0.0):
                alpha_1 = alpha_now[1]

            if np.isclose(u_2, 0.0):
                alpha_2 = alpha_now[2]

        # Choose the azimuth direction that requires the least rotation
        if alpha_now is not None:

            # Azimuth thruster 1
            rotation_1 = normalize_angle(alpha_1 - alpha_now[1])
            rotation_1_reverse = normalize_angle(alpha_1 + np.pi - alpha_now[1])

            if abs(rotation_1_reverse) < abs(rotation_1):
                alpha_1 = normalize_angle(alpha_1 + np.pi)
                u_1 = -u_1

            # Azimuth thruster 2
            rotation_2 = normalize_angle(alpha_2 - alpha_now[2])
            rotation_2_reverse = normalize_angle(alpha_2 + np.pi - alpha_now[2])

            if abs(rotation_2_reverse) < abs(rotation_2):
                alpha_2 = normalize_angle(alpha_2 + np.pi)
                u_2 = -u_2



        u_cmd = np.array([u_T, u_1, u_2])
        alpha_cmd = np.array([np.pi / 2, alpha_1, alpha_2])

        # Maximum thrust for each thruster
        u_max = np.array([thruster.u_max for thruster in self.thrusters])

        # Limit thrust commands to the allowed range
        u_cmd = np.clip(u_cmd, -u_max, u_max)

        return u_cmd, alpha_cmd


"""
Thrust Allocation template

Students should implement an algorithm that maps the desired body-frame
wrench to individual thruster commands. The simulator calls, once per step:

    allocator.allocate(t, dt, tau_d, u_now, alpha_now) -> (u_cmd, alpha_cmd)

Inputs (full actuator state — use what your algorithm needs):
    t         : current simulation time [s]
    dt        : time step [s]              (rate-aware/dynamic allocation)
    tau_d     : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz]
                (the 3-DOF wrench to allocate is tau_d[[0, 1, 5]]
                 = [Fx, Fy, Mz]; the other components are zero)
    u_now     : current actual thrusts [N]     (rate-aware allocation)
    alpha_now : current thruster angles [rad]  (minimize azimuth slewing)

Outputs:
    u_cmd     : signed thrust command for each thruster [N]
    alpha_cmd : thruster angle command for each thruster [rad]

Students may implement, for example:
    - pseudo-inverse allocation,
    - weighted least-squares allocation,
    - optimization-based allocation,
    - power-minimizing allocation.
"""

"""
class ThrustAllocator:
    Template for student thrust allocation.

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.thrusters)

        # TODO: Replace this placeholder with your thrust allocation algorithm.
        # The placeholder commands zero thrust and alpha for all thrusters.
        u_cmd = np.zeros(n)
        alpha_cmd = np.zeros(n)

        return u_cmd, alpha_cmd
"""
