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

Contract with the checks, exactly as the public checks test it: ``allocate()``
must never command more thrust than a thruster can deliver, so for an
infeasible demand (the check uses 400 kN of sway) every |u_cmd| <= u_max.
*How* you handle an infeasible demand beyond that limit is your design
decision, not a checked criterion — state your rule and justify it in the
report.  The check also calls ``allocate()`` repeatedly from the initial
actuator state, applies each command through IDEAL actuators and feeds the
resulting actuator state back as ``u_now``/``alpha_now``; after 30 s of such
calls the achieved wrench must equal the requested one.  A steady-state
allocator therefore passes on its first call; a rate-aware allocator
(optional extension, and optional in the project text too) passes once it has
converged.  In the simulation the transient (thrust ramps, 2 rpm azimuth
slews) is shaped by the actuator model in ``models/thruster_dynamics.py``
whatever you command, so rate limiting your own commands is optional, not
required.  ``u_now``/``alpha_now`` are passed in because a thruster produces
the same force as (u, alpha) or as (-u, alpha + pi): which of the two you
command, and on what basis, is part of your allocation design.

class ThrustAllocator:
    #Template for student thrust allocation.

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

from typing import List, Optional, Tuple
import numpy as np

from models.thruster_dynamics import ThrusterConfig
import cvxpy as cp

# Normalize angle to the range [-pi, pi]
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

        # Desired 3-DOF wrench: [Fx, Fy, Mz]
        tau = tau_d[[0, 1, 5]]
        # Convert desired forces and moment to kN and kNm
        tau = tau / 1000.0

        # Extended configuration matrix
        B_e = np.array([
            [0.0,   1.0,   0.0,   1.0,   0.0],
            [1.0,   0.0,   1.0,   0.0,   1.0],
            [12.0, -3.0, -13.0,   3.0, -13.0]
        ])




        # Maximum thrust for each thruster [kN]
        u_max = np.array([thruster.u_max for thruster in self.thrusters])
        u_max = u_max / 1000.0

        

        # Weighting matrix for surge, sway and yaw errors
        L = 13.0
        #W = np.diag([1.0, 1.0, 1.0 / L])
        # Give yaw error higher priority
        #W = np.diag([1.0, 1.0, 5.0 / L])
        # Prioritize yaw accuracy over additional sway force
        W = np.diag([1.0, 1.0, 10.0 / L])

        
        # Optimization variables:
        # [Tunnel thrust, Fx1, Fy1, Fx2, Fy2] in kN
        z = cp.Variable(5)

        # Weighted least-squares objective function
        residual = W @ (B_e @ z - tau)
        objective = cp.Minimize(cp.sum_squares(residual))

        # Maximum thrust constraints
        constraints = [
            cp.abs(z[0]) <= u_max[0],
            cp.norm(z[1:3], 2) <= u_max[1],
            cp.norm(z[3:5], 2) <= u_max[2]
        ]

        # Preserve the desired force direction
        if tau[0] != 0.0 or tau[1] != 0.0:

            Fx_desired = tau[0]
            Fy_desired = tau[1]

            desired_magnitude = np.sqrt(
                Fx_desired**2 + Fy_desired**2
            )

            direction_x = Fx_desired / desired_magnitude
            direction_y = Fy_desired / desired_magnitude

            Fx_actual = z[1] + z[3]
            Fy_actual = z[0] + z[2] + z[4]

            # Actual force must be parallel to desired force
            constraints.append(
                direction_x * Fy_actual
                - direction_y * Fx_actual == 0
            )

            # Actual force must not point in the opposite direction
            constraints.append(
                direction_x * Fx_actual
                + direction_y * Fy_actual >= 0
            )

        # Solve the constrained WLS problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)




        # Check if the optimization was successful
        if problem.status != cp.OPTIMAL:
            print("Optimization status:", problem.status)
            raise RuntimeError("Thrust allocation optimization failed")

        # Extract the optimized force components
        z = z.value

        # Extract the individual force components
        u_T = z[0]

        Fx1 = z[1]
        Fy1 = z[2]

        Fx2 = z[3]
        Fy2 = z[4]

        # Convert Cartesian forces to thrust and azimuth angles
        u_1 = np.sqrt(Fx1**2 + Fy1**2)
        alpha_1 = np.arctan2(Fy1, Fx1)

        u_2 = np.sqrt(Fx2**2 + Fy2**2)
        alpha_2 = np.arctan2(Fy2, Fx2)

        # Choose the shortest azimuth rotation
        if alpha_now is not None:

            # Azimuth thruster 1
            if np.isclose(u_1, 0.0):
                alpha_1 = alpha_now[1]
            else:
                rotation_1 = normalize_angle(alpha_1 - alpha_now[1])
                reverse_rotation_1 = normalize_angle(
                    alpha_1 + np.pi - alpha_now[1]
                )

                if abs(reverse_rotation_1) < abs(rotation_1):
                    alpha_1 = normalize_angle(alpha_1 + np.pi)
                    u_1 = -u_1

            # Azimuth thruster 2
            if np.isclose(u_2, 0.0):
                alpha_2 = alpha_now[2]
            else:
                rotation_2 = normalize_angle(alpha_2 - alpha_now[2])
                reverse_rotation_2 = normalize_angle(
                    alpha_2 + np.pi - alpha_now[2]
                )

                if abs(reverse_rotation_2) < abs(rotation_2):
                    alpha_2 = normalize_angle(alpha_2 + np.pi)
                    u_2 = -u_2

        # Create thrust and azimuth commands
        u_cmd = np.array([u_T, u_1, u_2])
        alpha_cmd = np.array([np.pi / 2, alpha_1, alpha_2])

        # Convert thrust commands from kN back to N
        u_cmd = u_cmd * 1000.0

        return u_cmd, alpha_cmd

    