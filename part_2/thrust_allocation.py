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

# Pakker en vinkel inn i [-pi, pi), slik at vi kan sammenligne azimuth-vinkler
# og velge korteste rotasjon
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ThrustAllocator:
    """Template for student thrust allocation."""

    def __init__(self, thrusters: List[ThrusterConfig]):
        self.thrusters = thrusters
        # Forrige gyldige løsning [kN]. Brukes hvis CLARABEL feiler i et
        # tidssteg, slik at lange kjøringer (Sim 5 har ~2e5 kall) ikke stopper
        self.z_prev = np.zeros(5)

    def allocate(
        self,
        t: float,
        dt: float,
        tau_d: np.ndarray,
        u_now: Optional[np.ndarray] = None,
        alpha_now: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Plukk ut surge, sway og yaw fra 6-DOF-vektoren og regn i kN/kNm,
        # så optimeringsproblemet blir godt skalert
        tau = tau_d[[0, 1, 5]]
        tau = tau / 1000.0
        # Konfigurasjonsmatrise for z = [tunnel Fy, az1 Fx, az1 Fy, az2 Fx, az2 Fy].
        # Yaw-raden er momentarmene: tunnelen sitter 12 m foran, azimuthene
        # 13 m bak og 3 m til hver side (N = x*Fy - y*Fx)


        # Utvidet konfigurasjons matrise
        B_e = np.array([
            [0.0,   1.0,   0.0,   1.0,   0.0],
            [1.0,   0.0,   1.0,   0.0,   1.0],
            [12.0, -3.0, -13.0,   3.0, -13.0]
        ])




        # Maks thrust fra thrusterkonfigurasjonen: 32, 80 og 80 kN
        u_max = np.array([thruster.u_max for thruster in self.thrusters])
        u_max = u_max / 1000.0

        

        # Yaw-feilen er i kNm og deles på L = 13 m for å bli sammenlignbar med
        # kraftfeilene. Faktoren 10 gir yaw høyere prioritet, slik at heading
        # holdes og sway-kraften gis opp først når kravet er for stort (Sim 7)
        L = 13.0
        W = np.diag([1.0, 1.0, 10.0 / L])

        
        # Optimeringsvariabler i kN: [tunnel Fy, az1 Fx, az1 Fy, az2 Fx, az2 Fy]
        z = cp.Variable(5)



        # Weighted least-squares objective with a small thrust penalty.
        # eps * ||z||^2 makes the solution unique (5 unknowns, 3 equations)
        # and approximates minimum power. eps is small enough that the
        # wrench residual stays negligible when the demand is feasible.
        eps = 1e-4
        residual = W @ (B_e @ z - tau)
        objective = cp.Minimize(cp.sum_squares(residual) + eps * cp.sum_squares(z))

        # Tunnelen gir bare sideveis kraft, azimuthene kan gi kraft i alle
        # retninger, så for dem er grensen på lengden av kraftvektoren
        constraints = [
            cp.abs(z[0]) <= u_max[0],
            cp.norm(z[1:3], 2) <= u_max[1],
            cp.norm(z[3:5], 2) <= u_max[2]
        ]

        # Når kravet er større enn kapasiteten, skal kraften bli kortere, ikke
        # snu. Ellers kunne fartøyet bli presset i en helt annen retning enn
        # kontrolleren ber om. Hoppes over når ønsket horisontal kraft er null
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

            # Kryssproduktet er null: allokert kraft er parallell med ønsket
            constraints.append(
                direction_x * Fy_actual
                - direction_y * Fx_actual == 0
            )

            # Prikkproduktet er ikke negativt: allokert kraft peker samme vei
            constraints.append(
                direction_x * Fx_actual
                + direction_y * Fy_actual >= 0
            )


        # OPTIMAL_INACCURATE godtas: det skjer når kravet er langt over
        # kapasiteten (for eksempel Sim 7 ved 24 m/s), og løsningen er da
        # fortsatt brukbar
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL)
            solved = problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        except cp.SolverError:
            solved = False

        if solved:
            z = z.value
            self.z_prev = z.copy()
        else:
            # Gjenbruk forrige løsning i dette ene tidssteget
            print(f"t = {t:.1f} s: allocation status {problem.status}, "
                  "previous solution reused")
            z = self.z_prev.copy()

        # Extract the individual force components
        u_T = z[0]

        Fx1 = z[1]
        Fy1 = z[2]

        Fx2 = z[3]
        Fy2 = z[4]

        # Fra kraftvektor til azimuth-kommando:
        # - Vinkel: retningen til kraften, eller den motsatte (med negativ
        #   thrust) hvis den ligger nærmere gjeldende vinkel. Azimuthene dreier
        #   bare 12 grader/s, så korteste rotasjon betyr mye.
        # - Under u_dead = 0.5 kN holdes vinkelen, fordi retningen til en nesten
        #   null kraft bare er numerisk støy og ikke verdt å dreie etter.
        # - Thrust: kraften projiseres på GJELDENDE vinkel. Første versjon ga
        #   full thrust langs den fastholdte vinkelen, som skjøv i feil retning
        #   og ga +-13 kNm chattering i yaw i Sim 2. Med projeksjonen skyver
        #   thrusteren aldri feil vei, og kraften bygges opp mens den dreier.
        u_dead = 0.5  # [kN]
        u_az = []
        alpha_az = []
        for j, (Fx, Fy) in enumerate([(Fx1, Fy1), (Fx2, Fy2)], start=1):
            magnitude = np.hypot(Fx, Fy)
            a_cmd = np.arctan2(Fy, Fx)
            a_now = alpha_now[j] if alpha_now is not None else a_cmd

            if magnitude < u_dead:
                a_cmd = a_now
            elif (abs(normalize_angle(a_cmd + np.pi - a_now))
                  < abs(normalize_angle(a_cmd - a_now))):
                a_cmd = normalize_angle(a_cmd + np.pi)

            # Fortegnsbestemt thrust langs vinkelen thrusteren faktisk har nå
            u_az.append(Fx * np.cos(a_now) + Fy * np.sin(a_now))
            alpha_az.append(a_cmd)

        u_1, u_2 = u_az
        alpha_1, alpha_2 = alpha_az

        # Tunnelen står fast på pi/2 (tverrskips), så bare thrusten varierer
        u_cmd = np.array([u_T, u_1, u_2])
        alpha_cmd = np.array([np.pi / 2, alpha_1, alpha_2])

        # Simulatoren forventer thrust i N
        u_cmd = u_cmd * 1000.0

        return u_cmd, alpha_cmd

    