"""
Controller template

Students should implement a controller that maps the vessel state and the
full reference to a body-frame wrench. The simulator calls, once per step:

    controller.compute(t, dt, eta, nu, eta_ref, nu_ref, acc_ref) -> tau_d

All generalized vectors are 6-DOF, ordered [surge, sway, heave, roll, pitch,
yaw]. The 3-DOF model uses indices [0, 1, 5]; the remaining components are
zero on input and ignored on output.

Inputs (full loop state and full reference):
    t       : current simulation time [s]
    dt      : time step [s]
    eta     : (6,) vessel NED state [N, E, z, phi, theta, psi]
              (use N = eta[0], E = eta[1], psi = eta[5])
    nu      : (6,) vessel BODY velocities [u, v, w, p, q, r]
              (use u = nu[0], v = nu[1], r = nu[5])
    eta_ref : (6,) NED reference state
              (use N_d = eta_ref[0], E_d = eta_ref[1], psi_d = eta_ref[5])
    nu_ref  : (6,) NED-frame reference velocities
              (use Ndot_d = nu_ref[0], Edot_d = nu_ref[1], psidot_d = nu_ref[5])
    acc_ref : (6,) NED-frame reference accelerations, same layout as nu_ref
              (use for model-based / inertia feedforward)

Output:
    tau_d   : (6,) desired BODY wrench [Fx, Fy, Fz, Mx, My, Mz] (N, Nm)
              (fill in Fx = tau_d[0], Fy = tau_d[1], Mz = tau_d[5];
               leave the other components zero)

Optional hooks the simulator will use IF you define them (safe to omit):
    reset()                                  — called before each run
    apply_external_aw(tau_applied, psi, dt)  — anti-windup with the (6,)
                                               wrench actually applied after
                                               allocation and the actuator
                                               model (ideal in Part 1)
    last_pid_body  : {"P","I","D"} -> (6,) BODY components   (logged)
    int_ned (2,), int_psi (float)            — integrator states (logged)

Constructor contract — the automated checks (``python check.py``, ``pytest``,
``notebooks/part_1_demo.ipynb``) construct your controller as
``DPController()`` with NO arguments, so your final tuned gains must be the
constructor defaults. Tuning only inside ``run_case_part1.py`` will pass your
own runs but fail the checks.
"""
import numpy as np


class DPController: #lager oppskrift på kontroller, når programmet senere sier controller = DPcontroller opprettes en faktisk kontroller
    """
    Template for student DP controller.

    Students may implement any type of controller (PID, LQR, backstepping,
    ...). Only compute() is required; everything else is optional.
    """

    def __init__(self, *args, **kwargs): #self refererer til den kontrolleren som akkurat nå benyttes 
        self.Kp_x = 3500 #surge-fram/bak
        self.Kp_y = 5000 #sway-sideveis
        self.Kp_psi = 200000 #yaw-rotasjon
        #proportional gains, det forteller oss hvor kraftig controlleren skal reagere på feil i surge, sway og yaw

        #derivative gains
        self.Kd_x = 60000
        self.Kd_y = 60000
        self.Kd_psi = 50000
        #D-gain ser på hvor fort båten beveger seg, og bramser hvis det går veldig raskt

        #integral gains
        self.Ki_x = 10
        self.Ki_y = 15
        self.Ki_psi = 600
        #I-gains skal fjerne små feil som blir værende, feks strøm som presser konstant

        #antiwindup
        self.Kaw = 0.00005

        #integral states
        self.int_ned = np.zeros(2) #skal etterhvert lagre hvor mye feil som bygger seg opp i north og east
        self.int_psi = 0.0 #skal lagre hvor mte heading-feil som har bygger seg opp over tid
        #"hukommelsen" til controlleren, lagrer hvor mye feil som har bygget seg opp over tid

        self.tau_unsat = np.zeros(6)


    def reset(self) -> None:
        """Reset integrator states before a new simulation run."""
        self.int_ned = np.zeros(2)
        self.int_psi = 0.0
        self.tau_unsat = np.zeros(6)
        #legger dette inn for å sikre at hvis man kjører ny simulering uten å nulstille integralverdiene så skal vi ikke ta med gamle feil fra forrige kjøring

    def apply_external_aw(self, tau_applied, psi, dt):
        # forskjell mellom faktisk påført og ønsket wrench i BODY
        delta_tau = tau_applied - self.tau_unsat

        # BODY -> NED
        R = np.array([
            [np.cos(psi), -np.sin(psi)],
            [np.sin(psi),  np.cos(psi)]
        ])
        # translational wrench difference in BODY
        delta_force_body = np.array([
            delta_tau[0],
            delta_tau[1]
        ])

        # BODY -> NED because int_ned is stored in NED
        delta_force_ned = np.dot(R, delta_force_body)

        # back-calculation exactly according to the assignment equation
        force_mismatch = np.linalg.norm(delta_force_body)

        if force_mismatch > 10000.0:
            Kaw_active = 0.0003
        else:
            Kaw_active = self.Kaw

        self.int_ned += Kaw_active * delta_force_ned * dt
   

    #def compute kalles for hvert tidssteg i simuleringen
    def compute(
        self,
        t: float,
        dt: float,
        eta: np.ndarray,
        nu: np.ndarray,
        eta_ref: np.ndarray,
        nu_ref: np.ndarray | None = None,
        acc_ref: np.ndarray | None = None,
    ) -> np.ndarray:
    
        #faktiske posisjoner i NED (north,east,down)
        N = eta[0] #posisjon nord/sør
        E = eta[1] #posisjon øst/vest
        psi = eta[5] #hvilken vei båten peker

        #faktiske hastigheter
        u = nu[0] #surge
        v = nu[1] #sway
        r = nu[5] #yaw

        #ønskede posisjoner
        N_d = eta_ref[0]
        E_d = eta_ref[1]
        psi_d = eta_ref[5] 

        # hvis nu_ref ikke er gitt, bruker vi null som ønsket hastighet
        if nu_ref is None:
             nu_ref = np.zeros(6)

        # ønskede hastigheter i NED
        Ndot_d = nu_ref[0] #ønsket hastighet nordover
        Edot_d = nu_ref[1] #ønsket hastighet østover
        psidot_d = nu_ref[5] #ønsker yaw-rate

        #posisjonsfeil, dette er gitt i prosjektets egen feildefinisjon
        e_N = N_d - N
        e_E = E_d - E
        #hvor langt båten er fra målet i NED koordinater
        e_psi = np.arctan2(np.sin(psi_d - psi),np.cos(psi_d - psi)) #går fraa -pi til pi
        #har nå funnet ut hvor målet ligger i forhold til båten i nord/øst-retninger

        #samler posisjonsfeil over tid til I-leddet
        self.int_ned[0] += e_N * dt
        self.int_ned[1] += e_E * dt
        if abs(e_psi) < np.deg2rad(20.0):
            self.int_psi += e_psi * dt
        
        #dette gjør at hvis båten feks ligger 1 meter feil i nord i 1 sekund så vil integralet bygge opp 1mx1s. hvis feilen fortsetter blir integralet større, dette gjør at kontrolleren etterhvert kan gi ekstra kraft for å fjerne en liten fiel som har ligget dær lenge, feks pga konstant strøm eller vind.
        #controlleren "husker over tid", brukes senere til å gi ekstra kraft hvis det er en konstant forstyrrelse

        #rotasjonsmatrise BODY->NED : vi vil finne ut hvordan målet ligger i forhold til selve båten foran/bak,høyre/venstre
        R = np.array([[np.cos(psi), -np.sin(psi)],[np.sin(psi),np.cos(psi)]])
        #denne matrisa forteller hvordan båten er rotert i forhold til NED-systemet (verden)

        #må samle ønskede hastigheter i NED
        vel_ref_ned = np.array([Ndot_d, Edot_d]) #lager vektor for ønskede hastigheter
        #samler ønsket nord/øst

        #transformerer ønsket hastighet fra NED til BODY
        vel_ref_body = np.dot(R.T, vel_ref_ned) #må transformere fordi hastigheten er i BODY-systemet. u og v er relativt til båten Ndot_d og Edot_d er hastigheter i jordfaste retnigner
        #roterer den til båtens koordinatsystem

        #ønskede hastigheter i BODY
        u_d = vel_ref_body[0]
        v_d = vel_ref_body[1]

        #hastighetsfeil i BODY
        e_u = u_d -u
        e_v = v_d - v
        e_r = psidot_d - r

        #samler psosisjonsfeilene i body-systemet i en vektor
        e_ned = np.array([e_N, e_E])

        #transformerer feilen
        e_body = np.dot(R.T, e_ned) #går fra nord/øst til frem/sideveis
        #R.T er transpose av R
        #dette tar feilen som nå er utrrykt som nord/øst og regner den om til fremover/sideveis sett fra båten
        #e_body[0] = hvor langt målet ligger fremover/bakover
        #e_bode[1] = hvor langt målet ligger sideveis
        #transformerer integralfeilen fra NED til BODY
        int_body = np.dot(R.T, self.int_ned) #oppsamlede nord/øst feil og transformere til fremover og sideveis sett fra båten

        int_x = int_body[0] #oppsamled feil frem/bakover
        int_y = int_body[1] #oppsamled feil sideveis

        e_x = e_body[0]
        e_y = e_body[1]

        #Legger inn en P-kontroller, dette e en feedback-kontroller. P står for proportional, jo større feil, jo mer kraft ber kontrolleren om
        P_x = self.Kp_x * e_x
        P_y = self.Kp_y * e_y
        P_psi = self.Kp_psi * e_psi
        #jo større feil, jo mer kraft eller moment bes det om fra kontrolleren, detta e P-leddet. ser på hvor lang unna målet båten er.

        #D-ledd
        D_x = self.Kd_x * e_u
        D_y = self.Kd_y * e_v
        D_psi = self.Kd_psi * e_r
        #D-leddet ser på forskjellen mellom ønsket å faktisk hastighet. det gir demping, hvis båten beveger seg raskt vil D-leddet redusere kraften/bremse.


        # I-ledd
        I_x = self.Ki_x * int_x
        I_y = self.Ki_y * int_y
        I_psi = self.Ki_psi * self.int_psi 
        #ser på feil over tid, kontrolleren ber gradvis om mer kraft helt til den konstante feilen forsvinner

        # Lagrer PID-leddene separat for logging i simulatoren
        self.last_pid_body = {
            "P": np.array([P_x, P_y, 0.0, 0.0, 0.0, P_psi]),
            "I": np.array([I_x, I_y, 0.0, 0.0, 0.0, I_psi]),
            "D": np.array([D_x, D_y, 0.0, 0.0, 0.0, D_psi]),
        }

        #total ønsket kraft/moment fra PID-kontorller, dette er selve PID-kontrolleren
        Fx = P_x + I_x + D_x
        Fy = P_y + I_y + D_y
        Mz = P_psi + I_psi + D_psi

        #lager BODY wrenchen, 6DOF
        tau_d = np.zeros(6)

        tau_d[0] = Fx
        tau_d[1] = Fy
        tau_d[5] = Mz

        # lagrer ønsket wrench før eventuell begrensning i thrust allocation
        self.tau_unsat = tau_d.copy()

        return tau_d
    
  
    
