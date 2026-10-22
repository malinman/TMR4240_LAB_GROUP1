"""Student observer templates and their common simulator interface.

The simulator calls, once per step (pose measurement only, no velocity):

    observer.step(t, dt, eta_measured, tau_est) -> ObserverEstimate(eta, nu, bias)

    eta_measured : (6,) measured NED pose [N, E, z, phi, theta, psi]
    tau_est      : (6,) desired controller wrench (before thruster dynamics)
                   of the PREVIOUS step (zero at the first step)
    eta, nu      : (6,) low-frequency NED pose and BODY velocity estimates
    bias         : (6,) slowly varying bias estimate (NED force) — may be zeros

A plain ``(eta, nu, bias)`` tuple is accepted as well.  Before every run the
engine calls ``reset(eta0)`` with the TRUE initial pose, so an observer can
start from the vessel's actual position.  Select an implementation with
``Part2SimConfig(observer_type=...)`` (Simulations 4-7 use the
``SELECTED_OBSERVER`` constant of ``run_case_part_2.py``); extra constructor
arguments go through ``Part2SimConfig(observer_kwargs=...)``.  Both classes
intentionally start as pass-through placeholders; students must implement and
tune them, keeping the tuned parameters as constructor defaults because the
checks call ``select_observer(kind)`` without arguments.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ObserverEstimate:
    eta: np.ndarray
    nu: np.ndarray
    bias: np.ndarray


class Observer:
    """Interface shared by all Part 2 observers."""

    name = "observer"

    def reset(self, eta0: np.ndarray | None = None):
        pass

    def step(
        self, t: float, dt: float, eta_measured: np.ndarray, tau_est: np.ndarray
    ) -> ObserverEstimate:
        raise NotImplementedError

    @staticmethod
    def _placeholder(eta_measured: np.ndarray) -> ObserverEstimate:
        return ObserverEstimate(
            eta=np.asarray(eta_measured, dtype=float).reshape(6).copy(),
            nu=np.zeros(6),
            bias=np.zeros(6),
        )


class NonlinearPassiveObserver(Observer):
    """Template for a nonlinear passive wave-filtering observer.

    Suggested states are wave-frequency motion, low-frequency pose, BODY
    velocity, and slowly-varying bias.  Implement the correction and model
    propagation in :meth:`step`.
    """

    name = "nonlinear_passive"

    def __init__(self, M=None, D=None):
        #Vessel mass matrix from Appendix B
        if M is None:
            M = np.array([
                [6.007e5, 0.0, 0.0],
                [0.0, 7.067e5, -4.733e5],
                [0.0, -5.712e5, 5.456e7]
            ])

        #Linear damping matrix from Appendix B
        if D is None:
            D = np.diag([1.118e3, 2.229e4, 1.950e6])

        self.M = np.asarray(M, dtype=float)
        self.D = np.asarray(D, dtype=float)


        #Nominal wave model parameters
        omega = np.array([0.8, 0.8, 0.8])
        zeta = np.array([0.1, 0.1, 0.1])

        Omega = np.diag(omega)
        Z = np.diag(zeta)

        #Wave-frequency state-space matrices
        self.Aw = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [-Omega @ Omega, -2 * Z @ Omega]
        ])

        self.Ew = np.vstack([
            np.zeros((3, 3)),
            np.eye(3)
        ])

        self.Cw = np.hstack([
            np.eye(3),
            np.zeros((3, 3))
        ])
        #Wave state ordering: [eta_w, eta_w_dot]
        #Observer gains must be consistent with this ordering.
        #Final gains will be selected using the passivity condition.

        #Bias model time constants
        self.Tb = np.diag([1000.0, 1000.0, 1000.0])
        self.Tb_inv = np.diag(1.0 / np.diag(self.Tb))

        #NPO correction gains
        #Wave-filter gains based on the selected nominal frequency
        #and observer bandwidth.
        #The linear observer subsystem was checked using the KYP condition.
        omega_w = 0.8
        zeta_w = 0.1
        zeta_n = 1.0
        omega_c = 1.0

        ka = -2.0 * (zeta_n - zeta_w) * omega_c / omega_w
        kb = 2.0 * omega_w * (zeta_n - zeta_w)

        self.K1 = np.vstack([
            kb * np.eye(3),
            -(omega_w**2) * ka * np.eye(3)
        ])
        self.K2 = omega_c * np.eye(3)
        #Preliminary bias and velocity correction gains
        self.K3 = 0.02 * np.eye(3)
        self.K4 = 0.5 * np.eye(3)



        #Observer states
        self.xi_hat = np.zeros(6)    #Wave-frequency states
        self.eta_hat = np.zeros(3)   #Low-frequency position and heading
        self.nu_hat = np.zeros(3)    #Low-frequency BODY velocity
        self.b_hat = np.zeros(3)     #Slowly varying bias


    def reset(self, eta0=None):
        #Reset observer states before a new simulation
        self.xi_hat = np.zeros(6)
        self.eta_hat = np.zeros(3)
        self.nu_hat = np.zeros(3)
        self.b_hat = np.zeros(3)

        if eta0 is not None:
            eta0 = np.asarray(eta0, dtype=float).reshape(6)
            self.eta_hat = eta0[[0, 1, 5]].copy()


    @staticmethod
    def rotation_matrix(psi):
        """Transform 3-DOF BODY velocities to NED."""
        c = np.cos(psi)
        s = np.sin(psi)

        return np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0]
        ])



    def step(self, t, dt, eta_measured, tau_est):
        #Extract measured North, East and heading
        y = np.asarray(eta_measured, dtype=float)[[0, 1, 5]]

        #Predicted measurement: LF motion + WF motion
        y_hat = self.eta_hat + self.Cw @ self.xi_hat

        #Measurement innovation
        y_tilde = y - y_hat

        #Wrap heading innovation to (-pi, pi]
        y_tilde[2] = (y_tilde[2] - np.pi) % (2 * np.pi) - np.pi
        if y_tilde[2] <= -np.pi:
            y_tilde[2] += 2 * np.pi


        #Rotation from BODY to NED
        J = self.rotation_matrix(self.eta_hat[2])

        #NPO state derivatives
        xi_dot = self.Aw @ self.xi_hat + self.K1 @ y_tilde

        eta_dot = J @ self.nu_hat + self.K2 @ y_tilde

        b_dot = -self.Tb_inv @ self.b_hat + self.K3 @ y_tilde

        tau = np.asarray(tau_est, dtype=float).reshape(6)[[0, 1, 5]]

        nu_dot = np.linalg.solve(
            self.M,
            -self.D @ self.nu_hat
            + J.T @ self.b_hat
            + tau
            + J.T @ self.K4 @ y_tilde
        )

        #Euler integration of observer states
        self.xi_hat += dt * xi_dot
        self.eta_hat += dt * eta_dot
        self.nu_hat += dt * nu_dot
        self.b_hat += dt * b_dot

        #Wrap estimated heading to (-pi, pi]
        self.eta_hat[2] = (self.eta_hat[2] + np.pi) % (2 * np.pi) - np.pi


        #Convert 3-DOF estimates to simulator's 6-DOF interface
        eta_out = np.zeros(6)
        nu_out = np.zeros(6)
        bias_out = np.zeros(6)

        eta_out[[0, 1, 5]] = self.eta_hat
        nu_out[[0, 1, 5]] = self.nu_hat
        bias_out[[0, 1, 5]] = self.b_hat

        return ObserverEstimate(
            eta=eta_out,
            nu=nu_out,
            bias=bias_out
        )



class KalmanFilterObserver(Observer):
    """Template for a discrete KF/EKF observer.

    Students define the process/measurement models and tune Q, R and P0.
    The input ``tau_est`` is the desired controller wrench before thruster
    dynamics.
    """

    name = "kalman"

    def __init__(self, *args, **kwargs):
        from models.gunnerus_3dof import Gunnerus3DOF #hente fartøysparametere til EKF-modell fra R/V Gunnerus
        vessel = Gunnerus3DOF()

        #massematrise og lineær demping
        self.M = vessel._v._Mrb + vessel._v._Ma
        self.Minv = np.linalg.inv(self.M) #Invers massematrise
        self.D = vessel._v._Dl #Dempingsmatrise

        #EKF state vector, tilstansvektor med 15 elementer:
        #x[0:6] = 6 bølgetilstander
        #x[6:9] = nord, øst og heading (lavfrekvente)
        #x[9:12] = surge, sway og yaw rate
        #x[12:15] = miljøbias i nord, øst og yaw
        self.x = np.zeros(15) #tilstandsvektor med 15 elementer, (zeros lager matrise kun med nullere)
        self.P0 = np.eye(15) #P0 forklare kor usikker vi e ved oppstart, 15x15, (eye lager en identitetsmatrise)
        #miljøbias: nord- og øst kraft, og yaw-moment, vi kjenne ikke miljøbias ved oppstart, og gir derfor disse tilstandene stor usikkerhet
        self.P0[12, 12] = 5000**2
        self.P0[13, 13] = 5000**2
        self.P0[14, 14] = 500000.0 ** 2  
        self.P = self.P0.copy() #P er usikkkerheten som oppdateres underveis 
        self.Q = np.eye(15) #Q er usikkerheten i prosessmodellen
        #Bølgemodellen får prosesstøy i aksellerasjonene
        self.Q[0:6, 0:6] = np.zeros((6, 6))
        self.Q[3:6, 3:6] = np.diag([5.0,5.0,5.0])
        self.Q[9:12, 9:12] = np.eye(3) * 1e-8  #Gir lav prosessusikkerhet til hastigheten
        #prosessusikkerhet for miljøbias i nord, øst og yaw
        self.Q[12, 12] = 1000 
        self.Q[13, 13] = 1000
        self.Q[14, 14] = 100000
        #ANTATT måleusikkerhet for nord, øst og heading
        sigma_pos = 0.5  #Antatt standardavvik for posisjon [m]
        sigma_psi = np.deg2rad(2.0)  #Antatt standardavvik for heading [rad]

        self.R = np.diag([sigma_pos**2,sigma_pos**2,sigma_psi**2 ])  #R e måleusikkerheten skrevvet som varians

        self.H = np.zeros((3,15)) #H beskriv hvilke tilstander vi kan måle
        #Målt posisjon = bølgebevegelse + lavfrekvent posisjon
        self.H[0:3, 0:3] = np.eye(3) #de tre første e bølgeutslag i nord, øst og heading
        self.H[0:3, 6:9] = np.eye(3) #lavfrekvent posisjon og heading

        #paramnetere for bølgemodellen
        self.omega_w = 0.8 #naturlig frekvens [rad/s], foreløpig eks verdi MÅ TUNES
        self.zeta_w = 0.1 #dempingsforholdet til bølgemodellen, foreløpig eks verid MÅ tunes

        #tidskonstanter for langsomt varierende miljøbias
        self.Tb = np.diag([1000, 1000, 1000])
        self.Tb_inv = np.linalg.inv(self.Tb)

        #lage bølgemodellen Aw
        I3 = np.eye(3) #identitetsmatrise for nord, øst og heading
        self.Aw = np.zeros((6, 6)) #matrise som beskriver kordan de seks bølgetilstandene utvikler seg
        self.Aw[0:3, 3:6] = I3 #bølgeposisjon endres med bølgehastigheten
        self.Aw[3:6, 0:3] = -(self.omega_w**2) * I3 #bølgedynamikkens tilbareførende ledd
        self.Aw[3:6, 3:6] = -2*self.zeta_w*self.omega_w*I3 #demping av bølgebevegelsen  

    def reset(self, eta0=None):
        self.x = np.zeros(15) #nullstiller tilstandsvektor før ny simulerig
        self.P = self.P0.copy() #nullstiller kovarians/usikkerheten før ny simulering

        if eta0 is not None: #hvis startposisjonen e kjent, bruker vi den
            self.x[6:9] = np.array(eta0)[[0, 1, 5]] #North, East, Heading

    def rotation_matrix(self, psi):
        #Rotasjonsmatrise fra BODY-koordinater til NED-koordinater
        c = np.cos(psi)
        s = np.sin(psi)

        return np.array([[c, -s, 0],[s,  c, 0],[0,  0, 1]])

    def process_model(self, x, tau): #bergner endrignshastighetene til de 15 tilstandene
        # Henter de ulike tilstandene fra tilstandsvektoren
        xi = x[0:6]       #Bølgebevegelse
        eta = x[6:9]      #posisjon og heading
        nu = x[9:12]      #Hastighet i BODY-koordinater
        bias = x[12:15]   #Miljøbias i NED-koordinater

        J = self.rotation_matrix(eta[2]) #Beregner rotasjonsmatrisen ut fra estimert heading
        xi_dot = np.dot(self.Aw, xi) #Beregner endringen i bølgetilstandene
        eta_dot = np.dot(J, nu) #Beregner endringen i posisjon og heading

        #fartøyets bevegelsesligning: M * nu_dot = tau - D * nu + J.T * bias
        damping = np.dot(self.D, nu)
        environmental_force = np.dot(J.T, bias)
        total_force = tau - damping + environmental_force
        nu_dot = np.dot(self.Minv, total_force)
        bias_dot = -np.dot(self.Tb_inv, bias) #miljeøbias avtar sakte

        return np.concatenate([xi_dot, eta_dot, nu_dot, bias_dot]) #Samler alle endringene i én vektor med 15 elementer

    def jacobian(self, x, tau): #jacobian beskrver kordan modellen påvirkes når tilstandene endres litt
        #heading, hastighet og miljøbias
        psi = x[8]
        nu = x[9:12]
        bias = x[12:15]

        c = np.cos(psi)
        s = np.sin(psi)

        J = self.rotation_matrix(psi) #rotasjonsmatrise

        dJ = np.array([[-s, -c, 0],[ c, -s, 0],[ 0,  0, 0]]) #deriverte av rotasjonsmatrisen

        F = np.zeros((15, 15)) #tom 15x15 jacobimatrise
        F[0:6, 0:6] = self.Aw #hvordan bølgene påvirker hverandre
        F[6:9, 8] = np.dot(dJ, nu) #hvordan heading påvirker posisjonsendringen
        F[6:9, 9:12] = J #hvordan hastighet påvirker posisjonsendringen
        heading_effect = np.dot(dJ.T, bias) #hvordan heading påvirker aksellerajsone
        F[9:12, 8] = np.dot(self.Minv, heading_effect)
        F[9:12, 9:12] = -np.dot(self.Minv, self.D) #hvordan demping påvirker aksellerasjonen
        F[9:12, 12:15] = np.dot(self.Minv, J.T) #hvordan miljøbias påvirker aksellerasjonen
        F[12:15, 12:15] = -self.Tb_inv #utviklking av miljøbias over tid

        return F

    def step(self, t, dt, eta_measured, tau_est): #ett EKF steg: først prediksjon->korreksjon

        tau = np.array(tau_est)[[0, 1, 5]] #hente kontrollkraft og moment for surge, sway og yaw
        x_dot = self.process_model(self.x, tau) #regenr hvor raskt tilstander endrer seg
        x_pred = self.x + dt*x_dot #forutsier neste tilstand Euler-metoden

        F = self.jacobian(self.x,tau) #bergner Jacobi-matrisen ved den nåværende tilstandden
        Fd = np.eye(15) + dt*F #diskretiserer jacobimatrisen vha Euler-mdetoden

        #diskret prosessusikkerhet
        FQ = np.dot(F, self.Q) 
        Qd = (self.Q * dt + 0.5*dt**2 * (FQ + FQ.T) + (dt**3 / 3) * np.dot(FQ, F.T))

        #forutsier usikkerhet ved neste steg
        P_pred = np.dot(Fd, self.P)
        P_pred = np.dot(P_pred, Fd.T) + Qd 

        y = np.array(eta_measured)[[0, 1, 5]] # Henter målt North, East og heading
        y_pred = np.dot(self.H, x_pred)#Beregner hva EKF forventer at målingen skal vær i den predikerte tilstanden
        innovation = y - y_pred #Beregner forskjellen mellom målt og forventet verdi
        #Sørger for at heading feilen ligger mellom -pi og pi
        innovation[2] = (innovation[2] - np.pi) % (2*np.pi) - np.pi
        if innovation[2] <= -np.pi:
            innovation[2] += 2*np.pi  

        #beregner usikkerheten i måleforskjellen
        S = np.dot(self.H, P_pred)
        S = np.dot(S, self.H.T) + self.R 

        #bergne Kalman-forsterkningen
        K = np.dot(P_pred,self.H.T) 
        K = np.dot(K, np.linalg.inv(S))

        #Korrigerer vha målingen
        correction = np.dot(K, innovation)
        self.x = x_pred + correction
        self.x[8] = (self.x[8] + np.pi) % (2*np.pi) - np.pi #estimert heading må ligge mellom -pi og pi

        #oppdaterer usikkerheten etter korreksjonen
        I = np.eye(15)
        A = I - np.dot(K, self.H)
        part_1 = np.dot(A, P_pred)
        part1 = np.dot(part_1, A.T)
        part_2 = np.dot(K, self.R)
        part2 = np.dot(part_2, K.T)
        self.P = part1 + part2

        #lage 6 elementvektorer som simulatoren forventer
        eta_est = np.zeros(6)
        nu_est = np.zeros(6)
        bias_est = np.zeros(6)
        eta_est[[0, 1, 5]] = self.x[6:9] #fyller inn nord, øst og heading
        nu_est[[0, 1, 5]] = self.x[9:12] #fyller inn surge, sway og yaw rate
        bias_est[[0, 1, 5]] = self.x[12:15] #fyller inn estimert miljøbias
        return ObserverEstimate(eta=eta_est, nu=nu_est, bias=bias_est) #sender estimatene tilbake til simulatoren


def select_observer(kind: str, **kwargs) -> Observer:
    """Construct one observer without changing simulator code."""
    choices = {
        "nonlinear_passive": NonlinearPassiveObserver,
        "kalman": KalmanFilterObserver,
    }
    try:
        return choices[kind](**kwargs)
    except KeyError as exc:
        raise ValueError(f"observer_type must be one of {tuple(choices)}") from exc

