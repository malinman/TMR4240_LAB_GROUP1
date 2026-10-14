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


        #Wave model parameters (to be tuned)
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

        #Bias model time constants (preliminary values)
        self.Tb = np.diag([1000.0, 1000.0, 1000.0])
        self.Tb_inv = np.diag(1.0 / np.diag(self.Tb))

        #NPO correction gains
        #Preliminary wave-filter gains
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
        pass

    def step(self, t, dt, eta_measured, tau_est):
        # TODO: implement the Kalman filter or extended Kalman filter.
        return self._placeholder(eta_measured)


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

