"""Thrust utilisation for Project Part 2, Simulation 5 — student work.

Definitions (from the project text, Part 2, Simulation 5):

* **Thrust utilisation** at time t is the sum of the magnitudes of the
  individual thruster forces, as a percentage of the sum of the maximum
  nominal thrusts of all active thrusters,

      U_T(t) = 100 * sum_i |u_i(t)| / sum_i u_max,i .

* **Average thrust utilisation** is the time average of U_T(t) over the
  evaluation window — by default the second half of the run, so the
  transient is excluded.

``simulation/capability.py`` (provided) runs the directional sweep, computes
the deviation measures and draws the polar plots; it calls the two functions
below for every direction.  Part 2, Simulation 5 raises
``NotImplementedError`` until you implement them.  The public check
(``python check.py --part 2``) verifies them on a synthetic log.

Inputs: ``logs`` carries ``logs.t`` (n,) time [s] and ``logs.u``
(n, n_thrusters) ACTUAL thrusts [N] (one signed column per thruster);
``thrusters`` is the list of ``ThrusterConfig`` with ``u_max`` [N].
"""
from __future__ import annotations

import numpy as np


def thrust_utilization(logs, thrusters) -> np.ndarray:
    """Instantaneous thrust utilisation U_T(t) in percent, from ACTUAL thrusts.

    TODO (students): implement the definition above — the sum of the
    magnitudes of the individual actuator forces (``logs.u``, one column per
    thruster) as a percentage of the sum of the maximum nominal thrusts of
    all active thrusters (``ThrusterConfig.u_max``).  Return one value per
    time sample, shape ``(len(logs.t),)``.
    """
    # raise NotImplementedError(
    #    "implement thrust_utilization() in part_2/utilization.py — the "
    #    "definition is in the project text (Part 2, Simulation 5)")

    
    """Calculate instantaneous thrust utilisation [%]."""

    actual_thrust = np.asarray(logs.u, dtype=float)

    total_thrust = np.sum(np.abs(actual_thrust), axis=1)

    max_thrust = sum(thruster.u_max for thruster in thrusters)

    if max_thrust <= 0:
        raise ValueError("Total maximum thrust must be positive")

    return 100.0 * total_thrust / max_thrust


def average_utilization(logs, thrusters, t_from: float | None = None) -> float:
    """Average thrust utilisation [%] over ``t >= t_from`` (default: the second
    half of the run, so the transient is excluded).

    TODO (students): implement — the time average of
    :func:`thrust_utilization` over the evaluation window.
    """
    # raise NotImplementedError(
    #    "implement average_utilization() in part_2/utilization.py — the "
    #    "definition is in the project text (Part 2, Simulation 5)")

    """Calculate average thrust utilisation over the evaluation window."""

    utilization = thrust_utilization(logs, thrusters)

    if t_from is None:
        t_from = 0.5 * logs.t[-1]

    mask = logs.t >= t_from

    if not np.any(mask):
        raise ValueError("No time samples in evaluation window")

    return float(np.mean(utilization[mask]))