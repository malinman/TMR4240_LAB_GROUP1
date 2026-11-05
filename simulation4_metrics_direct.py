"""Run Simulation 4 and calculate report metrics directly from simulator logs.

Place next to run_case_part_2.py and run: python simulation4_metrics_direct.py
Requires the project dependencies, including numpy and scipy.

LF 'truth' is a low-pass-filtered proxy from simulated total motion, NOT an
independently simulated low-frequency state. Report this methodological limit.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_case_part_2 as rc

DT = rc.DT
CUTOFF_RAD_S = 0.25  # LF proxy cutoff, below design wave frequency ~0.785 rad/s
WAVE_BAND_RAD_S = (0.45, 1.2)
EDGE_SECONDS = 30.0
THRUST_CAPACITY_N = 192000.0  # same total denominator as print_sim6_metrics
CONVERGENCE_HOLD_SECONDS = 20.0

def wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))

def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=float)**2)))

def filtered(x, t, cutoff=CUTOFF_RAD_S, band=None):
    dt = float(np.median(np.diff(t)))
    nyquist = np.pi / dt
    if band is None:
        sos = butter(4, cutoff / nyquist, btype='lowpass', output='sos')
    else:
        sos = butter(4, np.array(band) / nyquist, btype='bandpass', output='sos')
    return sosfiltfilt(sos, x, axis=0)

def data(log):
    t = np.asarray(log.t)
    eta = np.asarray(log.eta)[:, [0, 1, 5]].copy()
    eta[:, 2] = np.unwrap(eta[:, 2])
    nu = np.asarray(log.nu)[:, [0, 1, 5]]
    lf_eta = filtered(eta, t)
    lf_nu = filtered(nu, t)
    mask = (t >= EDGE_SECONDS) & (t <= t[-1] - EDGE_SECONDS)
    if not np.any(mask):
        raise ValueError('Run too short for edge exclusion')
    return t, eta, nu, lf_eta, lf_nu, mask

def open_metrics(log):
    t, eta, nu, lf_eta, lf_nu, m = data(log)
    est_eta = np.asarray(log.eta_est)[:, [0, 1, 5]].copy()
    est_eta[:, 2] = np.unwrap(est_eta[:, 2])
    est_nu = np.asarray(log.nu_est)[:, [0, 1, 5]]
    pose_err = est_eta - lf_eta
    pose_err[:, 2] = wrap(pose_err[:, 2])
    vel_err = est_nu - lf_nu
    # Wave-band attenuation: estimated vs measured pose amplitudes, in dB.
    # Negative values mean attenuation. Uses actual measured pose where available.
    meas = np.asarray(log.eta_measured)[:, [0, 1, 5]].copy()
    meas[:, 2] = np.unwrap(meas[:, 2])
    meas_wf = filtered(meas, t, band=WAVE_BAND_RAD_S)
    est_wf = filtered(est_eta, t, band=WAVE_BAND_RAD_S)
    attenuation = [20*np.log10(max(rms(est_wf[m, i]), 1e-12) /
                                max(rms(meas_wf[m, i]), 1e-12)) for i in range(3)]
    # Smoothness: RMS of estimated pose first derivative (N/E in m/s, psi in deg/s).
    # This is an explicit operational definition, not a universal metric.
    smooth = [rms(np.gradient(est_eta[:, i], t)[m]) for i in range(3)]
    return {
        'North RMSE [m]': rms(pose_err[m, 0]),
        'East RMSE [m]': rms(pose_err[m, 1]),
        'Heading RMSE [deg]': np.degrees(rms(pose_err[m, 2])),
        'Surge velocity RMSE [m/s]': rms(vel_err[m, 0]),
        'Sway velocity RMSE [m/s]': rms(vel_err[m, 1]),
        'Yaw rate RMSE [deg/s]': np.degrees(rms(vel_err[m, 2])),
        'Wave attenuation N/E/psi [dB]': attenuation,
        'Estimate smoothness N/E/psi [m/s,m/s,deg/s]': [smooth[0], smooth[1], np.degrees(smooth[2])],
    }

def convergence_time(log):
    t, eta, nu, lf_eta, lf_nu, m = data(log)
    est = np.asarray(log.eta_est)[:, [0, 1, 5]]
    error = est - lf_eta
    error[:, 2] = wrap(error[:, 2])
    # Sustained within 1m in N/E and 2deg in heading, for 20s.
    ok = ((np.abs(error[:, 0]) < 1.0) &
          (np.abs(error[:, 1]) < 1.0) &
          (np.abs(error[:, 2]) < np.deg2rad(2.0)))
    valid = np.flatnonzero((t >= EDGE_SECONDS) & (t <= t[-1] - EDGE_SECONDS))
    hold = int(np.ceil(CONVERGENCE_HOLD_SECONDS / np.median(np.diff(t))))
    for i in valid:
        if i + hold < len(t) and np.all(ok[i:i+hold]):
            return float(t[i] - t[0])
    return None

def find_azimuth(log):
    # Different simulator versions may use different names; inspect log attributes.
    for name in ('alpha', 'azimuth', 'azimuth_angles', 'thruster_angles', 'theta'):
        value = getattr(log, name, None)
        if value is None:
            continue
        a = np.asarray(value)
        if a.ndim == 2 and a.shape[0] == len(log.t) and a.shape[1] >= 2:
            return name, a[:, -2:]
    return None, None

def closed_metrics(log):
    t = np.asarray(log.t)
    eta = np.asarray(log.eta)
    # Sim4 closed-loop commands are zero, but use cmd to avoid hard-coding.
    cmd = np.asarray(log.cmd)
    pos_err = np.hypot(eta[:, 0] - cmd[:, 0], eta[:, 1] - cmd[:, 1])
    heading_err = wrap(eta[:, 5] - cmd[:, 5])
    u = np.asarray(log.u)
    utilisation = 100.0 * np.sum(np.abs(u), axis=1) / THRUST_CAPACITY_N
    name, az = find_azimuth(log)
    # Unwrap actual azimuth angles, then sum absolute increments for both units.
    rotation = None if az is None else float(np.degrees(np.abs(np.diff(np.unwrap(az, axis=0), axis=0)).sum()))
    return {
        'Position RMSE [m]': rms(pos_err),
        'Maximum position error [m]': float(pos_err.max()),
        'Heading RMSE [deg]': np.degrees(rms(heading_err)),
        'Mean thrust utilisation [%]': float(utilisation.mean()),
        'Total azimuth rotation [deg]': rotation,
        'azimuth_source': name,
    }

def fmt(value):
    if value is None:
        return 'N/A (azimuth log not found, or no sustained convergence)'
    if isinstance(value, list):
        return ' / '.join(f'{v:.3f}' for v in value)
    if isinstance(value, (int, float, np.number)):
        return f'{value:.3f}'
    return str(value)

def print_comparison(title, a_name, a, b_name, b):
    print('\n' + title)
    print(f'{"Metric":46s} {a_name:>22s} {b_name:>22s}')
    print('-'*94)
    for key in a:
        if key == 'azimuth_source':
            continue
        print(f'{key:46s} {fmt(a[key]):>22s} {fmt(b[key]):>22s}')

def main():
    print('Running Simulation 4 (all open-loop, convergence and closed-loop cases)...')
    logs = rc.sim4()
    plt.close('all')
    npo = open_metrics(logs[('nonlinear_passive', 'with waves')])
    ekf = open_metrics(logs[('kalman', 'with waves')])
    npo['Convergence time [s]'] = convergence_time(logs[('nonlinear_passive', 'offset start')])
    ekf['Convergence time [s]'] = convergence_time(logs[('kalman', 'offset start')])
    print_comparison('OPEN-LOOP (WITH WAVES)', 'NPO', npo, 'EKF', ekf)
    raw = closed_metrics(logs[('closed loop', 'raw measurements')])
    obs_name = f'observer {rc.selected_observer()}'
    obs = closed_metrics(logs[('closed loop', obs_name)])
    print_comparison('CLOSED-LOOP STATION KEEPING', 'Raw', raw, rc.selected_observer(), obs)
    print('\nAzimuth log fields:', raw['azimuth_source'], obs['azimuth_source'])
    print('NOTE: LF truth is a 0.25 rad/s low-pass proxy, not a true independent LF state.')
    print('NOTE: WF attenuation is shown separately for N/E/heading, negative dB means filtering.')
    print('NOTE: convergence requires 1m N, 1m E, 2deg heading for 20 consecutive seconds.')
    print('NOTE: smoothness is RMS of first derivative, not a unitless subjective score.')
    print('NOTE: thrust utilisation uses total capacity 192000 N, as in run_case_part_2.py.')

if __name__ == '__main__':
    main()
