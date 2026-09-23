import numpy as np

from part_1.config import default_thrusters_gunnerus3
from part_1.thrust_allocation import ThrustAllocator
from models.thruster_dynamics import ThrusterSet


def calculate_actual_wrench(allocator, thrusters, tau_d, alpha_now):
    """Run the allocator and calculate the actual produced wrench."""

    u_cmd, alpha_cmd = allocator.allocate(
        t=0.0,
        dt=0.05,
        tau_d=tau_d,
        alpha_now=alpha_now
    )

    thruster_set = ThrusterSet(thrusters, dynamics=False)

    _, _, tau_actual = thruster_set.step(
        u_cmd=u_cmd,
        alpha_cmd=alpha_cmd,
        dt=0.05
    )

    return u_cmd, alpha_cmd, tau_actual


def test_pure_surge():
    thrusters = default_thrusters_gunnerus3()
    allocator = ThrustAllocator(thrusters)

    tau_d = np.array([
        1000.0,  # Fx
        0.0,     # Fy
        0.0,     # Fz
        0.0,     # Mx
        0.0,     # My
        0.0      # Mz
    ])

    alpha_now = np.array([np.pi / 2, 0.0, 0.0])

    _, _, tau_actual = calculate_actual_wrench(
        allocator,
        thrusters,
        tau_d,
        alpha_now
    )

    desired_wrench = tau_d[[0, 1, 5]]

    assert np.allclose(tau_actual, desired_wrench)

def test_pure_sway():
    thrusters = default_thrusters_gunnerus3()
    allocator = ThrustAllocator(thrusters)

    # Request 1000 N pure sway
    tau_d = np.array([0.0, 1000.0, 0.0, 0.0, 0.0, 0.0])

    # Current thruster angles
    alpha_now = np.array([np.pi / 2, 0.0, 0.0])

    _, _, tau_actual = calculate_actual_wrench(
        allocator,
        thrusters,
        tau_d,
        alpha_now
    )

    desired_wrench = tau_d[[0, 1, 5]]

    assert np.allclose(tau_actual, desired_wrench)

def test_pure_yaw():
    thrusters = default_thrusters_gunnerus3()
    allocator = ThrustAllocator(thrusters)

    # Request 1000 Nm pure yaw moment
    tau_d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1000.0])

    # Current thruster angles
    alpha_now = np.array([np.pi / 2, 0.0, 0.0])

    _, _, tau_actual = calculate_actual_wrench(
        allocator,
        thrusters,
        tau_d,
        alpha_now
    )

    desired_wrench = tau_d[[0, 1, 5]]

    assert np.allclose(tau_actual, desired_wrench)

def test_thrust_limits():
    thrusters = default_thrusters_gunnerus3()
    allocator = ThrustAllocator(thrusters)

    # Request a wrench large enough to reach the thruster limits
    tau_d = np.array([180000.0, 0.0, 0.0, 0.0, 0.0, 300000.0])

    alpha_now = np.array([np.pi / 2, 0.0, 0.0])

    u_cmd, _, _ = calculate_actual_wrench(
        allocator,
        thrusters,
        tau_d,
        alpha_now
    )

    # Maximum allowed thrust for each thruster
    u_max = np.array([thruster.u_max for thruster in thrusters])

    # Check that no commanded thrust exceeds its limit
    assert np.all(np.abs(u_cmd) <= u_max)

def test_minimum_azimuth_rotation():
    thrusters = default_thrusters_gunnerus3()
    allocator = ThrustAllocator(thrusters)

    # Request 1000 N pure surge
    tau_d = np.array([1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Assume both azimuth thrusters are currently pointing backwards (180 degrees)
    alpha_now = np.array([np.pi / 2, np.pi, np.pi])

    u_cmd, alpha_cmd, tau_actual = calculate_actual_wrench(
        allocator,
        thrusters,
        tau_d,
        alpha_now
    )

    desired_wrench = tau_d[[0, 1, 5]]

    # The azimuth thrusters should stay close to 180 degrees
    # and use negative thrust instead of rotating 180 degrees
    assert np.isclose(abs(alpha_cmd[1]), np.pi)
    assert np.isclose(abs(alpha_cmd[2]), np.pi)
    assert u_cmd[1] < 0.0
    assert u_cmd[2] < 0.0

    # The produced wrench must still be correct
    assert np.allclose(tau_actual, desired_wrench)

def test_zero_thrust_keeps_current_angles():
    thrusters = default_thrusters_gunnerus3()
    allocator = ThrustAllocator(thrusters)

    # No requested force or moment
    tau_d = np.zeros(6)

    # Assume the azimuth thrusters are currently at 90 degrees
    alpha_now = np.array([np.pi / 2, np.pi / 2, np.pi / 2])

    u_cmd, alpha_cmd, tau_actual = calculate_actual_wrench(
        allocator,
        thrusters,
        tau_d,
        alpha_now
    )

    # No thrust should be produced
    assert np.allclose(u_cmd, 0.0)

    # Azimuth thrusters should keep their current angles
    assert np.isclose(alpha_cmd[1], alpha_now[1])
    assert np.isclose(alpha_cmd[2], alpha_now[2])

    # No wrench should be produced
    assert np.allclose(tau_actual, np.zeros(3))


def test_combined_wrench():
    thrusters = default_thrusters_gunnerus3()
    allocator = ThrustAllocator(thrusters)

    # Request simultaneous surge, sway and yaw
    tau_d = np.array([
        10000.0,     # Fx
        10000.0,     # Fy
        0.0,         # Fz
        0.0,         # Mx
        0.0,         # My
        20000.0      # Mz
    ])

    alpha_now = np.array([np.pi / 2, 0.0, 0.0])

    _, _, tau_actual = calculate_actual_wrench(
        allocator,
        thrusters,
        tau_d,
        alpha_now
    )

    desired_wrench = tau_d[[0, 1, 5]]

    # The allocator should reproduce the requested wrench
    assert np.allclose(tau_actual, desired_wrench)


def test_tunnel_thruster_limit():
    thrusters = default_thrusters_gunnerus3()
    allocator = ThrustAllocator(thrusters)

    # Large sway/yaw request that can require high tunnel thrust
    tau_d = np.array([
        50000.0,      # Fx
        50000.0,      # Fy
        0.0,          # Fz
        0.0,          # Mx
        0.0,          # My
        200000.0      # Mz
    ])

    alpha_now = np.array([np.pi / 2, 0.0, 0.0])

    u_cmd, _, _ = calculate_actual_wrench(
        allocator,
        thrusters,
        tau_d,
        alpha_now
    )

    # Bow tunnel thruster has a maximum absolute thrust of 32 kN
    assert abs(u_cmd[0]) <= 32000.0

def test_configuration_matrix_rank():
    # Extended configuration matrix for the Gunnerus thruster setup
    B_e = np.array([
        [0.0,   1.0,   0.0,   1.0,   0.0],
        [1.0,   0.0,   1.0,   0.0,   1.0],
        [12.0, -3.0, -13.0,   3.0, -13.0]
    ])

    rank = np.linalg.matrix_rank(B_e)

    # The matrix must have rank 3 to control surge, sway and yaw
    assert rank == 3