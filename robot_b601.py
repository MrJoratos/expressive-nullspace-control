#!/usr/bin/env python3
"""
Kinematics of the Seeed reBot Arm B601 (6-DOF revolute arm).

Pure-NumPy forward kinematics + geometric Jacobian, ported directly from the
verified URDF geometry in `environment/kgdt/fk.py` (same joint origins / axes).
No torch dependency.

Frames follow the URDF: each movable joint i has a constant origin transform
O_i (xyz, rpy) followed by a rotation R_i(theta_i) about a unit axis. A final
fixed transform E maps link6 -> end_link (the tool point).

    T_end(q) = O_1 R_1 O_2 R_2 ... O_6 R_6 E
    p_ee     = T_end[:3, 3]            # end-effector position in base frame (m)

Geometric position Jacobian for a revolute joint:
    Jv_i = z_i x (p_ee - p_i)
with z_i the joint axis (base frame) and p_i a point on that axis (the joint
frame origin). Angles here are in RADIANS.
"""
import numpy as np

# (xyz, rpy, axis) per movable joint, base -> link6, then the fixed end joint.
# rpy is URDF fixed-axis roll-pitch-yaw:  R = Rz(yaw) Ry(pitch) Rx(roll).
_JOINTS = [
    (np.array([-8.416e-05, 0.0, 0.08465]), np.array([0.0, 0.0, 0.0]),        np.array([0.0, 0.0, 1.0])),
    (np.array([0.020084, 0.031625, 0.05555]), np.array([-1.5708, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    (np.array([-0.264, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]),                np.array([0.0, 0.0, 1.0])),
    (np.array([0.2426, -0.054, -0.001625]), np.array([0.0, 0.0, 0.0]),       np.array([0.0, 0.0, 1.0])),
    (np.array([0.078308, -0.0375, -0.03]), np.array([-1.5708, 0.0, 0.0]),    np.array([0.0, 0.0, 1.0])),
    (np.array([0.028008, 0.0, 0.04]), np.array([0.0, 1.5708, 0.0]),          np.array([0.0, 0.0, 1.0])),
]
_END = (np.array([0.0, 0.0, 0.15539]), np.array([0.0, -1.5708, 3.1415]))

NUM_DOF = 6

# Conservative joint limits (rad) for clipping during the demo (URDF-style).
JOINT_LIMITS = np.deg2rad(np.array([
    [-170, 170], [-120, 120], [-160, 160], [-170, 170], [-120, 120], [-175, 175],
], dtype=float))


def _rpy(rpy):
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _origin_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy(rpy)
    T[:3, 3] = xyz
    return T


_ORIGIN_T = [_origin_T(xyz, rpy) for (xyz, rpy, _ax) in _JOINTS]
_AXES = [ax / np.linalg.norm(ax) for (_xyz, _rpy, ax) in _JOINTS]
_END_T = _origin_T(_END[0], _END[1])


def _axis_angle_T(axis, theta):
    x, y, z = axis
    c, s, C = np.cos(theta), np.sin(theta), 1.0 - np.cos(theta)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ])
    return T


def fk_frames(q):
    """q: (6,) radians. Returns list of cumulative transforms [T1..T6, T_end] (base frame)."""
    q = np.asarray(q, float)
    T = np.eye(4)
    frames = []
    for i in range(NUM_DOF):
        T = T @ _ORIGIN_T[i] @ _axis_angle_T(_AXES[i], q[i])
        frames.append(T.copy())
    frames.append(T @ _END_T)         # end-effector / tool frame
    return frames


def ee_position(q):
    return fk_frames(q)[-1][:3, 3]


def link_points(q):
    """Polyline of the kinematic chain in base frame: (8, 3) = base + 6 joints + end."""
    frames = fk_frames(q)
    pts = [np.zeros(3)] + [f[:3, 3] for f in frames]
    return np.array(pts)


def position_jacobian(q):
    """3x6 position Jacobian d(p_ee)/dq  (q in radians)."""
    frames = fk_frames(q)
    p_ee = frames[-1][:3, 3]
    J = np.zeros((3, NUM_DOF))
    for i in range(NUM_DOF):
        Ti = frames[i]
        z_i = Ti[:3, :3] @ _AXES[i]     # joint axis in base frame
        p_i = Ti[:3, 3]                 # point on the axis
        J[:, i] = np.cross(z_i, p_ee - p_i)
    return J


def clip_limits(q):
    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


if __name__ == "__main__":
    # geometry sanity
    for name, qd in [("home", np.zeros(6)),
                     ("j1=90", np.array([90., 0, 0, 0, 0, 0])),
                     ("j2=-90", np.array([0, -90., 0, 0, 0, 0]))]:
        p = ee_position(np.deg2rad(qd))
        print(f"{name:7s} ee=[{p[0]:+.4f} {p[1]:+.4f} {p[2]:+.4f}]  reach={np.linalg.norm(p):.4f} m")

    # finite-difference Jacobian check
    q = np.deg2rad([10., -40, 50, 20, 30, -15])
    J = position_jacobian(q)
    Jn = np.zeros((3, 6)); e = 1e-6
    for i in range(6):
        dq = q.copy(); dq[i] += e
        Jn[:, i] = (ee_position(dq) - ee_position(q)) / e
    print("max |J_analytic - J_finitediff| =", np.abs(J - Jn).max())

    # task = 3D position (3 rows) -> redundancy = 6 - rank(J)
    print("rank(J) =", np.linalg.matrix_rank(J), "-> null-space dim =", 6 - np.linalg.matrix_rank(J))
