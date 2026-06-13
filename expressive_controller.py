#!/usr/bin/env python3
"""
Expressive null-space controller for the 6-DOF reBot B601.

TASK (3 DoF):   keep the end-effector at a fixed 3-D position  x_des.
REDUNDANCY:     6 joints - 3 task constraints = 3 null-space DoF (>= 2).

The redundant motion that does NOT disturb the end-effector is used to *express
emotion*, parameterised by PAD = (Pleasure, Arousal, Dominance), each in [-1, 1].

Controller (resolved-rate, redundancy resolution):
    J   = position Jacobian (3x6)
    J+  = J^T (J J^T + lambda^2 I)^-1            (damped right pseudo-inverse)
    N   = I - J+ J                               (null-space projector, rank 3)
    q_dot = J+ (Kp_task (x_des - x))             # primary: hold the task point
          + N ( q_ref_dot + Kp_null (q_ref - q)) # secondary: expressive posture
The N(...) term lives entirely in the null space, so it cannot move the
end-effector -- the task is preserved by construction while the body "emotes".

PAD -> expression mapping (documented design, not unique):
    Arousal  A : energy of the motion  -> oscillation amplitude & frequency
                 (high A = large, fast; low A = small, slow / still)
    Dominance D: postural assertiveness -> expansion / elevation of the arm
                 (high D = tall, open, extended; low D = low, contracted)
    Pleasure P : valence -> openness + smoothness
                 (P>0 = open posture, smooth sine; P<0 = closed, agitated/jerky
                  via an added higher harmonic)
"""
import numpy as np
import robot_b601 as R

# ---- neutral pose & task target -------------------------------------------------
# Chosen within the REAL B601 joint limits (shoulder_lift, elbow_flex are negative;
# their hardware range is (-170,1) and (-200,1) deg respectively).
Q0_DEG = np.array([0.0, -50.0, -60.0, 0.0, 35.0, 0.0])    # natural presenting pose
Q0 = np.deg2rad(Q0_DEG)
X_DES = R.ee_position(Q0)                                  # the fixed task point

# ---- expressive joint-space directions (unit) ----------------------------------
# "expansion / elevation": raise shoulder, extend elbow, lift wrist
_U_POST = np.array([0.0, -1.0, -0.7, 0.0, 0.8, 0.0]); _U_POST /= np.linalg.norm(_U_POST)
# "sway / breathe": base yaw + shoulder bob + wrist
_U_OSC = np.array([1.0, 0.5, -0.4, 0.0, 0.5, 0.6]);    _U_OSC /= np.linalg.norm(_U_OSC)

# ---- mapping gains --------------------------------------------------------------
POST_SCALE = 0.60     # rad, max static postural bias
AMP_OSC    = 0.45     # rad, max oscillation amplitude
F_MIN, F_MAX = 0.15, 0.85   # Hz, oscillation frequency range


def pad_to_reference(pad, amp_scale=1.0, q0=Q0):
    """Return functions q_ref(t), q_ref_dot(t) for a given PAD = (P, A, D).

    amp_scale shrinks both the posture bias and the oscillation (use < 1 for
    gentle real-hardware motion). q0 is the neutral pose the expression rides on.
    """
    P, A, D = [float(np.clip(v, -1, 1)) for v in pad]
    a01 = (A + 1.0) / 2.0                       # arousal in [0,1]
    amp  = amp_scale * AMP_OSC * (0.10 + 0.90 * a01)   # amplitude grows with arousal
    freq = F_MIN + (F_MAX - F_MIN) * a01        # frequency grows with arousal
    posture = amp_scale * POST_SCALE * (0.7 * D + 0.3 * P) * _U_POST
    jerk = 0.35 * max(0.0, -P)                  # negative valence -> agitation
    w     = 2.0 * np.pi * freq

    def q_ref(t):
        s = np.sin(w * t) + jerk * np.sin(3.0 * w * t)
        return R.clip_limits(Q0 + posture + amp * s * _U_OSC)

    def q_ref_dot(t):
        sd = w * np.cos(w * t) + jerk * 3.0 * w * np.cos(3.0 * w * t)
        return amp * sd * _U_OSC

    return q_ref, q_ref_dot


def damped_pinv(J, lam=0.02):
    return J.T @ np.linalg.inv(J @ J.T + (lam ** 2) * np.eye(J.shape[0]))


def simulate(pad, duration=10.0, dt=0.02, Kp_task=6.0, Kp_null=4.0, amp_scale=1.0):
    """Run the expressive null-space controller. Returns trajectory + diagnostics.

    amp_scale < 1 produces gentler motion (used for real-hardware playback)."""
    q_ref, q_ref_dot = pad_to_reference(pad, amp_scale=amp_scale)
    n = int(duration / dt)
    q = Q0.copy()
    T, Q, EE, ERR = [], [], [], []
    for k in range(n):
        t = k * dt
        frames = R.fk_frames(q)
        p_ee = frames[-1][:3, 3]
        J = R.position_jacobian(q)
        Jp = damped_pinv(J)
        N = np.eye(R.NUM_DOF) - Jp @ J
        e = X_DES - p_ee
        qd = Jp @ (Kp_task * e) + N @ (q_ref_dot(t) + Kp_null * (q_ref(t) - q))
        q = R.clip_limits(q + qd * dt)
        T.append(t); Q.append(q.copy()); EE.append(p_ee.copy()); ERR.append(np.linalg.norm(e))
    return {
        "pad": np.asarray(pad, float), "t": np.array(T), "Q": np.array(Q),
        "EE": np.array(EE), "err_mm": np.array(ERR) * 1000.0,
        "target": X_DES, "Q0": Q0,
    }


if __name__ == "__main__":
    print(f"Task target x_des = {X_DES.round(4)} (m),  null-space dim = "
          f"{R.NUM_DOF - np.linalg.matrix_rank(R.position_jacobian(Q0))}")
    pads = {"baseline [0,0,0]": [0, 0, 0], "[1,1,1]": [1, 1, 1], "[-1,-1,-1]": [-1, -1, -1],
            "[-1,1,-1]": [-1, 1, -1], "[-1,1,1]": [-1, 1, 1]}
    print(f"\n{'PAD':18s} {'maxEEerr(mm)':>13s} {'posture range(deg)':>20s} {'mean|qdot|':>12s}")
    for name, pad in pads.items():
        s = simulate(pad)
        rng = np.rad2deg(s["Q"].max(0) - s["Q"].min(0)).max()
        qd = np.abs(np.diff(s["Q"], axis=0)).mean() / 0.02
        print(f"{name:18s} {s['err_mm'].max():13.4f} {rng:20.1f} {np.rad2deg(qd):12.2f}")
