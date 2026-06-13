# Expressive Null-Space Control on a 6-DOF Arm

A redundant robot performs a fixed task while its **spare (null-space) degrees of
freedom** are used to express emotion, parameterised by **PAD** (Pleasure, Arousal,
Dominance). Implemented and run both in kinematic simulation and on the real arm.

| | |
|---|---|
| **Robot** | Seeed reBot Arm B601 — **6-DOF** revolute serial arm |
| **Task** | hold the end-effector at a fixed **3-D position** (3 DoF) |
| **Redundancy** | 6 − 3 = **3 null-space DoF**  (≥ 2, as required) |
| **Controller** | resolved-rate with null-space projection `N = I − J⁺J` |
| **Expression** | PAD ∈ [−1,1]³ shapes the null-space reference motion |

## Why there are ≥ 2 redundant DoF
The task is only the tool **position**, so the position Jacobian `J` is **3×6** with
rank 3. The null space of `J` has dimension `6 − 3 = 3`: three independent
joint-velocity directions move the body **without moving the end-effector**. That
redundancy is where the expression lives. (Verified numerically in `robot_b601.py`;
the NumPy FK was also checked against the official URDF model and agrees to 0.00 mm.)

## Controller
```
q̇ = J⁺ · Kt (x_des − x)                # primary: hold the task point
   + N · ( q̇_ref + Kn (q_ref − q) )     # secondary: express, in the null space
N  = I − J⁺J            J⁺ = Jᵀ (J Jᵀ + λ²I)⁻¹   (damped pseudo-inverse)
```
The secondary term is projected by `N`, so by construction it cannot disturb the task
(end-effector error stays < 1 mm in simulation).

## PAD → expression mapping (`expressive_controller.py`)
- **Arousal** → motion energy: oscillation amplitude & frequency (still ↔ large/fast)
- **Dominance** → posture: expansion & elevation (contracted/low ↔ tall/open/extended)
- **Pleasure** → openness & smoothness (closed + jerky higher-harmonic ↔ open smooth sine)

## Real robot
`real_robot_run.py` runs the **same controller** on the physical B601 over the
LeRobot / Damiao CAN stack. Safety: defaults to a dry-run; `--execute` moves the arm
with reduced amplitude (×0.4), low joint speed (30°/s), a 6° per-step clamp and a gentle
ramp. Measured result: the **arousal→energy trend holds on hardware** — sad ≈ still
(~0.6°/s) vs high-arousal (~5.3°/s).

## Files
| file | purpose |
|---|---|
| `robot_b601.py` | pure-NumPy forward kinematics + geometric Jacobian (B601 URDF geometry) |
| `expressive_controller.py` | PAD→reference mapping + null-space controller + `simulate()` |
| `real_robot_run.py` | real-arm execution (LeRobot/Damiao CAN); safe dry-run default, `--execute` to move |

## Run
```bash
pip install -r requirements.txt          # numpy
python3 robot_b601.py                     # FK + Jacobian check, null-space dim = 3
python3 expressive_controller.py          # per-PAD metrics (max EE error < 1 mm)

# real arm (needs the lab LeRobot/ROS stack + a powered B601):
#   source environment/setup_ros_env.sh
#   python3 real_robot_run.py             # dry-run (no motion)
#   python3 real_robot_run.py --execute   # moves the arm (supervise!)
```

> The B601 URDF geometry is reproduced from the lab description package. The dataset /
> hardware and the lab LeRobot stack are not redistributed here.
