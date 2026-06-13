#!/usr/bin/env python3
"""
Real-robot execution of the expressive null-space controller on the Seeed
reBot B601, with synchronized Orbbec RGB recording and tracking logging.

SAFETY (read this):
  * Defaults to --dry-run: computes & prints the trajectory, NO motion, NO connect.
  * --record-test: records a few seconds of the camera only (still NO motion).
  * --execute: actually moves the arm. Conservative by design:
      - amplitudes scaled down (--amp-scale, default 0.4)
      - low joint velocity (--vel, default 30 deg/s)
      - per-step relative clamp (--max-rel, default 6 deg) via the SDK
      - gentle linear RAMP from the current pose to the start pose
    Clear the workspace and keep a hand on the power switch before --execute.

Run from the ROS-clean env:
    source environment/setup_ros_env.sh
    /usr/bin/python3 real_robot_run.py            # dry-run
    /usr/bin/python3 real_robot_run.py --record-test
    /usr/bin/python3 real_robot_run.py --execute  # moves the arm
"""
import argparse
import os
import sys
import time
import threading
from pathlib import Path
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "real")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import robot_b601 as R
import expressive_controller as C

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
CLIPS = [([0, 0, 0], "baseline"), ([1, 1, 1], "happy"), ([-1, -1, -1], "sad"),
         ([-1, 1, -1], "angry"), ([-1, 1, 1], "tense")]
FPS = 30
COLOR_TOPIC = "/camera/color/image_raw"


# --------------------------------------------------------------------------- camera
class CameraRecorder:
    """Background ROS subscriber buffering RGB frames; write per-clip mp4."""
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.br = CvBridge()
        self.lock = threading.Lock()
        self.recording = False
        self.frames = []
        qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.node = Node("expressive_rec")
        self.node.create_subscription(Image, COLOR_TOPIC, self._cb, qos)
        self._spin = threading.Thread(target=self._spin_loop, daemon=True)
        self._alive = True
        self._spin.start()

    def _spin_loop(self):
        while self._alive and self._rclpy.ok():
            self._rclpy.spin_once(self.node, timeout_sec=0.05)

    def _cb(self, msg):
        if self.recording:
            img = self.br.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.frames.append((time.time(), img))

    def start(self):
        with self.lock:
            self.frames = []
        self.recording = True

    def stop_and_save(self, path):
        self.recording = False
        time.sleep(0.1)
        import cv2
        with self.lock:
            frames = list(self.frames)
        if not frames:
            print(f"  [camera] no frames for {path}")
            return 0
        h, w = frames[0][1].shape[:2]
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
        for _, im in frames:
            vw.write(im)
        vw.release()
        print(f"  [camera] wrote {path}  ({len(frames)} frames)")
        return len(frames)

    def close(self):
        self._alive = False
        time.sleep(0.1)


# --------------------------------------------------------------------------- robot
def build_follower(port, max_rel, vel):
    for p in ["~/rebot_lerobot", "~/rebot_lerobot/lerobot/src",
              "~/rebot_lerobot/lerobot-robot-seeed-b601"]:
        sys.path.insert(0, str(Path(p).expanduser()))
    from lerobot_robot_seeed_b601.seeed_b601_dm_follower import SeeedB601DMFollower
    from lerobot_robot_seeed_b601.config_seeed_b601_dm_follower import SeeedB601DMFollowerConfig
    cfg = SeeedB601DMFollowerConfig(port=port, id="follower1", can_adapter="damiao",
                                    max_relative_target=float(max_rel),
                                    pos_vel_velocity=[float(vel)] * 7,
                                    disable_torque_on_disconnect=True)
    return SeeedB601DMFollower(cfg)


def connect_with_retries(port, max_rel, vel, tries=8):
    """The Damiao CAN handshake (ensure_mode) intermittently times out; the docs
    say 2-4 retries is normal. Recreate the follower each attempt."""
    last = None
    for i in range(tries):
        robot = build_follower(port, max_rel, vel)
        try:
            robot.connect(calibrate=False)
            print(f"  connected on attempt {i + 1}")
            return robot
        except Exception as e:
            last = e
            print(f"  connect attempt {i + 1}/{tries} failed: {str(e)[:70]}")
            try:
                robot.disconnect()
            except Exception:
                pass
            time.sleep(1.2)
    raise RuntimeError(f"could not connect after {tries} tries: {last}")


def read_pose_deg(robot):
    obs = robot.get_observation()
    return np.array([obs[f"{m}.pos"] for m in MOTORS], float)


def send_pose_deg(robot, q_deg, gripper):
    action = {f"{m}.pos": float(q_deg[i]) for i, m in enumerate(MOTORS)}
    action["gripper.pos"] = float(gripper)
    return robot.send_action(action)


def ramp(robot, q_from, q_to, gripper, seconds=3.0):
    n = int(seconds * FPS)
    for k in range(n + 1):
        a = (k / n)
        send_pose_deg(robot, (1 - a) * q_from + a * q_to, gripper)
        time.sleep(1.0 / FPS)


def play_clip(robot, Q_deg, gripper, cam, tag):
    """Stream a joint trajectory at FPS, log measured poses, record camera."""
    meas = []
    if cam:
        cam.start()
    for q in Q_deg:
        send_pose_deg(robot, q, gripper)
        meas.append(read_pose_deg(robot))
        time.sleep(1.0 / FPS)
    if cam:
        cam.stop_and_save(os.path.join(OUT, f"real_{tag}.mp4"))
    return np.array(meas)


# --------------------------------------------------------------------------- main
def trajectories(amp_scale, duration):
    """Precompute joint trajectories (deg) for every clip."""
    out = {}
    for pad, name in CLIPS:
        s = C.simulate(pad, duration=duration, dt=1.0 / FPS, amp_scale=amp_scale)
        out[name] = (pad, np.rad2deg(s["Q"]), s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="ACTUALLY move the arm")
    ap.add_argument("--record-test", action="store_true", help="record camera only, no motion")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--amp-scale", type=float, default=0.4)
    ap.add_argument("--vel", type=float, default=30.0, help="joint velocity deg/s")
    ap.add_argument("--max-rel", type=float, default=6.0, help="per-step clamp deg")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--gripper", type=float, default=0.0)
    args = ap.parse_args()

    trajs = trajectories(args.amp_scale, args.duration)
    print("=== Expressive null-space — real-robot plan ===")
    print(f"start pose Q0(deg) = {C.Q0_DEG.round(1)}   task x_des = {C.X_DES.round(3)} m")
    for name, (pad, Qd, s) in trajs.items():
        rng = (Qd.max(0) - Qd.min(0)).max()
        print(f"  {name:9s} PAD={pad}  joints span [{Qd.min():+.1f},{Qd.max():+.1f}] deg, "
              f"max excursion {rng:.1f} deg, sim EE err < {s['err_mm'].max():.2f} mm")

    if args.record_test:
        cam = CameraRecorder()
        print("\n[record-test] recording 3 s of the static scene (NO motion)...")
        cam.start(); time.sleep(3.0); cam.stop_and_save(os.path.join(OUT, "record_test.mp4"))
        cam.close(); return

    if not args.execute:
        print("\nDRY-RUN only (no connection, no motion). Pass --execute to move the arm.")
        return

    # ---- real motion (gated) -------------------------------------------------
    print("\n[execute] connecting follower ...")
    robot = connect_with_retries(args.port, args.max_rel, args.vel)
    cam = CameraRecorder()
    try:
        q_now = read_pose_deg(robot)
        obs = robot.get_observation()
        grip_hold = float(obs.get("gripper.pos", args.gripper))   # hold current gripper
        args.gripper = grip_hold
        print(f"  current pose (deg) = {q_now.round(1)}   gripper held at {grip_hold:.1f}")
        # adaptive ramp: respect the joint-velocity limit so the arm REACHES Q0
        t_ramp = max(4.0, float(np.abs(C.Q0_DEG - q_now).max()) / args.vel * 1.8)
        print(f"  ramping to start pose over {t_ramp:.1f} s ...")
        ramp(robot, q_now, C.Q0_DEG, args.gripper, seconds=t_ramp)
        for _ in range(int(1.5 * FPS)):                 # settle at Q0 before recording
            send_pose_deg(robot, C.Q0_DEG, args.gripper); time.sleep(1.0 / FPS)
        reached = read_pose_deg(robot)
        print(f"  reached (deg) = {reached.round(1)}   |Q0-reached| = "
              f"{np.abs(C.Q0_DEG - reached).round(1)}")
        logs = {}
        for name, (pad, Qd, s) in trajs.items():
            print(f"  >>> {name}  PAD={pad}")
            meas = play_clip(robot, Qd, args.gripper, cam, name)
            logs[name] = dict(pad=pad, cmd=Qd, meas=meas, sim=s["err_mm"])
            ramp(robot, Qd[-1], C.Q0_DEG, args.gripper, seconds=1.5)  # back to neutral
            time.sleep(0.5)
        np.savez(os.path.join(OUT, "real_logs.npz"),
                 **{f"{k}_cmd": v["cmd"] for k, v in logs.items()},
                 **{f"{k}_meas": v["meas"] for k, v in logs.items()})
        print("  ramping back to a parked pose ...")
        ramp(robot, C.Q0_DEG, q_now, args.gripper, seconds=3.0)
    finally:
        cam.close()
        robot.disconnect()
        print("[execute] disconnected (torque released).")


if __name__ == "__main__":
    main()
