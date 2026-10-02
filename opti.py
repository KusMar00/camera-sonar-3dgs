"""Loading utilities for the opti-acoustic turbid-water dataset
(https://github.com/ivanacollg/sonar_camera_reconstruction, ROS2 branch; data on Google Drive).

The ROS1 bags are read with `rosbags`, so no ROS install is needed. ROS1 bags embed their message
definitions, so the custom `sonar_oculus/OculusPing` type is registered straight from each bag.

Topic names are matched without the leading "/": the emulated-turbidity bags (`*_5C`, `*_7C`, `*_9C`)
log the odometry as `bruce/slam/localization/odom` instead of `/bruce/slam/localization/odom`.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

CAMERA_TOPIC = "camera/image_raw/compressed"   # sensor_msgs/CompressedImage, 1920x1080 JPEG
SONAR_TOPIC = "sonar_oculus_node/M750d/ping"   # sonar_oculus/OculusPing, polar JPEG + bearings
ODOM_TOPIC = "bruce/slam/localization/odom"    # nav_msgs/Odometry, from the authors' SLAM (not ground truth)
TOPICS = {"camera": CAMERA_TOPIC, "sonar": SONAR_TOPIC, "odom": ODOM_TOPIC}


# Calibration from the authors' config files (sonar_camera_reconstruction_pkg/config/params_*.yaml).
# Camera: 1920x1080, pinhole + plumb_bob distortion (k1, k2, p1, p2, k3). Both marina scenarios share one calibration.
CAMERA_INTRINSICS = {
    "tank": (np.array([[1096.404968, 0, 936.318685], [0, 1094.906739, 502.602195], [0, 0, 1]]),
             np.array([0.028292, -0.006851, -0.004749, -0.004993, 0.0])),
    "marina": (np.array([[1048.9128, 0, 938.53824], [0, 1044.4716, 495.39543], [0, 0, 1]]),
               np.array([-0.028647, 0.001474, -0.007251, -0.003957, 0.0])),
}
# "Transformation Matrix Sonar to Camera" (Ts_c) from the config. The authors' merge.py applies it as
# p_camera = R @ p_sonar + t, so it maps sonar-frame points into the camera frame: T_camera_sonar.
# Sonar frame: x forward, y left, z up. Camera frame (OpenCV): x right, y down, z forward.
# The camera looks along the sonar's x axis and sits 15 cm behind the sonar.
T_CAMERA_SONAR = np.array([[0.0, -1.0, 0.0, 0.0],
                           [0.0, 0.0, -1.0, 0.0],
                           [1.0, 0.0, 0.0, 0.15],
                           [0.0, 0.0, 0.0, 1.0]])
# Oculus M750d settings used by the authors: 130 deg horizontal FOV, 20 deg vertical aperture
SONAR_FOV_DEG = {"horizontal": 130.0, "vertical": 20.0}


def calibration_name(scenario: str) -> str:
    """Which camera calibration a scenario folder uses ("tank" or "marina")."""
    return "tank" if scenario.startswith("tank") else "marina"


def _norm(topic: str) -> str:
    return topic.lstrip("/")


# ---------------------------------------------------------------------------
# Bags


def make_typestore(bag: Path):
    """Typestore with the standard ROS1 types plus any custom types embedded in the bag."""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    ts = get_typestore(Stores.ROS1_NOETIC)
    with Reader(bag) as r:
        for c in r.connections:
            if c.msgtype not in ts.types:
                ts.register(get_types_from_msg(c.msgdef.data, c.msgtype))
    return ts


def read_topic(bag: Path, topic: str, typestore=None, limit: int | None = None):
    """Yield (bag_time_ns, deserialized msg) for one topic. `topic` is matched without its leading "/"."""
    from rosbags.rosbag1 import Reader

    typestore = typestore or make_typestore(bag)
    with Reader(bag) as r:
        conns = [c for c in r.connections if _norm(c.topic) == _norm(topic)]
        for n, (conn, t, raw) in enumerate(r.messages(connections=conns)):
            if limit is not None and n >= limit:
                break
            yield t, typestore.deserialize_ros1(raw, conn.msgtype)


def stamp_ns(msg) -> int:
    """Header timestamp (acquisition time) in ns. Use this for syncing, not the bag write time."""
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def inventory(bags: list[Path]) -> pd.DataFrame:
    """One row per bag: scenario, turbidity, duration, message counts and rates, sonar range, ground truth.

    Reads only the bag index and the first ping, so it is fast.
    """
    from rosbags.rosbag1 import Reader

    rows = []
    for bag in bags:
        with Reader(bag) as r:
            counts = {name: sum(c.msgcount for c in r.connections if _norm(c.topic) == _norm(topic))
                      for name, topic in TOPICS.items()}
            duration = r.duration / 1e9
        _, ping = next(read_topic(bag, SONAR_TOPIC))
        turbidity = re.search(r"_(\d)C$", bag.stem)
        rows.append({
            "scenario": bag.parent.name,
            "sequence": bag.stem,
            "turbidity": f"{turbidity.group(1)}C (emulated)" if turbidity else "none",
            "duration [s]": round(duration, 1),
            "camera frames": counts["camera"],
            "sonar pings": counts["sonar"],
            "odom msgs": counts["odom"],
            "camera [Hz]": round(counts["camera"] / duration, 1),
            "sonar [Hz]": round(counts["sonar"] / duration, 1),
            "odom [Hz]": round(counts["odom"] / duration),
            "sonar range [m]": round(ping.num_ranges * ping.range_resolution, 2),
            "ground truth": ", ".join(p.name for p in sorted(bag.parent.glob("*.stl"))),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Decoding


def decode_image(msg) -> np.ndarray:
    """RGB image from a sensor_msgs/CompressedImage."""
    import cv2

    return cv2.cvtColor(cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def decode_ping(ping) -> dict:
    """Polar sonar image and its axes from a sonar_oculus/OculusPing.

    Returns {"polar": (num_ranges, num_beams) uint8, row 0 = closest range,
             "ranges": (num_ranges,) bin ranges [m], "bearings": (num_beams,) beam angles [rad]}.
    """
    import cv2

    polar = cv2.imdecode(np.frombuffer(ping.ping.data, np.uint8), cv2.IMREAD_GRAYSCALE)
    return {
        "polar": polar,
        "ranges": np.arange(ping.num_ranges) * ping.range_resolution,
        "bearings": np.radians(np.asarray(ping.bearings, dtype=np.float64) / 100),  # stored in 1/100 degree
    }


def polar_to_fan(polar: np.ndarray, bearings: np.ndarray, range_resolution: float) -> np.ndarray:
    """Cartesian fan image (sonar at the bottom centre, looking up), one pixel per range bin.

    Same mapping as the authors' `ImagingSonar.generate_map_xy`. Pixels outside the fan are 0.
    """
    import cv2

    rows = polar.shape[0]
    max_range = rows * range_resolution
    cols = int(np.ceil(2 * np.sin((bearings[-1] - bearings[0]) / 2) * max_range / range_resolution))
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    forward = range_resolution * (rows - yy)              # distance ahead of the sonar
    lateral = range_resolution * (xx + 0.5 - cols / 2)     # distance to the side
    bearing = np.arctan2(lateral, forward)
    map_y = (np.hypot(forward, lateral) / range_resolution).astype(np.float32)
    map_x = np.interp(bearing, bearings, np.arange(len(bearings)), left=-1, right=-1).astype(np.float32)
    return cv2.remap(polar, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------
# Sequences and poses


def load_sequence(bag: Path) -> dict:
    """All camera frames of a bag, each paired with the nearest sonar ping and the interpolated odometry pose.

    Sync is by header timestamp. Camera frames outside the odometry time range are dropped. Returns
    {"frames": DataFrame, one row per kept camera frame (t [s], camera/ping index, ping time offset,
               interpolated pose x, y, z, qx, qy, qz, qw),
     "cameras": camera messages, "pings": sonar messages, "odom": DataFrame of the raw odometry}.
    """
    from scipy.spatial.transform import Rotation, Slerp

    ts = make_typestore(bag)
    cameras = [m for _, m in read_topic(bag, CAMERA_TOPIC, ts)]
    pings = [m for _, m in read_topic(bag, SONAR_TOPIC, ts)]
    odom = pd.DataFrame([
        dict(t_ns=stamp_ns(m), x=m.pose.pose.position.x, y=m.pose.pose.position.y, z=m.pose.pose.position.z,
             qx=m.pose.pose.orientation.x, qy=m.pose.pose.orientation.y,
             qz=m.pose.pose.orientation.z, qw=m.pose.pose.orientation.w)
        for _, m in read_topic(bag, ODOM_TOPIC, ts)
    ]).sort_values("t_ns").drop_duplicates("t_ns").reset_index(drop=True)

    t_cam = np.array([stamp_ns(m) for m in cameras])
    t_ping = np.array([stamp_ns(m) for m in pings])
    keep = np.flatnonzero((t_cam >= odom["t_ns"].iloc[0]) & (t_cam <= odom["t_ns"].iloc[-1]))

    # Nearest ping per camera frame
    order = np.argsort(t_ping)
    j = np.clip(np.searchsorted(t_ping[order], t_cam[keep]), 1, len(order) - 1)
    left, right = order[j - 1], order[j]
    ping_idx = np.where(np.abs(t_ping[left] - t_cam[keep]) <= np.abs(t_ping[right] - t_cam[keep]), left, right)

    # Pose at the camera time: linear for position, slerp for rotation
    t_odom = odom["t_ns"].to_numpy()
    t_rel = lambda t: (t - t_odom[0]) / 1e9   # seconds, to keep the interpolation well conditioned
    pos = np.column_stack([np.interp(t_cam[keep], t_odom, odom[c]) for c in "xyz"])
    rot = Slerp(t_rel(t_odom), Rotation.from_quat(odom[["qx", "qy", "qz", "qw"]].to_numpy()))(t_rel(t_cam[keep]))

    frames = pd.DataFrame({
        "t": t_cam[keep] / 1e9,
        "camera_idx": keep,
        "ping_idx": ping_idx,
        "ping_dt_ms": (t_ping[ping_idx] - t_cam[keep]) / 1e6,
        **dict(zip("xyz", pos.T)),
        **dict(zip(["qx", "qy", "qz", "qw"], rot.as_quat().T)),
    })
    return {"frames": frames, "cameras": cameras, "pings": pings, "odom": odom}


def sonar_poses(frames: pd.DataFrame, pitch: bool = True) -> np.ndarray:
    """(N, 4, 4) T_world_sonar from the interpolated odometry.

    The authors treat the odometry pose as the sonar pose. With `pitch=False`, pitch is zeroed as in
    their merge.py (euler_matrix(roll, 0, yaw), static xyz convention).
    """
    from scipy.spatial.transform import Rotation

    rot = Rotation.from_quat(frames[["qx", "qy", "qz", "qw"]].to_numpy())
    if not pitch:
        rpy = rot.as_euler("xyz")
        rpy[:, 1] = 0
        rot = Rotation.from_euler("xyz", rpy)
    T = np.tile(np.eye(4), (len(frames), 1, 1))
    T[:, :3, :3] = rot.as_matrix()
    T[:, :3, 3] = frames[["x", "y", "z"]].to_numpy()
    return T


def camera_poses(frames: pd.DataFrame, pitch: bool = True) -> np.ndarray:
    """(N, 4, 4) T_world_camera (OpenCV camera frame) = T_world_sonar @ inv(T_camera_sonar)."""
    return sonar_poses(frames, pitch) @ np.linalg.inv(T_CAMERA_SONAR)


def sonar_points(ping, threshold: int = 100, min_range: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Bright sonar returns as 3D points in the sonar frame, assuming zero elevation.

    An imaging sonar measures range and bearing but not elevation, so each return is placed in the
    sonar's horizontal plane. Pixels below `threshold` or closer than `min_range` (near-field noise)
    are skipped. Returns ((M, 3) points, (M,) intensities). Positive bearing (right side of the fan)
    maps to negative y, as in the authors' imaging_sonar.py.
    """
    s = decode_ping(ping)
    r_idx, b_idx = np.nonzero(s["polar"] >= threshold)
    r, b = s["ranges"][r_idx], s["bearings"][b_idx]
    near = r >= min_range
    r, b = r[near], b[near]
    points = np.column_stack([r * np.cos(b), -r * np.sin(b), np.zeros_like(r)])
    return points, s["polar"][r_idx[near], b_idx[near]]
