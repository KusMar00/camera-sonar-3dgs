"""Sonar-MASt3R dataset (Phung & Camilli, ICRA 2026, arXiv 2603.13585; https://sonar-mast3r.github.io/): sonar
geometry and fan -> polar resampling.

The authors' ROS bags record the sonar only as the viewer's rendered fan image (/oculus_wrist/oculus_viewer/image,
722x398 bgr8), not as raw polar pings, so the polar (range x bearing) image that export.py and Z-Splat expect has to
be resampled from the fan. Sonar: Oculus M3000d at 1.2 MHz (paper: "MT3000d"), 512 beams 0.25 deg apart, 0.6 deg
angular resolution, 130 x 20 deg aperture, max range set to 2 m (paper Sec. IV, Blueprint Subsea spec sheet).

Fan geometry, measured on the dataset1 frames (not documented by the authors):
    apex at the bottom centre of the image, (360.4, 397.8) px from a line fit of both fan edges;
    edges at +-65.4 deg from vertical (130.8 deg; the extra 0.8 deg is edge anti-aliasing);
    outer arc at 397.4-397.6 px from the apex at every bearing, i.e. the 2 m range spans the image height.
Assumed: the viewer draws each beam at its true bearing (image angle = bearing). fan_to_polar keeps the image's
left-right order (column 0 = image left); relative to the recorded oculus_link pose the bearings turned out to be
mirrored, which export.export_sonar_mast3r corrects (evidence there).

Intensity: the viewer tints a grey value (R ~= G, B boosted), so (R + G) / 2 recovers the grey value. That is
displayed brightness after the viewer's gain and display curve, monotonic in echo strength but not proportional.
"""

from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

FAN_SIZE = (722, 398)                  # (width, height) of the viewer image
FAN_APEX = (360.5, 397.5)              # (x, y) px, pixel-centre coordinates: bottom edge, centred
FAN_RADIUS_PX = 398.0                  # apex to outer arc (top edge of the image) = MAX_RANGE
MAX_RANGE = 2.0                        # [m]
HORIZONTAL_FOV = np.radians(130.0)
VERTICAL_FOV = np.radians(20.0)
# 398 range bins = one per fan pixel along the radius (5.03 mm, the rendering's resolution; native is 2.5 mm).
# 256 beams 0.51 deg apart: close to the 0.6 deg beam width. The fan resolves that spacing from 0.57 m outward;
# 512 beams would only interpolate neighbours below ~1.1 m, where the rendering has merged them.
N_RANGES, N_BEAMS = 398, 256


def polar_axes(n_ranges: int = N_RANGES, n_beams: int = N_BEAMS) -> tuple[np.ndarray, np.ndarray]:
    """Bin centres: ranges [m] (row 0 = closest) and bearings [rad] (column 0 = leftmost, positive to the right)."""
    ranges = (np.arange(n_ranges) + 0.5) * MAX_RANGE / n_ranges
    bearings = (np.arange(n_beams) + 0.5) * HORIZONTAL_FOV / n_beams - HORIZONTAL_FOV / 2
    return ranges, bearings


def _fan_pixel(r: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fan image pixel (x, y) of range r [m] and bearing b [rad]: bearing measured from vertical, right positive."""
    rho = r * FAN_RADIUS_PX / MAX_RANGE
    return FAN_APEX[0] + rho * np.sin(b), FAN_APEX[1] - rho * np.cos(b)


def fan_gray(fan_bgr: np.ndarray) -> np.ndarray:
    """Grey value of the tinted viewer image, float32: (R + G) / 2."""
    assert fan_bgr.shape[1::-1] == FAN_SIZE, f"expected a {FAN_SIZE} viewer image, got {fan_bgr.shape}"
    return (fan_bgr[..., 2].astype(np.float32) + fan_bgr[..., 1]) / 2


def fan_to_polar(fan_bgr: np.ndarray, n_ranges: int = N_RANGES, n_beams: int = N_BEAMS) -> np.ndarray:
    """Polar sonar image uint8 (n_ranges, n_beams) from a viewer fan image (BGR, as cv2 reads it), bilinear."""
    ranges, bearings = polar_axes(n_ranges, n_beams)
    x, y = _fan_pixel(ranges[:, None], bearings[None, :])
    polar = cv2.remap(fan_gray(fan_bgr), x.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.clip(np.rint(polar), 0, 255).astype(np.uint8)


def polar_to_fan(polar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Render a polar image back into the viewer's fan layout (float32 grey, and the mask of pixels inside the fan).
    Used to check the round trip fan -> polar -> fan."""
    ranges, bearings = polar_axes(*polar.shape)
    W, H = FAN_SIZE
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    r = np.hypot(xx - FAN_APEX[0], FAN_APEX[1] - yy) * MAX_RANGE / FAN_RADIUS_PX
    b = np.arctan2(xx - FAN_APEX[0], FAN_APEX[1] - yy)
    inside = (r <= MAX_RANGE) & (np.abs(b) <= HORIZONTAL_FOV / 2)
    rows = np.interp(r, ranges, np.arange(len(ranges)))       # clamps at the outermost bin centres
    cols = np.interp(b, bearings, np.arange(len(bearings)))
    fan = cv2.remap(polar.astype(np.float32), cols.astype(np.float32), rows.astype(np.float32), cv2.INTER_LINEAR)
    return np.where(inside, fan, 0).astype(np.float32), inside


# ---------------------------------------------------------------------------
# Sequences (as exported from the bag by prepare_sonar_mast3r.ipynb)


def load_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(K, plumb_bob distortion, (w, h)) of the raw camera from the authors' camera.yaml (ROS camera_calibration)."""
    c = yaml.safe_load(Path(path).read_text())
    assert c["distortion_model"] == "plumb_bob"
    return (np.array(c["camera_matrix"]["data"]).reshape(3, 3), np.array(c["distortion_coefficients"]["data"]),
            (c["image_width"], c["image_height"]))


def pose_matrices(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """(N, 4, 4) sensor-to-world matrices from positions (N, 3) and xyzw quaternions (N, 4)."""
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:, :3, 3] = pos
    return T


def load_sequence(seq: Path, max_dt: float = 0.05) -> dict:
    """Camera frames paired with their nearest sonar frame, from a bag export (cam_frames/, sonar_frames/, poses.npz).

    Returns "t" [s since the first camera frame], "cam_idx", "sonar_idx", "dt" [s, sonar minus camera time],
    "T_world_dwe_cam" and "T_world_oculus" (4x4 forward-kinematics poses in the arm base frame, as recorded:
    dwe_cam with the camera looking along +z, oculus_link with the boresight along +x).
    """
    p = np.load(Path(seq) / "poses.npz")
    cam_t, son_t = p["cam_stamp_ns"] / 1e9, p["sonar_stamp_ns"] / 1e9
    k = np.clip(np.searchsorted(son_t, cam_t), 1, len(son_t) - 1)
    near = np.where(np.abs(son_t[k] - cam_t) < np.abs(son_t[k - 1] - cam_t), k, k - 1)
    dt = son_t[near] - cam_t
    assert np.abs(dt).max() <= max_dt, f"camera frame without a sonar frame within {max_dt} s"
    return {"t": cam_t - cam_t[0], "cam_idx": np.arange(len(cam_t)), "sonar_idx": near, "dt": dt,
            "T_world_dwe_cam": pose_matrices(p["cam_pos"], p["cam_quat"]),
            "T_world_oculus": pose_matrices(p["sonar_pos"], p["sonar_quat"])[near]}
