"""Check candidate sonar poses of a UXO scene against an independent 3D model of the same scene.

Z-Splat pulls Gaussians towards where a sonar pose says the returns are, so a wrong pose silently damages
the fusion. COLMAP gives 3D points computed from the camera images alone, so it can test the sonar poses:
project the points into each sonar frame and ask whether they land on bright returns.

The score of a frame is the mean percentile rank of the pixels the points land on, within that sonar image:
0.5 = no better than a random pixel, 1.0 = always the brightest pixel. Ranks instead of raw intensity make
frames comparable, and uint8 ties (most pixels are dark) share their middle rank.

Repeating the score with the points shifted in range and bearing separates three cases:
    peak at zero shift                 the pose is right, as far as range and bearing can tell
    peak at a shift, same in all frames  a fixed calibration offset, and the peak measures it
    no clear peak                      inconsistent poses, or the points do not explain the returns

Candidates. COLMAP has no scale or frame of its own, so a similarity transform fitted on the camera positions
(COLMAP -> gantry world) brings points and cameras to metres:
    gantry   the exported sonar poses, from the gantry and the transforms.yaml chain
    colmap   the aligned COLMAP camera pose + the camera-to-sonar transform of the transforms.yaml chain
Each is run with one alignment for all passes and with one per pass, because the passes differ in scale by 5%.

Limits. The sonar does not measure elevation, so an elevation error only shows indirectly, through the
points that enter or leave the 14 deg aperture. Occlusion is ignored. Sparse COLMAP points lie on textured
surfaces, which are not necessarily the strongest sonar reflectors.

Usage: python check_sonar_poses.py [sonar_scene] [colmap_sparse_dir]
Writes example frames to renders/sonar_pose_check/.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import cv2
import numpy as np

OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])  # same as export.py
RANGE_SHIFTS = np.arange(-100, 101, 5)              # [range bins], 3 mm each: +- 30 cm
BEARING_SHIFTS = np.arange(-40, 41, 2)              # [beams], about 0.23 deg each: +- 9 deg
MAX_POINTS_PER_FRAME = 2000
FRAME_STEP = 3


def read_colmap(sparse: Path) -> tuple[dict, np.ndarray]:
    """COLMAP binary model as ({image name: camera-to-world, OpenCV axes}, (N, 3) points, (N, 3) uint8 rgb)."""
    cameras = {}
    with open(sparse / "images.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            _, qw, qx, qy, qz, tx, ty, tz, _ = struct.unpack("<i7di", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\0":
                name += c
            f.seek(24 * struct.unpack("<Q", f.read(8))[0], 1)
            R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                          [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
                          [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = R.T, -R.T @ np.array([tx, ty, tz])
            cameras[name.decode()] = c2w
    points, colors = [], []
    with open(sparse / "points3D.bin", "rb") as f:
        for _ in range(struct.unpack("<Q", f.read(8))[0]):
            data = struct.unpack("<Q3d3BdQ", f.read(51))
            points.append(data[1:4])
            colors.append(data[4:7])
            f.seek(8 * data[-1], 1)
    return cameras, np.array(points), np.array(colors, np.uint8)


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 4x4 similarity with dst = s R src + t (Umeyama), from corresponding (N, 3) positions."""
    ms, md = src.mean(0), dst.mean(0)
    U, S, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / ((src - ms) ** 2).sum()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, md - s * R @ ms
    return T


def apply_similarity(S: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    """Pose in the target frame of S: rotation without the scale, position scaled, so still a rigid pose."""
    out = S @ c2w
    out[:3, :3] /= np.cbrt(np.linalg.det(S[:3, :3]))
    return out


def rank_image(polar: np.ndarray) -> np.ndarray:
    """Percentile rank (0..1) of every pixel of a uint8 image; equal values share their middle rank."""
    cdf = np.cumsum(np.bincount(polar.ravel(), minlength=256)) / polar.size
    return ((cdf + np.concatenate([[0.0], cdf[:-1]])) / 2)[polar]


def project(points: np.ndarray, s2w: np.ndarray, sonar: dict) -> tuple[np.ndarray, np.ndarray]:
    """Row and column indices in the polar image of the points inside the sonar's field of view."""
    p = (points - s2w[:3, 3]) @ s2w[:3, :3]  # world -> sonar frame (x right, y down, z forward)
    r = np.linalg.norm(p, axis=1)
    bearing, elevation = np.arctan2(p[:, 0], p[:, 2]), np.arctan2(-p[:, 1], np.hypot(p[:, 0], p[:, 2]))
    ranges, bearings = sonar["ranges"], sonar["bearings"]
    keep = ((r >= ranges[0]) & (r <= ranges[-1]) & (bearing >= bearings[0]) & (bearing <= bearings[-1])
            & (np.abs(elevation) <= sonar["vertical_fov"] / 2))
    rows = np.interp(r[keep], ranges, np.arange(len(ranges)))
    cols = np.interp(bearing[keep], bearings, np.arange(len(bearings)))
    return np.round(rows).astype(int), np.round(cols).astype(int)


def shift_scores(rank: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(range shifts, bearing shifts) mean rank of the points after shifting them; NaN if none stay inside."""
    out = np.full((len(RANGE_SHIFTS), len(BEARING_SHIFTS)), np.nan)
    for i, dr in enumerate(RANGE_SHIFTS):
        for j, db in enumerate(BEARING_SHIFTS):
            a, b = rows + dr, cols + db
            ok = (a >= 0) & (a < rank.shape[0]) & (b >= 0) & (b < rank.shape[1])
            if ok.any():
                out[i, j] = rank[a[ok], b[ok]].mean()
    return out


def recording_of_frames(frames: list) -> np.ndarray:
    """Recording (pass) number of every frame: source_index restarts at each new recording."""
    index = np.array([f["source_index"] for f in frames])
    return np.concatenate([[0], np.cumsum(np.diff(index) < 0)])


def candidate_poses(frames: list, cameras: dict, passes: np.ndarray, per_pass: bool) -> dict:
    """For every frame its aligned points transform and both candidate sonar poses (see module docstring)."""
    cam_gantry = np.array([np.array(f["transform_matrix"]) @ OPENCV_TO_OPENGL for f in frames])
    son_gantry = np.array([np.array(f["sonar_transform_matrix"]) for f in frames])
    cam_colmap = np.array([cameras[Path(f["file_path"]).name] for f in frames])
    cam_to_sonar = np.linalg.inv(cam_gantry) @ son_gantry
    print(f"camera-to-sonar transform across frames: std of translation {cam_to_sonar[:, :3, 3].std(0).max() * 1000:.2f} mm, "
          f"of rotation entries {cam_to_sonar[:, :3, :3].std(0).max():.5f}")
    if per_pass:
        fits = {k: fit_similarity(cam_colmap[passes == k, :3, 3], cam_gantry[passes == k, :3, 3]) for k in np.unique(passes)}
    else:
        fits = {k: fit_similarity(cam_colmap[:, :3, 3], cam_gantry[:, :3, 3]) for k in np.unique(passes)}
    print("  scale per pass:", " ".join(f"{np.cbrt(np.linalg.det(S[:3, :3])):.4f}" for S in fits.values()))
    aligned = np.array([apply_similarity(fits[k], c) for k, c in zip(passes, cam_colmap)])
    return {"fits": fits, "gantry": son_gantry, "colmap": aligned @ cam_to_sonar[np.arange(len(frames))]}


def overlay(polar: np.ndarray, rows: np.ndarray, cols: np.ndarray, title: str) -> np.ndarray:
    """Polar image (closest range at the top, leftmost beam on the left) with the points drawn on it."""
    img = cv2.cvtColor(cv2.resize(polar, (512, 882), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    for r, c in zip(rows, cols):
        cv2.circle(img, (int((c + 0.5) * 4), int((r + 0.5) * 2)), 2, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.putText(img, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 255, 80), 2, cv2.LINE_AA)
    return img


def main(scene: Path, sparse: Path, out: Path):
    meta = json.load(open(scene / "transforms.json"))
    frames = meta["frames"]
    sonar = {"ranges": np.array(meta["sonar"]["ranges"]), "bearings": np.array(meta["sonar"]["bearings"]),
             "vertical_fov": meta["sonar"]["vertical_fov"]}
    cameras, points, _ = read_colmap(sparse)
    passes = recording_of_frames(frames)
    rng = np.random.default_rng(0)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(points)} COLMAP points, {len(frames)} frames, passes {np.bincount(passes)}, every {FRAME_STEP}rd frame scored")

    examples = [int(np.nonzero(passes == k)[0][np.sum(passes == k) // 2]) for k in np.unique(passes)]
    panels = {i: [] for i in examples}
    for per_pass in (False, True):
        print(f"\n=== {'one alignment per pass' if per_pass else 'one alignment for all passes'}")
        poses = candidate_poses(frames, cameras, passes, per_pass)
        world = {k: (S[:3, :3] @ points.T).T + S[:3, 3] for k, S in poses["fits"].items()}
        for name in ("gantry", "colmap"):
            maps, counts = {k: [] for k in np.unique(passes)}, []
            for i in range(0, len(frames), FRAME_STEP):
                k = passes[i]
                rows, cols = project(world[k], poses[name][i], sonar)
                counts.append(len(rows))
                if len(rows) == 0:
                    continue
                if len(rows) > MAX_POINTS_PER_FRAME:
                    keep = rng.choice(len(rows), MAX_POINTS_PER_FRAME, replace=False)
                    rows, cols = rows[keep], cols[keep]
                maps[k].append(shift_scores(rank_image(np.load(scene / frames[i]["sonar_file_path"])), rows, cols))
            print(f"--- {name}: median {np.median(counts):.0f} points in the field of view per frame")
            zero = (np.argmin(np.abs(RANGE_SHIFTS)), np.argmin(np.abs(BEARING_SHIFTS)))
            per_frame_peaks = []
            for k, m in maps.items():
                mean = np.nanmean(m, axis=0)
                pi, pj = np.unravel_index(np.nanargmax(mean), mean.shape)
                bin_m, beam_deg = np.median(np.diff(sonar["ranges"])), np.degrees(np.median(np.diff(sonar["bearings"])))
                print(f"  pass {k}: score at zero shift {mean[zero]:.3f} | best {mean[pi, pj]:.3f} at "
                      f"{RANGE_SHIFTS[pi] * bin_m * 100:+.1f} cm, {BEARING_SHIFTS[pj] * beam_deg:+.1f} deg | "
                      f"median over all shifts {np.nanmedian(mean):.3f}")
                peaks = [np.unravel_index(np.nanargmax(f), f.shape) for f in m]
                per_frame_peaks += [(RANGE_SHIFTS[a] * bin_m * 100, BEARING_SHIFTS[b] * beam_deg) for a, b in peaks]
            q = np.percentile(per_frame_peaks, [25, 50, 75], axis=0)
            print(f"  best shift of single frames, quartiles: range {q[0, 0]:+.1f} / {q[1, 0]:+.1f} / {q[2, 0]:+.1f} cm, "
                  f"bearing {q[0, 1]:+.1f} / {q[1, 1]:+.1f} / {q[2, 1]:+.1f} deg")
            if per_pass:
                continue
            for i in examples:
                rows, cols = project(world[passes[i]], poses[name][i], sonar)
                polar = np.load(scene / frames[i]["sonar_file_path"])
                if not panels[i]:
                    panels[i].append(overlay(polar, [], [], "sonar image"))
                panels[i].append(overlay(polar, rows, cols, name))
    for i, row in panels.items():
        cv2.imwrite(str(out / f"frame_{i:06d}_pass{passes[i]}.jpg"), np.hstack(row))
    print(f"\nExample frames: {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0] if args else "prepared/uxo/incendiary_3heights"),
         Path(args[1] if len(args) > 1 else "prepared/uxo/incendiary_3heights_colmap/colmap/sparse/0"),
         Path("renders/sonar_pose_check"))
