"""Loading and pose utilities for the UXO dataset (https://github.com/dfki-ric/uxo-dataset2024).

Frames are keyed by the ARIS frame index everywhere: sonar frames, GoPro frames, labels,
`gantry.csv` rows, and `aris_frame_meta.csv` rows all share it.

Pose chain, following the dataset authors' `demo/transforms_nb.ipynb`:
    world -> setup/portal_crane   translation = gantry (x, y, z), no rotation
    portal_crane -> setup/ar3     static translation from transforms.yaml; rotation is the
                                  static one composed with Euler 'xyz' (SonarRoll, SonarTilt, SonarPan)
    ar3 -> setup/aris -> setup/sonar, ar3 -> ... -> setup/camera   static, from transforms.yaml
All transforms are 4x4 homogeneous matrices mapping child-frame coordinates into the parent frame
(i.e. T_world_camera is camera-to-world).
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

# Recording folder (recordings/<target_type>/) -> target frame in transforms.yaml.
# Keyed by folder, not label class: the floor recordings reuse the "100lbs_aircraft_bomb" label
# class but sit at target/100lbs_floor, and some recordings have no labels at all.
TARGET_FRAMES = {
    "100lbs_bomb": "target/100lbs",
    "100lbs_bomb_(floor)": "target/100lbs_floor",
    "15cm_mortar": "target/mortar_shell",
    "20lbs_incendiary": "target/incendiary",
    "test_cylinder": "target/cylinder",
}

LABEL_SCALE = 3  # labels are in 640x360 (SD) pixels, GoPro frames are 1920x1080 (FHD)

# ARIS Explorer 3000, 128-beam mode: measured beam centre angles [deg], from the dataset authors'
# scripts/common/aris_definitions.py (SoundMetrics SDK). Column k of an aris_raw frame looks at
# -ARIS_BEAMS_128_DEG[k] (positive = right), as in the authors' aris_frame_to_polar2: the columns run
# right to left, so column 0 is the rightmost beam. Checked against the gantry poses by projecting the
# 100 lbs bomb into the raw frames.
ARIS_BEAMS_128_DEG = np.array([
    -15.0068, -14.7768, -14.5462, -14.3150, -14.0833, -13.8511, -13.6186, -13.3858,
    -13.1528, -12.9196, -12.6861, -12.4523, -12.2182, -11.9838, -11.7491, -11.5141,
    -11.2789, -11.0435, -10.8079, -10.5721, -10.3361, -10.0999,  -9.8635,  -9.6269,
     -9.3902,  -9.1534,  -8.9165,  -8.6795,  -8.4424,  -8.2053,  -7.9682,  -7.7310,
     -7.4938,  -7.2566,  -7.0193,  -6.7820,  -6.5446,  -6.3072,  -6.0698,  -5.8324,
     -5.5949,  -5.3574,  -5.1199,  -4.8823,  -4.6447,  -4.4071,  -4.1695,  -3.9318,
     -3.6941,  -3.4564,  -3.2187,  -2.9809,  -2.7430,  -2.5050,  -2.2669,  -2.0287,
     -1.7904,  -1.5520,  -1.3135,  -1.0749,  -0.8362,  -0.5974,  -0.3585,  -0.1196,
      0.1194,   0.3584,   0.5973,   0.8362,   1.0750,   1.3137,   1.5523,   1.7908,
      2.0292,   2.2675,   2.5057,   2.7438,   2.9818,   3.2197,   3.4575,   3.6952,
      3.9329,   4.1706,   4.4083,   4.6459,   4.8835,   5.1211,   5.3587,   5.5962,
      5.8337,   6.0712,   6.3086,   6.5460,   6.7834,   7.0208,   7.2581,   7.4954,
      7.7326,   7.9698,   8.2070,   8.4441,   8.6812,   8.9183,   9.1553,   9.3922,
      9.6290,   9.8657,  10.1023,  10.3387,  10.5749,  10.8109,  11.0467,  11.2823,
     11.5177,  11.7529,  11.9879,  12.2226,  12.4570,  12.6911,  12.9249,  13.1584,
     13.3916,  13.6246,  13.8574,  14.0899,  14.3221,  14.5538,  14.7850,  15.0156,
])
# ARIS Explorer 3000 apertures (manufacturer spec, 3 MHz / 128 beams)
SONAR_FOV_DEG = {"horizontal": 30.0, "vertical": 14.0}


# ---------------------------------------------------------------------------
# Recordings


def load_recording(rec: Path, polar: bool = True) -> dict:
    """Sonar frames, GoPro frames, labels, gantry rows, and per-frame ARIS metadata of one recording.

    Some recordings lost their GoPro footage (see notes.txt); they get empty `gopro_frames`.
    """
    sonar_dir = rec / "aris_polar" if polar and (rec / "aris_polar").is_dir() else rec / "aris_raw"
    notes = rec / "notes.txt"
    gopro_dir = rec / "gopro"
    labels_dir = rec / "labels"
    return {
        "name": rec.name,
        "target": rec.parent.name,
        "target_frame": TARGET_FRAMES[rec.parent.name],
        "sonar_frames": {int(f.stem): f for f in sorted(sonar_dir.iterdir())},
        "gopro_frames": {int(f.stem): f for f in sorted(gopro_dir.iterdir())} if gopro_dir.is_dir() else {},
        "labels": {int(f.stem): json.loads(f.read_text()) for f in sorted(labels_dir.glob("*.json"))}
        if labels_dir.is_dir() else {},
        "gantry": pd.read_csv(rec / "gantry.csv", index_col="aris_frame_idx"),
        "frame_meta": pd.read_csv(rec / "aris_frame_meta.csv", index_col="FrameIndex"),
        "file_meta": yaml.safe_load(open(rec / "aris_file_meta.yaml")),
        "notes": notes.read_text() if notes.exists() else "",
    }


def inventory(recordings: list[Path]) -> pd.DataFrame:
    """One row per recording: what it shows, how many frames each modality has, and whether it is usable.

    Only counts files and reads a few metadata columns, so it is fast for the whole dataset.
    `usable` is "camera+sonar" (GoPro frames, sonar frames, and poses all present), "sonar only"
    (GoPro footage lost), or "unusable" (no sonar frames or no poses).
    """
    def count(d: Path) -> set[int]:
        return {int(f.stem) for f in d.iterdir()} if d.is_dir() else set()

    rows = []
    for rec in recordings:
        sonar_dir = rec / "aris_polar" if (rec / "aris_polar").is_dir() else rec / "aris_raw"
        sonar, gopro, labels = count(sonar_dir), count(rec / "gopro"), count(rec / "labels")
        gantry = pd.read_csv(rec / "gantry.csv", index_col="aris_frame_idx")
        meta = pd.read_csv(rec / "aris_frame_meta.csv", index_col="FrameIndex", usecols=["FrameIndex", "SonarTilt", "SonarPan"])
        posed = gantry.index.intersection(meta.index)
        notes = (rec / "notes.txt").read_text().splitlines() if (rec / "notes.txt").exists() else []
        trajectory = next((l.split(":", 1)[1].strip() for l in notes if l.lower().startswith("- trajectory")), "")
        pan = np.degrees(np.unwrap(np.radians(meta["SonarPan"].sort_index().to_numpy())))  # flybys sit at +-180

        missing = [name for name, n in (("sonar", sonar), ("gopro", gopro), ("labels", labels), ("poses", posed)) if not len(n)]
        usable = ("unusable" if not sonar or posed.empty else "camera+sonar" if gopro else "sonar only")
        rows.append({
            "target": rec.parent.name,
            "recording": rec.name,
            "trajectory": trajectory,
            "gantry z [m]": round(gantry["z"].median(), 2),
            "tilt [deg]": round(meta["SonarTilt"].median()),
            "pan [deg]": f"{pan.min():.0f} to {pan.max():.0f}",
            "pan span [deg]": round(np.ptp(pan)),  # how far around the target the view sweeps
            "sonar frames": len(sonar),
            "gopro frames": len(gopro),
            "labelled frames": len(labels & gopro),
            "missing": ", ".join(missing),
            "usable": usable,
        })
    return pd.DataFrame(rows)


def sonar_geometry(rec: dict) -> dict:
    """Range and bearing axes of the raw ARIS frames (`aris_raw`, samples x beams) of one recording.

    Returns {"ranges": (num_samples,) range of each row centre [m], "bearings": (128,) bearing of each
    column [rad], positive to the right, i.e. descending: column 0 is the rightmost beam}.
    The range window drifts by under a millimetre within a recording (sound speed), so the median is used.
    """
    meta = rec["frame_meta"]
    assert (meta["PingMode"].isin([9, 10, 11, 12])).all(), "expected the 128-beam ping modes"
    n = int(meta["SamplesPerBeam"].iloc[0])
    assert (meta["SamplesPerBeam"] == n).all()
    start, length = meta["WindowStart"].median(), meta["WindowLength"].median()
    return {
        "ranges": start + (np.arange(n) + 0.5) * length / n,
        "bearings": -np.radians(ARIS_BEAMS_128_DEG),
    }


def label_box(label: dict) -> np.ndarray:
    """Label bounding box as [x_min, y_min, x_max, y_max] in full-HD GoPro pixels."""
    return LABEL_SCALE * np.array([label["x_min"], label["y_min"], label["x_max"], label["y_max"]], float)


# ---------------------------------------------------------------------------
# Calibration


def _pq_to_matrix(translation: dict, rotation: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R.from_quat([rotation[k] for k in "xyzw"]).as_matrix()
    T[:3, 3] = [translation[k] for k in "xyz"]
    return T


def load_transforms(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """Static transforms from calibration/transforms.yaml as {frame_id: (parent_frame_id, T_parent_frame)}."""
    data = yaml.safe_load(open(path))
    return {
        tf["frame_id"]: (tf["parent_frame_id"], _pq_to_matrix(tf["translation"], tf["rotation"]))
        for group in ("setup", "targets") for tf in data[group]
    }


def chain(tfs: dict, frame: str, root: str) -> np.ndarray:
    """T_root_frame by walking parent links up from `frame` until `root`."""
    T = np.eye(4)
    while frame != root:
        parent, T_parent_frame = tfs[frame]
        T = T_parent_frame @ T
        frame = parent
    return T


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(K, dist) from one of the calibration/camera_parameters/*.txt files (numpy-printed arrays)."""
    text = path.read_text()
    arrays = [np.array([float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", block)])
              for block in re.findall(r"=\s*(\[.*?\]\])", text, flags=re.S)]
    return arrays[0].reshape(3, 3), arrays[1]


# ---------------------------------------------------------------------------
# Poses


def rig_poses(rec: dict, tfs: dict, frames: tuple[str, ...] = ("setup/camera", "setup/sonar")) -> dict[str, np.ndarray]:
    """Per-frame world poses of the given rig frames.

    Returns {"frame_idx": (N,), "<frame>": (N, 4, 4) T_world_frame, ...}, one row per gantry row
    that also has ARIS metadata.
    """
    idx = rec["gantry"].index.intersection(rec["frame_meta"].index)
    xyz = rec["gantry"].loc[idx, ["x", "y", "z"]].to_numpy()
    rpy = rec["frame_meta"].loc[idx, ["SonarRoll", "SonarTilt", "SonarPan"]].to_numpy()

    # world -> portal_crane is a pure translation, so ar3's static offset and rotation are fixed
    parent, T_crane_ar3 = tfs["setup/ar3"]
    assert parent == "setup/portal_crane"
    T_world_ar3 = np.tile(np.eye(4), (len(idx), 1, 1))
    T_world_ar3[:, :3, 3] = xyz + T_crane_ar3[:3, 3]
    T_world_ar3[:, :3, :3] = (R.from_matrix(T_crane_ar3[:3, :3]) * R.from_euler("xyz", rpy, degrees=True)).as_matrix()

    out = {"frame_idx": idx.to_numpy(), "setup/ar3": T_world_ar3}
    for f in frames:
        out[f] = T_world_ar3 @ chain(tfs, f, "setup/ar3")
    return out


def project(points_world: np.ndarray, T_world_cam: np.ndarray, K: np.ndarray,
            dist: np.ndarray | None = None, fisheye: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Project (M, 3) world points with a camera-to-world pose (OpenCV camera frame: z forward, y down).

    Returns (M, 2) pixel coordinates and (M,) depths along the optical axis.
    """
    import cv2

    T_cam_world = np.linalg.inv(T_world_cam)
    p_cam = points_world @ T_cam_world[:3, :3].T + T_cam_world[:3, 3]
    depth = p_cam[:, 2]
    if dist is None:
        uv = p_cam[:, :2] / depth[:, None] @ K[:2, :2].T + K[:2, 2]
    elif fisheye:
        uv = cv2.fisheye.projectPoints(p_cam[None].astype(np.float64), np.zeros(3), np.zeros(3), K, dist.reshape(4, 1))[0][0]
    else:
        uv = cv2.projectPoints(p_cam.astype(np.float64), np.zeros(3), np.zeros(3), K, dist)[0][:, 0]
    return uv, depth


def view_geometry(rec: dict, tfs: dict, K: np.ndarray, dist: np.ndarray,
                  image_size: tuple[int, int] = (1920, 1080)) -> pd.DataFrame:
    """Where the camera is relative to the target, per frame.

    Camera position in the target frame (z up) as azimuth (direction around the target, 0-360 deg),
    elevation (angle above the target), and distance [m], plus whether the target origin projects
    in front of the camera and inside the image.
    """
    import cv2

    poses = rig_poses(rec, tfs, frames=("setup/camera",))
    T_world_cam = poses["setup/camera"]
    T_world_target = chain(tfs, rec["target_frame"], "world")

    # Camera position in the target frame
    p = (np.linalg.inv(T_world_target) @ T_world_cam[:, :, 3].T).T[:, :3]
    distance = np.linalg.norm(p, axis=1)

    # Target origin in each camera frame, projected in one call
    target = T_world_target[:3, 3]
    T_cam_world = np.linalg.inv(T_world_cam)
    p_cam = np.einsum("nij,j->ni", T_cam_world[:, :3, :3], target) + T_cam_world[:, :3, 3]
    uv = cv2.fisheye.projectPoints(p_cam[None].astype(np.float64), np.zeros(3), np.zeros(3), K, dist.reshape(4, 1))[0][0]
    w, h = image_size
    in_view = (p_cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)

    return pd.DataFrame({
        "frame_idx": poses["frame_idx"],
        "azimuth": np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360,
        "elevation": np.degrees(np.arcsin(p[:, 2] / distance)),
        "distance": distance,
        "in_view": in_view,
    })


# ---------------------------------------------------------------------------
# 3D models


def load_model(path: Path, max_faces: int | None = 40_000):
    """Return (vertices, faces, per-vertex RGB) for a textured OBJ, optionally decimated for display.

    Two .mtl files reference textures by old names that aren't in the archive, so the texture is
    always read from the .jpg next to the .obj.
    """
    import trimesh
    from PIL import Image
    from scipy.spatial import cKDTree

    mesh = trimesh.load(path, process=False)
    mesh.visual.material.image = Image.open(path.with_suffix(".jpg"))
    colors = mesh.visual.to_color().vertex_colors[:, :3]
    verts, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if max_faces is not None and len(faces) > max_faces:
        small = mesh.simplify_quadric_decimation(face_count=max_faces)
        _, nearest = cKDTree(verts).query(small.vertices)
        verts, faces, colors = np.asarray(small.vertices), np.asarray(small.faces), colors[nearest]
    return verts, faces, colors
