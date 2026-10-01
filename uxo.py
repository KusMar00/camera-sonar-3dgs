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
