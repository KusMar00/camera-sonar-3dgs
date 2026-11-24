"""Export camera + sonar sequences of both datasets to one common scene format.

Layout of one scene (prepared/<dataset>/<sequence>/):
    images/000000.jpg     undistorted pinhole images with the principal point in the image centre
                          (3DGS rasterizers ignore cx, cy)
    sonar/000000.npy      polar sonar intensity as recorded, uint8 (num_ranges, num_beams):
                          row 0 = closest range, column 0 = leftmost beam
    points3d.ply          initial point cloud from bright sonar returns (3DGS layout: xyz, normals, rgb)
    transforms.json       intrinsics, sonar geometry, per-frame poses

transforms.json is a nerfstudio transforms file for the camera, so nerfstudio and gsplat read the images
as-is. Per frame:
    "transform_matrix"        camera-to-world, OpenGL camera axes (x right, y up, z backward), as nerfstudio
    "sonar_file_path"
    "sonar_transform_matrix"  sonar-to-world, sonar axes x right, y down, z forward (boresight). A point
                              (x, y, z) in the sonar frame has bearing atan2(x, z), positive to the right,
                              and elevation atan2(-y, hypot(x, z)), positive up.
Top level "sonar": "ranges" [m] of every row and "bearings" [rad] of every column (both ascending), and
the horizontal and vertical aperture [rad].

World frame: each dataset's own (UXO: gantry world, z up; opti-acoustic: SLAM odometry frame; Sonar-MASt3R:
arm base frame, z up). Metres,
no rescaling or recentring.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import opti
import sonar_mast3r
import uxo

OPENCV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])  # flips camera y and z axes (right-multiply a c2w pose)
# Opti-acoustic sonar frame (x forward, y left, z up) -> common sonar frame (x right, y down, z forward):
# the same axis swap as the camera, so the rotation of T_CAMERA_SONAR
T_FLU_RDF = np.linalg.inv(np.block([[opti.T_CAMERA_SONAR[:3, :3], np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]))


# ---------------------------------------------------------------------------
# Building blocks


def undistorter(K: np.ndarray, dist: np.ndarray, size: tuple[int, int], fisheye: bool, scale: float = 1.0):
    """Function that undistorts an image to a pinhole camera with the principal point in the centre.

    The output keeps the aspect ratio, is `scale` times the input size, and gets the widest field of view
    that has no pixels outside the original image (no black borders for 3DGS to reconstruct). Returns
    (undistort(image) -> image, K_new, (w, h)).
    """
    # Image border in normalized (undistorted) coordinates. The largest centred rectangle with the image
    # aspect ratio inside it sets the focal length: its half-width c * w/2 maps to w/2 pixels, so f = 1 / c.
    t = np.linspace(0, 1, 200)[:, None]
    W, H = size
    border = np.vstack([np.hstack([t * W, 0 * t]), np.hstack([t * W, 0 * t + H]),
                        np.hstack([0 * t, t * H]), np.hstack([0 * t + W, t * H])])[:, None].astype(np.float64)
    if fisheye:
        xy = cv2.fisheye.undistortPoints(border, K, dist.reshape(4, 1))[:, 0]
    else:
        xy = cv2.undistortPoints(border, K, dist)[:, 0]
    c = np.min(np.maximum(np.abs(xy[:, 0]) / (W / 2), np.abs(xy[:, 1]) / (H / 2)))
    w, h = round(W * scale), round(H * scale)
    f = scale / c
    K_new = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    if fisheye:
        maps = cv2.fisheye.initUndistortRectifyMap(K, dist.reshape(4, 1), np.eye(3), K_new, (w, h), cv2.CV_16SC2)
    else:
        maps = cv2.initUndistortRectifyMap(K, dist, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
    return (lambda img: cv2.remap(img, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)), K_new, (w, h)


def sonar_returns(polar: np.ndarray, ranges: np.ndarray, bearings: np.ndarray, vertical_fov: float,
                  rng: np.random.Generator, percentile: float = 99.5, min_range: float = 0.3,
                  max_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Up to `max_points` of the brightest returns of one ping as 3D points in the sonar frame.

    Elevation is not measured, so each point gets a random one within the vertical aperture. Returns
    ((M, 3) points, (M,) intensities).
    """
    bright = (polar >= max(np.percentile(polar, percentile), 1)) & (ranges[:, None] >= min_range)
    r_idx, b_idx = np.nonzero(bright)
    if len(r_idx) > max_points:
        keep = rng.choice(len(r_idx), max_points, replace=False)
        r_idx, b_idx = r_idx[keep], b_idx[keep]
    r, b = ranges[r_idx], bearings[b_idx]
    e = rng.uniform(-vertical_fov / 2, vertical_fov / 2, len(r))
    points = r[:, None] * np.column_stack([np.cos(e) * np.sin(b), -np.sin(e), np.cos(e) * np.cos(b)])
    return points, polar[r_idx, b_idx]


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    """Binary PLY with the vertex layout 3DGS expects (x, y, z, nx, ny, nz, red, green, blue)."""
    dtype = [(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")] + [(k, "u1") for k in ("red", "green", "blue")]
    v = np.zeros(len(xyz), dtype)
    for k, col in zip("xyz", xyz.T):
        v[k] = col
    for k, col in zip(("red", "green", "blue"), rgb.T):
        v[k] = col
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
              + "".join(f"property {'float' if t == '<f4' else 'uchar'} {k}\n" for k, t in dtype)
              + "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(v.tobytes())


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(xyz, rgb) of a PLY written by `write_ply`."""
    data = Path(path).read_bytes()
    n = int(data.split(b"element vertex ")[1].split(b"\n")[0])
    dtype = [(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")] + [(k, "u1") for k in ("red", "green", "blue")]
    v = np.frombuffer(data, dtype, count=n, offset=data.index(b"end_header\n") + len(b"end_header\n"))
    return np.column_stack([v["x"], v["y"], v["z"]]), np.column_stack([v["red"], v["green"], v["blue"]])


def write_scene(out: Path, frames, *, K: np.ndarray, dist: np.ndarray, fisheye: bool, image_size: tuple[int, int],
                sonar: dict, meta: dict, image_scale: float = 1.0, jpeg_quality: int = 95,
                max_init_points: int = 100_000, seed: int = 0) -> dict:
    """Write one scene in the common format (see the module docstring).

    `frames` yields dicts (read lazily, one image in memory at a time) with "image" (distorted RGB uint8),
    "sonar" (polar uint8, columns in the order of sonar["bearings"]), "T_world_camera" (4x4, OpenCV camera
    axes), "T_world_sonar" (4x4, sonar axes x right, y down, z forward), "time" [s] and "source_index".
    `sonar` has "ranges", "bearings" (ascending or descending), "horizontal_fov", "vertical_fov" [rad] and
    "model". An existing scene in `out` is replaced. Returns the transforms.json content.
    """
    out = Path(out)
    for d in ("images", "sonar"):  # no stale frames from an earlier export with another step
        shutil.rmtree(out / d, ignore_errors=True)
        (out / d).mkdir(parents=True)
    undistort, K_new, (w, h) = undistorter(K, dist, image_size, fisheye, image_scale)
    ranges, bearings = np.asarray(sonar["ranges"]), np.asarray(sonar["bearings"])
    flip = bearings[0] > bearings[-1]                      # store columns left to right
    bearings = bearings[::-1] if flip else bearings
    assert (np.diff(ranges) > 0).all() and (np.diff(bearings) > 0).all()

    rng = np.random.default_rng(seed)
    records, points, intensity = [], [], []
    for n, fr in enumerate(frames):
        name = f"{n:06d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), cv2.cvtColor(undistort(fr["image"]), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        polar = fr["sonar"][:, ::-1] if flip else fr["sonar"]
        np.save(out / "sonar" / f"{name}.npy", np.ascontiguousarray(polar, dtype=np.uint8))

        T_ws = fr["T_world_sonar"]
        p, i = sonar_returns(polar, ranges, bearings, sonar["vertical_fov"], rng)
        points.append(p @ T_ws[:3, :3].T + T_ws[:3, 3])
        intensity.append(i)
        records.append({
            "file_path": f"images/{name}.jpg",
            "transform_matrix": (fr["T_world_camera"] @ OPENCV_TO_OPENGL).tolist(),
            "sonar_file_path": f"sonar/{name}.npy",
            "sonar_transform_matrix": T_ws.tolist(),
            "time": float(fr["time"]),
            "source_index": int(fr["source_index"]),
        })

    xyz, val = np.vstack(points), np.concatenate(intensity)
    if len(xyz) > max_init_points:
        keep = rng.choice(len(xyz), max_init_points, replace=False)
        xyz, val = xyz[keep], val[keep]
    write_ply(out / "points3d.ply", xyz, np.repeat(val[:, None], 3, axis=1))
    transforms = {
        **meta,
        "camera_model": "PINHOLE",
        "fl_x": K_new[0, 0], "fl_y": K_new[1, 1], "cx": K_new[0, 2], "cy": K_new[1, 2], "w": w, "h": h,
        "ply_file_path": "points3d.ply",
        "sonar": {
            "model": sonar["model"],
            "horizontal_fov": float(sonar["horizontal_fov"]),
            "vertical_fov": float(sonar["vertical_fov"]),
            "ranges": ranges.tolist(),
            "bearings": bearings.tolist(),
        },
        "frames": records,
    }
    (out / "transforms.json").write_text(json.dumps(transforms, indent=1))
    return transforms


# ---------------------------------------------------------------------------
# Datasets


def export_uxo(rec_path: Path | list[Path], tfs: dict, K: np.ndarray, dist: np.ndarray, out_root: Path,
               step: int = 1, name: str | None = None, init_half_size: float = 1.3, **kwargs) -> dict:
    """Export one UXO recording: every `step`-th frame that has a GoPro image, a raw sonar frame and a pose.

    K, dist are the `wide` fisheye intrinsics. Sonar frames are the raw ARIS samples (`aris_raw`), not the
    fan images in `aris_polar`. A list of recordings of the same target (e.g. passes at several heights, all
    in the gantry world) is merged into one scene called `name`; their sonar range windows must match.
    Frames keep the recordings' order, listed in the top-level "recordings".
    "init_bounds" (random start points in the Z-Splat reader, no sensor prior): a cube of +-init_half_size [m]
    around the target origin, as the earlier UXO scenes (prepare_uxo.ipynb step 6).
    """
    paths = [Path(p) for p in rec_path] if isinstance(rec_path, (list, tuple)) else [Path(rec_path)]
    recs = [uxo.load_recording(p, polar=False) for p in paths]
    geometry = uxo.sonar_geometry(recs[0])
    for rec in recs[1:]:
        assert rec["target"] == recs[0]["target"], "recordings of different targets"
        ranges = uxo.sonar_geometry(rec)["ranges"]
        assert len(ranges) == len(geometry["ranges"]) and np.allclose(ranges, geometry["ranges"], atol=0.005), \
            f"{rec['name']}: different sonar range window"

    def frames():
        for rec in recs:
            poses = uxo.rig_poses(rec, tfs)
            rows = [k for k, i in enumerate(poses["frame_idx"]) if i in rec["gopro_frames"] and i in rec["sonar_frames"]][::step]
            for k in rows:
                i = poses["frame_idx"][k]
                yield {
                    "image": cv2.cvtColor(cv2.imread(str(rec["gopro_frames"][i])), cv2.COLOR_BGR2RGB),
                    "sonar": cv2.imread(str(rec["sonar_frames"][i]), cv2.IMREAD_UNCHANGED),
                    "T_world_camera": poses["setup/camera"][k],
                    "T_world_sonar": poses["setup/sonar"][k],  # already x right, y down, z forward
                    "time": rec["frame_meta"].loc[i, "FrameTime"] / 1e6,
                    "source_index": i,
                }

    rec = recs[0]
    name = name or rec["name"]
    target = uxo.chain(tfs, rec["target_frame"], "world")
    return write_scene(
        Path(out_root) / "uxo" / name, frames(), K=K, dist=dist, fisheye=True, image_size=(1920, 1080),
        sonar={**geometry, "model": "ARIS Explorer 3000 (128 beams)",
               "horizontal_fov": np.radians(uxo.SONAR_FOV_DEG["horizontal"]),
               "vertical_fov": np.radians(uxo.SONAR_FOV_DEG["vertical"])},
        meta={"dataset": "uxo", "sequence": name, "target": rec["target"],
              **({"recordings": [r["name"] for r in recs]} if len(recs) > 1 else {}),
              "target_transform_matrix": target.tolist(),
              "init_bounds": [(target[:3, 3] - init_half_size).tolist(), (target[:3, 3] + init_half_size).tolist()]},
        **kwargs,
    )


def export_uxo_colmap(scene: Path, colmap_model: Path, tfs: dict, name: str | None = None) -> dict:
    """COLMAP variant of an exported UXO scene (export_uxo): same images, sonar frames and start points (links),
    only the poses change. As prepare_uxo.ipynb step 7 (user decision 2026-11-11: COLMAP poses, option A):

      camera  COLMAP pose of every registered frame (run_colmap on this scene's images, camera held fixed),
              similarity-aligned (Umeyama, positions only) to the scene's gantry camera positions: metres, gantry world
      sonar   COLMAP camera pose x the calibration's camera -> sonar transform (transforms.yaml, setup/camera and
              setup/sonar both under setup/ar3)
    Why: the gantry orientations disagree with COLMAP by ~4.5 deg (part steady pitch, part varying with the rotator),
    which blurred camera-only reconstructions even on training views. Unregistered frames are dropped.
    Written next to `scene` as <scene>_colmap (or `name`).
    """
    scene = Path(scene)
    meta = json.loads((scene / "transforms.json").read_text())
    col = read_colmap_images(colmap_model)
    keep = [(n, f) for n, f in enumerate(meta["frames"]) if Path(f["file_path"]).stem in col]
    C = np.array([col[Path(f["file_path"]).stem] for _, f in keep])               # OpenCV camera axes
    gantry = np.array([f["transform_matrix"] for _, f in keep]) @ OPENCV_TO_OPENGL   # back to OpenCV axes
    scale, R, tr = umeyama(C[:, :3, 3], gantry[:, :3, 3])
    T_wc = np.tile(np.eye(4), (len(C), 1, 1))
    T_wc[:, :3, :3] = R @ C[:, :3, :3]
    T_wc[:, :3, 3] = scale * (R @ C[:, :3, 3].T).T + tr
    residual = np.linalg.norm(T_wc[:, :3, 3] - gantry[:, :3, 3], axis=1)
    T_cam_sonar = np.linalg.inv(uxo.chain(tfs, "setup/camera", "setup/ar3")) @ uxo.chain(tfs, "setup/sonar", "setup/ar3")
    # rotation between the gantry and the COLMAP camera orientation, per frame [deg]
    dR = np.einsum("nji,njk->nik", gantry[:, :3, :3], T_wc[:, :3, :3])
    angle = np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))

    out = scene.with_name(name or f"{scene.name}_colmap")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for d in ("images", "sonar", meta["ply_file_path"]):
        (out / d).symlink_to(Path("..") / scene.name / d)
    frames = [{**f, "transform_matrix": (T_wc[k] @ OPENCV_TO_OPENGL).tolist(),
               "sonar_transform_matrix": (T_wc[k] @ T_cam_sonar).tolist()} for k, (_, f) in enumerate(keep)]
    transforms = {**meta, "sequence": out.name, "frames": frames, "poses": {
        "camera": "COLMAP (fixed pinhole camera), similarity-aligned to the gantry camera positions",
        "sonar": "COLMAP camera pose x calibration camera -> sonar",
        "colmap_model": str(colmap_model), "alignment_scale": float(scale), "alignment_R": R.tolist(),
        "alignment_t": tr.tolist(),
        "alignment_residual_m": {"median": float(np.median(residual)), "max": float(residual.max())},
        "gantry_vs_colmap_rotation_deg": {"median": float(np.median(angle)), "max": float(angle.max())},
        "T_camera_sonar": T_cam_sonar.tolist(),
        "frames": f"{len(keep)} of {len(meta['frames'])} (COLMAP registered)"}}
    (out / "transforms.json").write_text(json.dumps(transforms, indent=1))
    return transforms


def export_opti(bag: Path, out_root: Path, step: int = 1, pitch: bool = True, **kwargs) -> dict:
    """Export one opti-acoustic bag: every `step`-th camera frame with its nearest ping and interpolated pose."""
    bag = Path(bag)
    seq = opti.load_sequence(bag)
    f = seq["frames"].iloc[::step]
    T_world_sonar = opti.sonar_poses(f, pitch) @ T_FLU_RDF
    T_world_camera = opti.camera_poses(f, pitch)
    # A few bags switch between two range settings of the same 3 m span (e.g. 378 x 7.916 mm and
    # 379 x 7.905 mm); pings are resampled onto the most common one
    settings = pd.Series([(p.num_ranges, p.range_resolution) for p in seq["pings"]])
    n_ranges, resolution = settings.mode()[0]
    ranges = np.arange(n_ranges) * resolution
    K, dist = opti.CAMERA_INTRINSICS[opti.calibration_name(bag.parent.name)]

    def polar(ping) -> np.ndarray:
        s = opti.decode_ping(ping)
        if len(s["ranges"]) == n_ranges and ping.range_resolution == resolution:
            return s["polar"]
        rows = np.interp(ranges, s["ranges"], np.arange(len(s["ranges"])), right=-1)  # -1: beyond range -> 0
        map_x, map_y = np.meshgrid(np.arange(s["polar"].shape[1]), rows)
        return cv2.remap(s["polar"], map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def frames():
        for k, row in enumerate(f.itertuples()):
            yield {
                "image": opti.decode_image(seq["cameras"][row.camera_idx]),
                "sonar": polar(seq["pings"][row.ping_idx]),
                "T_world_camera": T_world_camera[k],
                "T_world_sonar": T_world_sonar[k],
                "time": row.t,
                "source_index": row.camera_idx,
            }

    return write_scene(
        Path(out_root) / "opti_acoustic" / bag.stem, frames(), K=K, dist=dist, fisheye=False, image_size=(1920, 1080),
        sonar={"ranges": ranges, "bearings": opti.decode_ping(seq["pings"][0])["bearings"], "model": "Oculus M750d",
               "horizontal_fov": np.radians(opti.SONAR_FOV_DEG["horizontal"]),
               "vertical_fov": np.radians(opti.SONAR_FOV_DEG["vertical"])},
        meta={"dataset": "opti_acoustic", "sequence": bag.stem, "scenario": bag.parent.name, "pitch": pitch},
        **kwargs,
    )


def export_sonar_mast3r(seq: Path, calibration: Path, out_root: Path, step: int = 1, name: str | None = None,
                        **kwargs) -> dict:
    """Export one Sonar-MASt3R sequence from its bag export (prepare_sonar_mast3r.ipynb): every `step`-th camera
    frame with its nearest sonar frame (fan image resampled to polar) and the forward-kinematics poses stamped at
    each image.

    Frames: dwe_cam is used as an OpenCV camera (z forward, x right, y down); oculus_link as a ROS sensor frame
    (x forward = boresight, y left, z up), converted with the same axis swap as the opti-acoustic sonar.

    Bearings are mirrored (fan image left = sonar right in this frame). Evidence, dataset1: bright returns of all
    559 frames placed in the world at zero elevation form a sharp ring of 2.23 m diameter (the 2.1 m tank wall; zero
    elevation overestimates it) with the mirror, and a 1.23 m smear without (densest 2% of 1 cm cells: 21% vs 16% of
    the returns). Whether the viewer mirrors the fan or oculus_link has y right / z down cannot be told apart: they
    differ by a 180 deg rotation about the boresight, which only flips the unmeasured elevation.
    """
    seq = Path(seq)
    s = sonar_mast3r.load_sequence(seq)
    K, dist, size = sonar_mast3r.load_calibration(calibration)
    T_world_sonar = s["T_world_oculus"] @ T_FLU_RDF

    def frames():
        for i in range(0, len(s["t"]), step):
            ci, si = s["cam_idx"][i], s["sonar_idx"][i]
            yield {
                "image": cv2.cvtColor(cv2.imread(str(seq / f"cam_frames/frame_{ci:06d}.png")), cv2.COLOR_BGR2RGB),
                "sonar": sonar_mast3r.fan_to_polar(cv2.imread(str(seq / f"sonar_frames/frame_{si:06d}.png")))[:, ::-1],
                "T_world_camera": s["T_world_dwe_cam"][i],
                "T_world_sonar": T_world_sonar[i],
                "time": s["t"][i],
                "source_index": ci,
            }

    name = name or seq.name
    return write_scene(Path(out_root) / "sonar_mast3r" / name, frames(), K=K, dist=dist, fisheye=False,
                       image_size=size, sonar=_mast3r_sonar(), meta={**MAST3R_META, "sequence": name}, **kwargs)


def _mast3r_sonar() -> dict:
    ranges, bearings = sonar_mast3r.polar_axes()
    return {"ranges": ranges, "bearings": bearings, "model": "Oculus M3000d (from fan video)",
            "horizontal_fov": sonar_mast3r.HORIZONTAL_FOV, "vertical_fov": sonar_mast3r.VERTICAL_FOV}


# init_bounds: box for random start points (Z-Splat reader, underwater-splatting/scene/sonar_camera_reader.py): the 2.1 m tank (centre at (0.64, -0.04) m, fitted to
# the sonar wall echoes) from below the floor (z ~ -0.30 m, mode of the sonar returns) to above the camera path
# (z 0.30-0.42 m). Arm base frame of dataset1.
MAST3R_META = {"dataset": "sonar_mast3r", "init_bounds": [[-0.45, -1.15, -0.45], [1.75, 1.05, 0.50]]}


def _interp_poses(t: np.ndarray, T: np.ndarray, tq: np.ndarray) -> np.ndarray:
    """4x4 poses T (N, 4, 4) at increasing times t, interpolated at times tq: positions linear, rotations slerp."""
    from scipy.spatial.transform import Rotation, Slerp

    out = np.tile(np.eye(4), (len(tq), 1, 1))
    out[:, :3, :3] = Slerp(t, Rotation.from_matrix(T[:, :3, :3]))(tq).as_matrix()
    out[:, :3, 3] = np.stack([np.interp(tq, t, T[:, i, 3]) for i in range(3)], axis=1)
    return out


def export_sonar_mast3r_colmap(seq: Path, calibration: Path, colmap_model: Path, out_root: Path,
                               name: str | None = None, camera_latency: float = 0.185, sonar_latency: float = 0.110,
                               max_gap: float = 0.25, **kwargs) -> dict:
    """Export a Sonar-MASt3R sequence with COLMAP camera poses and a timing model for both sensors.

    Why: the forward-kinematics (FK) camera poses disagree with COLMAP by 2.2 cm / 4.2 deg (median, dataset1). Most of
    it is timing: the images are ~185 ms older than their stamps (COLMAP vs FK, position error 2.15 -> 0.40 cm when
    the FK poses are shifted), the sonar frames ~110 ms (sharpest world map of the sonar returns, broad optimum).
    ~2 deg of FK rotation error remains after the shift, so the camera poses come from COLMAP, not from FK.

    colmap_model: COLMAP text model run on the images of the FK export (export_sonar_mast3r, step 1), so an image
    name is the camera frame index; the camera model was held fixed to that export's pinhole camera.
      camera   COLMAP pose of every registered frame, similarity-aligned (Umeyama, positions only) to the FK camera
               positions at the frame's true capture time (stamp - camera_latency): metres, arm base frame.
      pairing  each camera frame gets the sonar frame closest in true time (sonar stamp - sonar_latency).
      sonar    the aligned COLMAP camera trajectory interpolated at the sonar frame's true time, times the fixed
               camera -> sonar transform of the FK model (median translation, mean rotation over all frames; FK
               camera and sonar poses compared at the same instant). Frames whose sonar time is not between two
               registered camera frames at most `max_gap` apart are dropped (no interpolation across gaps).
    Frame conventions and the bearing mirror as in export_sonar_mast3r.
    """
    seq = Path(seq)
    p = np.load(seq / "poses.npz")
    t0 = p["cam_stamp_ns"][0]
    tc_stamp, ts_stamp = (p["cam_stamp_ns"] - t0) / 1e9, (p["sonar_stamp_ns"] - t0) / 1e9
    tc, ts = tc_stamp - camera_latency, ts_stamp - sonar_latency                      # true capture times
    T_fk_cam = sonar_mast3r.pose_matrices(p["cam_pos"], p["cam_quat"])
    T_fk_son = sonar_mast3r.pose_matrices(p["sonar_pos"], p["sonar_quat"]) @ T_FLU_RDF

    # camera -> sonar of the FK model: both FK poses at the camera stamps (FK poses are timed by their stamps)
    from scipy.spatial.transform import Rotation
    inside = (tc_stamp >= ts_stamp[0]) & (tc_stamp <= ts_stamp[-1])
    T_cs = np.linalg.inv(T_fk_cam[inside]) @ _interp_poses(ts_stamp, T_fk_son, tc_stamp[inside])
    T_cam_sonar = np.eye(4)
    T_cam_sonar[:3, :3] = Rotation.from_matrix(T_cs[:, :3, :3]).mean().as_matrix()
    T_cam_sonar[:3, 3] = np.median(T_cs[:, :3, 3], axis=0)

    # COLMAP camera poses, aligned to the FK camera positions at the true capture times
    col = read_colmap_images(colmap_model)
    reg = np.array(sorted(int(k) for k in col))
    C = np.array([col[f"{i:06d}"] for i in reg])
    fk_pos = _interp_poses(tc_stamp, T_fk_cam, np.clip(tc[reg], tc_stamp[0], tc_stamp[-1]))[:, :3, 3]
    scale, R, tr = umeyama(C[:, :3, 3], fk_pos)
    T_wc = np.tile(np.eye(4), (len(reg), 1, 1))
    T_wc[:, :3, :3] = R @ C[:, :3, :3]
    T_wc[:, :3, 3] = scale * (R @ C[:, :3, 3].T).T + tr
    residual = np.linalg.norm(T_wc[:, :3, 3] - fk_pos, axis=1)

    # pairing in true time, sonar pose from the camera trajectory at the sonar's true time
    k = np.clip(np.searchsorted(ts, tc[reg]), 1, len(ts) - 1)
    son = np.where(np.abs(ts[k] - tc[reg]) < np.abs(ts[k - 1] - tc[reg]), k, k - 1)
    b = np.searchsorted(tc[reg], ts[son])                                              # bracketing camera frames
    ok = (b > 0) & (b < len(reg))
    ok[ok] &= (tc[reg][b[ok]] - tc[reg][b[ok] - 1]) <= max_gap
    T_ws = _interp_poses(tc[reg], T_wc, ts[son][ok]) @ T_cam_sonar
    K, dist, size = sonar_mast3r.load_calibration(calibration)

    def frames():
        for n, (ci, si) in enumerate(zip(reg[ok], son[ok])):
            yield {
                "image": cv2.cvtColor(cv2.imread(str(seq / f"cam_frames/frame_{ci:06d}.png")), cv2.COLOR_BGR2RGB),
                "sonar": sonar_mast3r.fan_to_polar(cv2.imread(str(seq / f"sonar_frames/frame_{si:06d}.png")))[:, ::-1],
                "T_world_camera": T_wc[ok][n],
                "T_world_sonar": T_ws[n],
                "time": tc_stamp[ci],
                "source_index": int(ci),
            }

    name = name or f"{seq.name.removesuffix('_bag')}_colmap"
    meta = {**MAST3R_META, "sequence": name, "poses": {
        "camera": "COLMAP (fixed intrinsics), similarity-aligned to FK camera positions at true capture time",
        "sonar": "COLMAP camera trajectory at the sonar's true time x FK camera->sonar",
        "colmap_model": str(colmap_model), "camera_latency_s": camera_latency, "sonar_latency_s": sonar_latency,
        "alignment_scale": float(scale), "alignment_R": R.tolist(), "alignment_t": tr.tolist(),
        "alignment_residual_m": {"median": float(np.median(residual)), "max": float(residual.max())},
        "T_camera_sonar": T_cam_sonar.tolist(),
        "frames": f"{ok.sum()} of {len(tc)} camera frames (COLMAP registered {len(reg)}; "
                  f"{(~ok).sum()} dropped: sonar time in a gap of unregistered frames)"}}
    return write_scene(Path(out_root) / "sonar_mast3r" / name, frames(), K=K, dist=dist, fisheye=False,
                       image_size=size, sonar=_mast3r_sonar(), meta=meta, **kwargs)


# ---------------------------------------------------------------------------
# COLMAP poses


def run_colmap(scene: Path, out: Path, colmap: str = "colmap", gpu: bool = False) -> Path:
    """Structure from motion on an exported scene's images with its pinhole camera held FIXED (no focal, principal
    point or distortion refinement), as prepare_uxo.ipynb step 7: single camera, exhaustive matching, mapper.
    Writes <out>/sparse/<model>/ as text and returns the largest model's folder. An existing `out` is replaced.
    Image names are the scene's (000000.jpg, ...), so a COLMAP image name is the scene frame number."""
    import subprocess

    meta = json.loads((Path(scene) / "transforms.json").read_text())
    out = Path(out)
    shutil.rmtree(out, ignore_errors=True)
    (out / "sparse").mkdir(parents=True)
    db, images = str(out / "database.db"), str(Path(scene).resolve() / "images")
    run = lambda *a: subprocess.run([colmap, *a], check=True, capture_output=True, text=True)
    run("feature_extractor", "--database_path", db, "--image_path", images,
        "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "1",
        "--ImageReader.camera_params", ",".join(f"{meta[k]:.6f}" for k in ("fl_x", "fl_y", "cx", "cy")),
        "--SiftExtraction.use_gpu", str(int(gpu)))
    run("exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", str(int(gpu)))
    run("mapper", "--database_path", db, "--image_path", images, "--output_path", str(out / "sparse"),
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0")
    models = sorted((out / "sparse").iterdir())
    for m in models:
        run("model_converter", "--input_path", str(m), "--output_path", str(m), "--output_type", "TXT")
    return max(models, key=lambda m: len(read_colmap_images(m)))


def read_colmap_images(model_dir: Path) -> dict[str, np.ndarray]:
    """images.txt of a COLMAP text model -> {image name stem: camera-to-world 4x4, OpenCV camera axes}.
    COLMAP stores world-to-camera as quaternion (w first) and translation."""
    from scipy.spatial.transform import Rotation

    lines = [l for l in (Path(model_dir) / "images.txt").read_text().splitlines() if not l.startswith("#")]
    poses = {}
    for l in lines[::2]:
        v = l.split()
        w2c = np.eye(4)
        w2c[:3, :3] = Rotation.from_quat([float(v[2]), float(v[3]), float(v[4]), float(v[1])]).as_matrix()
        w2c[:3, 3] = [float(x) for x in v[5:8]]
        poses[Path(v[9]).stem] = np.linalg.inv(w2c)
    return poses


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) minimising sum |s R src + t - dst|^2 over point pairs (Umeyama 1991), no reflection."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a ** 2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s

