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

World frame: each dataset's own (UXO: gantry world, z up; opti-acoustic: SLAM odometry frame). Metres,
no rescaling or recentring.
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import opti
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


def export_uxo(rec_path: Path, tfs: dict, K: np.ndarray, dist: np.ndarray, out_root: Path,
               step: int = 1, **kwargs) -> dict:
    """Export one UXO recording: every `step`-th frame that has a GoPro image, a raw sonar frame and a pose.

    K, dist are the `wide` fisheye intrinsics. Sonar frames are the raw ARIS samples (`aris_raw`), not the
    fan images in `aris_polar`.
    """
    rec = uxo.load_recording(Path(rec_path), polar=False)
    poses = uxo.rig_poses(rec, tfs)
    rows = [k for k, i in enumerate(poses["frame_idx"]) if i in rec["gopro_frames"] and i in rec["sonar_frames"]][::step]
    geometry = uxo.sonar_geometry(rec)

    def frames():
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

    return write_scene(
        Path(out_root) / "uxo" / rec["name"], frames(), K=K, dist=dist, fisheye=True, image_size=(1920, 1080),
        sonar={**geometry, "model": "ARIS Explorer 3000 (128 beams)",
               "horizontal_fov": np.radians(uxo.SONAR_FOV_DEG["horizontal"]),
               "vertical_fov": np.radians(uxo.SONAR_FOV_DEG["vertical"])},
        meta={"dataset": "uxo", "sequence": rec["name"], "target": rec["target"],
              "target_transform_matrix": uxo.chain(tfs, rec["target_frame"], "world").tolist()},
        **kwargs,
    )


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
