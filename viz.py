"""Rerun visualisation of exported scenes (prepared/<dataset>/<sequence>/, see export.py).

Reads only the exported files, so it shows exactly what a 3DGS method gets. One recording per scene:
    world/points           initial point cloud (points3d.ply), grey by sonar intensity
    world/camera_path      camera centres of all frames (blue); world/sonar_path: sonar centres (orange)
    world/frustums/<k>     a static camera frustum every few frames, to see all viewpoints at once
    world/camera           the current camera pose, with its frustum and image (world/camera/image)
    world/sonar            the current sonar pose: fan outline and the current bright returns, drawn as
                           arcs over the vertical aperture because the sonar does not measure elevation
    world/target           UXO only: the target frame from transforms.yaml (axes 15 cm, RGB = xyz)
    sonar                  the current polar sonar frame (far range at the top); UXO: target as range/bearing
Timelines: "frame" (index into transforms.json) and "time" (seconds since the first frame).

The camera view (blueprint) projects the sonar returns and the target through the camera, so with correct
poses and extrinsics they land on the object in the image.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

import export

CAMERA_COLOR = (42, 120, 214)
SONAR_COLOR = (235, 104, 52)
TARGET_COLORS = {"100lbs_bomb": (42, 120, 214), "100lbs_bomb_(floor)": (235, 104, 52),
                 "15cm_mortar": (27, 175, 122), "20lbs_incendiary": (237, 161, 0), "test_cylinder": (232, 123, 164)}


def load_scene(path: Path) -> dict:
    return json.loads((Path(path) / "transforms.json").read_text())


def _rgb_intensity(v: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    """Colour scaled by intensity (uint8), dark returns stay visible at 30%."""
    a = 0.3 + 0.7 * np.clip(v / 255.0, 0, 1)
    return (a[:, None] * np.array(color)).astype(np.uint8)


def _sonar_point(r, b, e) -> np.ndarray:
    """Point at range r, bearing b, elevation e in the sonar frame (x right, y down, z forward)."""
    return np.stack(np.broadcast_arrays(r * np.cos(e) * np.sin(b), -r * np.sin(e), r * np.cos(e) * np.cos(b)), -1)


def fan_outline(sonar: dict, n: int = 24) -> list[np.ndarray]:
    """Line strips outlining the sonar's field of view (max range, both apertures)."""
    r, h, v = sonar["ranges"][-1], sonar["horizontal_fov"] / 2, sonar["vertical_fov"] / 2
    b, e = np.linspace(-h, h, n), np.linspace(-v, v, n // 3)
    strips = [_sonar_point(r, b, ev) for ev in (-v, 0.0, v)]                    # far arcs
    strips += [_sonar_point(r, bv, e) for bv in (-h, h)]                        # far edges
    strips += [np.vstack([np.zeros(3), _sonar_point(r, bv, ev)]) for bv in (-h, h) for ev in (-v, v)]
    return strips


def sonar_arcs(polar: np.ndarray, sonar: dict, max_returns: int = 60, percentile: float = 99.5,
               min_range: float = 0.3, n: int = 5) -> tuple[list[np.ndarray], np.ndarray]:
    """The brightest returns of one frame as arcs over the vertical aperture (sonar frame), and their intensity."""
    ranges, bearings = np.asarray(sonar["ranges"]), np.asarray(sonar["bearings"])
    bright = (polar >= max(np.percentile(polar, percentile), 1)) & (ranges[:, None] >= min_range)
    r_idx, b_idx = np.nonzero(bright)
    top = np.argsort(polar[r_idx, b_idx])[::-1][:max_returns]
    r_idx, b_idx = r_idx[top], b_idx[top]
    e = np.linspace(-sonar["vertical_fov"] / 2, sonar["vertical_fov"] / 2, n)
    arcs = _sonar_point(ranges[r_idx, None], bearings[b_idx, None], e[None])     # (M, n, 3)
    return list(arcs), polar[r_idx, b_idx]


def _transform(T: np.ndarray) -> rr.Transform3D:
    T = np.asarray(T)
    return rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3])


def _pinhole(t: dict, scale: float, plane: float, **kwargs) -> rr.Pinhole:
    return rr.Pinhole(focal_length=[t["fl_x"] * scale, t["fl_y"] * scale],
                      principal_point=[t["cx"] * scale, t["cy"] * scale],
                      resolution=[round(t["w"] * scale), round(t["h"] * scale)],
                      camera_xyz=rr.ViewCoordinates.RUB,  # transform_matrix uses OpenGL camera axes
                      image_plane_distance=plane, **kwargs)


def _jpeg(path: Path, width: int) -> bytes:
    img = cv2.imread(str(path))
    img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()


def blueprint(t: dict) -> rrb.Blueprint:
    """3D scene on the left; camera image with the projected sonar returns and target, and the sonar, on the right."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="3D"),
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera/image", name="camera",
                                  contents=["$origin/**", "/world/sonar/returns", "/world/target/**"]),
                rrb.Spatial2DView(origin="sonar", name=f"sonar ({t['sonar']['model']})"),
                rrb.TextDocumentView(origin="info", name="scene"),
                row_shares=[3, 3, 1],
            ),
            column_shares=[3, 2],
        ),
        rrb.TimePanel(state="expanded"),
        collapse_panels=True,
    )


def log_scene(scene: Path, rec: rr.RecordingStream, max_frames: int = 150, n_frustums: int = 25,
              image_width: int = 960):
    """Log one exported scene to `rec`. Long scenes are subsampled to `max_frames` evenly spaced frames."""
    scene = Path(scene)
    t = load_scene(scene)
    frames, sonar = t["frames"], t["sonar"]
    ranges, bearings = np.asarray(sonar["ranges"]), np.asarray(sonar["bearings"])
    scale = image_width / t["w"]
    plane = 0.1 * ranges[-1]                                           # frustum depth, scaled to the scene
    cams = np.array([f["transform_matrix"] for f in frames])
    sonars = np.array([f["sonar_transform_matrix"] for f in frames])
    shown = np.unique(np.linspace(0, len(frames) - 1, min(max_frames, len(frames))).astype(int))

    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)  # both world frames have z up
    xyz, rgb = export.read_ply(scene / t["ply_file_path"])
    rec.log("world/points", rr.Points3D(xyz, colors=_rgb_intensity(rgb[:, 0].astype(float)), radii=-1.5), static=True)
    rec.log("world/camera_path", rr.LineStrips3D([cams[:, :3, 3]], colors=CAMERA_COLOR, radii=-2.0), static=True)
    rec.log("world/sonar_path", rr.LineStrips3D([sonars[:, :3, 3]], colors=SONAR_COLOR, radii=-2.0), static=True)
    for k in np.unique(np.linspace(0, len(frames) - 1, n_frustums).astype(int)):
        rec.log(f"world/frustums/{k:06d}", _transform(cams[k]),
                _pinhole(t, scale, 0.6 * plane, color=(160, 160, 160)), static=True)
    rec.log("world/camera/image", _pinhole(t, scale, plane, color=CAMERA_COLOR), static=True)
    rec.log("world/sonar/fan", rr.LineStrips3D(fan_outline(sonar), colors=SONAR_COLOR, radii=-1.5), static=True)

    target = None
    if "target_transform_matrix" in t:
        T_target = np.array(t["target_transform_matrix"])
        target = T_target[:3, 3]
        rec.log("world/target", _transform(T_target), rr.TransformAxes3D(0.15), static=True)
        rec.log("world/target/origin", rr.Points3D([[0, 0, 0]], labels=[t["target"]], colors=(255, 220, 0),
                                                   radii=0.015), static=True)

    info = {k: t[k] for k in ("dataset", "sequence", "target", "scenario", "pitch") if k in t}
    info |= {"frames": f"{len(frames)} exported, {len(shown)} shown", "image": f"{t['w']}x{t['h']} pinhole",
             "camera hfov": f"{np.degrees(2 * np.arctan(t['w'] / 2 / t['fl_x'])):.1f} deg", "sonar": sonar["model"],
             "sonar range": f"{ranges[0]:.2f} to {ranges[-1]:.2f} m, {len(bearings)} beams over "
                            f"{np.degrees(bearings[-1] - bearings[0]):.0f} deg"}
    rec.log("info", rr.TextDocument("\n".join(f"- **{k}**: {v}" for k, v in info.items()), media_type="text/markdown"),
            static=True)

    t0 = frames[0]["time"]
    for k in shown:
        fr = frames[k]
        rec.set_time("frame", sequence=int(k))
        rec.set_time("time", duration=fr["time"] - t0)
        rec.log("world/camera", _transform(cams[k]))
        rec.log("world/camera/image", rr.EncodedImage(contents=_jpeg(scene / fr["file_path"], image_width),
                                                      media_type="image/jpeg"))
        polar = np.load(scene / fr["sonar_file_path"])
        rec.log("world/sonar", _transform(sonars[k]))
        arcs, val = sonar_arcs(polar, sonar)
        rec.log("world/sonar/returns", rr.LineStrips3D(arcs, colors=_rgb_intensity(val, SONAR_COLOR), radii=-1.0))
        rec.log("sonar/image", rr.Image(polar[::-1]))                  # row 0 = far range, at the top
        if target is not None:  # target origin as range and bearing, in pixel coordinates of sonar/image
            p = np.linalg.inv(sonars[k]) @ np.append(target, 1)
            r, b = np.linalg.norm(p[:3]), np.arctan2(p[0], p[2])
            col = np.interp(b, bearings, np.arange(len(bearings)))
            row = len(ranges) - 1 - np.interp(r, ranges, np.arange(len(ranges)))
            inside = ranges[0] <= r <= ranges[-1] and bearings[0] <= b <= bearings[-1]
            rec.log("sonar/target", rr.Points2D([[col, row]] if inside else np.zeros((0, 2)), colors=(255, 220, 0),
                                                radii=3, labels=[t["target"]] if inside else None))


def log_uxo_overview(scenes: list[Path], rec: rr.RecordingStream, points_per_scene: int = 3000, seed: int = 0):
    """All UXO scenes in the shared gantry world: camera paths coloured by target, target frames, sonar points."""
    rng = np.random.default_rng(seed)
    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    targets = {}
    for scene in scenes:
        t = load_scene(scene)
        color = TARGET_COLORS.get(t["target"], (200, 200, 200))
        cams = np.array([f["transform_matrix"] for f in t["frames"]])
        rec.log(f"world/{t['target']}/{t['sequence']}/camera_path",
                rr.LineStrips3D([cams[:, :3, 3]], colors=color, radii=-2.0, labels=[t["sequence"]], show_labels=False),
                static=True)
        xyz, rgb = export.read_ply(Path(scene) / t["ply_file_path"])
        keep = rng.choice(len(xyz), min(points_per_scene, len(xyz)), replace=False)
        rec.log(f"world/{t['target']}/{t['sequence']}/points",
                rr.Points3D(xyz[keep], colors=_rgb_intensity(rgb[keep, 0].astype(float), color), radii=-1.5),
                static=True)
        targets[t["target"]] = np.array(t["target_transform_matrix"])
    for name, T in targets.items():
        rec.log(f"world/{name}/target", _transform(T), rr.TransformAxes3D(0.15), static=True)
        rec.log(f"world/{name}/target/origin", rr.Points3D([[0, 0, 0]], labels=[name], colors=TARGET_COLORS.get(name),
                                                           radii=0.02), static=True)
