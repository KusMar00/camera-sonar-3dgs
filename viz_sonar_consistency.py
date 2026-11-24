"""Rerun view of the sonar consistency check (check_sonar_consistency.py): where the camera model and the sonar put
the target echo, in 3D.

Why: the check reduces each frame to one range/bearing shift. This shows the same comparison in the scene, so one
can see on which part of the casing the measured echo lies relative to the camera model's geometry.

What is drawn, in the world frame of the scene (UXO: target at the origin, z up):
    world/gaussians          the target Gaussians the check rendered (camera-only model, TARGET_BOX), grey
    world/sonar_path         all sonar centres
    world/sonar/fov          the current sonar's field of view as Z-Splat models it (pinhole, r_min to r_max)
    world/sonar/predicted    bright pixels of the predicted image, blue
    world/sonar/measured     bright pixels of the measured image, yellow
    sonar/overlay            the 2D overlay of examples.png (blue predicted, yellow measured, white both)
    info                     the shift the check found for the current frame
Timeline "frame" = sonar frame index.

A sonar pixel fixes range and column but not elevation, so it is drawn as an arc over the vertical aperture: the set
of 3D points that would all produce that pixel. With the rasterizer's pinhole columns, column c is the plane
x/z = u_c, and a point at range r on it is r * (u_c, v, 1) / |(u_c, v, 1)| with |v| <= tan(vfov/2). A predicted arc
passes through the Gaussians that made it; if the poses are consistent, the measured arcs do too, and a range error
shows as yellow arcs running parallel to the blue ones, just outside or inside the casing.

Colours are blue / yellow so they stay distinguishable with red-green colour blindness.

Runs in the main environment (needs rerun):
    python viz_sonar_consistency.py <check output dir> <scene dir with sonar.npz> [out.rrd]
then open the recording with `rerun <out.rrd>` (default: <check output dir>/consistency.rrd).
"""

import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

PRED_COLOR = (64, 153, 255)      # sky blue, as in the 2D overlay
MEAS_COLOR = (255, 200, 0)       # yellow
GAUSS_COLOR = (150, 150, 150)
SONAR_COLOR = (200, 200, 200)
MIN_REL = 0.3                    # pixels at least this fraction of their column's peak are drawn (as in training,
                                 # images are compared column-normalised)
MAX_ARCS = 400                   # brightest pixels per image, keeps the 3D view readable
ARC_POINTS = 7


def column_max_one(im: np.ndarray) -> np.ndarray:
    peak = im.max(axis=0, keepdims=True)
    return np.divide(im, peak, out=np.zeros_like(im), where=peak > 0)


def pixel_arcs(im: np.ndarray, ranges: np.ndarray, u: np.ndarray, tan_v: float) -> np.ndarray:
    """(M, ARC_POINTS, 3) arcs in the sonar frame (x right, y down, z forward) for the bright pixels of `im`."""
    n = column_max_one(im)
    r_idx, c_idx = np.nonzero(n >= MIN_REL)
    top = np.argsort(im[r_idx, c_idx])[::-1][:MAX_ARCS]
    r_idx, c_idx = r_idx[top], c_idx[top]
    v = np.linspace(-tan_v, tan_v, ARC_POINTS)
    d = np.stack(np.broadcast_arrays(u[c_idx, None], v[None], 1.0), -1)          # (M, P, 3) ray directions
    return ranges[r_idx, None, None] * d / np.linalg.norm(d, axis=-1, keepdims=True)


def fov_lines(r_min: float, r_max: float, tan_h: float, tan_v: float) -> list[np.ndarray]:
    """Edges of the pinhole sonar frustum between r_min and r_max."""
    corners = np.array([[sx * tan_h, sy * tan_v, 1.0] for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    corners /= np.linalg.norm(corners, axis=1, keepdims=True)
    near, far = r_min * corners, r_max * corners
    return [np.vstack([near, near[:1]]), np.vstack([far, far[:1]])] + [np.vstack([a, b]) for a, b in zip(near, far)]


def main(check_dir: Path, scene_dir: Path, out: Path):
    son = np.load(scene_dir / "sonar.npz")
    pred_npz = np.load(check_dir / "predicted.npz")
    meas_all, preds, c2ws = son["images"], pred_npz["images"], son["c2w"]
    hfov, vfov, width = float(son["hfov"]), float(son["vfov"]), int(son["width"])
    r_min, r_max = float(son["r_min"]), float(son["r_max"])
    n_bins = meas_all.shape[1]
    dr = (r_max - r_min) / n_bins
    ranges = r_min + (np.arange(n_bins) + 0.5) * dr                              # bin centres
    tan_h, tan_v = np.tan(hfov / 2), np.tan(vfov / 2)
    u = ((np.arange(width) + 0.5) / width * 2 - 1) * tan_h                       # x/z of each column centre
    res = np.loadtxt(check_dir / "per_frame.csv", delimiter=",", ndmin=2)
    dc_deg = json.loads((check_dir / "summary.json").read_text())["column_deg"]

    rec = rr.RecordingStream("sonar_consistency")
    rec.save(str(out))
    rec.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="3D"),
            rrb.Vertical(rrb.Spatial2DView(origin="sonar", name="sonar overlay (far range at top)"),
                         rrb.TextDocumentView(origin="info", name="frame"), row_shares=[4, 1]),
            column_shares=[3, 2]),
        rrb.TimePanel(state="expanded"), collapse_panels=True))

    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("world/gaussians", rr.Points3D(pred_npz["gaussians_xyz"], colors=GAUSS_COLOR, radii=0.0015), static=True)
    rec.log("world/sonar_path", rr.LineStrips3D([c2ws[:, :3, 3]], colors=SONAR_COLOR, radii=-1.0), static=True)
    rec.log("world/sonar/fov", rr.LineStrips3D(fov_lines(r_min, r_max, tan_h, tan_v), colors=SONAR_COLOR,
                                               radii=-1.0), static=True)

    for k, (c2w, meas, pred) in enumerate(zip(c2ws, meas_all, preds)):
        rec.set_time("frame", sequence=k)
        rec.log("world/sonar", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
        for name, im, color in (("predicted", pred, PRED_COLOR), ("measured", meas, MEAS_COLOR)):
            arcs = pixel_arcs(im.astype(np.float64), ranges, u, tan_v) if im.any() else np.zeros((0, 2, 3))
            rec.log(f"world/sonar/{name}", rr.LineStrips3D(list(arcs), colors=color, radii=-1.5))
        p, m = column_max_one(pred.astype(np.float64)), column_max_one(meas.astype(np.float64))
        rec.log("sonar/overlay", rr.Image((255 * np.clip(np.dstack([m + 0.25 * p, m + 0.6 * p, p]), 0, 1)[::-1]).astype(np.uint8)))
        sr, sc = res[k, 1], res[k, 2]
        text = (f"**frame {k}**\n\nbest shift: range {sr * dr * 100:+.1f} cm (+ = measured further), "
                f"bearing {sc * dc_deg:+.1f} deg\n\ncorrelation {res[k, 4]:.2f} at zero shift, {res[k, 3]:.2f} at best"
                if not np.isnan(sr) else f"**frame {k}**\n\nnot compared (empty prediction or measurement)")
        rec.log("info", rr.TextDocument(text, media_type="text/markdown"))
    print(f"wrote {out}\nopen with: rerun {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    check_dir = Path(a[0]).resolve()
    main(check_dir, Path(a[1]).resolve(), Path(a[2]).resolve() if len(a) > 2 else check_dir / "consistency.rrd")
