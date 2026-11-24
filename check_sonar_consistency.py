"""Do camera and sonar agree on where things are? Predict the sonar images from a camera-only Z-Splat model and
compare them with the measured ones.

Why: adding sonar to a UXO Z-Splat run smears the target in the camera renders, also on training views, i.e. the
two losses pull the Gaussians to different places. A camera-only model trained on the same camera poses has a sharp
target, so its geometry is what the camera says. Rendering it at the sonar poses gives the sonar image the camera
"expects"; comparing with the measured sonar image shows whether the sonar poses (camera pose x T_camera_sonar)
put the echoes in the same place.

How: every sonar view is rendered with Z-Splat's own rasterizer, exactly as in training (scene/cameras.Camera with
the sonar pose and aperture, gaussian_renderer.render, output z_density_w = range bins x pinhole columns). Per frame,
the 2D cross-correlation of predicted and measured image over range and column shifts gives the shift that best
lines them up:
    shift_r > 0   the measured echo is further away than the camera model predicts
    shift_c > 0   the measured echo is to the right (higher column) of the prediction
Raw images (not column-normalised) are correlated, so strong echoes such as the target dominate, not noise columns.

Only the target is rendered (TARGET_BOX, world frame: target at the origin, z up): the camera-only model also fills
the water with large faint Gaussians, which look right to the camera but are no reflectors, and the cleaned
measured images contain almost only the target echo. With all Gaussians, the correlation matched the target arc
against that haze (shifts stuck at the search limit). Gaussians outside the box are made fully transparent.

Reading the result:
    same shift in all frames      a fixed camera -> sonar calibration error, and the shift measures it
    near zero                     poses consistent; look at the sonar model / loss instead
    shift varying between frames  per-frame pose or timing errors

Limits: the camera model has no acoustic properties (opacity stands in for echo strength, no occlusion, as in
Z-Splat), so the predicted image will differ in brightness; only the location of structure is compared. Elevation
errors are only seen indirectly (they change which part of the scene falls inside the vertical aperture).

Runs in the Z-Splat environment (Python 3.8), from any directory:
    python check_sonar_consistency.py <camera-only model dir> <scene dir with sonar.npz> [out dir]
Writes summary.json, per_frame.csv, shifts.png, examples.png and predicted.npz (all predicted images and the kept
target Gaussians), which viz_sonar_consistency.py shows in 3D with Rerun (not installed in this environment).
Colours: predicted blue, measured yellow, overlap white (readable with red-green colour blindness).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ZSPLAT = Path.home() / "Dev" / "underwater-splatting"   # method repo (Z-Splat + thesis changes)
MAX_SHIFT_R = 0.20     # [m] largest range shift searched
MAX_SHIFT_C = 16       # [columns] largest column shift searched
MIN_PRED = 1e-3        # frames whose predicted image is (almost) empty in the range window are skipped
# UXO 20lbs_incendiary (gantry world, target at the origin, z up): casing Gaussians at z 0.05-0.12 m, floor at
# z -0.37 to -0.30 m (checked on the camera-only COLMAP model). Keep casing and stand, not the floor.
# max_scale: the casing is built of small Gaussians (median largest axis 0.3 cm); ~1100 Gaussians centred in the box
# are 3-52 cm large and nearly transparent (median opacity 0.08): haze that spreads over the whole sonar view.
TARGET_BOX = {"radius_xy": 0.5, "z_min": -0.25, "z_max": 0.4, "max_scale": 0.03}


def load_model(model_dir: Path, target_only: bool = True):
    """GaussianModel of the last saved iteration of a trained run, and that iteration. With target_only, Gaussians
    outside TARGET_BOX get opacity ~0, so the rasterizer skips them (alpha < 1/255)."""
    import torch
    from scene.gaussian_model import GaussianModel
    it = max(int(p.name.split("_")[1]) for p in (model_dir / "point_cloud").iterdir())
    g = GaussianModel(3)
    g.load_ply(str(model_dir / "point_cloud" / f"iteration_{it}" / "point_cloud.ply"))
    if target_only:
        with torch.no_grad():
            x = g.get_xyz
            keep = ((x[:, :2].norm(dim=1) < TARGET_BOX["radius_xy"]) & (x[:, 2] > TARGET_BOX["z_min"])
                    & (x[:, 2] < TARGET_BOX["z_max"]) & (g.get_scaling.max(dim=1).values < TARGET_BOX["max_scale"]))
            g._opacity[~keep] = -30.0                                 # stored before the sigmoid
        print(f"target only: {int(keep.sum())} of {len(keep)} Gaussians kept")
    else:
        keep = torch.ones(g.get_xyz.shape[0], dtype=torch.bool, device=g.get_xyz.device)
    return g, it, keep.cpu().numpy()


def render_sonar(gaussians, c2w: np.ndarray, hfov: float, vfov: float, width: int, height: int,
                 z_range: tuple) -> np.ndarray:
    """Predicted sonar image (range bins, columns), raw: the same rasterizer call as a sonar view in training.
    z_range = (bins, first range, last range [m]), as the scene reader sets it on every view."""
    import torch
    from types import SimpleNamespace
    from gaussian_renderer import render
    from scene.cameras import Camera

    w2c = np.linalg.inv(c2w)
    cam = Camera(colmap_id=0, R=w2c[:3, :3].T, T=w2c[:3, 3], FoVx=hfov, FoVy=vfov,
                 image=torch.zeros(3, height, width), gt_alpha_mask=None, image_name="sonar", uid=0,
                 depth=[None, None], is_sonar=True)
    cam.z_range = z_range
    pipe = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    with torch.no_grad():
        out = render(cam, gaussians, pipe, torch.zeros(3, device="cuda"))
    return out["z_density_w"].cpu().numpy()


def best_shift(pred: np.ndarray, meas: np.ndarray, max_r: int, max_c: int) -> tuple[int, int, float, float]:
    """(shift_r, shift_c, normalised correlation at that shift, at zero shift): meas[r + sr, c + sc] ~ pred[r, c]."""
    from scipy.signal import correlate
    H, W = pred.shape
    xc = correlate(meas, pred, mode="full", method="fft")      # xc[H-1+sr, W-1+sc] = sum pred[r,c] meas[r+sr, c+sc]
    xc /= np.linalg.norm(pred) * np.linalg.norm(meas)
    win = xc[H - 1 - max_r:H + max_r, W - 1 - max_c:W + max_c]
    i, j = np.unravel_index(np.argmax(win), win.shape)
    return int(i - max_r), int(j - max_c), float(win[i, j]), float(xc[H - 1, W - 1])


def column_max_one(im: np.ndarray) -> np.ndarray:
    peak = im.max(axis=0, keepdims=True)
    return np.divide(im, peak, out=np.zeros_like(im), where=peak > 0)


def main(model_dir: Path, scene_dir: Path, out_dir: Path):
    sys.path.insert(0, str(ZSPLAT))
    os.chdir(ZSPLAT)                                               # the renderer imports relative to the clone
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    son = np.load(scene_dir / "sonar.npz")
    meas_all, c2ws = son["images"], son["c2w"]
    hfov, vfov, width, height = float(son["hfov"]), float(son["vfov"]), int(son["width"]), int(son["height"])
    r_min, r_max = float(son["r_min"]), float(son["r_max"])
    n_bins = meas_all.shape[1]
    dr = (r_max - r_min) / n_bins
    dc_deg = np.degrees(hfov) / width                              # column width near the centre (pinhole: ~uniform at 30 deg)
    max_r = int(round(MAX_SHIFT_R / dr))

    gaussians, it, keep = load_model(model_dir)
    print(f"model {model_dir.name} iteration {it}, {gaussians.get_xyz.shape[0]} Gaussians; "
          f"{len(c2ws)} sonar frames, {n_bins} bins of {1000 * dr:.1f} mm, {width} columns of ~{dc_deg:.2f} deg")

    rows, preds = [], {}
    for k, (c2w, meas) in enumerate(zip(c2ws, meas_all)):
        pred = render_sonar(gaussians, c2w, hfov, vfov, width, height, (n_bins, r_min, r_max))
        assert pred.shape == meas.shape, f"rendered {pred.shape} vs measured {meas.shape}: rasterizer bins differ"
        if pred.sum() < MIN_PRED or meas.sum() == 0:
            rows.append((k, np.nan, np.nan, np.nan, np.nan))
            continue
        sr, sc, peak, zero = best_shift(pred.astype(np.float64), meas.astype(np.float64), max_r, MAX_SHIFT_C)
        rows.append((k, sr, sc, peak, zero))
        preds[k] = pred

    res = np.array(rows, dtype=float)
    ok = ~np.isnan(res[:, 1])
    sr_m, sc_deg = res[ok, 1] * dr, res[ok, 2] * dc_deg
    at_edge = (np.abs(res[ok, 1]) >= max_r) | (np.abs(res[ok, 2]) >= MAX_SHIFT_C)
    summary = {
        "model": str(model_dir), "iteration": it, "scene": str(scene_dir), "frames": int(len(res)),
        "frames_compared": int(ok.sum()), "range_bin_m": dr, "column_deg": dc_deg,
        "shift_range_cm": {"median": 100 * float(np.median(sr_m)), "p10": 100 * float(np.percentile(sr_m, 10)),
                           "p90": 100 * float(np.percentile(sr_m, 90))},
        "shift_bearing_deg": {"median": float(np.median(sc_deg)), "p10": float(np.percentile(sc_deg, 10)),
                              "p90": float(np.percentile(sc_deg, 90))},
        "correlation": {"at_zero_shift_median": float(np.median(res[ok, 4])),
                        "at_best_shift_median": float(np.median(res[ok, 3]))},
        "frames_at_search_limit": int(at_edge.sum()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    pred_all = np.zeros(meas_all.shape, np.float32)                  # skipped frames stay empty
    for k, pred in preds.items():
        pred_all[k] = pred
    np.savez_compressed(out_dir / "predicted.npz", images=pred_all,
                        gaussians_xyz=gaussians.get_xyz.detach().cpu().numpy()[keep].astype(np.float32))
    np.savetxt(out_dir / "per_frame.csv", res, delimiter=",", fmt="%.6g",
               header="frame,shift_range_bins,shift_columns,corr_best,corr_zero")
    print(json.dumps(summary, indent=1))

    # shifts over the recording
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(res[:, 0], res[:, 1] * dr * 100, ".-", ms=3, lw=0.6)
    ax[0].set(ylabel="range shift [cm]\n(+ = measured further)")
    ax[1].plot(res[:, 0], res[:, 2] * dc_deg, ".-", ms=3, lw=0.6)
    ax[1].set(ylabel="bearing shift [deg]\n(+ = measured right)")
    ax[2].plot(res[:, 0], res[:, 4], ".", ms=3, label="at zero shift")
    ax[2].plot(res[:, 0], res[:, 3], ".", ms=3, label="at best shift")
    ax[2].set(ylabel="correlation", xlabel="sonar frame")
    ax[2].legend(fontsize=8)
    for a in ax[:2]:
        a.axhline(0, color="k", lw=0.5)
    fig.suptitle(f"{scene_dir.name}: measured vs predicted from {model_dir.name}")
    fig.tight_layout()
    fig.savefig(out_dir / "shifts.png", dpi=120)

    # example frames: predicted | measured | overlay, column-normalised as in training. Overlay: predicted in the blue
    # (sky blue), measured in red + green (yellow), so overlap is white: readable with red-green colour blindness.
    show = [k for k in np.array(sorted(preds))[np.linspace(0, len(preds) - 1, 6).astype(int)]]
    fig, ax = plt.subplots(len(show), 3, figsize=(10, 2.1 * len(show)))
    ext = [0, width, r_min, r_max]
    for row, k in zip(ax, show):
        p, m = column_max_one(preds[k]), column_max_one(meas_all[k].astype(np.float64))
        row[0].imshow(p, origin="lower", aspect="auto", extent=ext, cmap="gray")
        row[1].imshow(m, origin="lower", aspect="auto", extent=ext, cmap="gray")
        row[2].imshow(np.clip(np.dstack([m + 0.25 * p, m + 0.6 * p, p]), 0, 1), origin="lower", aspect="auto", extent=ext)
        row[0].set_ylabel(f"frame {k}\nrange [m]")
        row[2].set_title(f"shift {res[k, 1] * dr * 100:+.1f} cm, {res[k, 2] * dc_deg:+.1f} deg", fontsize=8)
    for a, t in zip(ax[0], ("predicted (camera model)", "measured", "overlay: blue pred, yellow meas, white both")):
        a.set_title(t if a is not ax[0][2] else a.get_title() + "\n" + t, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "examples.png", dpi=110)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    a = sys.argv[1:]
    repo = Path(__file__).resolve().parent
    main(Path(a[0]).resolve(), Path(a[1]).resolve(),
         Path(a[2]).resolve() if len(a) > 2 else repo / "renders" / "sonar_consistency" / Path(a[0]).name)
