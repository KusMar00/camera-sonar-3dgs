"""Rewrite an exported UXO scene with COLMAP camera poses in metres, for training Z-Splat with sonar.

The gantry camera poses are not consistent enough for 3DGS, and COLMAP's are, but COLMAP has no scale, while
Z-Splat's sonar loss works in metres. So one similarity transform, fitted on the camera positions over all
recordings, brings the COLMAP model into the gantry world. One for all recordings, because COLMAP's model is
consistent across them and a fit per recording would break that.

Sonar poses are not observed by COLMAP. They follow from the rigid camera-to-sonar transform of the
transforms.yaml chain: T_world_sonar = T_world_camera(COLMAP) * T_camera_sonar. check_sonar_poses.py found
this good in bearing but up to about 10 cm off in range, differing per recording, so this scene is an
experiment, not verified calibration. The initial point cloud is COLMAP's, not the sonar cloud.

Usage: python export_colmap_poses.py [scene] [colmap_sparse_dir] [output_scene]
Images and sonar frames are linked to the source scene, not copied.
"""

import json
import sys
from pathlib import Path

import numpy as np

from check_sonar_poses import OPENCV_TO_OPENGL, apply_similarity, fit_similarity, read_colmap
from export import write_ply


def main(scene: Path, sparse: Path, out: Path):
    meta = json.load(open(scene / "transforms.json"))
    frames = meta["frames"]
    cameras, points, colors = read_colmap(sparse)
    cam_gantry = np.array([np.array(f["transform_matrix"]) @ OPENCV_TO_OPENGL for f in frames])
    son_gantry = np.array([np.array(f["sonar_transform_matrix"]) for f in frames])
    cam_colmap = np.array([cameras[Path(f["file_path"]).name] for f in frames])

    S = fit_similarity(cam_colmap[:, :3, 3], cam_gantry[:, :3, 3])
    cam_world = np.array([apply_similarity(S, c) for c in cam_colmap])
    son_world = cam_world @ np.linalg.inv(cam_gantry) @ son_gantry
    world = (S[:3, :3] @ points.T).T + S[:3, 3]
    scale = float(np.cbrt(np.linalg.det(S[:3, :3])))
    residual = np.linalg.norm(cam_world[:, :3, 3] - cam_gantry[:, :3, 3], axis=1)
    print(f"scale {scale:.4f}, camera position residual to gantry: median {np.median(residual) * 1000:.0f} mm, "
          f"max {residual.max() * 1000:.0f} mm")

    for f, c, s in zip(frames, cam_world, son_world):
        f["transform_matrix"] = (c @ OPENCV_TO_OPENGL).tolist()
        f["sonar_transform_matrix"] = s.tolist()
    meta["sequence"] = out.name
    meta["poses"] = {"source": "COLMAP aligned to the gantry world; sonar from T_camera_sonar of the transforms.yaml chain",
                     "colmap_to_world": S.tolist(), "colmap_scale": scale}
    out.mkdir(parents=True, exist_ok=True)
    for name in ("images", "sonar"):
        link = out / name
        if not link.exists():
            link.symlink_to(scene.resolve() / name)
    write_ply(out / meta["ply_file_path"], world, colors)
    json.dump(meta, open(out / "transforms.json", "w"), indent=1)
    print(f"{len(frames)} frames, {len(world)} points -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(Path(args[0] if args else "prepared/uxo/incendiary_3heights"),
         Path(args[1] if len(args) > 1 else "prepared/uxo/incendiary_3heights_colmap/colmap/sparse/0"),
         Path(args[2] if len(args) > 2 else "prepared/uxo/incendiary_3heights_zsplat"))
