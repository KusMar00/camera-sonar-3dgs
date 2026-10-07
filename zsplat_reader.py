"""Z-Splat scene reader for the common camera + sonar format written by export.py.

Z-Splat (Qu et al. 2024, https://github.com/QuintonQu/gaussian-splatting-with-depth) trains real
forward-looking sonar on branch `gaussian-splatting-with-depth-FLS-2`, with its rasterizer fork
(https://github.com/QuintonQu/diff-gs-with-depth, branch `diff-gs-main-FLS-2`). That branch only reads the
AONeuS scene through hard-coded paths. This file adds a reader for our scenes.

Install, in the Z-Splat repo:
  1. Copy this file to scene/sonar_camera_reader.py
  2. At the end of scene/dataset_readers.py:
         from scene.sonar_camera_reader import readSonarCameraSceneInfo
         sceneLoadTypeCallbacks["SonarCamera"] = readSonarCameraSceneInfo
  3. In scene/__init__.py, as the first scene type check:
         if os.path.exists(os.path.join(args.source_path, "transforms.json")):
             scene_info = sceneLoadTypeCallbacks["SonarCamera"](args.source_path, args.eval)
         elif ...  (the existing checks)
  4. Z-Splat hard-codes the sonar depth histogram: 512 bins of view depth from 0.75 to 3.0 m, in
     cuda_rasterizer/forward.cu and backward.cu (z_index_max, z_view_min, z_view_max), rasterize_points.cu
     (D) and scene/cameras.py (znear, zfar). Z_BINS, Z_MIN, Z_MAX below must match, and Z_MAX should
     be the sonar's maximum range ("ranges" in transforms.json; the reader warns otherwise). The defaults
     fit the 3 m scenes: UXO 100lbs_bomb, opti-acoustic tank and marina_pier. Other UXO targets need
     Z_MAX = 2.0 (100lbs_bomb_(floor): 2.5 or 3.0), opti-acoustic marina_sea_wall Z_MAX = 4.99.
  5. Train with the default resolution (or -r 1): the sonar views must not be resized.

How the sonar becomes a Z-Splat view: Z-Splat renders a sonar as a pinhole camera (FovX = horizontal
aperture, FovY = vertical aperture) and, for every image column, a histogram of Gaussian density over view
depth z. Its loss compares that (Z_BINS, width) histogram with the measurement, each column normalised to
max 1. The measured polar image (range x bearing) is resampled onto the same grid: column u looks along
bearing atan(tan_u), and a return at range r in that column has depth z = r cos(bearing), assuming zero
elevation (the sonar does not measure it; the same far-field approximation Z-Splat makes).
"""

import json
import math
import os

import cv2
import numpy as np
from PIL import Image

# Must match the rasterizer, see step 4 above
Z_BINS, Z_MIN, Z_MAX = 512, 0.75, 3.0

USE_CAMERA = True        # set one to False for camera-only or sonar-only training
USE_SONAR = True
SONAR_WIDTH = None       # columns of the sonar view; None = one per beam
SONAR_MIN_RANGE = 0.3    # [m] near-field returns (ringing, mount) are cleared
SONAR_THRESHOLD = 0      # raw intensity below this is cleared, before per-column normalisation
LLFFHOLD = 8             # with --eval, every 8th camera frame is a test view; sonar views always train


def sonar_depth_histogram(polar: np.ndarray, ranges: np.ndarray, bearings: np.ndarray,
                          horizontal_fov: float, width: int) -> np.ndarray:
    """Resample a polar sonar image (range x bearing, both axes ascending) to Z-Splat's (Z_BINS, width)
    depth-per-column histogram, each column normalised to max 1 (all zero if the column is empty)."""
    tan_u = ((2 * np.arange(width) + 1) / width - 1) * math.tan(horizontal_fov / 2)
    theta = np.arctan(tan_u)
    z = Z_MIN + (np.arange(Z_BINS) + 0.5) * (Z_MAX - Z_MIN) / Z_BINS
    r = z[:, None] / np.cos(theta)[None, :]
    rows = np.interp(r, ranges, np.arange(len(ranges)), left=-1, right=-1)
    cols = np.broadcast_to(np.interp(theta, bearings, np.arange(len(bearings)), left=-1, right=-1), r.shape)
    img = polar.astype(np.float32)
    img[ranges < SONAR_MIN_RANGE] = 0
    img[img < SONAR_THRESHOLD] = 0
    hist = cv2.remap(img, cols.astype(np.float32), rows.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    peak = hist.max(axis=0, keepdims=True)
    return np.divide(hist, peak, out=np.zeros_like(hist), where=peak > 0).astype(np.float32)


def readSonarCameraSceneInfo(path, eval, llffhold=LLFFHOLD):
    from scene.dataset_readers import CameraInfo, SceneInfo, fetchPly, getNerfppNorm
    from utils.graphics_utils import focal2fov

    meta = json.load(open(os.path.join(path, "transforms.json")))
    son = meta["sonar"]
    ranges, bearings = np.array(son["ranges"]), np.array(son["bearings"])
    hfov, vfov = son["horizontal_fov"], son["vertical_fov"]
    if ranges[-1] < Z_MAX - 0.05 or ranges[0] > Z_MIN + 0.05:
        # Depth bins the sonar cannot see would be supervised as empty
        print(f"[ WARNING ] the sonar sees {ranges[0]:.2f}-{ranges[-1]:.2f} m but the depth histogram spans "
              f"{Z_MIN}-{Z_MAX} m: set Z_MAX = {math.floor(ranges[-1] * 100) / 100} (and Z_MIN >= {ranges[0]:.2f}) "
              f"here and in the rasterizer, see step 4 in sonar_camera_reader.py")
    width = SONAR_WIDTH or len(bearings)
    height = max(1, round(width * math.tan(vfov / 2) / math.tan(hfov / 2)))  # square pixels

    train, test = [], []
    for n, frame in enumerate(meta["frames"]):
        name = os.path.splitext(os.path.basename(frame["file_path"]))[0]
        if USE_CAMERA:
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1  # OpenGL -> COLMAP camera axes
            w2c = np.linalg.inv(c2w)
            image_path = os.path.join(path, frame["file_path"])
            cam = CameraInfo(uid=n, R=w2c[:3, :3].T, T=w2c[:3, 3],
                             FovY=focal2fov(meta["fl_y"], meta["h"]), FovX=focal2fov(meta["fl_x"], meta["w"]),
                             image=Image.open(image_path), image_path=image_path, image_name=name,
                             width=meta["w"], height=meta["h"], depth=[None, None], is_sonar=False)
            (test if eval and n % llffhold == 0 else train).append(cam)
        if USE_SONAR:
            w2s = np.linalg.inv(np.array(frame["sonar_transform_matrix"]))  # sonar axes are already x right, y down, z forward
            polar = np.load(os.path.join(path, frame["sonar_file_path"]))
            hist = sonar_depth_histogram(polar, ranges, bearings, hfov, width)
            train.append(CameraInfo(uid=n, R=w2s[:3, :3].T, T=w2s[:3, 3], FovY=vfov, FovX=hfov,
                                    image=Image.new("RGB", (width, height)), image_path=None,
                                    image_name=f"sonar_{name}", width=width, height=height,
                                    depth=[None, hist], is_sonar=True))

    print(f"Read {len(meta['frames'])} frames: {len(train)} train views, {len(test)} test views "
          f"(sonar view {width}x{height}, {Z_BINS} depth bins {Z_MIN}-{Z_MAX} m)")
    return SceneInfo(point_cloud=fetchPly(os.path.join(path, meta["ply_file_path"])),
                     train_cameras=train, test_cameras=test,
                     nerf_normalization=getNerfppNorm(train),
                     ply_path=os.path.join(path, meta["ply_file_path"]))
