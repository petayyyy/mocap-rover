"""Known-size opponent localization from YOLO boxes, with explicit ambiguity.

Fits projected cuboid bounds. Yaw is only a nuisance parameter and is not
published as an observed heading. Dimensions are model configuration, not truth.
"""

import numpy as np
from scipy.optimize import least_squares


def localize_box(bbox, camera, dimensions=(1.0, 0.72, 0.483), camera_model=None):
    """Fit a known cuboid to an axis-aligned pixel box; returns base-centre XY.

    Rays and projections go through ``CameraModel`` so the box is read with
    the camera's own lens.  ``cv2.undistortPoints`` with the calibration's D
    read an ideal fisheye (D = [0, 0, 0, 0]) as a pinhole and put the box
    centre up to a metre off near the edge of an IMX219 frame.
    """
    from .camera_model import CameraModel
    from .ray_plane import pixel_rays, ray_plane

    if camera_model is None:
        K = np.asarray(camera["K"], dtype=float).reshape(3, 3)
        size = camera.get("image_size") or (int(round(2 * K[0, 2])), int(round(2 * K[1, 2])))
        camera_model = CameraModel(K, camera.get("D", [0.0] * 5), size,
                                   model=camera.get("distortion_model"))
    model = camera_model
    R = np.asarray(camera["R_world_optical"], dtype=float)
    origin = np.asarray(camera["position_world"], dtype=float)
    bbox = np.asarray(bbox, dtype=float)
    center = (bbox[:2] + bbox[2:]) / 2
    midpoint, _ = ray_plane(pixel_rays(model, center)[0], R, origin, dimensions[2] / 2)
    if midpoint is None:
        raise ValueError("box center ray does not meet arena")
    pts = np.array(
        [
            [x, y, z]
            for x in (-dimensions[0] / 2, dimensions[0] / 2)
            for y in (-dimensions[1] / 2, dimensions[1] / 2)
            for z in (0, dimensions[2])
        ]
    )

    def residual(p):
        c, s = np.cos(p[2]), np.sin(p[2])
        Q = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        cp = (pts @ Q.T + np.r_[p[:2], 0.0] - origin) @ R
        if (cp[:, 2] <= 1e-6).any():
            return np.full(4, 1e4)
        uv = model.project(cp)
        return np.r_[uv.min(axis=0), uv.max(axis=0)] - bbox

    fits = [
        least_squares(residual, [*midpoint[:2], yaw], max_nfev=20)
        for yaw in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    ]
    fit = min(fits, key=lambda f: np.dot(f.fun, f.fun))
    return {
        "position_m": (*map(float, fit.x[:2]), 0.14),
        "box_fit_rms_px": float(np.sqrt(np.mean(fit.fun**2))),
        "yaw_valid": False,
    }
