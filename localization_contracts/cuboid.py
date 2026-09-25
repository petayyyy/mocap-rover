"""Known-size opponent localization from YOLO boxes, with explicit ambiguity.

Fits projected cuboid bounds. Yaw is only a nuisance parameter and is not
published as an observed heading. Dimensions are model configuration, not truth.
"""

import numpy as np
from scipy.optimize import least_squares


def localize_box(bbox, camera, dimensions=(1.0, 0.72, 0.483)):
    K = np.asarray(camera["K"], dtype=float).reshape(3, 3)
    R = np.asarray(camera["R_world_optical"])
    origin = np.array(camera["position_world"])
    bbox = np.asarray(bbox, dtype=float)
    center = (bbox[:2] + bbox[2:]) / 2
    import cv2

    ray = cv2.undistortPoints(
        center.reshape(1, 1, 2), K, np.asarray(camera.get("D", [0] * 5), dtype=float)
    )
    direction = R @ np.r_[ray.reshape(2), 1.0]
    if direction[2] >= -1e-6:
        raise ValueError("box center ray does not meet arena")
    midpoint = origin + direction * ((dimensions[2] / 2 - origin[2]) / direction[2])
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
        uv, _ = cv2.projectPoints(
            cp,
            np.zeros(3),
            np.zeros(3),
            K,
            np.asarray(camera.get("D", [0] * 5), dtype=float),
        )
        uv = uv[:, 0]
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
