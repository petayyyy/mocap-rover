from __future__ import annotations
import numpy as np

def rotate_twist_and_covariance(linear, angular, covariance, rotation):
    """Rotate twist and its 6x6 covariance from child frame into another frame."""
    r = np.asarray(rotation, dtype=float); v = np.asarray(linear, dtype=float); w = np.asarray(angular, dtype=float); c = np.asarray(covariance, dtype=float)
    if r.shape != (3,3) or v.shape != (3,) or w.shape != (3,) or c.shape != (6,6): raise ValueError("expected R[3,3], vectors[3], covariance[6,6]")
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(r), 1, atol=1e-8): raise ValueError("rotation is not proper orthonormal")
    t = np.zeros((6,6)); t[:3,:3] = r; t[3:,3:] = r
    return r @ v, r @ w, t @ c @ t.T
