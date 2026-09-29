"""Deterministic AprilTag observation geometry boundary for simulation/replay."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .contracts import Observation, SCHEMA_VERSION, FRAME_ARENA
from .marker_families import normalize_marker_family, marker_method

@dataclass(frozen=True)
class TagConfig:
    family: str = "tag36h11"
    ids: tuple = (0, 1)
    size_m: float = .40
    calibration_version: str = ""

@dataclass(frozen=True)
class Detection:
    camera_id: str; frame_seq: int; tag_id: int; corners: tuple
    capture_time_ns: int; receive_time_ns: int; processed_time_ns: int
    timestamp_uncertainty_ns: int = 0; exposure_duration_ns: int = 0

def marker_plane_z(placement, offset_z_m, base_z_nominal_m, inverted_base_z_m):
    """Arena height of one marker's surface for a level rover.

    Flipping the rover mirrors the marker offset through the base and changes
    the height the base itself rests at, so both follow the placement.  Shared
    so the ROI planner aims at the same planes the pose gate solves on.
    """
    if str(placement).lower() == "bottom":
        return float(inverted_base_z_m) - float(offset_z_m)
    return float(base_z_nominal_m) + float(offset_z_m)


class AprilTagObserver:
    def __init__(self, config: TagConfig, fx, fy, cx, cy, quality_min=.07):
        family = normalize_marker_family(config.family)
        if config.size_m <= 0: raise ValueError("invalid tag configuration")
        self.config = TagConfig(family, tuple(config.ids), config.size_m, config.calibration_version)
        self.fx, self.fy, self.cx, self.cy, self.quality_min = fx, fy, cx, cy, quality_min
    def observe(self, d: Detection):
        if d.tag_id not in self.config.ids or len(d.corners) != 4: return None
        pts = d.corners
        if any(len(p) != 2 or not all(math.isfinite(x) for x in p) for p in pts): return None
        side = sum(math.hypot(pts[(i+1)%4][0]-pts[i][0], pts[(i+1)%4][1]-pts[i][1]) for i in range(4))/4
        if side <= 1: return None
        z = self.fx * self.config.size_m / side
        x = (sum(p[0] for p in pts)/4-self.cx)*z/self.fx
        y = (sum(p[1] for p in pts)/4-self.cy)*z/self.fy
        yaw = math.atan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0])
        quality = min(1., side/100.)
        if quality < self.quality_min or z <= 0: return None
        return Observation(SCHEMA_VERSION, d.camera_id, d.frame_seq, f"{d.camera_id}:{d.frame_seq}:{d.tag_id}",
            "tag_rover", d.capture_time_ns, "sim", d.timestamp_uncertainty_ns, d.exposure_duration_ns,
            d.receive_time_ns, d.processed_time_ns, self.config.calibration_version, FRAME_ARENA,
            (x,y,z), (0.01,0.01,0.04), quality, marker_method(self.config.family), self.config.family, d.tag_id,
            # This boundary only estimates a planar centre/range from corners.  It
            # does not solve PnP, so advertising a 6-D pose here is unsafe.
            pose_6d_valid=False, attitude_state="unknown", pixel_features={"yaw_rad": yaw, "side_px": side}).validate()


class PnpAprilTagObserver(AprilTagObserver):
    """Corner-to-metric-pose boundary for a marker lying on a known plane.

    Two independent XY estimates are formed from the same corners:

    ``pnp``
        IPPE square PnP.  Six degrees of freedom, so its range is set by the
        apparent marker size and inherits every focal-length and distortion
        error.  A range error ``dz`` reaches XY as ``dz * tan(incidence)``,
        which is why the recorded tilted cameras showed a 0.85 m XY P95 from a
        0.48 m height bias.

    ``ray``
        The centre bearing intersected with the plane the marker actually lies
        on.  The rover is planar and its marker height is known to a
        centimetre, so this replaces the weakest quantity in the PnP solution
        with a measured constant.

    The two disagreeing is the strongest available signal that a solution is
    wrong, so both are always computed and their separation is a gate.  PnP is
    kept for yaw and for the base-height check; XY is published from the ray by
    default.
    """

    XY_SOURCES = ("ray", "pnp")

    def __init__(self, config, K, D, camera_pose, base_tag=None, quality_min=.07,
                 max_reprojection_px=2.0, max_planar_tilt_deg=40.0,
                 min_side_px=20.0, *, camera_model=None, image_size=None,
                 base_z_nominal_m=0.14, base_z_tolerance_m=0.25,
                 base_z_sigma_gate=6.0,
                 max_incidence_deg=65.0, pnp_ray_disagreement_m=0.35,
                 min_edge_distance_px=8.0, valid_radius_margin=0.85,
                 min_quad_aspect=0.35, xy_source="ray", quality_reference_m=0.02,
                 plane_normal_world=(0.0, 0.0, 1.0),
                 tag_placement=None, inverted_base_z_m=0.225,
                 sigma_px_corner=None, sigma_plane_m=0.02,
                 sigma_extrinsic_m=0.01, line_time_ns=0):
        super().__init__(config, float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]), quality_min)
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(D, dtype=np.float64).reshape(-1)
        if xy_source not in self.XY_SOURCES:
            raise ValueError(f"xy_source must be one of {self.XY_SOURCES}")
        self.xy_source = xy_source
        self.max_reprojection_px = float(max_reprojection_px)
        self.max_planar_tilt_rad = math.radians(float(max_planar_tilt_deg))
        self.min_side_px = float(min_side_px)
        self.base_z_nominal_m = float(base_z_nominal_m)
        self.base_z_tolerance_m = float(base_z_tolerance_m)
        self.base_z_sigma_gate = float(base_z_sigma_gate)
        self.max_incidence_rad = math.radians(float(max_incidence_deg))
        self.pnp_ray_disagreement_m = float(pnp_ray_disagreement_m)
        self.min_edge_distance_px = float(min_edge_distance_px)
        self.valid_radius_margin = float(valid_radius_margin)
        self.min_quad_aspect = float(min_quad_aspect)
        self.quality_reference_m = float(quality_reference_m)
        # The rover stands on a known plane, so its base Z axis must equal that
        # plane's normal.  Taking abs() of the alignment instead, as this class
        # used to, also accepts the mirrored reconstruction -- exactly the
        # branch the planarity gate exists to reject.
        normal = np.asarray(plane_normal_world, dtype=float).reshape(3)
        self.plane_normal_world = normal / max(np.linalg.norm(normal), 1e-12)
        # The rover carries a marker on top and another underneath.  Seeing
        # the underside one from a ceiling camera means the rover is upside
        # down, so the marker id states the orientation instead of the gate
        # having to accept both and hope.  That is strictly stronger than
        # taking abs() of the alignment: each id still has exactly one
        # expected sign, so a mirrored PnP branch is rejected as before.
        self.tag_placement = {int(k): str(v).lower()
                              for k, v in (tag_placement or {}).items()}
        self.inverted_base_z_m = float(inverted_base_z_m)
        self.sigma_plane_m = float(sigma_plane_m)
        self.sigma_extrinsic_m = float(sigma_extrinsic_m)
        # Corner localization noise as a function of apparent marker size.  A
        # 77 px marker refines to about a third of a pixel; a 22 px one does
        # not, and pretending otherwise is what made the old fixed 4 mm
        # covariance reject good handoffs.
        self.sigma_px_corner = tuple(sigma_px_corner or ((60.0, 0.30), (30.0, 0.50), (0.0, 0.80)))
        # Rolling shutter: row r of the frame is exposed r * line_time after
        # the frame stamp.  The IMX219 reads a line in about 9.5 us, so the
        # bottom of a 1232-row frame is 11.7 ms -- 13 cm at 11 m/s -- later
        # than the top.  Gazebo renders a global shutter, hence 0 by default.
        self.line_time_ns = float(line_time_ns)
        if self.line_time_ns < 0:
            raise ValueError("line_time_ns must be nonnegative")
        self.last_diagnostic = None
        self.R_arena_camera, self.t_arena_camera = self._transform(camera_pose)
        self.camera_model = camera_model or self._default_model(image_size)
        if isinstance(base_tag, dict) and 'rotation' not in base_tag:
            self.tag_transforms = {int(k): self._matrix(v) for k, v in base_tag.items()}
            self.T_base_tag = np.eye(4)
        else:
            self.T_base_tag = self._matrix(base_tag) if base_tag is not None else np.eye(4)
            self.tag_transforms = {i: self.T_base_tag for i in config.ids}

    def _default_model(self, image_size):
        from .camera_model import CameraModel
        if image_size is None:
            # Nothing declares the frame, so the radius gate cannot run; assume
            # a frame large enough to hold the principal point symmetrically.
            image_size = (int(round(2 * self.K[0, 2])) or 2, int(round(2 * self.K[1, 2])) or 2)
        return CameraModel(self.K, self.D, image_size)

    @staticmethod
    def _matrix(value):
        if value is None: return np.eye(4)
        if isinstance(value, dict):
            R = np.asarray(value.get("rotation"), dtype=np.float64).reshape(3, 3)
            t = np.asarray(value.get("translation"), dtype=np.float64).reshape(3)
        else:
            a = np.asarray(value, dtype=np.float64)
            if a.shape == (4, 4): return a
            raise ValueError("transform must be 4x4 or {rotation,translation}")
        out = np.eye(4); out[:3, :3] = R; out[:3, 3] = t; return out

    @classmethod
    def _transform(cls, value):
        T = cls._matrix(value)
        R, t = T[:3, :3], T[:3, 3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1, atol=1e-6):
            raise ValueError("camera rotation must be proper orthonormal")
        return R, t

    # ------------------------------------------------------------- geometry

    def base_orientation(self, tag_id):
        """+1 when this marker implies an upright rover, -1 when inverted."""
        return -1.0 if self.tag_placement.get(int(tag_id)) == "bottom" else 1.0

    def nominal_base_z(self, tag_id):
        """Expected base height for the orientation this marker implies.

        An upside-down rover rests on whatever was highest, so its base link
        sits roughly a top-plate offset above the floor rather than a wheel
        radius.  The default is an estimate; measure it if flips matter.
        """
        return (self.base_z_nominal_m if self.base_orientation(tag_id) > 0
                else self.inverted_base_z_m)

    def tag_plane_z(self, tag_id):
        """Arena height of the marker surface for a level rover.

        Flipping the rover mirrors the marker offset through the base, so the
        sign of the offset follows the orientation.
        """
        return marker_plane_z(self.tag_placement.get(int(tag_id), "top"),
                              float(self.tag_transforms[tag_id][2, 3]),
                              self.base_z_nominal_m, self.inverted_base_z_m)

    def corner_sigma_px(self, side_px):
        for threshold, sigma in self.sigma_px_corner:
            if side_px >= threshold:
                return sigma
        return self.sigma_px_corner[-1][1]

    def centre_ray(self, corners_px):
        """Normalized bearing of the marker centre from its four corners.

        The centre of a planar square is where its diagonals cross, and a
        perspective projection keeps that true, so the crossing is taken on
        the undistorted rays.  The pixel mean of the corners is not the
        centre: perspective and, far more, a fisheye lens pull it towards the
        image centre -- 1.2 cm of XY at 40 degrees on the IMX219 lens.
        """
        rays = self.camera_model.undistort(corners_px)
        h = np.hstack([rays, np.ones((4, 1))])
        crossing = np.cross(np.cross(h[0], h[2]), np.cross(h[1], h[3]))
        if abs(crossing[2]) < 1e-12 or not np.isfinite(crossing).all():
            return rays.mean(axis=0)
        return crossing[:2] / crossing[2]

    def _ray_plane(self, ray, plane_z):
        """Intersect the normalized bearing ``ray`` with ``z = plane_z``."""
        direction = self.R_arena_camera @ np.array([ray[0], ray[1], 1.0])
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            return None, None
        direction = direction / norm
        drop = plane_z - self.t_arena_camera[2]
        # A ray parallel to the plane never meets it, and a negative parameter
        # means the plane is behind the camera.
        if abs(direction[2]) < 1e-3 or drop / direction[2] <= 0:
            return None, None
        point = self.t_arena_camera + direction * (drop / direction[2])
        incidence = math.acos(min(1.0, abs(direction[2])))
        return point, incidence

    def _xy_covariance(self, point_xy, incidence, side_px, plane_z):
        """Anisotropic 2x2 block in arena axes.

        Bearing noise maps to ``h*sigma_theta/cos(incidence)`` across the ray
        and ``h*sigma_theta/cos^2(incidence)`` along it; the unknown marker
        height adds ``sigma_plane*tan(incidence)`` along the ray only.
        """
        focal = float(min(self.K[0, 0], self.K[1, 1]))
        sigma_theta = self.corner_sigma_px(side_px) / focal / 2.0
        height = abs(self.t_arena_camera[2] - plane_z)
        cos_i = max(math.cos(incidence), 1e-3)
        sigma_radial = height * sigma_theta / cos_i**2 + self.sigma_plane_m * math.tan(incidence)
        sigma_tangential = height * sigma_theta / cos_i
        offset = np.asarray(point_xy, dtype=float) - self.t_arena_camera[:2]
        norm = np.linalg.norm(offset)
        if norm < 1e-6:
            radial = np.array([1.0, 0.0])
        else:
            radial = offset / norm
        tangential = np.array([-radial[1], radial[0]])
        basis = np.column_stack([radial, tangential])
        block = basis @ np.diag([sigma_radial**2, sigma_tangential**2]) @ basis.T
        return block + np.eye(2) * self.sigma_extrinsic_m**2

    def range_sigma_m(self, side_px, range_m):
        """1-sigma PnP range precision from apparent marker size.

        ``z = f * S / side``, so ``dz = z * dside / side``: a marker half as
        wide in pixels is twice as uncertain in depth.  A fixed base-height
        tolerance would either reject every distant marker or accept the
        1.08 m reconstruction the recorded run produced.
        """
        return float(range_m) * self.corner_sigma_px(side_px) / max(float(side_px), 1.0)

    def _reject(self, reason, **fields):
        self.last_diagnostic = {"accepted": False, "reason": reason, **fields}
        return None

    # -------------------------------------------------------------- observe

    def observe(self, d: Detection):
        self.last_diagnostic = {"accepted": False, "reason": "invalid_detection"}
        if d.tag_id not in self.config.ids or len(d.corners) != 4: return None
        if d.tag_id not in self.tag_transforms: return None
        image = np.asarray(d.corners, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(image).all(): return None
        sides = np.asarray([np.linalg.norm(image[(i + 1) % 4] - image[i]) for i in range(4)])
        side = float(np.mean(sides))
        aspect = float(np.min(sides) / max(np.max(sides), 1e-9))
        if side < self.min_side_px:
            return self._reject("too_small", side_px=side)
        if aspect < self.min_quad_aspect:
            return self._reject("degenerate_quad", side_px=side, quad_aspect=aspect)

        # A corner outside the model's domain has no meaningful undistorted
        # ray, and OpenCV answers with a diverged one rather than an error.
        inside = self.camera_model.inside_valid_radius(image, self.valid_radius_margin)
        if not bool(np.all(inside)):
            return self._reject("outside_model_domain", side_px=side)
        edge = float(np.min(self.camera_model.edge_distance_px(image)))
        if edge < self.min_edge_distance_px:
            return self._reject("frame_edge", side_px=side, edge_distance_px=edge)

        plane_z = self.tag_plane_z(d.tag_id)
        sign = self.base_orientation(d.tag_id)
        nominal_base_z = self.nominal_base_z(d.tag_id)
        ray_point, incidence = self._ray_plane(self.centre_ray(image), plane_z)
        if ray_point is None:
            return self._reject("ray_misses_plane", side_px=side)
        if incidence > self.max_incidence_rad:
            return self._reject("incidence", side_px=side,
                                incidence_deg=math.degrees(incidence))

        pnp = self._solve_pnp(image, d.tag_id, side)
        if pnp is None:
            return None
        T_arena_base, reproj, tilt, candidates = pnp

        base_z = float(T_arena_base[2, 3])
        ray_base = np.array([
            ray_point[0], ray_point[1],
            plane_z - sign * float(self.tag_transforms[d.tag_id][2, 3])])
        disagreement = float(np.linalg.norm(T_arena_base[:2, 3] - ray_base[:2]))
        diagnostic = {
            "side_px": side, "quad_aspect": aspect, "edge_distance_px": edge,
            "reprojection_error_px": reproj, "planar_tilt_deg": math.degrees(tilt),
            "incidence_deg": math.degrees(incidence), "candidate_count": candidates,
            "pnp_base_z_m": base_z, "pnp_xy": [float(v) for v in T_arena_base[:2, 3]],
            "placement": self.tag_placement.get(int(d.tag_id), "top"),
            "base_inverted": sign < 0,
            "ray_xy": [float(v) for v in ray_base[:2]],
            "pnp_ray_disagreement_m": disagreement,
        }
        # These are gross-error gates, not precision tests.  Planar PnP depth
        # is weakly observable near fronto-parallel: rendered markers with a
        # 6 mm ray error routinely show a 0.1 m base-height error at 1 px
        # reprojection, so a tight bound throws away good measurements to
        # punish the one quantity the ray estimate does not use.  The bound is
        # set to catch a wrong branch or a misidentified plane -- the recorded
        # failure sat 0.94 m off -- and scales with the depth precision the
        # apparent marker size can actually deliver.
        range_m = float(np.linalg.norm(ray_point - self.t_arena_camera))
        sigma_range = self.range_sigma_m(side, range_m)
        cos_i, sin_i = math.cos(incidence), math.sin(incidence)
        base_z_limit = max(self.base_z_tolerance_m,
                           self.base_z_sigma_gate * sigma_range * cos_i)
        disagreement_limit = max(self.pnp_ray_disagreement_m,
                                 self.base_z_sigma_gate * sigma_range * sin_i)
        diagnostic.update(range_sigma_m=sigma_range, base_z_limit_m=base_z_limit,
                          pnp_ray_limit_m=disagreement_limit)
        if abs(base_z - nominal_base_z) > base_z_limit:
            return self._reject("base_height", **diagnostic)
        if disagreement > disagreement_limit:
            return self._reject("pnp_ray_disagreement", **diagnostic)

        position = ray_base if self.xy_source == "ray" else T_arena_base[:3, 3]
        yaw = math.atan2(T_arena_base[1, 0], T_arena_base[0, 0])
        block = self._xy_covariance(position[:2], incidence, side, plane_z)
        sigma_yaw = min(math.radians(30.0), max(
            math.radians(0.5),
            math.sqrt(2.0) * self.corner_sigma_px(side) / max(side, 1.0)
            / max(math.cos(incidence), 1e-3),
        ))
        covariance = (
            float(block[0, 0]), float(block[0, 1]), 0.0,
            float(block[1, 0]), float(block[1, 1]), 0.0,
            0.0, 0.0, float(self.sigma_plane_m**2),
        )

        # Quality is a monotone reading of the covariance actually published,
        # not an independent score.  The old product included a planarity term
        # that collapsed on the weakly observable PnP branch tilt, which is
        # unrelated to how well the ray fixes XY: rendered views with a 6 mm
        # ray error scored 0.02 and were thrown away.
        sigma_xy = math.sqrt(max(block[0, 0], block[1, 1]))
        quality = max(0.0, min(1.0, self.quality_reference_m
                               / (self.quality_reference_m + sigma_xy)))
        diagnostic["quality"] = quality
        if quality < self.quality_min:
            return self._reject("quality", **diagnostic)

        row = float(image[:, 1].mean())
        exposure_ns = int(d.capture_time_ns + round(row * self.line_time_ns))
        self.last_diagnostic = {"accepted": True, "reason": "accepted", **diagnostic}
        return Observation(
            SCHEMA_VERSION, d.camera_id, d.frame_seq,
            f"{d.camera_id}:{d.frame_seq}:{d.tag_id}", "tag_rover",
            d.capture_time_ns, "sim", d.timestamp_uncertainty_ns,
            d.exposure_duration_ns, d.receive_time_ns, d.processed_time_ns,
            self.config.calibration_version, FRAME_ARENA,
            tuple(float(x) for x in position), covariance, quality,
            marker_method(self.config.family, pnp=True), self.config.family,
            d.tag_id, pose_6d_valid=True, attitude_state="valid",
            pixel_features={
                "yaw_rad": yaw, "yaw_sigma_rad": sigma_yaw,
                "xy_sigma_m": float(math.sqrt(max(block[0, 0], block[1, 1]))),
                "xy_source": self.xy_source,
                # When the marker was actually exposed; the filter uses this,
                # not the frame stamp and never the arrival time.
                "exposure_time_ns": exposure_ns, "exposure_row_px": row,
                "line_time_ns": self.line_time_ns,
                "rotation_arena_base": T_arena_base[:3, :3].tolist(),
                "pnp_candidate_count": candidates,
                **diagnostic,
            },
        ).validate()

    def _solve_pnp(self, image, tag_id, side):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("PnpAprilTagObserver requires OpenCV") from exc
        s = self.config.size_m / 2.0
        # tag frame: x right, y up, z outward; image order is top-left first.
        object_points = np.asarray([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        # Solve in the tag frame turned half a turn about its x axis.  A level
        # marker seen from the ceiling faces the camera, so camera-from-tag is
        # a rotation by pi -- exactly where the Rodrigues vector is singular.
        # There IPPE returned NaN or a branch 2-4 px off on noise-free
        # corners, and LM refinement settled in the mirrored minimum, so the
        # reprojection gate refused 140 of 252 exact views of a level marker.
        # In the turned frame the same pose is near the identity.  Turning the
        # frame reverses the corner order IPPE_SQUARE expects; the pose is
        # turned back below, so nothing downstream sees the change.
        turn = np.diag([1.0, -1.0, -1.0, 1.0])
        order = [3, 2, 1, 0]
        distortion = self.camera_model.solve_pnp_distortion()
        if self.camera_model.model == "fisheye":
            # solvePnP has no fisheye path: undistort to ideal pinhole rays and
            # solve against an identity camera instead.
            rays = self.camera_model.undistort(image)
            image_pnp = rays * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]]
        else:
            image_pnp = image
        image_pnp = np.ascontiguousarray(image_pnp[order])
        try:
            ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                object_points, image_pnp, self.K, distortion,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            starts = [(r, t) for r, t in zip(rvecs, tvecs) if ok
                      and np.isfinite(r).all() and np.isfinite(t).all()]
            if not starts:
                # IPPE's closed form degenerates on a few exactly symmetric
                # views; SQPnP has no such case and gives the global minimum.
                ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    object_points, image_pnp, self.K, distortion,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                starts = [(r, t) for r, t in zip(rvecs, tvecs) if ok
                          and np.isfinite(r).all() and np.isfinite(t).all()]
        except cv2.error:
            return self._reject("opencv_error", side_px=side)
        if not starts:
            return self._reject("pnp_failed", side_px=side)
        T_arena_camera = np.eye(4)
        T_arena_camera[:3, :3] = self.R_arena_camera
        T_arena_camera[:3, 3] = self.t_arena_camera
        tag_points = np.asarray([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
        candidates = []
        for rvec, tvec in starts:
            rvec, tvec = cv2.solvePnPRefineLM(
                object_points, image_pnp, self.K, distortion, rvec.copy(), tvec.copy()
            )
            if not np.isfinite(tvec).all() or float(tvec[2, 0]) <= 0:
                continue
            R_camera_turned, _ = cv2.Rodrigues(rvec)
            T_camera_tag = np.eye(4)
            T_camera_tag[:3, :3] = R_camera_turned
            T_camera_tag[:3, 3] = tvec.reshape(3)
            T_camera_tag = T_camera_tag @ turn
            # Residual in the camera's own pixels, through its own model, so
            # the 2 px gate means the same thing at the centre of a fisheye
            # frame and at its edge, where undistorted pixels are stretched.
            points_camera = tag_points @ T_camera_tag[:3, :3].T + T_camera_tag[:3, 3]
            if (points_camera[:, 2] <= 1e-9).any():
                continue
            projected = self.camera_model.project(points_camera)
            reproj = float(np.sqrt(np.mean(np.sum((projected - image) ** 2, axis=1))))
            # Refinement started from the ill-conditioned IPPE branch can walk
            # away entirely and land at a near-zero tilt with a huge residual.
            # Ordering by tilt first let those win and then killed the frame on
            # reprojection, which is what removed 4116 otherwise good frames.
            if reproj > self.max_reprojection_px:
                continue
            T_arena_base = (T_arena_camera @ T_camera_tag
                            @ np.linalg.inv(self.tag_transforms[tag_id]))
            alignment = float(np.clip(
                np.dot(T_arena_base[:3, 2], self.plane_normal_world)
                * self.base_orientation(tag_id), -1.0, 1.0))
            tilt = math.acos(alignment)
            candidates.append((tilt, reproj, T_arena_base))
        if not candidates:
            return self._reject("no_physical_solution", side_px=side)
        tilt, reproj, T_arena_base = min(candidates, key=lambda item: (item[0], item[1]))
        if tilt > self.max_planar_tilt_rad:
            return self._reject("nonplanar_pose", side_px=side,
                                reprojection_error_px=reproj,
                                planar_tilt_deg=math.degrees(tilt),
                                candidate_count=len(candidates))
        return T_arena_base, reproj, tilt, len(candidates)
