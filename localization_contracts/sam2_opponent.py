"""SAM2 as a second opinion on the opponent: a mask per camera from video memory.

The background path (``opponent_camera.py``) finds the opponent as whatever
differs from the empty arena.  It is blind exactly where two bodies touch
(one blob), where a body stands still long enough to be learned, and after a
loss it can only come back through the operator or a blob far from
tag_rover.  SAM2 (``facebookresearch/sam2``, used as published, never trained
or fine-tuned here) follows one object through a video from one box prompt
and does not look at the background at all.

Pieces, from the GPU outwards:

``Sam2Engine``      the model (lazy ``torch``/``sam2`` import), one image
                    encoder pass for every camera of an instant, then one
                    memory-conditioned decoder step per camera.
``Sam2Tracker``     one camera's video memory: the prompt frame plus a window
                    of the most recent frames, so time and VRAM do not grow
                    with the length of the run.  A new prompt resets it.
``Sam2Opponent``    the policy, free of torch: when to prompt (the operator's
                    box at the start; the filter's predicted body when the
                    mask is unsure, empty, broken or was refused), the mask
                    -> ``SilhouetteObserver.measure`` -> an opponent reading
                    with its own sigma, and the identity guards (gate around
                    the opponent's prediction, nearer to it than to
                    tag_rover's, not on tag_rover's marker).

Nothing here reads truth.  The prompts come from the operator's rectangle
(the replay's one truth read, made before the match) and the filter.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .foreground import SilhouetteObserver
from .opponent_camera import body_hull_px

# Defaults of the published SAM2.1 tiny model, relative to the sam2 package.
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
DEFAULT_CHECKPOINT = "models/sam2/sam2.1_hiera_tiny.pt"
CONFIG_BY_CHECKPOINT = {
    "sam2.1_hiera_tiny.pt": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small.pt": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus.pt": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large.pt": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

# Rejection reasons, counted in the replay's observations.jsonl.
LOW_SCORE = "low_score"
EMPTY_MASK = "empty_mask"
FRAGMENTED = "fragmented_mask"
OUT_OF_GATE = "out_of_gate"
NEARER_TAG = "nearer_tag_rover"
ON_TAG_MARKER = "on_tag_marker"
NEAR_TAG_LOST = "near_tag_rover_while_lost"


def sam2_available():
    """True when ``torch`` with CUDA and ``sam2`` import; never raises."""
    try:
        import torch  # noqa: F401
        import sam2  # noqa: F401
    except Exception:
        return False
    return True


def config_for_checkpoint(checkpoint):
    name = str(checkpoint).replace("\\", "/").rsplit("/", 1)[-1]
    return CONFIG_BY_CHECKPOINT.get(name, DEFAULT_CONFIG)


@dataclass
class StepOutput:
    """What the model says about one camera's frame."""
    mask: np.ndarray            # bool, frame size
    score: float                # object presence probability, 0..1
    prompted: bool
    gpu_ms: float = 0.0


class Sam2Tracker:
    """One camera's SAM2 memory: the prompt frame and a window of recent frames."""

    def __init__(self, camera_id, window):
        self.camera_id = camera_id
        self.window = int(window)
        self.reset()

    def reset(self):
        self.output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        self.frame_idx = -1
        self.prompted = False

    def prune(self):
        """Keep only the last ``window`` non-prompt frames."""
        old = [k for k in self.output_dict["non_cond_frame_outputs"]
               if k <= self.frame_idx - self.window]
        for k in old:
            del self.output_dict["non_cond_frame_outputs"][k]

    @property
    def stored_frames(self):
        return (len(self.output_dict["cond_frame_outputs"])
                + len(self.output_dict["non_cond_frame_outputs"]))


class Sam2Engine:
    """SAM2 on the GPU; ``step`` takes every camera of one instant at once.

    ``torch`` and ``sam2`` are imported here, not at module import, so the
    package and its tests run without them.
    """

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, config=None, device="cuda",
                 dtype="bfloat16", window=None, image_size=None):
        import torch
        from sam2.build_sam import build_sam2
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("SAM2 asked for CUDA, but torch sees no CUDA device")
        self.torch = torch
        self.device = torch.device(device)
        self.config = config or config_for_checkpoint(checkpoint)
        self.checkpoint = str(checkpoint)
        # apply_postprocessing=False: the published hole filling needs the
        # optional CUDA extension; a mask is read as a blob anyway.
        # image_size: the square the frame is resized to (published 1024).  The
        # small stream is 640x480, so 1024 upsamples; a smaller square is an
        # inference setting, not a change of the weights.
        overrides = [] if not image_size else [f"++model.image_size={int(image_size)}"]
        self.model = build_sam2(self.config, self.checkpoint, device=str(self.device),
                                apply_postprocessing=False, hydra_overrides_extra=overrides)
        self.dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                      "float32": torch.float32}[dtype]
        self.dtype_name = dtype
        self.image_size = int(self.model.image_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        # Memory frames the model ever looks back at (mask memories and object
        # pointers); older ones are dropped, so the state is bounded.
        need = max(int(self.model.num_maskmem), int(self.model.max_obj_ptrs_in_encoder))
        self.window = int(window or need)
        self.calls = 0

    def new_tracker(self, camera_id):
        return Sam2Tracker(camera_id, self.window)

    def device_name(self):
        torch = self.torch
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(self.device)
        return str(self.device)

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory_mb(self):
        if self.device.type != "cuda":
            return None
        return self.torch.cuda.max_memory_allocated(self.device) / 2 ** 20

    def _batch(self, images):
        torch = self.torch
        F = torch.nn.functional
        tensors = []
        for image in images:
            array = np.ascontiguousarray(image)
            if array.ndim == 2:
                array = np.repeat(array[..., None], 3, axis=2)
            tensors.append(torch.from_numpy(array).to(self.device, non_blocking=True))
        x = torch.stack(tensors).permute(0, 3, 1, 2).float() / 255.0
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear",
                          align_corners=False, antialias=True)
        return (x - self.mean) / self.std

    def step(self, items):
        """``items``: list of (tracker, image HxWx3 uint8, box xyxy px or None).

        A box resets the tracker and makes this frame its prompt.  Returns a
        ``StepOutput`` per item; ``gpu_ms`` is the wall time of the whole call
        (every camera), synchronised with the device.
        """
        torch = self.torch
        F = torch.nn.functional
        if not items:
            return []
        begin = time.perf_counter()
        model = self.model
        outputs = []
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=self.dtype,
                                                    enabled=self.dtype != torch.float32):
            batch = self._batch([image for _, image, _ in items])
            backbone = model.forward_image(batch)
            _, feats, pos, sizes = model._prepare_backbone_features(backbone)
            for i, (tracker, image, box) in enumerate(items):
                height, width = image.shape[:2]
                if box is not None:
                    tracker.reset()
                elif not tracker.prompted:
                    raise ValueError(f"{tracker.camera_id}: no prompt yet")
                tracker.frame_idx += 1
                frame_idx = tracker.frame_idx
                point_inputs = None
                if box is not None:
                    scale = torch.tensor([self.image_size / width, self.image_size / height],
                                         device=self.device)
                    coords = torch.tensor(np.asarray(box, dtype=np.float32).reshape(1, 2, 2),
                                          device=self.device) * scale
                    point_inputs = {"point_coords": coords,
                                    "point_labels": torch.tensor([[2, 3]], dtype=torch.int32,
                                                                 device=self.device)}
                out = model.track_step(
                    frame_idx=frame_idx, is_init_cond_frame=box is not None,
                    current_vision_feats=[f[:, i:i + 1] for f in feats],
                    current_vision_pos_embeds=[p[:, i:i + 1] for p in pos],
                    feat_sizes=sizes, point_inputs=point_inputs, mask_inputs=None,
                    output_dict=tracker.output_dict, num_frames=1 << 30,
                    track_in_reverse=False, run_mem_encoder=True)
                compact = {"maskmem_features": out["maskmem_features"],
                           "maskmem_pos_enc": out["maskmem_pos_enc"],
                           "pred_masks": out["pred_masks"], "obj_ptr": out["obj_ptr"],
                           "object_score_logits": out["object_score_logits"]}
                key = "cond_frame_outputs" if box is not None else "non_cond_frame_outputs"
                tracker.output_dict[key][frame_idx] = compact
                tracker.prompted = True
                tracker.prune()
                logits = F.interpolate(out["pred_masks"].float(), size=(height, width),
                                       mode="bilinear", align_corners=False)
                score = torch.sigmoid(out["object_score_logits"].float()).reshape(-1)[0]
                outputs.append((logits[0, 0] > 0.0, score, box is not None))
            masks = [m.cpu().numpy() for m, _, _ in outputs]
            scores = [float(s) for _, s, _ in outputs]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        ms = (time.perf_counter() - begin) * 1e3
        self.calls += 1
        return [StepOutput(mask, score, prompted, ms)
                for mask, score, (_, _, prompted) in zip(masks, scores, outputs)]


@dataclass
class Sam2Reading:
    """One accepted SAM2 measurement of the opponent, arena metres."""
    camera_id: str
    x: float
    y: float
    covariance_xy: tuple
    score: float
    incidence_rad: float
    pixels: int
    prompted: bool
    detail: dict = field(default_factory=dict)


@dataclass
class Sam2Refusal:
    camera_id: str
    reason: str
    x: float | None = None
    y: float | None = None
    score: float | None = None
    prompted: bool = False


class Sam2Opponent:
    """Policy around the engine for the cameras of one rig; no torch here.

    ``cameras``: {camera_id: (camera_model of the small frame, R_world_optical,
    position_world)}.  ``marker``: (offset_x, offset_y, height, side) of
    tag_rover's top marker in its body frame.
    """

    def __init__(self, engine, cameras, *, size_m=(0.9, 0.52, 0.483),
                 marker=(0.0, 0.0, 0.3654, 0.4), score_threshold=0.5, min_mask_px=150,
                 min_main_component=0.8, gate_m=0.8, marker_overlap_max=0.30,
                 clear_of_tag_m=1.0, sigma_scale=1.0, along_sigma_scale=0.20,
                 extent_plane_z=None, prompt_margin_px=8, stale_ns=500_000_000):
        self.engine = engine
        self.size = tuple(float(v) for v in size_m)
        self.marker = tuple(float(v) for v in marker)
        self.score_threshold = float(score_threshold)
        self.min_mask_px = int(min_mask_px)
        self.min_main_component = float(min_main_component)
        self.gate_m = float(gate_m)
        self.marker_overlap_max = float(marker_overlap_max)
        self.clear_of_tag_m = float(clear_of_tag_m)
        self.sigma_scale = float(sigma_scale)
        self.prompt_margin_px = float(prompt_margin_px)
        self.stale_ns = int(stale_ns)
        self.cameras = {}
        for cid, (model, R, C) in cameras.items():
            R = np.asarray(R, dtype=float)
            C = np.asarray(C, dtype=float)
            observer = SilhouetteObserver(model, R, C, size_m=self.size, estimator="extent",
                                          extent_plane_z=extent_plane_z,
                                          along_sigma_scale=along_sigma_scale)
            self.cameras[cid] = {"model": model, "R": R, "C": C, "observer": observer,
                                 "tracker": engine.new_tracker(cid), "need_prompt": True,
                                 "last_ns": None}
        self.prompts = {"operator": 0, "prediction": 0}

    # ------------------------------------------------------------ geometry

    def predicted_box(self, cid, pose):
        """Pixel box (x0, y0, x1, y1) of the opponent body at ``pose`` = (x, y, yaw)."""
        cam = self.cameras[cid]
        hull = body_hull_px(cam["model"], cam["R"], cam["C"], pose[:2], pose[2], self.size)
        if hull is None or not np.isfinite(hull).all():
            return None
        width, height = cam["model"].width, cam["model"].height
        m = self.prompt_margin_px
        x0, y0 = hull.min(axis=0) - m
        x1, y1 = hull.max(axis=0) + m
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(width - 1.0, x1), min(height - 1.0, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return (float(x0), float(y0), float(x1), float(y1))

    def marker_rect(self, cid, tag_pose):
        """Pixel rectangle (x0, y0, x1, y1) around tag_rover's top marker, or None."""
        cam = self.cameras[cid]
        ox, oy, z, side = self.marker
        x, y, yaw = tag_pose[:3]
        c, s = math.cos(yaw), math.sin(yaw)
        centre = np.array([x + c * ox - s * oy, y + s * ox + c * oy])
        half = side / 2.0
        corners = np.array([[a * half, b * half] for a in (-1, 1) for b in (-1, 1)])
        corners = corners @ np.array([[c, s], [-s, c]]) + centre
        world = np.column_stack([corners, np.full(4, z)])
        optical = (world - cam["C"]) @ cam["R"]
        if (optical[:, 2] <= 1e-6).any():
            return None
        uv = cam["model"].project(optical)
        if not np.isfinite(uv).all():
            return None
        x0, y0 = uv.min(axis=0)
        x1, y1 = uv.max(axis=0)
        return (float(x0), float(y0), float(x1), float(y1))

    @staticmethod
    def rect_covered(mask, rect):
        """Fraction of ``rect`` (x0, y0, x1, y1) covered by ``mask``."""
        h, w = mask.shape
        x0, y0, x1, y1 = rect
        area = max((x1 - x0) * (y1 - y0), 1e-9)
        ix0, iy0 = int(max(0, math.floor(x0))), int(max(0, math.floor(y0)))
        ix1, iy1 = int(min(w, math.ceil(x1))), int(min(h, math.ceil(y1)))
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        return float(mask[iy0:iy1, ix0:ix1].sum()) / area

    # -------------------------------------------------------------- policy

    def prompt_for(self, cid, now_ns, operator_box=None, prediction=None, may_prompt=False):
        """Box prompt for this frame, or None (keep the memory), or False (skip)."""
        cam = self.cameras[cid]
        tracker = cam["tracker"]
        if cam["last_ns"] is not None and now_ns - cam["last_ns"] > self.stale_ns:
            # Not looked at for a while (another camera had the slot): the
            # memory describes a scene long gone.
            tracker.reset()
            cam["need_prompt"] = True
        if operator_box is not None:
            self.prompts["operator"] += 1
            return tuple(float(v) for v in operator_box)
        if (cam["need_prompt"] or not tracker.prompted) and may_prompt and prediction is not None:
            box = self.predicted_box(cid, prediction)
            if box is not None:
                self.prompts["prediction"] += 1
                return box
        if not tracker.prompted:
            return False
        return None

    def process(self, frames, now_ns, *, opp_prediction=None, opp_gate_m=None,
                tag_prediction=None, may_prompt=False, operator_boxes=None):
        """One instant.  ``frames``: [(camera_id, small image)].

        ``opp_prediction`` (x, y, yaw) of the opponent track or None when it is
        lost; ``may_prompt``: the track is good enough to aim a prompt with.
        Returns (readings, refusals, step outputs by camera, gpu ms or None).
        """
        operator_boxes = operator_boxes or {}
        items, order = [], []
        for cid, image in frames:
            prompt = self.prompt_for(cid, now_ns, operator_boxes.get(cid), opp_prediction,
                                     may_prompt)
            if prompt is False:
                continue
            cam = self.cameras[cid]
            items.append((cam["tracker"], image, prompt))
            order.append(cid)
        if not items:
            return [], [], {}, None
        steps = self.engine.step(items)
        readings, refusals, by_camera = [], [], {}
        for cid, step in zip(order, steps):
            cam = self.cameras[cid]
            cam["last_ns"] = now_ns
            by_camera[cid] = step
            result = self.read(cid, step, opp_prediction, opp_gate_m, tag_prediction)
            if isinstance(result, Sam2Refusal):
                cam["need_prompt"] = True
                refusals.append(result)
            else:
                cam["need_prompt"] = False
                readings.append(result)
        return readings, refusals, by_camera, (steps[0].gpu_ms if steps else None)

    def read(self, cid, step, opp_prediction, opp_gate_m, tag_prediction):
        """Mask -> reading through the silhouette observer and the identity guards."""
        cam = self.cameras[cid]
        mask = np.asarray(step.mask, dtype=bool)
        if step.score < self.score_threshold:
            return Sam2Refusal(cid, LOW_SCORE, score=step.score, prompted=step.prompted)
        area = int(mask.sum())
        if area < self.min_mask_px:
            return Sam2Refusal(cid, EMPTY_MASK, score=step.score, prompted=step.prompted)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        if count > 2:
            main = int(stats[1:, cv2.CC_STAT_AREA].max())
            if main < self.min_main_component * area:
                return Sam2Refusal(cid, FRAGMENTED, score=step.score, prompted=step.prompted)
            mask = labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
        h, w = mask.shape
        observer = cam["observer"]
        found = observer.measure(mask, (0, 0, w, h), None, gate_m=1e9)
        if found is None:
            return Sam2Refusal(cid, f"silhouette:{observer.last_reason}", score=step.score,
                               prompted=step.prompted)
        x, y = found.x, found.y
        if opp_prediction is not None:
            d_opp = math.hypot(x - opp_prediction[0], y - opp_prediction[1])
            if d_opp > (opp_gate_m or self.gate_m):
                return Sam2Refusal(cid, OUT_OF_GATE, x, y, step.score, step.prompted)
        if tag_prediction is not None:
            d_tag = math.hypot(x - tag_prediction[0], y - tag_prediction[1])
            if opp_prediction is not None:
                if d_tag < d_opp:
                    return Sam2Refusal(cid, NEARER_TAG, x, y, step.score, step.prompted)
            elif d_tag < self.clear_of_tag_m:
                return Sam2Refusal(cid, NEAR_TAG_LOST, x, y, step.score, step.prompted)
            rect = self.marker_rect(cid, tag_prediction)
            if rect is not None and self.rect_covered(mask, rect) > self.marker_overlap_max:
                return Sam2Refusal(cid, ON_TAG_MARKER, x, y, step.score, step.prompted)
        cov = tuple(float(v) * self.sigma_scale ** 2 for v in found.covariance_xy)
        return Sam2Reading(cid, float(x), float(y), cov, float(step.score),
                           float(found.incidence_rad), int(found.pixels), step.prompted,
                           {"length_m": found.length_m, "width_m": found.width_m,
                            "mask_px": area})

    def memory_frames(self):
        return {cid: cam["tracker"].stored_frames for cid, cam in self.cameras.items()}
