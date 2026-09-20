"""Evaluation-only metrics. It deliberately has no runtime detector/tracker API."""
import json, math
from pathlib import Path

def evaluate(estimate_path, truth_path):
    estimates = json.loads(Path(estimate_path).read_text())
    truth = json.loads(Path(truth_path).read_text())
    by_stamp = {int(x["stamp_ns"]): x for x in truth}
    errors = []
    for item in estimates:
        ref = by_stamp.get(int(item["stamp_ns"]))
        if ref is not None:
            errors.append(math.dist(item["position_m"], ref["position_m"]))
    return {"matched": len(errors), "p95_xy_m": sorted(errors)[max(0, math.ceil(.95*len(errors))-1)] if errors else None,
            "truth_used_only_here": True}
