#!/usr/bin/env python3
"""Train and evaluate on the scene-disjoint Gazebo dataset; no simulator required."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--model", required=True, type=Path, help="Local pretrained YOLO checkpoint"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("pretrained checkpoint does not exist")
    if (args.output / "rover").exists():
        parser.error("training output exists; choose a new directory")
    from ultralytics import YOLO
    import ultralytics
    import torch

    manifest = args.dataset / "manifest.json"
    dataset = args.dataset / "data.yaml"
    if not manifest.is_file() or not dataset.is_file():
        parser.error("dataset/manifest incomplete")
    provenance = {
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "pretrained_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "seed": args.seed,
        "epochs": args.epochs,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "hardware_verified": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    model = YOLO(str(args.model))
    model.train(
        data=str(dataset.resolve()),
        epochs=args.epochs,
        imgsz=640,
        batch=12,
        device=0,
        workers=2,
        project=str(args.output),
        name="rover",
        seed=args.seed,
        patience=12,
        amp=True,
        plots=True,
        cache=False,
    )
    best = args.output / "rover/weights/best.pt"
    model = YOLO(str(best))
    model.val(
        data=str(dataset.resolve()),
        split="test",
        device=0,
        project=str(args.output),
        name="holdout",
    )
    provenance["weights_sha256"] = hashlib.sha256(best.read_bytes()).hexdigest()
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
