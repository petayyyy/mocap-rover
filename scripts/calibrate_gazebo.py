#!/usr/bin/env python3
"""Image calibration using surveyed movable checkerboards, never camera truth.

Board dimensions and commanded surveyed poses define the metric arena gauge.
K/D and camera extrinsics are estimated from rendered checkerboard corners.
"""
import argparse, json, sys, time, os, subprocess
from google.protobuf.text_format import MessageToString
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.capture import LatestFrames, rgb_array
from localization_contracts.gazebo_services import ServiceClient


def board_sdf(name):
    visuals = [
        '<visual name="border"><pose>0 0 -.002 0 0 0</pose><geometry><box><size>1.8 1.32 .002</size></box></geometry><material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><emissive>1 1 1 1</emissive></material></visual>'
    ]
    for y in range(7):
        for x in range(10):
            color = "1 1 1 1" if (x + y) % 2 == 0 else "0 0 0 1"
            visuals.append(
                f'<visual name="s{x}_{y}"><pose>{(x-4.5)*.16} {(3-y)*.16} 0 0 0 0</pose><geometry><box><size>.16 .16 .002</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse><emissive>{color}</emissive></material></visual>'
            )
    return (
        f'<sdf version="1.9"><model name="{name}"><static>true</static><link name="board">'
        + "".join(visuals)
        + "</link></model></sdf>"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--views", type=int, default=18)
    a = p.parse_args()
    cv2.setNumThreads(2)
    cfg = json.loads(Path(a.config).read_text())
    cams = {c["name"]: c for c in cfg["cameras"]}
    node = Node()
    latest = LatestFrames(cams)
    topics = []
    names = []
    services = ServiceClient()

    def request(service, msg):
        if not services.request(service, msg):
            raise RuntimeError(service)

    for cid in cams:
        topic = f"/cameras/{cid}/image"
        topics.append(topic)
        assert node.subscribe(Image, topic, lambda msg, cid=cid: latest.push(cid, msg))
        name = "calibration_" + cid
        services.request("remove", Entity(name=name, type=Entity.MODEL))
        names.append(name)
        request("create", EntityFactory(sdf=board_sdf(name), name=name))
    obj = np.array(
        [[(x - 4) * 0.16, (2.5 - y) * 0.16, 0] for y in range(6) for x in range(9)],
        np.float32,
    )
    records = {cid: [] for cid in cams}
    rng = np.random.default_rng(912)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    sample_dir = out.parent / "calibration_views"
    sample_dir.mkdir(exist_ok=True)
    try:
        for v in range(a.views):
            poses = {}
            for cid, c in cams.items():
                r = Rotation.from_euler(
                    "xyz",
                    [
                        rng.uniform(-0.45, 0.45),
                        rng.uniform(-0.45, 0.45),
                        rng.uniform(-0.18, 0.18),
                    ],
                )
                pos = np.array(c["position_world"]) + np.array(
                    [
                        rng.uniform(-0.45, 0.45),
                        rng.uniform(-0.3, 0.3),
                        -rng.uniform(1.4, 2.2),
                    ]
                )
                T = np.eye(4)
                T[:3, :3] = r.as_matrix()
                T[:3, 3] = pos
                poses[cid] = T
                msg = Pose(name="calibration_" + cid)
                msg.position.x, msg.position.y, msg.position.z = pos
                (
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w,
                ) = r.as_quat()
                request("set_pose", msg)
            time.sleep(0.6)
            for cid, c in cams.items():
                latest.take(cid)
                item = latest.take(cid, 2)
                if item is None:
                    continue
                image = rgb_array(item[0])
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                ok, corners = cv2.findChessboardCorners(
                    gray,
                    (9, 6),
                    cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
                )
                if not ok:
                    cv2.imwrite(
                        str(sample_dir / f"{cid}_failed.png"),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    )
                    print(f"{cid} view {v}: corners not found", flush=True)
                    continue
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (5, 5),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001),
                )
                # Resolve the chessboard 180-degree ambiguity using surveyed
                # board poses and rough nominal mounting, never actual camera poses.
                T = poses[cid]
                world = obj @ T[:3, :3].T + T[:3, 3]
                R = np.asarray(c["R_world_optical"])
                camera = (world - np.asarray(c["position_world"])) @ R
                K = np.asarray(c["K"]).reshape(3, 3)
                approx = camera @ K.T
                approx = approx[:, :2] / approx[:, 2:]
                if np.linalg.norm(corners[::-1, 0] - approx) < np.linalg.norm(
                    corners[:, 0] - approx
                ):
                    corners = corners[::-1].copy()
                records[cid].append(
                    {
                        "view": v,
                        "image_points": corners[:, 0].tolist(),
                        "board_pose": T.tolist(),
                    }
                )
                if v in (0, a.views - 1):
                    cv2.imwrite(
                        str(sample_dir / f"{cid}_{v}.png"),
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                    )
            out.with_suffix(".observations.json").write_text(
                json.dumps(
                    {"object_points": obj.tolist(), "cameras": records}, indent=2
                )
                + "\n"
            )
            print("views", v + 1, {k: len(x) for k, x in records.items()}, flush=True)
        report = {}
        for cid, c in cams.items():
            rows = records[cid]
            if len(rows) < 10:
                raise RuntimeError(f"{cid}: only {len(rows)} usable views")
            train = rows[:-3]
            holdout = rows[-3:]
            K = np.asarray(c["K"]).reshape(3, 3).copy()
            D = np.zeros(5)
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
                [obj] * len(train),
                [np.asarray(x["image_points"], np.float32) for x in train],
                tuple(c["image_size"]),
                K,
                D,
                flags=cv2.CALIB_USE_INTRINSIC_GUESS
                | cv2.CALIB_ZERO_TANGENT_DIST
                | cv2.CALIB_FIX_K3,
            )
            world = np.concatenate(
                [
                    obj @ np.array(row["board_pose"])[:3, :3].T
                    + np.array(row["board_pose"])[:3, 3]
                    for row in train
                ]
            )
            pixels = np.concatenate([np.array(row["image_points"]) for row in train])
            ok, rv, tv, inliers = cv2.solvePnPRansac(
                world, pixels, K, D, reprojectionError=1.5, iterationsCount=200
            )
            if not ok:
                raise RuntimeError(cid + " extrinsic PnP failed")
            rv, tv = cv2.solvePnPRefineLM(
                world[inliers[:, 0]], pixels[inliers[:, 0]], K, D, rv, tv
            )
            R, _ = cv2.Rodrigues(rv)
            pos = -R.T @ tv.reshape(3)
            errors = []
            for row in holdout:
                T = np.array(row["board_pose"])
                w = obj @ T[:3, :3].T + T[:3, 3]
                uv, _ = cv2.projectPoints(w, rv, tv, K, D)
                errors.extend(
                    np.linalg.norm(uv[:, 0] - np.array(row["image_points"]), axis=1)
                )
            p95 = float(np.percentile(errors, 95))
            if p95 > 2:
                raise RuntimeError(f"{cid}: held-out reprojection p95 {p95:.2f}px")
            c.update(
                K=K.reshape(-1).tolist(),
                D=D.reshape(-1).tolist(),
                position_world=pos.tolist(),
                R_world_optical=R.T.tolist(),
            )
            report[cid] = {
                "training_views": len(train),
                "holdout_views": len(holdout),
                "rms_px": rms,
                "holdout_p95_px": p95,
                "inliers": len(inliers),
            }
        cfg["role"] = "image_calibrated"
        cfg["calibration_version"] = "gazebo-checkerboard-v1"
        cfg["calibration_report"] = report
        out.write_text(json.dumps(cfg, indent=2) + "\n")
        out.with_suffix(".observations.json").write_text(
            json.dumps({"object_points": obj.tolist(), "cameras": records}, indent=2)
            + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
    finally:
        for name in names:
            request("remove", Entity(name=name, type=Entity.MODEL))
        for topic in topics:
            node.unsubscribe(topic)
        latest.close()
        services.close()
    os._exit(0)


if __name__ == "__main__":
    main()
