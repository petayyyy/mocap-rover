#!/usr/bin/env python3
"""Gazebo tag-only scenario driver; HTTP results and ground truth stay evaluator-only.

Run beside --tag-only runtime. Teleports the tagged model, covers the tag, pauses
the world, tests calibration apply/rollback, and restores the initial scene.
Never commands the opponent. Output is evidence, not a full S15 acceptance claim.
"""
import argparse
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localization_contracts.gazebo_services import ServiceClient
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--port", type=int, default=8081)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    client = ServiceClient()
    rows = []
    base = f"http://127.0.0.1:{a.port}"
    def get(path):
        return json.load(urllib.request.urlopen(base + path, timeout=5))
    def post(value):
        req = urllib.request.Request(base + "/api/settings", json.dumps(value).encode(),
                                     {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=5))
    def sample(label, duration=3, expected=None):
        begin = time.monotonic()
        samples = []
        while time.monotonic() - begin < duration:
            s = get("/api/status")
            samples.append(s)
            time.sleep(.2)
        row = dict(scenario=label, expected=expected, samples=samples)
        rows.append(row)
        (out / "scenarios.json").write_text(json.dumps(rows, indent=2) + "\n")
        track = samples[-1]["tracks"].get("tag_rover", {})
        print(label, track.get("tracking_state"), track.get("state"), flush=True)
    def pose(x, y, yaw=0, inverted=False):
        q = Pose(name="tag_rover")
        q.position.x, q.position.y, q.position.z = x, y, .226 if inverted else .14
        if inverted:
            q.orientation.x = math.cos(yaw / 2)
            q.orientation.y = math.sin(yaw / 2)
        else:
            q.orientation.w = math.cos(yaw / 2)
            q.orientation.z = math.sin(yaw / 2)
        assert client.request("set_pose", q)
        time.sleep(1)
    cover = False
    try:
        for i, (x, y) in enumerate([(3,2),(9,2),(3,6),(9,6),(3,10),(9,10),
                                    (6,4),(6,8),(.8,.8),(11.2,.8),
                                    (.8,11.2),(11.2,11.2),(6,6)]):
            yaw = (i % 4) * math.pi / 2
            pose(x,y,yaw)
            sample(f"point_{i:02}", expected=dict(x=x,y=y,yaw=yaw))
        for yaw in (0, math.pi/2, -math.pi/2):
            pose(3,2,yaw,True)
            sample("bottom_tag_"+str(round(yaw,3)), expected=dict(x=3,y=2,yaw=yaw,tag_id=1))
        pose(3,2)
        factory = EntityFactory()
        factory.sdf = """<sdf version="1.9"><model name="acceptance_tag_cover"><static>true</static>
<pose>3 2 0.65 0 0 0</pose><link name="cover"><visual name="cover">
<geometry><box><size>1.5 1.5 0.05</size></box></geometry>
<material><ambient>0.03 0.03 0.03 1</ambient><diffuse>0.03 0.03 0.03 1</diffuse></material>
</visual></link></model></sdf>"""
        cover = client.request("create", factory)
        assert cover
        time.sleep(1)
        sample("occluded", expected={"valid":False,"tracking_state":"LOST"})
        assert client.request("remove", Entity(name="acceptance_tag_cover",type=Entity.MODEL))
        cover = False
        time.sleep(1)
        sample("recovered", expected={"valid":True})
        assert client.request("control", WorldControl(pause=True))
        time.sleep(.5)
        sample("paused_clock", expected={"valid":False,"tracking_state":"TIME_UNCERTAIN"})
        assert client.request("control", WorldControl(pause=False))
        time.sleep(1)
        sample("resumed_clock", expected={"valid":True})
        settings = get("/api/settings")
        response = post(settings)
        assert response["ok"]
        time.sleep(1)
        sample("calibration_apply", expected={"valid":True})
        assert post({"action":"rollback"})["ok"]
        time.sleep(1)
        sample("calibration_rollback", expected={"valid":True})
        for i in range(1,7):
            (out/f"camera_{i}.jpg").write_bytes(urllib.request.urlopen(
                base+f"/preview/camera_{i}",timeout=5).read())
    finally:
        client.request("control", WorldControl(pause=False))
        if cover:
            client.request("remove", Entity(name="acceptance_tag_cover",type=Entity.MODEL))
        pose(3,2)
        client.close()


if __name__ == "__main__":
    main()
