#!/usr/bin/env python3
"""Intersect a pixel ray with a horizontal plane using calibrated extrinsics."""
import argparse
import json
from pathlib import Path


def project_pixel(camera, u, v, height):
    fx, fy = camera['K'][0], camera['K'][4]
    cx, cy = camera['K'][2], camera['K'][5]
    ray = [(u-cx)/fx, (v-cy)/fy, 1.0]
    rotation = camera['R_world_optical']
    direction = [sum(row[k]*ray[k] for k in range(3)) for row in rotation]
    origin = camera['position_world']
    if abs(direction[2]) < 1e-10:
        raise ValueError('Ray is parallel to the plane')
    distance = (height-origin[2])/direction[2]
    if distance <= 0:
        raise ValueError('Plane intersection is behind the camera')
    return [origin[k]+distance*direction[k] for k in range(3)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('camera', help='camera_1 through camera_6')
    parser.add_argument('u', type=float)
    parser.add_argument('v', type=float)
    parser.add_argument('--height', type=float, default=0)
    parser.add_argument('--calibration', type=Path,
                        default=Path(__file__).resolve().parents[1]/'config/cameras.json')
    args = parser.parse_args()
    data = json.loads(args.calibration.read_text())
    camera = next((c for c in data['cameras'] if c['name']==args.camera), None)
    if camera is None:
        parser.error('Unknown camera')
    if any(camera['D']):
        parser.error('Undistort the pixel first; this helper assumes zero distortion')
    try:
        print(json.dumps(dict(position_world=project_pixel(camera,args.u,args.v,args.height),
                              calibration_role=data.get('role','user_calibration'))))
    except ValueError as error:
        parser.error(str(error))
