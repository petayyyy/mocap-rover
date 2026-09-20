"""Generator integration checks; uses Gazebo's independent rotation implementation."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from project_pixel import project_pixel
from gz.math7 import Quaterniond, Vector3d

class Scenarios(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)

    def generate(self,*args):
        subprocess.run([sys.executable,str(ROOT/'scripts/generate_world.py'),
                        '--output-dir',str(self.path),*args],check=True,stdout=subprocess.DEVNULL)
        return json.loads((self.path/'config/cameras_ground_truth.json').read_text())

    def test_styles_preserve_installation(self):
        previous=None; colors=[]
        for style in ('dark','light','steel','sand','two-tone'):
            data=self.generate('--style',style)
            if previous is not None: self.assertEqual(previous,data)
            previous=data
            subprocess.run(['gz','sdf','-k',str(self.path/'worlds/mocap_arena.sdf')],check=True,stdout=subprocess.DEVNULL)
            tree=ET.parse(self.path/'worlds/mocap_arena.sdf')
            colors.append(tree.find(".//model[@name='yolo_rover']/link[@name='base_link']/visual[@name='body_visual']/material/diffuse").text)
        self.assertEqual(len(set(colors)),5)

    def test_actual_extrinsics_match_sdf(self):
        data=self.generate()
        tree=ET.parse(self.path/'worlds/mocap_arena.sdf')
        for camera in data['cameras']:
            model=tree.find(f".//model[@name='{camera['name']}']")
            pose=list(map(float,model.find('pose').text.split()))
            self.assertEqual(pose[:3],camera['position_world'])
            sensor=list(map(float,model.find('.//sensor/pose').text.split()))
            q=Quaterniond(*pose[3:])*Quaterniond(*sensor[3:])
            # Gazebo camera: +X forward, +Y left, +Z up.
            for j,axis in enumerate((Vector3d(0,-1,0),Vector3d(0,0,-1),Vector3d(1,0,0))):
                vector=q.rotate_vector(axis)
                for i,value in enumerate((vector.x(),vector.y(),vector.z())):
                    self.assertAlmostEqual(value,camera['R_world_optical'][i][j],places=10)
            for value in camera['installation_offset_world_m']: self.assertLessEqual(abs(value),.03)
            for value,bound in zip(camera['installation_rpy_world_deg'],(.5,5,.5)): self.assertLessEqual(abs(value),bound)

    def test_seed_and_ideal_mode(self):
        a=self.generate('--seed','42')
        self.assertEqual(a,self.generate('--seed','42'))
        self.assertNotEqual(a,self.generate('--seed','43'))
        ideal=self.generate('--ideal-cameras')
        nominal=json.loads((self.path/'config/cameras.json').read_text())
        for a,b in zip(ideal['cameras'],nominal['cameras']):
            self.assertEqual(a['position_world'],b['position_world'])
            self.assertEqual(a['R_world_optical'],b['R_world_optical'])

    def test_ray_intersection_and_nominal_error(self):
        data=self.generate()
        nominal=json.loads((self.path/'config/cameras.json').read_text())
        errors=[]
        for actual,design in zip(data['cameras'],nominal['cameras']):
            point=[*design['position_world'][:2],.3654]
            delta=[point[i]-actual['position_world'][i] for i in range(3)]
            r=actual['R_world_optical']
            optical=[sum(r[i][j]*delta[i] for i in range(3)) for j in range(3)]
            u=actual['K'][0]*optical[0]/optical[2]+actual['K'][2]
            v=actual['K'][4]*optical[1]/optical[2]+actual['K'][5]
            estimate=project_pixel(actual,u,v,point[2])
            self.assertLess(math.dist(point,estimate),1e-9)
            errors.append(math.dist(point,project_pixel(design,u,v,point[2])))
        self.assertGreater(max(errors),.02)

if __name__=='__main__': unittest.main()
