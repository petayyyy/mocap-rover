import unittest
import numpy as np
from localization_contracts.image_pipeline import MultiCameraImagePipeline

class SimulationS09MultiImage(unittest.TestCase):
 def test_one_channel_disable_does_not_stop_other_pipelines(self):
  import cv2
  marker=np.zeros((240,240),dtype=np.uint8); (getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),0,240,marker,1)
  image=np.full((480,640),255,dtype=np.uint8); image[120:360,200:440]=marker
  K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
  spec={cid:{"K":K,"D":[0]*5,"R_world_optical":np.eye(3),"position_world":[0,0,0]} for cid in ('camera_1','camera_2')}
  p=MultiCameraImagePipeline(spec,{"rotation":np.eye(3),"translation":[0,0,0]},'c1')
  self.assertEqual(len(p.process('camera_1',image,1_000_000)),1); self.assertEqual(len(p.process('camera_2',image,1_000_000)),1)
  p.disable('camera_2'); out=p.publish(1_000_000); self.assertEqual(out['disabled'],['camera_2']); self.assertEqual(out['selected_camera'],'camera_1')
  p.enable('camera_2'); self.assertEqual(set(p.publish(1_000_000)['cameras']),{'camera_1','camera_2'})
 def test_calibration_apply_resets_only_reconfigured_camera_session(self):
  import cv2
  marker=np.zeros((240,240),dtype=np.uint8); (getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),0,240,marker,1)
  image=np.full((480,640),255,dtype=np.uint8); image[120:360,200:440]=marker; K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
  spec={cid:{"K":K,"D":[0]*5,"R_world_optical":np.eye(3),"position_world":[0,0,0]} for cid in ('camera_1','camera_2')}; p=MultiCameraImagePipeline(spec,{"rotation":np.eye(3),"translation":[0,0,0]},'c1')
  p.process('camera_1',image,1); p.process('camera_2',image,1); old1=p.cameras['camera_1'].fusion.session; old2=p.cameras['camera_2'].fusion.session
  p.reconfigure_camera('camera_1',K,[0]*5,'c2',{'rotation':np.eye(3),'translation':[0,0,0]},{'rotation':np.eye(3),'translation':[0,0,0]})
  self.assertGreater(p.cameras['camera_1'].fusion.session,old1); self.assertEqual(p.cameras['camera_2'].fusion.session,old2)

if __name__=='__main__': unittest.main()
