import unittest
import numpy as np
from localization_contracts.image_pipeline import OneCameraImagePipeline
from simulation.replay_image import ReplayImageFrame, ReplayImageRunner

class SimulationS05ReplayImage(unittest.TestCase):
 def test_reset_creates_new_session_and_reuses_shared_pipeline(self):
  import cv2
  marker=np.zeros((240,240),dtype=np.uint8); (getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),0,240,marker,1); image=np.full((480,640),255,dtype=np.uint8); image[120:360,200:440]=marker
  K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
  p=OneCameraImagePipeline('camera_1',K,[0]*5,{'rotation':np.eye(3),'translation':[0,0,0]},{'rotation':np.eye(3),'translation':[0,0,0]},'replay-c1')
  r=ReplayImageRunner(p); out=r.run([('new_session',None),('frame',1,ReplayImageFrame(image,1,2,1)),('new_session',None),('frame',2,ReplayImageFrame(image,3,4,2))])
  self.assertEqual(out['session'],2); self.assertEqual(out['processed'],2); self.assertEqual(r.pipeline.fusion.session,2)

if __name__=='__main__': unittest.main()
