import unittest
import numpy as np

from localization_contracts.image_pipeline import OneCameraImagePipeline

class SimulationS09ImagePipeline(unittest.TestCase):
 def test_image_detector_pnp_fusion_and_independent_publish(self):
  import cv2
  marker=np.zeros((240,240),dtype=np.uint8); (getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),0,240,marker,1)
  image=np.full((480,640),255,dtype=np.uint8); image[120:360,200:440]=marker
  K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
  p=OneCameraImagePipeline('camera_1',K,[0]*5,{'rotation':np.eye(3),'translation':[0,0,0]},
      {'rotation':np.eye(3),'translation':[0,0,0]},'c1')
  self.assertEqual(len(p.process(image,1_000_000)),1); item,status=p.publish(1_000_000)
  self.assertEqual(status.tracking_state,'TRACKING'); self.assertEqual(status.calibration_version,'c1'); self.assertTrue(item['valid'])
 def test_aruco_4x4_50_detector_to_pnp(self):
  import cv2
  marker=np.zeros((160,160),dtype=np.uint8)
  (getattr(cv2.aruco,'drawMarker',None) or cv2.aruco.generateImageMarker)(
      cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),7,160,marker,1)
  image=np.full((480,640),255,dtype=np.uint8); image[160:320,240:400]=marker
  K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
  p=OneCameraImagePipeline('camera_1',K,[0]*5,
      {'rotation':np.eye(3),'translation':[0,0,0]},
      {7:{'rotation':np.eye(3),'translation':[0,0,0]}},'aruco-c1',
      family='DICT_4X4_50',marker_ids=(7,))
  result=p.process(image,1_000_000)
  self.assertEqual(len(result),1)
  self.assertEqual(result[0].marker_family,'aruco4x4_50')
  self.assertEqual(result[0].method,'aruco4x4_50_pnp')

if __name__=='__main__': unittest.main()
