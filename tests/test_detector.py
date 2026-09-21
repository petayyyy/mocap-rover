import unittest
import numpy as np

from localization_contracts.detector import AprilTagImageDetector
from localization_contracts.apriltag import Detection, PnpAprilTagObserver, TagConfig


class SimulationS08Detector(unittest.TestCase):
    def test_real_image_detector_returns_corners_not_truth(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV unavailable")
        marker = np.zeros((240, 240), dtype=np.uint8)
        cv2.aruco.drawMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 240, marker, 1
        )
        canvas = np.full((480, 640), 255, dtype=np.uint8)
        canvas[120:360, 200:440] = marker
        detections = AprilTagImageDetector().detect(canvas)
        self.assertEqual([d.tag_id for d in detections], [0])
        self.assertEqual(len(detections[0].corners), 4)
        self.assertEqual(AprilTagImageDetector().backend, "opencv-aruco-apriltag-36h11-legacy")

    def test_detector_uses_bounded_contrast_preprocessing(self):
        import cv2, os
        image=cv2.imread('/tmp/mocap-s08-live/camera_1.png',0) if os.path.isfile('/tmp/mocap-s08-live/camera_1.png') else None
        if image is None: self.skipTest('live Gazebo snapshot unavailable')
        self.assertEqual([d.tag_id for d in AprilTagImageDetector().detect(image)], [0])

    def test_detector_to_pnp_observation_has_capture_provenance(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV unavailable")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker = np.zeros((240, 240), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, 0, 240, marker, 1)
        canvas = np.full((480, 640), 255, dtype=np.uint8); canvas[120:360, 200:440] = marker
        hit = AprilTagImageDetector().detect(canvas)[0]
        K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
        observer=PnpAprilTagObserver(TagConfig(calibration_version="image-c1"), K, [0]*5,
            {"rotation":np.eye(3),"translation":[0,0,0]})
        obs=observer.observe(Detection("camera_1",7,hit.tag_id,hit.corners,1_000_000,1_001_000,1_002_000))
        self.assertTrue(obs.pose_6d_valid); self.assertEqual(obs.capture_time_ns,1_000_000)
        self.assertEqual(obs.method,"apriltag36h11_pnp")


if __name__ == "__main__":
    unittest.main()
