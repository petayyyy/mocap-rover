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
        (getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), 0, 240, marker, 1
        )
        canvas = np.full((480, 640), 255, dtype=np.uint8)
        canvas[120:360, 200:440] = marker
        detections = AprilTagImageDetector().detect(canvas)
        self.assertEqual([d.tag_id for d in detections], [0])
        self.assertEqual(len(detections[0].corners), 4)
        self.assertIn(AprilTagImageDetector().backend, ("opencv-aruco-apriltag-36h11-legacy", "opencv-aruco-apriltag-36h11-modern"))

    def test_aruco_4x4_50_alias_and_allowed_ids(self):
        import cv2
        marker = np.zeros((120, 120), dtype=np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        (getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(
            dictionary, 23, 120, marker, 1
        )
        canvas = np.full((240, 320), 205, dtype=np.uint8)
        canvas[60:180, 100:220] = marker
        detector = AprilTagImageDetector("4x4_dict_50", allowed_ids=(23,))
        self.assertEqual(detector.family, "aruco4x4_50")
        self.assertEqual([d.tag_id for d in detector.detect(canvas)], [23])
        self.assertEqual(AprilTagImageDetector("DICT_4X4_50", allowed_ids=(7,)).detect(canvas), ())

    def test_coverage_profile_keeps_small_markers(self):
        import cv2
        for family, dictionary_id in (
            ("tag36h11", cv2.aruco.DICT_APRILTAG_36h11),
            ("aruco4x4_50", cv2.aruco.DICT_4X4_50),
        ):
            marker = np.zeros((20, 20), dtype=np.uint8)
            (getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(
                cv2.aruco.getPredefinedDictionary(dictionary_id), 0, 20, marker, 1
            )
            canvas = np.full((480, 640), 255, dtype=np.uint8)
            canvas[230:250, 310:330] = marker
            found = AprilTagImageDetector(
                family, profile="coverage", allowed_ids=(0,)
            ).detect(canvas)
            self.assertEqual([d.tag_id for d in found], [0])

    def test_detector_uses_bounded_contrast_preprocessing(self):
        import cv2, os
        image=cv2.imread('/tmp/mocap-s08-live/camera_1.png',0) if os.path.isfile('/tmp/mocap-s08-live/camera_1.png') else None
        if image is None: self.skipTest('live Gazebo snapshot unavailable')
        self.assertEqual([d.tag_id for d in AprilTagImageDetector().detect(image)], [0])
    def test_half_scale_preserves_tag_and_rescales_corners(self):
        import cv2
        image=cv2.imread('/tmp/mocap-s08-baseline/camera_1.ppm',0) if __import__('os').path.isfile('/tmp/mocap-s08-baseline/camera_1.ppm') else None
        if image is None: self.skipTest('Gazebo snapshot unavailable')
        full=AprilTagImageDetector().detect(image)[0]; half=AprilTagImageDetector(scale=.5).detect(image)[0]
        self.assertEqual(half.tag_id,full.tag_id); self.assertLess(float(np.linalg.norm(np.asarray(half.corners)-np.asarray(full.corners))),10)

    def test_detector_to_pnp_observation_has_capture_provenance(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV unavailable")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker = np.zeros((240, 240), dtype=np.uint8)
        (getattr(cv2.aruco, "drawMarker", None) or cv2.aruco.generateImageMarker)(dictionary, 0, 240, marker, 1)
        canvas = np.full((480, 640), 255, dtype=np.uint8); canvas[120:360, 200:440] = marker
        hit = AprilTagImageDetector().detect(canvas)[0]
        K=np.array([[500.,0,320],[0,500.,240],[0,0,1.]])
        # Toy rig: the camera sits at the origin looking along +Z, so the
        # marker plane normal points back at it rather than along arena +Z.
        observer=PnpAprilTagObserver(TagConfig(calibration_version="image-c1"), K, [0]*5,
            {"rotation":np.eye(3),"translation":[0,0,0]},
            base_z_nominal_m=500*0.4/240, image_size=[640,480],
            plane_normal_world=(0,0,-1))
        obs=observer.observe(Detection("camera_1",7,hit.tag_id,hit.corners,1_000_000,1_001_000,1_002_000))
        self.assertTrue(obs.pose_6d_valid); self.assertEqual(obs.capture_time_ns,1_000_000)
        self.assertEqual(obs.method,"apriltag36h11_pnp")


if __name__ == "__main__":
    unittest.main()
