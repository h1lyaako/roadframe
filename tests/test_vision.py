import unittest
from unittest.mock import patch

import numpy as np

from vision import review_candidate


class VisualConfirmationTests(unittest.TestCase):
    def test_fixed_camera_needs_agreement_with_motion(self):
        image = np.zeros((90, 160, 3), dtype=np.uint8)
        frames = [(t, image) for t in (7.5, 8, 8.5, 9, 9.5, 10, 10.5)]
        region = (.5, .5, .4)
        with patch("vision.samples", return_value=frames), patch("vision.ask") as ask:
            ask.return_value = {"outcome": "impact changes motion", "panel": 6}
            self.assertEqual(review_candidate("fixture", 20, 9, region)["time"], 9)
            ask.return_value = {"outcome": "impact changes motion", "panel": 7}
            self.assertIsNone(review_candidate("fixture", 20, 9, region))
            ask.return_value = {"outcome": "none", "panel": 4}
            self.assertIsNone(review_candidate("fixture", 20, 9, region))
