import unittest

import numpy as np

from exp46_pole_geometry import axis_geometry


class PoleGeometryTests(unittest.TestCase):
    def test_exact_pole_distances_and_segment_endpoints(self):
        negative = np.array([-.5, .5, 0.])
        positive = np.array([.5, .5, 0.])
        queries = np.array([[0., .5, .7], [.8, .5, .2], [-.5, .5, 0.]])
        geometry = axis_geometry(queries, negative, positive)
        np.testing.assert_allclose(geometry["along"], [0, .8, -.5])
        np.testing.assert_allclose(geometry["perpendicular"], [.7, .2, 0])
        np.testing.assert_allclose(geometry["segment_distance"], [.7, np.sqrt(.13), 0])
        for sign, side in ((-1, "negative"), (1, "positive")):
            reconstructed = np.sqrt((geometry["along"] - sign * .5) ** 2
                                    + geometry["perpendicular"] ** 2)
            np.testing.assert_allclose(reconstructed, geometry[f"{side}_distance"])

    def test_equal_cosine_difference_does_not_imply_equal_relevance(self):
        negative = np.array([-.6, .8, 0.])
        positive = np.array([.6, .8, 0.])
        near = np.array([.1, np.sqrt(.99), 0.])
        far = np.array([.1, 0., np.sqrt(.99)])
        self.assertAlmostEqual(near @ (positive - negative), far @ (positive - negative))
        geometry = axis_geometry(np.stack((near, far)), negative, positive)
        self.assertGreater(geometry["perpendicular"][1], geometry["perpendicular"][0])


if __name__ == "__main__":
    unittest.main()