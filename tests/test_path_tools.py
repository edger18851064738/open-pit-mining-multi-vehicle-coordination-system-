import unittest
from utils.path_tools import PathProcessor
from utils.geo_tools import GeoUtils
import numpy as np

class TestPathTools(unittest.TestCase):
    def setUp(self):
        self.geo = GeoUtils()
        self.processor = PathProcessor(self.geo)

    def test_path_smoothing(self):
        test_path = [(0,0), (3,4), (6,8)]
        smoothed = self.processor.smooth_path(test_path)
        self.assertGreater(len(smoothed), 5)
        self.assertAlmostEqual(smoothed[0][0], 0.0, delta=0.1)

    def test_path_length_calculation(self):
        diagonal_path = [(0,0), (1,1)]
        expected_length = np.sqrt(2) * self.geo.grid_size
        self.assertAlmostEqual(
            self.processor.calculate_path_length(diagonal_path),
            expected_length,
            delta=0.1
        )

    def test_optimization_error_handling(self):
        with self.assertRaises(Exception):
            self.processor.optimize_path((0,0), (100,100), None)

if __name__ == '__main__':
    unittest.main()