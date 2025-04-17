import unittest
from utils.geo_tools import GeoUtils

class TestGeoUtils(unittest.TestCase):
    def setUp(self):
        self.geo = GeoUtils()
        
    def test_grid_metres_conversion(self):
        test_cases = [
            ((50, 50), (5000.0, 5000.0)),
            ((0, 0), (0.0, 0.0)),
            ((100, 200), (10000.0, 20000.0))
        ]
        
        for grid, expected in test_cases:
            with self.subTest(grid=grid):
                metres = self.geo.grid_to_metres(*grid)
                restored = self.geo.metres_to_grid(*metres)
                self.assertEqual(restored, grid)
                self.assertAlmostEqual(metres[0], expected[0], delta=0.1)
                
    def test_bresenham_consistency(self):
        start = (0, 0)
        end = (3, 3)
        expected_points = [(0,0), (1,1), (2,2), (3,3)]
        self.assertEqual(self.geo.bresenham_line(start, end), expected_points)

if __name__ == '__main__':
    unittest.main()