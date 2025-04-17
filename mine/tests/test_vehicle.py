import unittest
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from models.vehicle import MiningVehicle, FuelExhaustedError, ConcurrentTaskError
from algorithm.map_service import MapService
from utils.geo_tools import GeoUtils
from unittest.mock import Mock

class TestMiningVehicle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map_service = MapService()
        cls.geo_utils = GeoUtils()
        cls.sample_config = {
            'max_capacity': 50000,
            'fuel_capacity': 300.0,
            'fuel_consumption_rate': 0.015,
            'base_location': (0.0, 0.0),
            'max_speed': 5.0
        }

    def setUp(self):
        self.vehicle = MiningVehicle(
            vehicle_id=1,
            map_service=self.map_service,
            config=self.sample_config
        )

    def test_initial_state(self):
        """验证车辆初始状态"""
        self.assertEqual(self.vehicle.status, 'idle')
        self.assertAlmostEqual(self.vehicle.current_location[0], 0.0)
        self.assertAlmostEqual(self.vehicle.current_location[1], 0.0)
        self.assertEqual(self.vehicle.current_fuel, 300.0)

    def test_plane_coordinate_movement(self):
        """测试平面坐标系移动"""
        # 向右移动100米，向上移动200米
        new_position = (100.0, 200.0)
        self.vehicle.update_position(new_position)
        
        # 验证位置更新
        self.assertTupleEqual(self.vehicle.current_location, new_position)
        
        # 验证燃油消耗（距离=math.hypot(100,200)=223.6068米）
        expected_fuel = 300.0 - 223.6068 * 0.015
        self.assertTrue(math.isclose(self.vehicle.current_fuel, expected_fuel, rel_tol=1e-4))
        
        # 验证累计里程
        self.assertTrue(math.isclose(self.vehicle.mileage, 223.6068, rel_tol=1e-4))

    def test_fuel_exhaustion(self):
        """测试燃油耗尽异常"""
        self.vehicle.current_fuel = 1.0  # 设置低油量
        with self.assertRaises(FuelExhaustedError):
            self.vehicle.update_position((5000.0, 5000.0))  # 长距离移动

    def test_coordinate_conversion_integration(self):
        """测试与GeoUtils的坐标转换集成"""
        test_point = (150.0, -80.0)
        
        # 转换到虚拟坐标系
        converted = self.geo_utils.metres_to_ll(*test_point, (0,0))
        # 转换回米坐标
        restored = self.geo_utils.ll_to_metres(*converted, (0,0))
        
        self.assertTrue(math.isclose(restored[0], 150.0, abs_tol=1e-5))
        self.assertTrue(math.isclose(restored[1], -80.0, abs_tol=1e-5))

    def test_concurrent_task_handling(self):
        """测试并发任务分配"""
        mock_task1 = Mock()
        mock_task1.task_id = "task_001"
        mock_task2 = Mock()
        mock_task2.task_id = "task_002"
        
        self.vehicle.register_task_assignment(mock_task1)
        with self.assertRaises(ConcurrentTaskError):
            self.vehicle.register_task_assignment(mock_task2)

    def test_maintenance_operations(self):
        """测试维护操作"""
        # 模拟行驶80000米触发维护
        self.vehicle.mileage = 80001.0
        self.assertTrue(self.vehicle._needs_maintenance())
        
        # 执行全面维护
        self.vehicle.perform_maintenance("full_service")
        self.assertEqual(self.vehicle.mileage, 0.0)
        self.assertEqual(len(self.vehicle.maintenance_records), 1)

    def test_real_time_status_report(self):
        """测试实时状态报告"""
        self.vehicle.update_position((150.5, 200.3))
        report = self.vehicle.get_status_report()
        
        self.assertEqual(report['location'], (150.5, 200.3))
        self.assertEqual(report['fuel'], "298.8/300.0L")
        self.assertEqual(report['load'], "0/50000kg")

    def test_path_planning_integration(self):
        """测试与路径规划器的集成"""
        from algorithm.path_planner import HybridPathPlanner
        
        planner = HybridPathPlanner(self.geo_utils)
        start = (0.0, 0.0)
        end = (100.0, 100.0)
        
        # 获取规划路径
        path = planner.optimize_path(start, end, self.map_service.road_network, self.vehicle.vehicle_id)
        self.assertGreater(len(path), 2)
        
        # 模拟沿路径移动
        for point in path:
            self.vehicle.update_position(point)
        
        self.assertTrue(self.vehicle.mileage > 140.0)

if __name__ == '__main__':
    unittest.main()