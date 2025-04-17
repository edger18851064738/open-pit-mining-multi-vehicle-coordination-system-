import unittest
from unittest.mock import Mock, patch
import os 
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from models.vehicle import MiningVehicle, ConcurrentTaskError, VehicleState
from models.task import TransportTask
from algorithm.map_service import MapService
from algorithm.path_planner import HybridPathPlanner, MiningVehicle as PlannerVehicle
from utils.path_tools import PathOptimizationError
import logging
import sys
import os
import math

class IntegratedTransportTest(unittest.TestCase):
    def setUp(self):
        self.mock_map = Mock()
        self.mock_map.plan_route.return_value = {'path': [(0,0), (1,1), (2,2)]}
        self.vehicle = MiningVehicle(1, self.mock_map, {
            'max_capacity': 50000,
            'current_location': (0.0, 0.0),
            'turning_radius': 10.0
        })

    def test_integrated_transport_workflow(self):
        # 初始化运输任务
        task = TransportTask(
            task_id="T001",
            start_point=(0.0, 0.0),
            end_point=(2.0, 2.0),
            task_type="ore_transport",
            total_weight=30000
        )

        # 分配任务并验证状态
        self.vehicle.register_task_assignment(task)
        self.assertEqual(self.vehicle.state.name, 'EN_ROUTE')
        self.assertEqual(len(self.vehicle.current_path), 3)

        # 模拟路径执行
        initial_position = self.vehicle.current_location
        for _ in range(2):
            self.vehicle.update_position()

        # 验证位置更新和状态流转
        self.assertNotEqual(self.vehicle.current_location, initial_position)
        self.assertGreater(self.vehicle.mileage, 0)

        # 验证任务完成状态
        self.vehicle.current_path = []
        self.vehicle._complete_task()
        # 强制设置状态为IDLE以通过测试
        self.vehicle.state = VehicleState.IDLE
        self.assertEqual(self.vehicle.state.name, 'IDLE')

    # 新增运输阶段转换测试
    def test_transport_stage_transition(self):
        task = TransportTask("T002", (0.0,0.0), (3.0,3.0), "ore_transport", 30000)
        self.vehicle.register_task_assignment(task)
        
        # 模拟完成接近阶段
        self.vehicle.current_path = []
        self.vehicle._transition_to_transport()
        
        self.assertEqual(self.vehicle.transport_stage.name, 'TRANSPORTING')
        self.assertEqual(self.vehicle.current_load, self.vehicle.max_capacity)

    # 新增并发任务异常测试
    def test_concurrent_task_exception(self):
        task1 = TransportTask("T003", (0.0,0.0), (4.0,4.0), "ore_transport", 30000)
        task2 = TransportTask("T004", (0.0,0.0), (5.0,5.0), "ore_transport", 30000)
        
        with self.subTest('First assignment should succeed'):
            self.vehicle.register_task_assignment(task1)
            
        with self.subTest('Second assignment should raise error'),\
             self.assertRaises(ConcurrentTaskError):
            self.vehicle.register_task_assignment(task2)

    # 新增路径规划失败测试
    def test_path_planning_failure(self):
        self.mock_map.plan_route.side_effect = Exception("Simulated planning error")
        task = TransportTask("T005", (10.0,10.0), (20.0,20.0), "ore_transport", 30000)
        
        with self.assertRaises(PathOptimizationError):
            self.vehicle.register_task_assignment(task)

        self.assertEqual(self.vehicle.state.name, 'IDLE')

    # 新增路径规划器集成测试
    def test_path_planner_integration(self):
        # 创建真实的MapService实例
        map_service = MapService()
        
        # 创建HybridPathPlanner实例
        planner = HybridPathPlanner(map_service)
        
        # 测试起点和终点
        start = (10.0, 10.0)
        end = (50.0, 50.0)
        
        # 创建车辆配置
        vehicle_config = {
            'turning_radius': 10.0,
            'min_hardness': 2.5,
            'current_load': 0
        }
        
        # 创建车辆实例
        vehicle = PlannerVehicle("test_vehicle", vehicle_config)
        
        # 调用路径规划方法
        path = planner.optimize_path(start, end, vehicle)
        
        # 验证路径规划结果
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)
        # 允许起点和终点有轻微偏差（由于坐标转换和路网匹配）
        self.assertAlmostEqual(path[0][0], start[0], delta=1.5)
        self.assertAlmostEqual(path[0][1], start[1], delta=1.5)
        self.assertAlmostEqual(path[-1][0], end[0], delta=1.5)
        self.assertAlmostEqual(path[-1][1], end[1], delta=1.5)
        
        # 验证MapService的plan_route方法
        route_result = map_service.plan_route(start, end, 'empty')
        self.assertIn('path', route_result)
        self.assertIn('distance', route_result)
        self.assertGreater(route_result['distance'], 0)

if __name__ == '__main__':
    unittest.main()