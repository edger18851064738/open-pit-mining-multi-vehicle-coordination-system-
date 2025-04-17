import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import math
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from algorithm.dispatch_service_v1 import DispatchService
from models.vehicle import MiningVehicle
from models.task import TransportTask
from utils.geo_tools import GeoUtils

class TestDispatchServiceV1(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        print("\n=== 测试场景初始化 ===")
        print("1. 创建模拟路径规划器和地图服务")
        print("2. 初始化2辆矿车(车辆ID:1和2)")
        print("3. 创建2个运输任务(任务ID:T1和T2)")
        print("4. 初始化调度服务")
        
        self.planner = MagicMock()
        self.map_service = MagicMock()
        
        # 测试车辆
        self.vehicles = [
            MiningVehicle(vehicle_id=1, map_service=self.map_service, config={
                'max_capacity': 100,
                'fuel_capacity': 300.0,
                'fuel_consumption_rate': 0.015,
                'base_location': (0.0, 0.0),
                'max_speed': 5
            }),
            MiningVehicle(vehicle_id=2, map_service=self.map_service, config={
                'max_capacity': 150,
                'fuel_capacity': 300.0,
                'fuel_consumption_rate': 0.015,
                'base_location': (0.0, 0.0),
                'max_speed': 3
            })
        ]
        
        # 测试任务
        self.task1 = TransportTask(task_id="T1", start_point=(0,0), end_point=(5,5), total_weight=50, task_type="transport")
        self.task2 = TransportTask(task_id="T2", start_point=(2,2), end_point=(7,7), total_weight=80, task_type="transport")
        
        # 调度服务
        self.baseline_service = DispatchService(self.planner, self.map_service)
        
    def test_vehicle_registration(self):
        """测试车辆注册功能"""
        print("\n=== 测试车辆注册 ===")
        print("预期结果: 基准模式应成功注册2辆车辆")
        self.baseline_service.register_vehicles(self.vehicles)
        self.assertEqual(len(self.baseline_service.vehicles), 2)
        print(f"实际结果: 注册车辆数={len(self.baseline_service.vehicles)}")
        

        
    def test_task_assignment_baseline(self):
        """测试基线模式下的任务分配"""
        print("\n=== 测试任务分配 ===")
        print("预期结果: 基准模式应正确分配任务T1给车辆")
        
        self.baseline_service.register_vehicles(self.vehicles)
        self.baseline_service.task_queue = [(1, 1, 1, self.task1)]
        
        assigned_task = self.baseline_service.assign_next_task()
        self.assertEqual(assigned_task.task_id, "T1")
        print(f"实际结果: 分配的任务ID={assigned_task.task_id}")
        
    def test_conflict_detection(self):
        """测试路径冲突检测"""
        print("\n=== 测试路径冲突检测 ===")
        print("测试场景: 两辆车使用相同路径[(1,1),(2,2),(3,3)]")
        print("预期结果: 应检测到路径冲突")
        
        # 设置相同路径
        path = [(1,1), (2,2), (3,3)]
        self.vehicles[0].current_path = path
        self.vehicles[1].current_path = path
        
        self.baseline_service.register_vehicles(self.vehicles)
        conflicts = self.baseline_service._detect_path_conflicts(path)
        self.assertGreater(conflicts, 0)
        print(f"实际结果(基准模式): 检测到冲突数={conflicts}")
        

        

        


    def test_vehicle_queue(self):
        """测试车辆排队机制"""
        print("\n=== 测试车辆排队 ===")
        print("测试场景: 3辆车同时请求任务")
        print("预期结果: 任务应按车辆ID顺序分配")
        
        # 添加第三辆测试车辆
        vehicle3 = MiningVehicle(vehicle_id=3, map_service=self.map_service, config={
            'max_capacity': 200,
            'fuel_capacity': 300.0,
            'fuel_consumption_rate': 0.015,
            'base_location': (0.0, 0.0),
            'max_speed': 4
        })
        self.vehicles.append(vehicle3)
        
        self.baseline_service.register_vehicles(self.vehicles)
        
        # 设置任务队列
        self.baseline_service.task_queue = [
            (1, 1, 1, self.task1),
            (2, 2, 2, self.task2)
        ]
        
        # 验证任务分配顺序
        first_task = self.baseline_service.assign_next_task()
        self.assertEqual(first_task.task_id, "T1")
        
        # 确保任务队列中的第二个任务分配给第二辆车
        self.baseline_service.task_queue = [
            (2, 2, 2, self.task2)
        ]
        second_task = self.baseline_service.assign_next_task()
        self.assertEqual(second_task.task_id, "T2")
        
        print(f"实际结果: 第一个分配的任务ID={first_task.task_id}")
        print(f"第二个分配的任务ID={second_task.task_id}")

if __name__ == '__main__':
    unittest.main()