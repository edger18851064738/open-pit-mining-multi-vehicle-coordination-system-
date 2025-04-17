"""
虚拟仿真环境模块
实现功能：
1. 地图坐标系和障碍物模型
2. 车辆运动模拟
3. 碰撞检测
4. 与调度系统集成
"""
from __future__ import annotations
from config.paths import PROJECT_ROOT
import os
import math
import time
import threading
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque
from models.vehicle import VehicleState
from models.vehicle import MiningVehicle
from algorithm.map_service import MapService
import random
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import json
class VirtualEnvironment:
    """虚拟仿真环境核心类"""
    
    def __init__(self, map_service: MapService):
        self.map_service = map_service
        self.vehicles: Dict[int, MiningVehicle] = {}
        self.obstacles: Set[Tuple[float, float]] = set()
        self.time_step = 0.1  # 仿真时间步长(秒)
        self.simulation_time = 0.0
        self.running = False
        self.thread = None
        
        # 初始化状态记录
        self.state_history = deque(maxlen=1000)
        import json
        import logging
        
        # 初始化基础障碍物
        self._init_obstacles()
    
    def _init_obstacles(self):
        """初始化地图障碍物"""
        # 从地图服务获取障碍物信息
        map_data = self.map_service.get_map_data()
        if 'obstacles' in map_data:
            self.obstacles.update(tuple(obs) for obs in map_data['obstacles'])
    
    def add_vehicle(self, vehicle: MiningVehicle):
        """添加车辆到仿真环境"""
        if vehicle.vehicle_id in self.vehicles:
            raise ValueError(f"车辆 {vehicle.vehicle_id} 已存在")
        self.vehicles[vehicle.vehicle_id] = vehicle
    
    def remove_vehicle(self, vehicle_id: int):
        """从仿真环境移除车辆"""
        if vehicle_id in self.vehicles:
            del self.vehicles[vehicle_id]
    
    def start_simulation(self):
        """启动仿真线程"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.thread.start()
    
    def stop_simulation(self):
        """停止仿真"""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _simulation_loop(self):
        """仿真主循环（包含联合训练）"""
        # 初始化性能监控
        perf_stats = {
            'update_time': 0.0,
            'collision_time': 0.0,
            'total_cycles': 0,
            'qmix_training_time': 0.0
        }
        
        # 初始化状态记录
        self.state_history = deque(maxlen=1000)
        
        while self.running:
            cycle_start = time.time()
            
            # 记录当前状态
            state = self.get_environment_state()
            self.state_history.append(state)
            
            # 随机事件生成
            if random.random() < 0.001:  # 0.1%概率生成随机事件
                self._generate_random_event()
            
            # 更新所有车辆状态
            update_start = time.time()
            for vehicle in self.vehicles.values():
                self._update_vehicle(vehicle)
            perf_stats['update_time'] += time.time() - update_start
            
            # 检测碰撞
            collision_start = time.time()
            self._detect_collisions()
            perf_stats['collision_time'] += time.time() - collision_start
            
            # QMIX联合训练
            if hasattr(self, 'qmix_training') and self.qmix_training:
                train_start = time.time()
                self._qmix_joint_training(state)
                perf_stats['qmix_training_time'] += time.time() - train_start
            
            # 更新时间
            self.simulation_time += self.time_step
            perf_stats['total_cycles'] += 1
            
            # 控制仿真速度
            elapsed = time.time() - cycle_start
            if elapsed < self.time_step:
                time.sleep(self.time_step - elapsed)
            
            # 每小时输出性能报告
            if self.simulation_time % 3600 < self.time_step:
                self._log_performance(perf_stats)
    
    def _update_vehicle(self, vehicle: MiningVehicle):
        """更新单个车辆状态"""
        if vehicle.status == VehicleState.EN_ROUTE:
            # 模拟车辆移动
            vehicle.update_position()
            
            # 检查是否到达路径终点
            if vehicle.path_index >= len(vehicle.current_path) - 1:
                vehicle.status = VehicleState.IDLE
    
    def _detect_collisions(self):
        """检测车辆间及车辆与障碍物的碰撞"""
        positions = defaultdict(list)
        
        # 收集所有车辆位置
        for vid, vehicle in self.vehicles.items():
            pos = vehicle.current_location
            positions[pos].append(vid)
            
            # 检查与障碍物的碰撞
            if pos in self.obstacles:
                vehicle.perform_emergency_stop()
        
        # 检查车辆间碰撞
        for pos, vids in positions.items():
            if len(vids) > 1:
                for vid in vids:
                    self.vehicles[vid].perform_emergency_stop()
    
    def get_environment_state(self) -> Dict:
        """获取当前环境状态快照"""
        state = {
            'time': self.simulation_time,
            'vehicles': [
                {
                    'id': vid,
                    'position': v.current_location,
                    'status': v.status.name,
                    'path': v.current_path,
                    'speed': v.speed if hasattr(v, 'speed') else 0,
                    'load': v.current_load if hasattr(v, 'current_load') else 0
                }
                for vid, v in self.vehicles.items()
            ],
            'obstacles': list(self.obstacles)
        }
        
        # 记录状态到日志文件
        if not hasattr(self, 'state_logger'):
            import logging
            self.state_logger = logging.getLogger('simulation_state')
            self.state_logger.setLevel(logging.INFO)
            handler = logging.FileHandler('simulation_state.log')
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.state_logger.addHandler(handler)
        
        self.state_logger.info(json.dumps(state))
        
        # 写入JSON文件
        with open('simulation_state.json', 'w') as f:
            json.dump(state, f, indent=2)
            
        return state

class MockMapService(MapService):
    """用于测试的模拟地图服务"""
    
    def __init__(self):
        super().__init__()
        self._obstacles = [
            (50.0, 50.0), 
            (-30.0, -30.0),
            (100.0, -100.0)
        ]
    
    def get_map_data(self) -> Dict:
        return {
            'obstacles': self._obstacles,
            'bounds': [(-200, -200), (200, 200)]
        }
    
    def plan_route(self, start, end, vehicle_type):
        """生成简单的直线路径"""
        return {
            'path': [start, end],
            'distance': np.linalg.norm(np.array(end) - np.array(start))
        }
        
    def plan_return_path(self, current_pos, base_pos, vehicle_type):
        """生成返回基地的简单直线路径"""
        return {
            'path': [current_pos, base_pos],
            'distance': np.linalg.norm(np.array(base_pos) - np.array(current_pos))
        }

if __name__ == "__main__":
    """测试虚拟环境"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟环境
    mock_map = MockMapService()
    env = VirtualEnvironment(mock_map)
    
    # 添加测试车辆
    config = {
        'max_capacity': 5000,
        'current_location': (0.0, 0.0),
        'base_location': (0.0, 0.0)
    }
    vehicle = MiningVehicle(1, mock_map, config)
    env.add_vehicle(vehicle)
    
    # 启动仿真
    env.start_simulation()
    
    try:
        # 模拟任务分配
        from models.task import TransportTask
        task = TransportTask(
            task_id="TEST-001",
            start_point=(0.0, 0.0),
            end_point=(100.0, 100.0),
            task_type="loading"
        )
        vehicle.assign_task(task)
        
        # 运行仿真一段时间
        time.sleep(5)
        
        # 打印状态
        print("当前环境状态:")
        print(env.get_environment_state())
        
    finally:
        env.stop_simulation()