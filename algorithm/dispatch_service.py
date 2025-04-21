import sys
import os
import logging
import threading
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from algorithm.optimized_path_planner import HybridPathPlanner
from algorithm.cbs import ConflictBasedSearch
from algorithm.map_service import MapService

class DispatchSystem:
    """调度系统核心 - 简化版，专注于调度和冲突解决"""
    
    def __init__(self, path_planner: HybridPathPlanner, map_service: MapService):
        """初始化调度系统"""
        # 核心组件
        self.path_planner = path_planner
        self.map_service = map_service
        self.cbs = ConflictBasedSearch(path_planner)
        
        # 设置path_planner的dispatch引用
        self.path_planner.dispatch = self
        
        # 车辆和任务管理
        self.vehicles = {}  # 车辆字典 {vehicle_id: vehicle}
        self.task_queue = deque()  # 等待分配的任务队列
        self.active_tasks = {}  # 正在执行的任务 {task_id: task}
        self.completed_tasks = {}  # 已完成的任务 {task_id: task}
        
        # 路径规划和冲突管理
        self.vehicle_paths = {}  # 车辆当前路径 {vehicle_id: path}
        
        # 性能指标
        self.metrics = {
            'tasks_completed': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'planning_count': 0
        }
        
        # 同步锁
        self.lock = threading.RLock()
        
        logging.info("调度系统初始化完成")
    
    def add_vehicle(self, vehicle: MiningVehicle):
        """添加车辆到调度系统"""
        with self.lock:
            self.vehicles[vehicle.vehicle_id] = vehicle
            logging.info(f"已添加车辆 {vehicle.vehicle_id}")
    
    def add_task(self, task: TransportTask):
        """添加任务到队列"""
        with self.lock:
            self.task_queue.append(task)
            logging.info(f"已添加任务 {task.task_id} 到队列")
    
    def scheduling_cycle(self):
        """调度循环主函数"""
        with self.lock:
            try:
                # 1. 为空闲车辆分配任务
                self._assign_tasks_to_idle_vehicles()
                
                # 2. 更新车辆位置
                self._update_vehicle_positions()
                
                # 3. 检查任务完成情况
                self._check_completed_tasks()
                
                # 4. 检测和解决路径冲突
                self._resolve_path_conflicts()
                
                return True
            except Exception as e:
                logging.error(f"调度循环执行出错: {str(e)}")
                return False
    
    def _assign_tasks_to_idle_vehicles(self):
        """为空闲车辆分配任务"""
        # 获取所有空闲车辆
        idle_vehicles = [v for v in self.vehicles.values() 
                        if v.state == VehicleState.IDLE and not v.current_task]
        
        if not idle_vehicles or not self.task_queue:
            return
        
        for vehicle in idle_vehicles:
            if not self.task_queue:
                break
                
            # 取出队列中的下一个任务
            task = self.task_queue.popleft()
            
            # 规划路径
            try:
                path = self.path_planner.plan_path(
                    vehicle.current_location, 
                    task.end_point,
                    vehicle
                )
                
                if path and len(path) > 1:
                    # 分配任务给车辆
                    vehicle.assign_task(task)
                    vehicle.assign_path(path)
                    
                    # 更新任务状态
                    self.active_tasks[task.task_id] = task
                    
                    # 更新路径记录
                    self.vehicle_paths[str(vehicle.vehicle_id)] = path
                    
                    logging.info(f"已将任务 {task.task_id} 分配给车辆 {vehicle.vehicle_id}，路径长度: {len(path)}")
                    self.metrics['planning_count'] += 1
                else:
                    # 路径规划失败，放回队列末尾
                    logging.warning(f"无法为任务 {task.task_id} 规划路径，放回队列")
                    self.task_queue.append(task)
            except Exception as e:
                logging.error(f"任务分配出错: {str(e)}")
                # 出错时放回队列
                self.task_queue.append(task)
    
    def _update_vehicle_positions(self):
        """更新车辆位置"""
        for vehicle in self.vehicles.values():
            # 检查车辆是否有路径
            if vehicle.current_path and vehicle.path_index < len(vehicle.current_path) - 1:
                # 移动到下一个路径点
                vehicle.path_index += 1
                vehicle.current_location = vehicle.current_path[vehicle.path_index]
    
    def _check_completed_tasks(self):
        """检查任务完成情况"""
        for vehicle in self.vehicles.values():
            # 检查是否到达终点
            if (vehicle.current_task and vehicle.current_path and 
                vehicle.path_index >= len(vehicle.current_path) - 1):
                
                task = vehicle.current_task
                task_id = task.task_id
                
                # 标记任务完成
                task.is_completed = True
                
                # 更新任务状态
                if task_id in self.active_tasks:
                    self.completed_tasks[task_id] = task
                    del self.active_tasks[task_id]
                
                # 更新车辆状态
                vehicle.current_task = None
                vehicle.state = VehicleState.IDLE
                vehicle.current_path = []
                vehicle.path_index = 0
                
                # 更新路径记录
                if str(vehicle.vehicle_id) in self.vehicle_paths:
                    del self.vehicle_paths[str(vehicle.vehicle_id)]
                
                # 更新指标
                self.metrics['tasks_completed'] += 1
                
                logging.info(f"车辆 {vehicle.vehicle_id} 已完成任务 {task_id}")
    
    def _resolve_path_conflicts(self):
        """检测并解决路径冲突"""
        # 收集所有车辆的路径
        active_paths = {}
        for vid, vehicle in self.vehicles.items():
            if (vehicle.current_path and vehicle.path_index < len(vehicle.current_path) - 1):
                # 只考虑当前位置之后的路径
                remaining_path = vehicle.current_path[vehicle.path_index:]
                if len(remaining_path) > 1:
                    active_paths[str(vid)] = remaining_path
        
        if len(active_paths) < 2:
            return  # 不足两条路径，无需检查冲突
        
        # 使用CBS检测冲突
        conflicts = self.cbs.find_conflicts(active_paths)
        
        if conflicts:
            # 更新指标
            self.metrics['conflicts_detected'] += len(conflicts)
            
            # 使用CBS解决冲突
            resolved_paths = self.cbs.resolve_conflicts(active_paths)
            
            # 统计修改的路径数
            changed_paths = 0
            
            # 应用解决方案
            for vid_str, new_path in resolved_paths.items():
                if new_path != active_paths.get(vid_str, []):
                    vid = int(vid_str)
                    vehicle = self.vehicles.get(vid)
                    
                    if vehicle:
                        # 确保新路径从当前位置开始
                        current_pos = vehicle.current_location
                        if new_path[0] != current_pos:
                            new_path.insert(0, current_pos)
                        
                        # 更新车辆路径
                        vehicle.assign_path(new_path)
                        vehicle.path_index = 0
                        
                        # 更新路径记录
                        self.vehicle_paths[vid_str] = new_path
                        changed_paths += 1
            
            # 更新指标
            self.metrics['conflicts_resolved'] += changed_paths
            
            if changed_paths > 0:
                logging.info(f"解决了 {len(conflicts)} 个冲突，修改了 {changed_paths} 条路径")
    
    def get_status(self):
        """获取系统状态概览"""
        with self.lock:
            return {
                'vehicles': {
                    'total': len(self.vehicles),
                    'idle': sum(1 for v in self.vehicles.values() if v.state == VehicleState.IDLE),
                    'active': sum(1 for v in self.vehicles.values() if v.state != VehicleState.IDLE)
                },
                'tasks': {
                    'queued': len(self.task_queue),
                    'active': len(self.active_tasks),
                    'completed': len(self.completed_tasks)
                },
                'metrics': self.metrics
            }
    
    def start_scheduling(self, interval=1.0):
        """启动调度循环"""
        self.running = True
        
        while self.running:
            self.scheduling_cycle()
            time.sleep(interval)
    
    def stop_scheduling(self):
        """停止调度循环"""
        self.running = False


if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    import logging
    
    # 设置日志级别
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 初始化基础组件
    print("初始化系统组件...")
    map_service = MapService()
    path_planner = HybridPathPlanner(map_service)
    
    # 创建调度系统
    dispatch = DispatchSystem(path_planner, map_service)
    
    # 创建测试用障碍物
    obstacles = []
    map_size = 200
    
    # 创建水平障碍物
    for y in range(80, 90):
        for x in range(20, 80):
            obstacles.append((x, y))
    
    for y in range(120, 130):
        for x in range(100, 180):
            obstacles.append((x, y))
    
    # 创建垂直障碍物
    for x in range(100, 110):
        for y in range(20, 80):
            obstacles.append((x, y))
    
    for x in range(150, 160):
        for y in range(130, 180):
            obstacles.append((x, y))
    
    # 设置障碍物
    path_planner.obstacle_grids = set(obstacles)
    
    # 创建测试用车辆
    vehicles = []
    start_positions = [
        (30, 30),   # 左下角
        (30, 170),  # 左上角
        (170, 30),  # 右下角
        (170, 170), # 右上角
        (100, 30),  # 下中
    ]
    
    for i in range(5):
        config = {
            'current_location': start_positions[i % len(start_positions)],
            'max_capacity': 50,
            'max_speed': random.uniform(5.0, 8.0),
            'base_location': (100, 190),  # 基地位置
            'status': VehicleState.IDLE
        }
        
        vehicle = MiningVehicle(
            vehicle_id=i+1,
            map_service=map_service,
            config=config
        )
        
        # 确保必要属性存在
        if not hasattr(vehicle, 'current_path'):
            vehicle.current_path = []
        if not hasattr(vehicle, 'path_index'):
            vehicle.path_index = 0
        
        # 添加到调度系统
        dispatch.add_vehicle(vehicle)
        vehicles.append(vehicle)
    
    # 创建测试用任务
    key_points = [
        (30, 30),   # 左下
        (30, 170),  # 左上
        (170, 30),  # 右下
        (170, 170), # 右上
        (100, 170), # 上中
        (100, 30),  # 下中
        (30, 100),  # 左中
        (170, 100), # 右中
    ]
    
    for i in range(10):
        # 随机选择起点和终点（确保不同）
        start_idx = random.randint(0, len(key_points)-1)
        end_idx = start_idx
        while end_idx == start_idx:
            end_idx = random.randint(0, len(key_points)-1)
            
        start_point = key_points[start_idx]
        end_point = key_points[end_idx]
        
        # 随机任务类型
        task_type = random.choice(["transport", "loading", "unloading"])
        
        # 创建任务
        task = TransportTask(
            task_id=f"TASK-{i+1}",
            start_point=start_point,
            end_point=end_point,
            task_type=task_type,
            priority=random.randint(1, 3)
        )
        
        # 添加到调度系统
        dispatch.add_task(task)
    
    # 设置可视化
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    
    # 绘制障碍物
    obstacle_x = [p[0] for p in obstacles]
    obstacle_y = [p[1] for p in obstacles]
    plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', alpha=0.5, s=10)
    
    # 绘制初始车辆位置
    for vehicle in vehicles:
        plt.scatter(
            vehicle.current_location[0], 
            vehicle.current_location[1], 
            c=f'C{vehicle.vehicle_id % 10}', 
            marker='o', 
            s=100, 
            label=f'Vehicle {vehicle.vehicle_id}'
        )
    
    plt.legend()
    plt.title("初始车辆位置和障碍物")
    plt.grid(True)
    plt.savefig("initial_state.png")
    plt.close()
    
    # 运行调度周期
    print("\n开始调度测试...")
    print(f"初始状态: {dispatch.get_status()}")
    
    # 运行10个周期
    for cycle in range(10):
        print(f"\n执行调度周期 {cycle+1}/10")
        
        # 执行一次调度
        start_time = time.time()
        dispatch.scheduling_cycle()
        elapsed = time.time() - start_time
        
        # 获取状态
        status = dispatch.get_status()
        
        # 打印状态
        print(f"调度耗时: {elapsed*1000:.2f}毫秒")
        print(f"车辆状态: 总数={status['vehicles']['total']}, "
              f"空闲={status['vehicles']['idle']}, "
              f"活动={status['vehicles']['active']}")
        print(f"任务状态: 队列={status['tasks']['queued']}, "
              f"活动={status['tasks']['active']}, "
              f"完成={status['tasks']['completed']}")
        print(f"冲突统计: 检测={status['metrics']['conflicts_detected']}, "
              f"解决={status['metrics']['conflicts_resolved']}")
        
        # 每隔几个周期可视化当前状态
        if cycle % 2 == 0:
            # 绘制当前状态
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            ax.set_xlim(0, map_size)
            ax.set_ylim(0, map_size)
            
            # 绘制障碍物
            plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', alpha=0.5, s=10)
            
            # 绘制车辆位置和路径
            for vehicle in vehicles:
                # 绘制车辆
                plt.scatter(
                    vehicle.current_location[0], 
                    vehicle.current_location[1], 
                    c=f'C{vehicle.vehicle_id % 10}', 
                    marker='o', 
                    s=100
                )
                
                # 标注车辆ID
                plt.text(
                    vehicle.current_location[0] + 2, 
                    vehicle.current_location[1] + 2, 
                    str(vehicle.vehicle_id),
                    fontsize=12
                )
                
                # 绘制路径
                if vehicle.current_path and len(vehicle.current_path) > 1:
                    path_x = [p[0] for p in vehicle.current_path]
                    path_y = [p[1] for p in vehicle.current_path]
                    plt.plot(path_x, path_y, c=f'C{vehicle.vehicle_id % 10}', linestyle='--', alpha=0.7)
            
            plt.title(f"周期 {cycle+1} - 车辆位置和路径")
            plt.grid(True)
            plt.savefig(f"cycle_{cycle+1}_state.png")
            plt.close()
        
        # 暂停一下，让模拟更真实
        time.sleep(0.5)
    
    # 最终状态
    final_status = dispatch.get_status()
    print("\n测试完成！")
    print(f"最终状态: {final_status}")
    print(f"完成任务数: {final_status['metrics']['tasks_completed']}")
    print(f"检测到的冲突: {final_status['metrics']['conflicts_detected']}")
    print(f"解决的冲突: {final_status['metrics']['conflicts_resolved']}")
    
    # 绘制最终状态
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    
    # 绘制障碍物
    plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', alpha=0.5, s=10)
    
    # 绘制车辆最终位置
    for vehicle in vehicles:
        plt.scatter(
            vehicle.current_location[0], 
            vehicle.current_location[1], 
            c=f'C{vehicle.vehicle_id % 10}', 
            marker='o', 
            s=100
        )
        
        # 标注车辆ID
        plt.text(
            vehicle.current_location[0] + 2, 
            vehicle.current_location[1] + 2, 
            str(vehicle.vehicle_id),
            fontsize=12
        )
    
    plt.title("最终车辆位置")
    plt.grid(True)
    plt.savefig("final_state.png")
    plt.close()
    
    print("\n测试过程中生成了以下图片:")
    print("- initial_state.png: 初始车辆位置和障碍物")
    print("- cycle_X_state.png: 各个周期的车辆位置和路径")
    print("- final_state.png: 最终车辆位置")