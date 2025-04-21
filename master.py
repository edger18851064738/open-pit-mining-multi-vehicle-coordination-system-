#!/usr/bin/env python3
"""
露天矿多车协同调度系统 - 主程序
==============================================

本系统实现了露天矿场景下的多车协同调度功能，包括：
1. 路径规划与优化
2. 冲突检测与解决 
3. 任务分配与执行
4. 实时可视化与监控

使用PyQt5和PyQtGraph实现高性能可视化，支持路径规划和任务执行过程的实时监控。
"""

import os
import sys
import time
import math
import random
import logging
import threading
import argparse
from typing import List, Tuple, Dict, Optional, Set, Any
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入项目模块
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from utils.geo_tools import GeoUtils
from algorithm.map_service import MapService
from algorithm.optimized_path_planner import HybridPathPlanner
from algorithm.cbs import ConflictBasedSearch

# PyQt和PyQtGraph导入
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, 
        QGridLayout, QSplitter, QTextEdit, QGroupBox, QComboBox, QSlider, QCheckBox
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    import pyqtgraph as pg
    visualization_available = True
except ImportError:
    visualization_available = False
    print("PyQt5或PyQtGraph未安装，将使用命令行模式运行")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mining_system")

# 设置PyQtGraph全局配置
if visualization_available:
    pg.setConfigOptions(antialias=True, background='w', foreground='k')


class SimulationThread(QThread if visualization_available else threading.Thread):
    """模拟线程，处理车辆移动和任务执行"""
    
    if visualization_available:
        update_signal = pyqtSignal(dict)  # 发送更新数据的信号
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = False
        self.paused = False
    
    def run(self):
        """线程运行函数"""
        self.running = True
        start_time = time.time()
        last_status_time = start_time
        last_conflict_check_time = start_time
        
        while self.running and time.time() - start_time < self.controller.duration:
            if not self.paused:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # 更新车辆位置
                self.controller.update_vehicles()
                
                # 分配新任务给空闲车辆
                self.controller.assign_new_tasks()
                
                # 定期检查冲突
                if current_time - last_conflict_check_time >= self.controller.conflict_check_interval:
                    self.controller.resolve_conflicts()
                    last_conflict_check_time = current_time
                
                # 定期输出状态
                if current_time - last_status_time >= self.controller.status_interval:
                    self.controller.print_status(elapsed)
                    last_status_time = current_time
                
                # 发送更新信号（如果可视化可用）
                if visualization_available:
                    update_data = self.controller.get_update_data()
                    self.update_signal.emit(update_data)
            
            # 控制模拟速度
            time.sleep(self.controller.update_interval)
        
        # 模拟结束，打印结果
        self.controller.print_final_results(time.time() - start_time)
        self.running = False
    
    def stop(self):
        """停止线程"""
        self.running = False
        if hasattr(self, 'wait'):
            self.wait()  # 仅在QThread中可用
    
    def pause(self):
        """暂停模拟"""
        self.paused = True
    
    def resume(self):
        """继续模拟"""
        self.paused = False


class SystemController:
    """系统控制器，管理模拟和调度过程"""
    
    def __init__(self, config=None):
        """初始化系统控制器"""
        self.config = config or self._get_default_config()
        
        # 提取配置参数
        self.duration = self.config.get('duration', 120)
        self.update_interval = self.config.get('update_interval', 0.1)
        self.status_interval = self.config.get('status_interval', 5.0)
        self.conflict_check_interval = self.config.get('conflict_check_interval', 1.0)
        self.num_vehicles = self.config.get('num_vehicles', 5)
        self.num_tasks = self.config.get('num_tasks', 10)
        
        # 初始化系统组件 (基础组件)
        logger.info("初始化系统组件...")
        self._init_basic_components()
        
        # 创建车辆和任务
        self.vehicles = self._create_vehicles()
        self.tasks = self._create_tasks()
        
        # 创建障碍物 (在创建任务和车辆之后)
        self.obstacles = self._create_obstacles()
        
        # 将障碍物应用到规划器
        self.path_planner.obstacle_grids = set(self.obstacles)
        
        # 任务和冲突计数
        self.tasks_completed = 0
        self.conflicts_detected = 0
        self.conflicts_resolved = 0
        
        # 已完成的任务
        self.completed_tasks = []
        
        # 路径和冲突相关
        self.vehicle_paths = {}  # 记录每个车辆的当前路径
        self.displayed_paths = {}  # 记录已显示的路径（可视化用）
        self.path_items = []  # 存储路径可视化项
        
        logger.info("系统控制器初始化完成")
    
    def _get_default_config(self):
        """获取默认配置"""
        return {
            'duration': 120,  # 模拟时长（秒）
            'update_interval': 0.1,  # 更新间隔（秒）
            'status_interval': 5.0,  # 状态输出间隔（秒）
            'conflict_check_interval': 1.0,  # 冲突检查间隔（秒）
            'num_vehicles': 5,  # 车辆数量
            'num_tasks': 10,  # 任务数量
            'map_size': 200,  # 地图尺寸
            'obstacle_density': 0.15,  # 障碍物密度
            'visualization': True  # 是否启用可视化
        }
    def _init_basic_components(self):
        """初始化系统基础组件"""
        try:
            # 创建地图和路径规划相关组件
            self.geo_utils = GeoUtils()
            self.map_service = MapService()
            self.path_planner = HybridPathPlanner(self.map_service)
            
            # 创建冲突解决器
            self.cbs = ConflictBasedSearch(self.path_planner)
            
            # 为路径规划器设置mock dispatch对象
            class MockDispatch:
                def __init__(self):
                    self.vehicles = {}
            
            self.mock_dispatch = MockDispatch()
            self.path_planner.dispatch = self.mock_dispatch
            
        except Exception as e:
            logger.error(f"初始化系统组件失败: {str(e)}")
            raise    
    def _init_system_components(self):
        """初始化系统核心组件"""
        try:
            # 创建地图和路径规划相关组件
            self.geo_utils = GeoUtils()
            self.map_service = MapService()
            self.path_planner = HybridPathPlanner(self.map_service)
            
            # 创建冲突解决器
            self.cbs = ConflictBasedSearch(self.path_planner)
            
            # 为路径规划器设置mock dispatch对象
            class MockDispatch:
                def __init__(self):
                    self.vehicles = {}
            
            self.mock_dispatch = MockDispatch()
            self.path_planner.dispatch = self.mock_dispatch
            
            # 创建障碍物
            self.obstacles = self._create_obstacles()
            
            # 将障碍物应用到规划器
            self.path_planner.obstacle_grids = set(self.obstacles)
            
        except Exception as e:
            logger.error(f"初始化系统组件失败: {str(e)}")
            raise
    
    def _create_vehicles(self):
        """创建测试车辆"""
        vehicles = []
        
        # 车辆起始位置，分散在地图边缘
        start_positions = [
            (180, 180),  # 右上角
            (20, 180),   # 左上角
            (20, 20),    # 左下角
            (180, 20),   # 右下角
            (100, 20),   # 下方中央
        ]
        
        # 确保有足够的起始位置
        while len(start_positions) < self.num_vehicles:
            start_positions.append(
                (random.randint(20, 180), random.randint(20, 180))
            )
        
        # 创建车辆
        for i in range(self.num_vehicles):
            config = {
                'current_location': start_positions[i],
                'max_capacity': 50,
                'max_speed': random.uniform(5.0, 8.0),
                'base_location': (100, 190),  # 基地位置
                'status': VehicleState.IDLE
            }
            
            vehicle = MiningVehicle(
                vehicle_id=i+1,
                map_service=self.map_service,
                config=config
            )
            
            # 确保必要属性存在
            if not hasattr(vehicle, 'current_path'):
                vehicle.current_path = []
            if not hasattr(vehicle, 'path_index'):
                vehicle.path_index = 0
            
            # 为路径规划器的dispatch添加车辆
            self.mock_dispatch.vehicles[i+1] = vehicle
            
            # 如果使用可视化，添加颜色属性
            if visualization_available:
                vehicle.color = pg.intColor(i % 10)
            
            vehicles.append(vehicle)
        
        logger.info(f"已创建{len(vehicles)}辆车辆")
        return vehicles
    
    def _create_tasks(self):
        """创建测试任务"""
        tasks = []
        
        # 任务起点和终点
        locations = [
            (30, 30),   # 左下区域
            (30, 170),  # 左上区域
            (170, 30),  # 右下区域
            (170, 170), # 右上区域
            (100, 30),  # 下方中央
            (100, 170), # 上方中央
            (30, 100),  # 左侧中央
            (170, 100), # 右侧中央
        ]
        
        # 创建任务
        for i in range(self.num_tasks):
            # 随机选择起点和终点（确保不同）
            start_idx = random.randint(0, len(locations)-1)
            end_idx = start_idx
            while end_idx == start_idx:
                end_idx = random.randint(0, len(locations)-1)
                
            start_point = locations[start_idx]
            end_point = locations[end_idx]
            
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
            
            tasks.append(task)
        
        logger.info(f"已创建{len(tasks)}个任务")
        return tasks
    
    def _create_obstacles(self):
        """创建迷宫式障碍物"""
        obstacles = []
        map_size = self.config.get('map_size', 200)
        
        # 获取所有任务的起点和终点，作为安全区域
        safe_points = []
        for task in self.tasks:
            safe_points.append(task.start_point)
            safe_points.append(task.end_point)
        
        # 添加车辆初始位置作为安全区域
        for vehicle in self.vehicles:
            safe_points.append(vehicle.current_location)
        
        # 定义安全半径（起点终点周围不放障碍物）
        safe_radius = 20
        
        # 创建迷宫式水平墙壁
        num_h_walls = 5
        wall_thickness = 10
        for i in range(num_h_walls):
            # 水平墙的Y坐标（均匀分布）
            y = int(map_size * (i + 1) / (num_h_walls + 1))
            # 墙的起始和结束X坐标（留一个通道）
            passage_start = random.randint(20, map_size - 80)
            passage_end = passage_start + random.randint(40, 60)
            
            # 左侧墙体
            for x in range(10, passage_start):
                for dy in range(wall_thickness):
                    point = (x, y + dy - wall_thickness//2)
                    # 检查该点是否在安全区域内
                    if not self._is_near_safe_points(point, safe_points, safe_radius):
                        obstacles.append(point)
            
            # 右侧墙体
            for x in range(passage_end, map_size - 10):
                for dy in range(wall_thickness):
                    point = (x, y + dy - wall_thickness//2)
                    # 检查该点是否在安全区域内
                    if not self._is_near_safe_points(point, safe_points, safe_radius):
                        obstacles.append(point)
        
        # 创建迷宫式垂直墙壁
        num_v_walls = 5
        for i in range(num_v_walls):
            # 垂直墙的X坐标
            x = int(map_size * (i + 1) / (num_v_walls + 1))
            # 墙的起始和结束Y坐标（留一个通道）
            passage_start = random.randint(20, map_size - 80)
            passage_end = passage_start + random.randint(40, 60)
            
            # 下部墙体
            for y in range(10, passage_start):
                for dx in range(wall_thickness):
                    point = (x + dx - wall_thickness//2, y)
                    # 检查该点是否在安全区域内
                    if not self._is_near_safe_points(point, safe_points, safe_radius):
                        obstacles.append(point)
            
            # 上部墙体
            for y in range(passage_end, map_size - 10):
                for dx in range(wall_thickness):
                    point = (x + dx - wall_thickness//2, y)
                    # 检查该点是否在安全区域内
                    if not self._is_near_safe_points(point, safe_points, safe_radius):
                        obstacles.append(point)
        
        # 添加一些随机小障碍物
        num_random_obstacles = 10
        for _ in range(num_random_obstacles):
            ox = random.randint(20, map_size - 20)
            oy = random.randint(20, map_size - 20)
            size = random.randint(5, 15)
            
            for dx in range(-size//2, size//2):
                for dy in range(-size//2, size//2):
                    point = (ox + dx, oy + dy)
                    # 检查该点是否在安全区域内
                    if not self._is_near_safe_points(point, safe_points, safe_radius):
                        obstacles.append(point)
        
        logger.info(f"已创建{len(obstacles)}个障碍物点")
        return obstacles

    def _is_near_safe_points(self, point, safe_points, safe_radius):
        """检查点是否靠近安全点"""
        x, y = point
        for sx, sy in safe_points:
            distance = math.sqrt((x - sx)**2 + (y - sy)**2)
            if distance < safe_radius:
                return True
        return False
    
    def _rasterize_polygon(self, polygon):
        """将多边形光栅化为点集"""
        if not polygon or len(polygon) < 3:
            return []
            
        points = []
        
        # 找出多边形的边界框
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 对边界框内的每个点检查是否在多边形内
        for x in range(int(min_x), int(max_x) + 1):
            for y in range(int(min_y), int(max_y) + 1):
                if self._point_in_polygon((x, y), polygon):
                    points.append((x, y))
                    
        return points
    
    def _point_in_polygon(self, point, polygon):
        """判断点是否在多边形内 (射线法)"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def assign_initial_tasks(self):
        """分配初始任务给车辆"""
        # 确保每辆车都有任务
        for i, vehicle in enumerate(self.vehicles):
            if i < len(self.tasks):
                task = self.tasks[i]
                
                # 规划路径
                path = self.path_planner.plan_path(vehicle.current_location, task.end_point, vehicle)
                
                # 保存车辆路径
                self.vehicle_paths[str(vehicle.vehicle_id)] = path
                
                # 检查冲突
                vehicle_paths = {str(v.vehicle_id): self.vehicle_paths.get(str(v.vehicle_id), []) 
                               for v in self.vehicles if str(v.vehicle_id) in self.vehicle_paths}
                
                # 解决冲突
                resolved_paths = self.cbs.resolve_conflicts(vehicle_paths)
                
                # 更新路径
                if str(vehicle.vehicle_id) in resolved_paths:
                    path = resolved_paths[str(vehicle.vehicle_id)]
                
                # 分配任务和路径
                vehicle.assign_task(task)
                vehicle.assign_path(path)
                task.assigned_to = vehicle.vehicle_id
                
                # 更新车辆路径记录
                self.vehicle_paths[str(vehicle.vehicle_id)] = path
                
                logger.info(f"已将任务{task.task_id}分配给车辆{vehicle.vehicle_id}，路径长度: {len(path)}")
    
    def update_vehicles(self):
        """更新车辆位置"""
        for vehicle in self.vehicles:
            # 检查车辆是否有路径
            if vehicle.current_path and vehicle.path_index < len(vehicle.current_path) - 1:
                # 移动到下一个路径点
                vehicle.path_index += 1
                vehicle.current_location = vehicle.current_path[vehicle.path_index]
                
                # 检查是否到达终点
                if vehicle.path_index >= len(vehicle.current_path) - 1:
                    # 车辆到达终点，任务完成
                    self._handle_task_completion(vehicle)
    
    def _handle_task_completion(self, vehicle):
        """处理任务完成"""
        if not vehicle.current_task:
            return
            
        # 获取任务信息
        task = vehicle.current_task
        task_id = task.task_id
        
        # 标记任务完成
        task.is_completed = True
        self.tasks_completed += 1
        self.completed_tasks.append(task)
        
        # 更新车辆状态
        vehicle.current_task = None
        vehicle.state = VehicleState.IDLE
        vehicle.current_path = []
        vehicle.path_index = 0
        
        # 从路径记录中移除
        if str(vehicle.vehicle_id) in self.vehicle_paths:
            del self.vehicle_paths[str(vehicle.vehicle_id)]
        
        logger.info(f"车辆{vehicle.vehicle_id}完成任务{task_id}")
    
    def assign_new_tasks(self):
        """分配新任务给空闲车辆"""
        # 获取空闲车辆
        idle_vehicles = [v for v in self.vehicles if v.state == VehicleState.IDLE]
        
        # 获取可用任务 (未完成且未分配)
        available_tasks = []
        for task in self.tasks:
            if hasattr(task, 'is_completed') and not task.is_completed:
                # 检查任务是否已被分配给正在执行的车辆
                already_assigned = False
                for v in self.vehicles:
                    if v.current_task and v.current_task.task_id == task.task_id:
                        already_assigned = True
                        break
                
                if not already_assigned:
                    available_tasks.append(task)
            elif not hasattr(task, 'is_completed') or (not task.is_completed and not hasattr(task, 'assigned_to')):
                available_tasks.append(task)
        
        # 分配任务
        for vehicle in idle_vehicles:
            if not available_tasks:
                break
                
            # 选择任务 (基于优先级和距离)
            best_task = None
            best_score = float('inf')
            
            for task in available_tasks:
                # 计算距离分数
                distance = math.dist(vehicle.current_location, task.end_point)
                
                # 计算优先级分数 (优先级越高，分数越低)
                priority_score = 4 - task.priority  # 优先级1-3变成分数3-1
                
                # 综合分数
                score = distance * priority_score
                
                if score < best_score:
                    best_score = score
                    best_task = task
            
            if best_task:
                # 规划路径
                path = self.path_planner.plan_path(vehicle.current_location, best_task.end_point, vehicle)
                
                # 检查冲突
                vehicle_paths = {str(v.vehicle_id): self.vehicle_paths.get(str(v.vehicle_id), []) 
                               for v in self.vehicles if hasattr(v, 'current_path') and v.current_path}
                vehicle_paths[str(vehicle.vehicle_id)] = path
                
                # 解决冲突
                resolved_paths = self.cbs.resolve_conflicts(vehicle_paths)
                
                # 更新路径
                if str(vehicle.vehicle_id) in resolved_paths:
                    path = resolved_paths[str(vehicle.vehicle_id)]
                
                # 分配任务和路径
                vehicle.assign_task(best_task)
                vehicle.assign_path(path)
                best_task.assigned_to = vehicle.vehicle_id
                
                # 更新车辆路径记录
                self.vehicle_paths[str(vehicle.vehicle_id)] = path
                
                # 从可用任务中移除
                available_tasks.remove(best_task)
                
                logger.info(f"已将任务{best_task.task_id}分配给空闲车辆{vehicle.vehicle_id}，路径长度: {len(path)}")
    
    def resolve_conflicts(self):
        """检测并解决路径冲突"""
        # 收集所有车辆的路径
        vehicle_paths = {}
        for vehicle in self.vehicles:
            if (hasattr(vehicle, 'current_path') and vehicle.current_path and 
                len(vehicle.current_path) > vehicle.path_index):
                # 只考虑当前位置之后的路径
                remaining_path = vehicle.current_path[vehicle.path_index:]
                if len(remaining_path) > 1:
                    vehicle_paths[str(vehicle.vehicle_id)] = remaining_path
        
        # 检测冲突
        if vehicle_paths:
            # 记录冲突检测前的路径数
            before_count = len(vehicle_paths)
            
            # 检测冲突
            conflicts = self.cbs.find_conflicts(vehicle_paths)
            self.conflicts_detected += len(conflicts)
            
            if conflicts:
                # 执行冲突解决
                resolved_paths = self.cbs.resolve_conflicts(vehicle_paths)
                
                # 计算修改的路径数
                changed_count = sum(1 for vid in vehicle_paths if vid in resolved_paths and 
                                   vehicle_paths[vid] != resolved_paths[vid])
                
                if changed_count > 0:
                    logger.info(f"解决了{changed_count}条路径的冲突")
                    self.conflicts_resolved += changed_count
                    
                    # 更新车辆路径
                    for vid_str, path in resolved_paths.items():
                        if path and path != vehicle_paths.get(vid_str, []):
                            vid = int(vid_str)
                            for vehicle in self.vehicles:
                                if vehicle.vehicle_id == vid:
                                    # 保留当前位置
                                    current_pos = vehicle.current_location
                                    # 确保新路径从当前位置开始
                                    if len(path) > 0 and path[0] != current_pos:
                                        path = [current_pos] + path
                                    vehicle.assign_path(path)
                                    vehicle.path_index = 0  # 重置路径索引
                                    
                                    # 更新路径记录
                                    self.vehicle_paths[vid_str] = path
                                    break
    
    def get_update_data(self):
        """获取用于可视化更新的数据"""
        return {
            'vehicles': {v.vehicle_id: v.current_location for v in self.vehicles},
            'paths': {v.vehicle_id: self.vehicle_paths.get(str(v.vehicle_id), []) 
                     for v in self.vehicles if hasattr(v, 'current_task') and v.current_task},
            'tasks_completed': self.tasks_completed,
            'total_tasks': len(self.tasks),
            'conflicts_detected': self.conflicts_detected,
            'conflicts_resolved': self.conflicts_resolved
        }
    
    def print_status(self, elapsed_time):
        """打印系统状态"""
        # 计算状态统计
        total_vehicles = len(self.vehicles)
        idle_count = sum(1 for v in self.vehicles if not v.current_task)
        moving_count = total_vehicles - idle_count
        
        total_tasks = len(self.tasks)
        completed_count = len(self.completed_tasks)
        in_progress_count = sum(1 for v in self.vehicles if v.current_task)
        pending_count = total_tasks - completed_count - in_progress_count
        
        # 打印状态
        print("\n" + "="*40)
        print(f"模拟时间: {int(elapsed_time)}秒")
        print(f"车辆: {moving_count}活动/{total_vehicles}总数 ({idle_count}空闲)")
        print(f"任务: {completed_count}完成/{total_tasks}总数 ({in_progress_count}进行中, {pending_count}等待)")
        
        # 打印车辆详情
        print("\n车辆状态:")
        for vehicle in self.vehicles:
            status = "空闲" if not vehicle.current_task else "执行任务"
            task_id = vehicle.current_task.task_id if vehicle.current_task else "无"
            position = f"({vehicle.current_location[0]:.1f}, {vehicle.current_location[1]:.1f})"
            progress = ""
            
            if vehicle.current_path and vehicle.path_index > 0:
                progress = f"{vehicle.path_index}/{len(vehicle.current_path)}点"
                
            print(f"  车辆{vehicle.vehicle_id}: {status} | 任务: {task_id} | 位置: {position} | 进度: {progress}")
        
        print("="*40)
    
    def print_final_results(self, elapsed_time):
        """打印最终结果"""
        # 计算完成任务
        completed_tasks = self.completed_tasks
        
        # 计算每辆车完成的任务数
        vehicle_completions = {}
        for task in completed_tasks:
            if hasattr(task, 'assigned_to') and task.assigned_to:
                vid = task.assigned_to
                vehicle_completions[vid] = vehicle_completions.get(vid, 0) + 1
        
        print("\n" + "="*60)
        print("模拟结束 - 最终结果")
        print("="*60)
        print(f"总时间: {elapsed_time:.1f}秒")
        print(f"总车辆: {len(self.vehicles)}")
        print(f"总任务: {len(self.tasks)}")
        print(f"完成任务: {len(completed_tasks)}/{len(self.tasks)} ({len(completed_tasks)/len(self.tasks)*100:.1f}%)")
        print(f"检测冲突: {self.conflicts_detected}")
        print(f"解决冲突: {self.conflicts_resolved}")
        
        # 打印每辆车的任务完成情况
        print("\n车辆任务完成情况:")
        for vehicle in self.vehicles:
            completions = vehicle_completions.get(vehicle.vehicle_id, 0)
            print(f"  车辆{vehicle.vehicle_id}: {completions}个任务")
        
        print("\n任务详情:")
        for task in self.tasks:
            status = "已完成" if (hasattr(task, 'is_completed') and task.is_completed) else "未完成"
            assigned = f"车辆{task.assigned_to}" if hasattr(task, 'assigned_to') and task.assigned_to else "未分配"
            print(f"  任务{task.task_id}: {status} | {assigned} | 优先级: {task.priority}")
        
        print("="*60)
    
    def run_simulation(self):
        """运行模拟"""
        # 分配初始任务
        self.assign_initial_tasks()
        
        # 创建模拟线程
        sim_thread = SimulationThread(self)
        
        # 开始模拟
        logger.info(f"开始模拟，持续时间: {self.duration}秒")
        sim_thread.start()
        
        # 如果不使用可视化，等待模拟完成
        if not visualization_available or not self.config.get('visualization', True):
            sim_thread.join()  # 等待线程结束
        
        return sim_thread


class MiningSystemUI(QMainWindow):
    """露天矿多车协同调度系统UI"""
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.sim_thread = None
        self.displayed_paths = {}
        self.path_items = []
        
        # 设置窗口
        self.setWindowTitle("露天矿多车协同调度系统")
        self.resize(1200, 800)
        
        # 创建UI
        self.setup_ui()
        
        # 更新地图显示
        self.update_map_display()
    
    def setup_ui(self):
        """设置用户界面"""
        # 主部件与布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 上下分割
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # 上部：地图显示与控制
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        splitter.addWidget(top_widget)
        
        # 下部：状态与日志
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        splitter.addWidget(bottom_widget)
        
        # 设置分割比例
        splitter.setSizes([600, 200])
        
        # ===== 地图显示区域 =====
        map_group = QGroupBox("路径规划与调度地图")
        map_layout = QVBoxLayout()
        map_group.setLayout(map_layout)
        
        # 创建地图视图
        self.map_view = pg.PlotWidget()
        self.map_view.setAspectLocked(True)
        self.map_view.setRange(xRange=(0, self.controller.config.get('map_size', 200)), 
                            yRange=(0, self.controller.config.get('map_size', 200)))
        self.map_view.showGrid(x=True, y=True, alpha=0.5)
        map_layout.addWidget(self.map_view)
        
        top_layout.addWidget(map_group, 3)  # 地图占3份宽度
        
        # ===== 控制面板 =====
        control_group = QGroupBox("系统控制")
        control_layout = QVBoxLayout()
        control_group.setLayout(control_layout)
        
        # 模拟控制
        sim_group = QGroupBox("模拟控制")
        sim_layout = QGridLayout()
        sim_group.setLayout(sim_layout)
        
        # 开始/停止按钮
        self.start_btn = QPushButton("开始模拟")
        self.start_btn.clicked.connect(self.toggle_simulation)
        sim_layout.addWidget(self.start_btn, 0, 0)
        
        # 暂停/继续按钮
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        sim_layout.addWidget(self.pause_btn, 0, 1)
        
        # 模拟速度
        sim_layout.addWidget(QLabel("模拟速度:"), 1, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(50)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        sim_layout.addWidget(self.speed_slider, 1, 1)
        
        control_layout.addWidget(sim_group)
        
        # 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QGridLayout()
        display_group.setLayout(display_layout)
        
        # 显示车辆路径
        display_layout.addWidget(QLabel("显示路径:"), 0, 0)
        self.show_paths_cb = QCheckBox()
        self.show_paths_cb.setChecked(True)
        self.show_paths_cb.stateChanged.connect(self.on_show_paths_changed)
        display_layout.addWidget(self.show_paths_cb, 0, 1)
        
        # 显示障碍物
        display_layout.addWidget(QLabel("显示障碍物:"), 1, 0)
        self.show_obstacles_cb = QCheckBox()
        self.show_obstacles_cb.setChecked(True)
        self.show_obstacles_cb.stateChanged.connect(self.update_map_display)
        display_layout.addWidget(self.show_obstacles_cb, 1, 1)
        
        # 显示热图
        display_layout.addWidget(QLabel("显示活动热图:"), 2, 0)
        self.show_heatmap_cb = QCheckBox()
        self.show_heatmap_cb.setChecked(False)
        self.show_heatmap_cb.stateChanged.connect(self.on_show_heatmap_changed)
        display_layout.addWidget(self.show_heatmap_cb, 2, 1)
        
        control_layout.addWidget(display_group)
        
        # 冲突检测与解决
        conflict_group = QGroupBox("冲突管理")
        conflict_layout = QVBoxLayout()
        conflict_group.setLayout(conflict_layout)
        
        # 检测冲突按钮
        self.detect_conflicts_btn = QPushButton("检测冲突")
        self.detect_conflicts_btn.clicked.connect(self.on_detect_conflicts)
        conflict_layout.addWidget(self.detect_conflicts_btn)
        
        # 解决冲突按钮
        self.resolve_conflicts_btn = QPushButton("解决冲突")
        self.resolve_conflicts_btn.clicked.connect(self.on_resolve_conflicts)
        conflict_layout.addWidget(self.resolve_conflicts_btn)
        
        control_layout.addWidget(conflict_group)
        
        # 添加弹簧
        control_layout.addStretch()
        
        # 统计信息
        stats_group = QGroupBox("系统统计")
        stats_layout = QGridLayout()
        stats_group.setLayout(stats_layout)
        
        stats_layout.addWidget(QLabel("总车辆数:"), 0, 0)
        self.total_vehicles_label = QLabel(str(len(self.controller.vehicles)))
        stats_layout.addWidget(self.total_vehicles_label, 0, 1)
        
        stats_layout.addWidget(QLabel("总任务数:"), 1, 0)
        self.total_tasks_label = QLabel(str(len(self.controller.tasks)))
        stats_layout.addWidget(self.total_tasks_label, 1, 1)
        
        stats_layout.addWidget(QLabel("完成任务:"), 2, 0)
        self.completed_tasks_label = QLabel("0")
        stats_layout.addWidget(self.completed_tasks_label, 2, 1)
        
        stats_layout.addWidget(QLabel("检测冲突:"), 3, 0)
        self.conflicts_detected_label = QLabel("0")
        stats_layout.addWidget(self.conflicts_detected_label, 3, 1)
        
        stats_layout.addWidget(QLabel("解决冲突:"), 4, 0)
        self.conflicts_resolved_label = QLabel("0")
        stats_layout.addWidget(self.conflicts_resolved_label, 4, 1)
        
        control_layout.addWidget(stats_group)
        
        top_layout.addWidget(control_group, 1)  # 控制面板占1份宽度
        
        # ===== 状态与日志区域 =====
        # 状态面板
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)
        
        bottom_layout.addWidget(status_group)
        
        # 车辆状态面板
        vehicle_status_group = QGroupBox("车辆状态")
        vehicle_status_layout = QVBoxLayout()
        vehicle_status_group.setLayout(vehicle_status_layout)
        
        self.vehicle_status_text = QTextEdit()
        self.vehicle_status_text.setReadOnly(True)
        vehicle_status_layout.addWidget(self.vehicle_status_text)
        
        bottom_layout.addWidget(vehicle_status_group)
        
        # 初始状态
        self.update_status("系统已初始化，准备就绪")

    def toggle_simulation(self):
        """开始/停止模拟"""
        if not self.sim_thread or not self.sim_thread.running:
            # 开始模拟
            self.sim_thread = self.controller.run_simulation()
            self.sim_thread.update_signal.connect(self.update_ui)
            
            # 更新UI状态
            self.start_btn.setText("停止模拟")
            self.pause_btn.setEnabled(True)
            self.update_status("模拟已启动")
        else:
            # 停止模拟
            self.sim_thread.stop()
            
            # 更新UI状态
            self.start_btn.setText("开始模拟")
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("暂停")
            self.update_status("模拟已停止")

    def toggle_pause(self):
        """暂停/继续模拟"""
        if not self.sim_thread:
            return
            
        if not self.sim_thread.paused:
            # 暂停模拟
            self.sim_thread.pause()
            self.pause_btn.setText("继续")
            self.update_status("模拟已暂停")
        else:
            # 继续模拟
            self.sim_thread.resume()
            self.pause_btn.setText("暂停")
            self.update_status("模拟已继续")

    def on_speed_changed(self, value):
        """更新模拟速度"""
        self.controller.update_interval = 1.0 / value
        self.update_status(f"模拟速度已调整为 {value}x")

    def on_show_paths_changed(self, state):
        """显示/隐藏路径"""
        show = state == Qt.Checked
        
        # 更新所有路径的可见性
        for item in self.path_items:
            if show:
                item.setOpacity(1.0)
            else:
                item.setOpacity(0.0)

    def on_show_heatmap_changed(self, state):
        """显示/隐藏热图"""
        show = state == Qt.Checked
        
        if show:
            if not hasattr(self, 'heatmap_img'):
                # 创建热图
                self.heatmap_data = np.zeros((
                    self.controller.config.get('map_size', 200),
                    self.controller.config.get('map_size', 200)
                ))
                self.heatmap_img = pg.ImageItem()
                self.map_view.addItem(self.heatmap_img)
                
                # 创建颜色映射
                pos = np.array([0.0, 0.33, 0.66, 1.0])
                color = np.array([
                    [0, 0, 0, 0],
                    [0, 0, 255, 50],
                    [255, 255, 0, 100],
                    [255, 0, 0, 150]
                ])
                cmap = pg.ColorMap(pos, color)
                self.heatmap_img.setLookupTable(cmap.getLookupTable())
                
                # 设置位置
                self.heatmap_img.setRect(pg.QtCore.QRectF(
                    0, 0, 
                    self.controller.config.get('map_size', 200),
                    self.controller.config.get('map_size', 200)
                ))
            
            # 显示热图
            self.heatmap_img.setOpacity(0.7)
        else:
            # 隐藏热图
            if hasattr(self, 'heatmap_img'):
                self.heatmap_img.setOpacity(0)

    def on_detect_conflicts(self):
        """手动检测冲突"""
        self.controller.resolve_conflicts()
        self.update_status("已手动检测冲突")
        self.update_ui(self.controller.get_update_data())

    def on_resolve_conflicts(self):
        """手动解决冲突"""
        # 检测和解决冲突
        self.controller.resolve_conflicts()
        self.update_status("已手动解决冲突")
        self.update_ui(self.controller.get_update_data())

    def update_map_display(self):
        """更新地图显示"""
        # 清除地图
        self.map_view.clear()
        
        # 绘制障碍物
        if self.show_obstacles_cb.isChecked() and self.controller.obstacles:
            obstacle_x = [p[0] for p in self.controller.obstacles]
            obstacle_y = [p[1] for p in self.controller.obstacles]
            
            obstacle_item = pg.ScatterPlotItem(
                obstacle_x, obstacle_y,
                size=3, pen=None, brush=pg.mkBrush(100, 100, 100, 150)
            )
            self.map_view.addItem(obstacle_item)
        
        # 绘制车辆
        for vehicle in self.controller.vehicles:
            # 车辆标记
            vehicle_item = pg.ScatterPlotItem(
                [vehicle.current_location[0]], [vehicle.current_location[1]],
                size=10, pen=pg.mkPen(vehicle.color), brush=pg.mkBrush(vehicle.color)
            )
            self.map_view.addItem(vehicle_item)
            
            # 车辆ID标签
            label = pg.TextItem(str(vehicle.vehicle_id), anchor=(0.5, 0.5))
            label.setPos(vehicle.current_location[0], vehicle.current_location[1])
            self.map_view.addItem(label)

    def update_ui(self, data):
        """更新UI显示"""
        # 更新车辆位置
        self.update_map_display()
        
        # 更新路径显示
        if self.show_paths_cb.isChecked():
            self.update_path_display(data.get('paths', {}))
        
        # 更新热图
        if self.show_heatmap_cb.isChecked():
            self.update_heatmap(data.get('paths', {}))
        
        # 更新统计信息
        self.completed_tasks_label.setText(str(data.get('tasks_completed', 0)))
        self.conflicts_detected_label.setText(str(data.get('conflicts_detected', 0)))
        self.conflicts_resolved_label.setText(str(data.get('conflicts_resolved', 0)))
        
        # 更新车辆状态
        self.update_vehicle_status()

    def update_path_display(self, paths):
        """更新路径显示，保留历史路径"""
        # 初始化路径存储
        if not hasattr(self, 'displayed_paths'):
            self.displayed_paths = {}
            self.path_items = []
        
        # 降低之前路径的透明度
        for i, item in enumerate(self.path_items):
            opacity = max(0.2, 1.0 - (len(self.path_items) - i) * 0.1)
            try:
                pen = item.opts['pen']
                pen.setWidth(max(1, pen.width() - 0.2))  # 逐渐减小线宽
                item.setPen(pen)
                item.setOpacity(opacity)
            except:
                pass
        
        # 绘制新的路径
        for vehicle_id, path in paths.items():
            if not path or len(path) < 2:
                continue
                
            # 找到对应的车辆
            vehicle = next((v for v in self.controller.vehicles if v.vehicle_id == vehicle_id), None)
            if not vehicle:
                continue
            
            # 检查此路径是否与上一个相同
            path_key = str(path)
            if vehicle_id in self.displayed_paths and self.displayed_paths[vehicle_id] == path_key:
                continue  # 如果路径相同，跳过
            
            # 创建路径线
            x_data = [p[0] for p in path]
            y_data = [p[1] for p in path]
            
            # 使用更明显的样式表示新路径
            path_line = pg.PlotDataItem(
                x_data, y_data,
                pen=pg.mkPen(color=vehicle.color, width=3, style=Qt.SolidLine),
                name=f"车辆{vehicle_id}-路径{len(self.path_items)}"
            )
            
            self.map_view.addItem(path_line)
            self.path_items.append(path_line)  # 添加到路径项列表
            self.displayed_paths[vehicle_id] = path_key  # 记录显示的路径
            
            # 如果路径项过多，限制数量以避免性能问题
            max_paths = 50
            if len(self.path_items) > max_paths:
                # 移除最旧的路径
                old_item = self.path_items.pop(0)
                self.map_view.removeItem(old_item)

    def update_heatmap(self, paths):
        """更新热图数据"""
        if not hasattr(self, 'heatmap_data') or not hasattr(self, 'heatmap_img'):
            return
            
        # 衰减现有热图数据
        self.heatmap_data *= 0.98
        
        # 为每个车辆的路径添加热度
        for _, path in paths.items():
            if not path or len(path) < 2:
                continue
                
            for point in path:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < self.heatmap_data.shape[0] and 0 <= y < self.heatmap_data.shape[1]:
                    self.heatmap_data[x, y] += 0.5
                    
                    # 添加周围点的热度（模糊效果）
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.heatmap_data.shape[0] and 0 <= ny < self.heatmap_data.shape[1]:
                                self.heatmap_data[nx, ny] += 0.1
        
        # 更新热图显示
        self.heatmap_img.setImage(self.heatmap_data.T)

    def update_vehicle_status(self):
        """更新车辆状态信息"""
        status_html = "<html><body><table width='100%'>"
        status_html += "<tr><th>车辆ID</th><th>状态</th><th>当前任务</th><th>位置</th><th>进度</th></tr>"
        
        for vehicle in self.controller.vehicles:
            status = "空闲" if not vehicle.current_task else "执行任务"
            task_id = vehicle.current_task.task_id if vehicle.current_task else "无"
            position = f"({vehicle.current_location[0]:.1f}, {vehicle.current_location[1]:.1f})"
            
            progress = ""
            if vehicle.current_path and vehicle.path_index > 0:
                progress_pct = min(100, vehicle.path_index / len(vehicle.current_path) * 100)
                progress = f"{progress_pct:.1f}%"
            
            # 设置行样式（基于状态）
            row_style = ""
            if not vehicle.current_task:
                row_style = "background-color: #f0f0f0;"
            
            status_html += f"<tr style='{row_style}'>"
            status_html += f"<td>{vehicle.vehicle_id}</td>"
            status_html += f"<td>{status}</td>"
            status_html += f"<td>{task_id}</td>"
            status_html += f"<td>{position}</td>"
            status_html += f"<td>{progress}</td>"
            status_html += "</tr>"
        
        status_html += "</table></body></html>"
        self.vehicle_status_text.setHtml(status_html)

    def update_status(self, message):
        """更新状态信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
        # 滚动到底部
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='露天矿多车协同调度系统')
    parser.add_argument('--duration', type=int, default=120, help='模拟持续时间(秒)')
    parser.add_argument('--vehicles', type=int, default=5, help='车辆数量')
    parser.add_argument('--tasks', type=int, default=10, help='任务数量')
    parser.add_argument('--no-gui', action='store_true', help='不使用图形界面')
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'duration': args.duration,
        'num_vehicles': args.vehicles,
        'num_tasks': args.tasks,
        'visualization': not args.no_gui and visualization_available
    }
    
    # 创建系统控制器
    controller = SystemController(config)
    
    # 根据配置决定使用GUI还是命令行模式
    if config['visualization']:
        # 创建Qt应用
        app = QApplication(sys.argv)
        
        # 创建主窗口
        main_window = MiningSystemUI(controller)
        main_window.show()
        
        # 运行应用
        sys.exit(app.exec_())
    else:
        # 命令行模式
        sim_thread = controller.run_simulation()
        sim_thread.join()  # 等待模拟结束

if __name__ == "__main__":
    main()