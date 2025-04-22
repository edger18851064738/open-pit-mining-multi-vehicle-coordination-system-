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

特点：
- 支持多种调度算法的插件式接入
- 模块化设计，便于算法替换和测试
- 实时可视化与性能监控
"""

import os
import sys
import time
import math
import random
import logging
import threading
import argparse
from typing import List, Tuple, Dict, Optional, Set, Any, Callable, Type
from datetime import datetime
from enum import Enum
import importlib
import numpy as np
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入项目模块
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from utils.geo_tools import GeoUtils
from algorithm.map_service import MapService
from algorithm.optimized_path_planner import HybridPathPlanner

# 导入调度算法接口
from algorithm.dispatcher_interface import DispatcherInterface
from algorithm.cbs import ConflictBasedSearch
from algorithm.dispatch_service import DispatchSystem

# 调度算法类型枚举
class DispatcherType(Enum):
    CBS = "cbs"  # Conflict-Based Search
    DISPATCH_SYSTEM = "dispatch_system"  # 基本调度系统
    CUSTOM = "custom"  # 自定义算法

# PyQt和PyQtGraph导入
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, 
        QGridLayout, QSplitter, QTextEdit, QGroupBox, QComboBox, QSlider, QCheckBox,
        QMessageBox, QFileDialog, QTabWidget, QRadioButton, QButtonGroup
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QPoint
    from PyQt5.QtGui import QFont, QIcon, QColor, QPen, QBrush
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
        error_signal = pyqtSignal(str)   # 发送错误信息的信号
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.running = False
        self.paused = False
        self.simulation_speed = 1.0  # 模拟速度倍率
        self.last_update_time = 0    # 上次UI更新时间
        self.min_update_interval = 0.05  # 最小UI更新间隔（秒）

    def run(self):
        """线程运行函数"""
        try:
            self.running = True
            start_time = time.time()
            last_status_time = start_time
            last_conflict_check_time = start_time
            self.last_update_time = start_time
            error_count = 0  # 错误计数器
            max_errors = 5   # 增加最大允许错误次数
            update_error_count = 0  # UI更新错误计数器
            max_update_errors = 3  # UI更新最大错误次数
            
            # 创建专用的线程锁
            self.thread_lock = threading.RLock()
            
            while self.running and time.time() - start_time < self.controller.duration:
                if not self.paused:
                    try:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        
                        # 使用可重入锁保护关键操作
                        with self.thread_lock:
                            # 更新车辆位置
                            try:
                                self.controller.update_vehicles()
                            except Exception as e:
                                logger.error(f"更新车辆位置失败: {str(e)}")
                                time.sleep(0.1)
                                continue
                            
                            # 分配新任务给空闲车辆
                            try:
                                self.controller.assign_new_tasks()
                            except Exception as e:
                                logger.error(f"分配任务失败: {str(e)}")
                            
                            # 定期检查冲突
                            if current_time - last_conflict_check_time >= self.controller.conflict_check_interval:
                                try:
                                    self.controller.resolve_conflicts()
                                    last_conflict_check_time = current_time
                                except Exception as e:
                                    logger.error(f"解决冲突失败: {str(e)}")
                        
                        # 定期输出状态
                        if current_time - last_status_time >= self.controller.status_interval:
                            try:
                                self.controller.print_status(elapsed)
                                last_status_time = current_time
                            except Exception as e:
                                logger.warning(f"状态输出失败: {str(e)}")
                        
                        # 发送更新信号（如果可视化可用），控制更新频率
                        if visualization_available and current_time - self.last_update_time >= self.min_update_interval:
                            try:
                                with self.thread_lock:
                                    update_data = self.controller.get_update_data()
                                if update_data and isinstance(update_data, dict):  # 严格验证数据有效性
                                    self.update_signal.emit(update_data)
                                    self.last_update_time = current_time
                                    update_error_count = 0  # 重置UI更新错误计数
                                    error_count = max(0, error_count - 1)  # 逐步减少错误计数
                            except Exception as e:
                                update_error_count += 1
                                logger.warning(f"UI更新错误 ({update_error_count}/{max_update_errors}): {str(e)}")
                                if update_error_count >= max_update_errors:
                                    logger.error("UI更新多次失败，尝试重置更新机制")
                                    self.last_update_time = current_time + 2.0  # 强制等待更长时间
                                    update_error_count = 0  # 重置计数器
                                time.sleep(0.2)  # 增加恢复时间
                                
                    except Exception as e:
                        error_count += 1
                        logger.error(f"模拟循环错误 ({error_count}/{max_errors}): {str(e)}")
                        if visualization_available:
                            try:
                                self.error_signal.emit(f"模拟错误: {str(e)}")
                            except:
                                pass  # 忽略信号发送失败
                        
                        if error_count >= max_errors:
                            logger.critical("达到最大错误次数，正在尝试恢复...")
                            time.sleep(1.0)  # 给系统更多恢复时间
                            error_count = max_errors - 2  # 降低错误计数而不是直接中断
                            continue
                            
                        time.sleep(0.5)  # 错误发生后等待一段时间再继续
                        continue
                
                # 动态调整模拟速度
                try:
                    actual_interval = self.controller.update_interval / max(0.1, self.simulation_speed)
                    actual_interval = min(max(0.01, actual_interval), 0.5)  # 限制延迟范围
                    time.sleep(actual_interval)
                except Exception as e:
                    logger.warning(f"调整模拟速度失败: {str(e)}")
                    time.sleep(0.1)  # 使用默认延迟
            
            # 模拟结束，打印结果
            try:
                if error_count < max_errors:  # 只有在正常结束时才打印结果
                    self.controller.print_final_results(time.time() - start_time)
            except Exception as e:
                logger.error(f"打印最终结果失败: {str(e)}")
                
        except Exception as e:
            logger.error(f"模拟线程发生严重错误: {str(e)}")
            if visualization_available:
                try:
                    self.error_signal.emit(f"模拟线程错误: {str(e)}")
                except:
                    pass  # 忽略信号发送失败
        finally:
            self.running = False
            logger.info("模拟线程已停止")
            try:
                # 清理资源
                if hasattr(self, 'thread_lock'):
                    del self.thread_lock
            except:
                pass

    
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
        
    def set_speed(self, speed):
        """设置模拟速度"""
        self.simulation_speed = max(0.1, min(10.0, speed))  # 限制在0.1-10倍之间


class SystemController:
    """系统控制器，管理模拟和调度过程"""
    
    def __init__(self, config=None, dispatcher_type=DispatcherType.CBS, custom_dispatcher=None):
        """
        初始化系统控制器
        
        Args:
            config: 配置字典
            dispatcher_type: 调度算法类型
            custom_dispatcher: 自定义调度器类或实例
        """
        self.config = config or self._get_default_config()
        
        # 提取配置参数
        self.duration = self.config.get('duration', 120)
        self.update_interval = self.config.get('update_interval', 0.1)
        self.status_interval = self.config.get('status_interval', 5.0)
        self.conflict_check_interval = self.config.get('conflict_check_interval', 1.0)
        self.num_vehicles = self.config.get('num_vehicles', 5)
        self.num_tasks = self.config.get('num_tasks', 10)
        
        # 调度算法类型和自定义调度器
        self.dispatcher_type = dispatcher_type
        self.custom_dispatcher = custom_dispatcher
        
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
        
        # 创建用于线程安全访问的锁
        self.data_lock = threading.RLock()
        
        # 系统统计信息
        self.stats = {
            'start_time': time.time(),
            'run_time': 0,
            'tasks_created': self.num_tasks,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'vehicle_states': {},
            'vehicle_utilization': 0.0,
            'path_planning_count': 0,
            'path_planning_failures': 0,
            'avg_planning_time': 0,
            'planning_times': []
        }
        
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
            'visualization': True,  # 是否启用可视化
            'dispatcher': {
                'type': 'cbs',  # 调度算法类型
                'params': {}  # 调度算法参数
            }
        }
    
    def _init_basic_components(self):
        """初始化系统基础组件"""
        try:
            # 创建地图和路径规划相关组件
            self.geo_utils = GeoUtils()
            self.map_service = MapService()
            self.path_planner = HybridPathPlanner(self.map_service)
            
            # 根据配置创建调度器
            if self.dispatcher_type == DispatcherType.CBS:
                # 创建冲突解决器
                self.cbs = ConflictBasedSearch(self.path_planner)
                # 使用新的调度系统
                self.dispatch = DispatchSystem(self.path_planner, self.map_service)
            elif self.dispatcher_type == DispatcherType.CUSTOM and self.custom_dispatcher:
                # 自定义调度器
                if isinstance(self.custom_dispatcher, type):
                    # 如果是类，创建实例
                    self.dispatch = self.custom_dispatcher(self.path_planner, self.map_service)
                else:
                    # 如果是实例，直接使用
                    self.dispatch = self.custom_dispatcher
                # 如果自定义调度器没有cbs属性，创建一个
                if not hasattr(self.dispatch, 'cbs'):
                    self.cbs = ConflictBasedSearch(self.path_planner)
                    self.dispatch.cbs = self.cbs
            else:
                # 默认使用基本调度系统
                self.dispatch = DispatchSystem(self.path_planner, self.map_service)
                self.cbs = ConflictBasedSearch(self.path_planner)
            
            # 为路径规划器设置dispatch对象
            self.path_planner.dispatch = self.dispatch
            
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
                'min_hardness': 2.5,
                'turning_radius': 10.0,
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
            
            # 将车辆添加到调度系统
            self.dispatch.add_vehicle(vehicle)
            
            # 如果使用可视化，添加颜色属性
            if visualization_available:
                vehicle.color = pg.intColor(i % 10)
            
            vehicles.append(vehicle)
        
        logger.info(f"已创建{len(vehicles)}辆车辆")
        return vehicles
    def _update_stats(self):
        """更新系统统计信息"""
        with self.data_lock:
            # 更新运行时间
            self.stats['run_time'] = time.time() - self.stats['start_time']
            
            # 更新车辆状态统计
            vehicle_states = {}
            active_count = 0
            for vehicle in self.vehicles:
                if hasattr(vehicle, 'state'):
                    state_name = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                    if state_name not in vehicle_states:
                        vehicle_states[state_name] = 0
                    vehicle_states[state_name] += 1
                    
                    # 统计活动车辆
                    if state_name != 'IDLE' and state_name != 'VehicleState.IDLE':
                        active_count += 1
            
            self.stats['vehicle_states'] = vehicle_states
            
            # 计算车辆利用率
            if len(self.vehicles) > 0:
                self.stats['vehicle_utilization'] = active_count / len(self.vehicles)
            
            # 更新路径规划统计
            if self.stats['planning_times']:
                self.stats['avg_planning_time'] = sum(self.stats['planning_times']) / len(self.stats['planning_times'])    
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
            
            # 添加到调度系统
            self.dispatch.add_task(task)
            tasks.append(task)
        
        logger.info(f"已创建{len(tasks)}个任务")
        return tasks
    def print_status(self, elapsed_time):
        """打印当前状态"""
        with self.data_lock:
            # 格式化运行时间
            minutes = int(elapsed_time) // 60
            seconds = int(elapsed_time) % 60
            time_str = f"{minutes:02d}:{seconds:02d}"
            
            # 打印状态信息
            print(f"\n== 系统状态 [{time_str}] ==")
            print(f"任务: {self.stats['tasks_completed']}/{self.stats['tasks_created']} 完成")
            print(f"冲突: {self.stats['conflicts_detected']} 检测, {self.stats['conflicts_resolved']} 解决")
            
            # 打印车辆状态
            print("\n车辆状态:")
            for state, count in self.stats['vehicle_states'].items():
                print(f"  {state}: {count}辆")
            
            # 打印性能指标
            print(f"\n车辆利用率: {self.stats['vehicle_utilization']*100:.1f}%")
            print(f"平均规划时间: {self.stats['avg_planning_time']*1000:.2f}毫秒")
            
            # 打印调度器信息
            print(f"\n当前调度算法: {self.dispatcher_type.value}")    
    def print_final_results(self, total_time):
        """打印最终结果"""
        with self.data_lock:
            self._update_stats()
        print("\n" + "="*50)
        print("露天矿多车协同调度系统 - 模拟结果")
        print("="*50)
        
        # 基本信息
        print(f"\n模拟时间: {total_time:.1f}秒")
        print(f"调度算法: {self.dispatcher_type.value}")
        
        # 任务统计
        print(f"\n总任务数: {self.stats['tasks_created']}")
        print(f"完成任务数: {self.stats['tasks_completed']}")
        print(f"任务完成率: {self.stats['tasks_completed']/max(1, self.stats['tasks_created'])*100:.1f}%")
        
        # 冲突统计
        print(f"\n检测到的冲突: {self.stats['conflicts_detected']}")
        print(f"解决的冲突: {self.stats['conflicts_resolved']}")
        if self.stats['conflicts_detected'] > 0:
            print(f"冲突解决率: {self.stats['conflicts_resolved']/self.stats['conflicts_detected']*100:.1f}%")
        
        # 性能统计
        print(f"\n车辆利用率: {self.stats['vehicle_utilization']*100:.1f}%")
        print(f"路径规划次数: {self.stats['path_planning_count']}")
        print(f"规划失败次数: {self.stats['path_planning_failures']}")
        print(f"平均规划时间: {self.stats['avg_planning_time']*1000:.2f}毫秒")
        
        # 保存结果到文件
        try:
            results_dir = os.path.join(PROJECT_ROOT, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"{self.dispatcher_type.value}_{timestamp}.txt")
            
            with open(filename, 'w') as f:
                f.write("露天矿多车协同调度系统 - 模拟结果\n")
                f.write("="*50 + "\n\n")
                f.write(f"模拟时间: {total_time:.1f}秒\n")
                f.write(f"调度算法: {self.dispatcher_type.value}\n\n")
                f.write(f"总任务数: {self.stats['tasks_created']}\n")
                f.write(f"完成任务数: {self.stats['tasks_completed']}\n")
                f.write(f"任务完成率: {self.stats['tasks_completed']/max(1, self.stats['tasks_created'])*100:.1f}%\n\n")
                f.write(f"检测到的冲突: {self.stats['conflicts_detected']}\n")
                f.write(f"解决的冲突: {self.stats['conflicts_resolved']}\n")
                if self.stats['conflicts_detected'] > 0:
                    f.write(f"冲突解决率: {self.stats['conflicts_resolved']/self.stats['conflicts_detected']*100:.1f}%\n\n")
                f.write(f"车辆利用率: {self.stats['vehicle_utilization']*100:.1f}%\n")
                f.write(f"路径规划次数: {self.stats['path_planning_count']}\n")
                f.write(f"规划失败次数: {self.stats['path_planning_failures']}\n")
                f.write(f"平均规划时间: {self.stats['avg_planning_time']*1000:.2f}毫秒\n")
            
            print(f"\n结果已保存到: {filename}")
        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            
        print("\n" + "="*50)
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
    # Improvements to SystemController class in master.py
    def get_update_data(self):
        """获取用于可视化更新的数据，确保线程安全和数据一致性"""
        try:
            # 使用可重入锁保护数据访问
            with self.data_lock:
                # 创建数据快照
                vehicles_data = {}
                paths_data = {}
                
                # 安全地获取车辆位置数据
                for v in self.vehicles:
                    try:
                        if v and hasattr(v, 'vehicle_id') and hasattr(v, 'current_location'):
                            if v.current_location and len(v.current_location) == 2:
                                vehicles_data[v.vehicle_id] = (float(v.current_location[0]), float(v.current_location[1]))
                                
                                # 同时更新路径数据 - 重要部分
                                if hasattr(v, 'current_path') and v.current_path and len(v.current_path) > 1:
                                    # 确保路径有效
                                    valid_path = []
                                    for p in v.current_path:
                                        if isinstance(p, (list, tuple)) and len(p) == 2:
                                            valid_path.append((float(p[0]), float(p[1])))
                                    
                                    if valid_path:
                                        paths_data[v.vehicle_id] = valid_path
                                        
                    except Exception as e:
                        logger.warning(f"处理车辆{getattr(v, 'vehicle_id', '未知')}数据时出错: {str(e)}")
                        continue
                
                # 更新状态统计信息
                self._update_stats()
                
                # 构建更新数据
                update_data = {
                    'vehicles': vehicles_data,
                    'paths': paths_data,
                    'tasks_completed': self.stats['tasks_completed'],
                    'total_tasks': len(self.tasks),
                    'conflicts_detected': self.stats['conflicts_detected'],
                    'conflicts_resolved': self.stats['conflicts_resolved'],
                    'stats': self.stats,
                    'obstacles': list(self.obstacles)
                }
                
                return update_data
                
        except Exception as e:
            logger.error(f"获取更新数据时发生错误: {str(e)}")
            return {
                'vehicles': {},
                'paths': {},
                'tasks_completed': 0,
                'total_tasks': 0,
                'conflicts_detected': 0,
                'conflicts_resolved': 0,
                'stats': {
                    'run_time': 0,
                    'vehicle_states': {},
                    'vehicle_utilization': 0.0
                },
                'obstacles': []
            }
    def _is_near_safe_points(self, point, safe_points, safe_radius):
        """检查点是否靠近安全点"""
        x, y = point
        for sx, sy in safe_points:
            distance = math.sqrt((x - sx)**2 + (y - sy)**2)
            if distance < safe_radius:
                return True
        return False
    
    def update_vehicles(self):
        """更新车辆位置"""
        try:
            with self.data_lock:
                for vehicle in self.vehicles:
                    # 检查车辆是否有路径和路径索引是否有效
                    if (vehicle.current_path and 
                        isinstance(vehicle.path_index, int) and 
                        0 <= vehicle.path_index < len(vehicle.current_path) - 1):
                        
                        # 移动到下一个路径点
                        vehicle.path_index += 1
                        
                        # 确保路径索引在有效范围内
                        if vehicle.path_index < len(vehicle.current_path):
                            vehicle.current_location = vehicle.current_path[vehicle.path_index]
                            
                            # 更新路径记录 - 添加此行
                            self.vehicle_paths[str(vehicle.vehicle_id)] = vehicle.current_path
                        
                            # 检查是否到达终点
                            if vehicle.path_index >= len(vehicle.current_path) - 1:
                                # 车辆到达终点，任务完成
                                self._handle_task_completion(vehicle)
                    else:
                        # 如果路径无效但车辆有任务，记录异常情况
                        if vehicle.current_task:
                            logger.warning(f"车辆{vehicle.vehicle_id}有任务但路径无效或已完成")
        except Exception as e:
            logger.error(f"更新车辆位置时发生错误: {str(e)}")
    
    def _handle_task_completion(self, vehicle):
        """处理任务完成"""
        try:
            # 验证车辆和任务
            if not vehicle or not hasattr(vehicle, 'current_task') or not vehicle.current_task:
                return
                    
            # 获取任务信息
            task = vehicle.current_task
            task_id = task.task_id if hasattr(task, 'task_id') else '未知'
            
            # 标记任务完成
            task.is_completed = True
            
            with self.data_lock:
                # 更新任务统计
                self.tasks_completed += 1
                self.stats['tasks_completed'] += 1
                
                # 避免重复添加到已完成任务列表
                if task not in self.completed_tasks:
                    self.completed_tasks.append(task)
                
                # 更新车辆状态
                vehicle.current_task = None
                vehicle.state = VehicleState.IDLE
                vehicle.current_path = []
                vehicle.path_index = 0
                
                # 清理路径记录 - 确保路径不再显示
                if str(vehicle.vehicle_id) in self.vehicle_paths:
                    del self.vehicle_paths[str(vehicle.vehicle_id)]
                
                # 打印明确的完成消息
                print(f"Vehicle {vehicle.vehicle_id} completed task {task_id}")
                
            logger.info(f"车辆{vehicle.vehicle_id}完成任务{task_id}")
        except Exception as e:
            logger.error(f"处理任务完成时发生错误: {str(e)}")
            if vehicle:
                vehicle.state = VehicleState.IDLE
                vehicle.current_path = []
                vehicle.path_index = 0
    
    def assign_new_tasks(self):
        """分配新任务给空闲车辆"""
        # 从调度系统获取空闲车辆和未分配任务，执行调度
        self.dispatch.scheduling_cycle()
    
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
        
        if len(vehicle_paths) < 2:
            return  # 不足两条路径，无需检查冲突
        
        # 使用调度系统解决冲突
        # 这里是调度算法的关键部分，应该根据使用的调度算法来调用相应的方法
        if hasattr(self.dispatch, 'resolve_path_conflicts'):
            # 调用调度系统的路径冲突解决方法
            self.dispatch.resolve_path_conflicts()
        elif hasattr(self.cbs, 'find_conflicts') and hasattr(self.cbs, 'resolve_conflicts'):
            # 使用CBS检测冲突
            conflicts = self.cbs.find_conflicts(vehicle_paths)
            
            if conflicts:
                # 更新指标
                with self.data_lock:
                    self.conflicts_detected += len(conflicts)
                    self.stats['conflicts_detected'] += len(conflicts)
                
                # 使用CBS解决冲突
                resolved_paths = self.cbs.resolve_conflicts(vehicle_paths)
                
                # 统计修改的路径数
                changed_paths = 0
                
                # 应用解决方案
                for vid_str, new_path in resolved_paths.items():
                    if new_path != vehicle_paths.get(vid_str, []):
                        vid = int(vid_str)
                        vehicle = next((v for v in self.vehicles if v.vehicle_id == vid), None)
                        
                        if vehicle:
                            # 确保新路径从当前位置开始
                            current_pos = vehicle.current_location
                            if new_path[0] != current_pos:
                                new_path.insert(0, current_pos)
                            
                            # 更新车辆路径
                            vehicle.assign_path(new_path)
                            vehicle.path_index = 0
                            
                            # 更新路径记录
                            with self.data_lock:
                                self.vehicle_paths[vid_str] = new_path
                            changed_paths += 1
                
                # 更新指标
                with self.data_lock:
                    self.conflicts_resolved += changed_paths
                    self.stats['conflicts_resolved'] += changed_paths
                
                if changed_paths > 0:
                    logger.info(f"解决了 {len(conflicts)} 个冲突，修改了 {changed_paths} 条路径")
        else:
            logger.warning("调度系统没有提供冲突解决方法")



# PyQt和PyQtGraph导入
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, 
        QGridLayout, QSplitter, QTextEdit, QGroupBox, QComboBox, QSlider, QCheckBox,
        QMessageBox, QFileDialog, QTabWidget, QRadioButton, QButtonGroup
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QPoint
    from PyQt5.QtGui import QFont, QIcon, QColor, QPen, QBrush
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




class MiningSystemUI(QMainWindow):
    """露天矿多车协同调度系统 - 用户界面"""
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.ui_lock = threading.RLock()
        self.init_ui()
        
        # 创建模拟线程
        self.sim_thread = SimulationThread(controller)
        self.sim_thread.update_signal.connect(self.update_ui)
        self.sim_thread.error_signal.connect(self.handle_error)
        
        # 初始化热图数据
        map_size = self.controller.config.get('map_size', 200)
        self.heatmap_data = np.zeros((map_size, map_size))
        
        # 处理视图缩放
        self.map_view.sigRangeChanged.connect(self.on_view_changed)
        
        self.setWindowTitle("露天矿多车协同调度系统")
        
    def init_ui(self):
        """初始化用户界面"""
        # 主窗口设置
        self.resize(1280, 800)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建上下分割窗口
        self.splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.splitter)
        
        # 上部分：地图和控制
        self.top_widget = QWidget()
        self.top_layout = QHBoxLayout(self.top_widget)
        self.splitter.addWidget(self.top_widget)
        
        # 下部分：状态信息
        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.splitter.addWidget(self.bottom_widget)
        
        # 左侧：地图视图
        self.map_widget = QWidget()
        self.map_layout = QVBoxLayout(self.map_widget)
        self.top_layout.addWidget(self.map_widget, 3)  # 3:1比例
        
        # 创建地图视图
        self.map_view = pg.PlotWidget(title="露天矿协同调度地图")
        self.map_view.setAspectLocked(True)
        self.map_view.setRange(xRange=(0, 200), yRange=(0, 200))
        self.map_view.showGrid(x=True, y=True, alpha=0.5)
        self.map_layout.addWidget(self.map_view)
        
        # 右侧：控制面板
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout(self.control_widget)
        self.top_layout.addWidget(self.control_widget, 1)  # 1:3比例
        
        # 模拟控制组
        self.sim_group = QGroupBox("模拟控制")
        self.sim_layout = QVBoxLayout(self.sim_group)
        self.control_layout.addWidget(self.sim_group)
        
        # 开始/停止按钮
        self.start_btn = QPushButton("开始模拟")
        self.start_btn.clicked.connect(self.toggle_simulation)
        self.sim_layout.addWidget(self.start_btn)
        
        # 暂停/继续按钮
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        self.sim_layout.addWidget(self.pause_btn)
        
        # 速度控制
        self.speed_layout = QHBoxLayout()
        self.speed_layout.addWidget(QLabel("模拟速度:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.change_speed)
        self.speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("1.0x")
        self.speed_layout.addWidget(self.speed_label)
        self.sim_layout.addLayout(self.speed_layout)
        
        # 显示控制组
        self.display_group = QGroupBox("显示控制")
        self.display_layout = QVBoxLayout(self.display_group)
        self.control_layout.addWidget(self.display_group)
        
        # 显示障碍物
        self.show_obstacles_cb = QCheckBox("显示障碍物")
        self.show_obstacles_cb.setChecked(True)
        self.show_obstacles_cb.stateChanged.connect(self.update_display_settings)
        self.display_layout.addWidget(self.show_obstacles_cb)
        
        # 显示路径
        self.show_paths_cb = QCheckBox("显示路径")
        self.show_paths_cb.setChecked(True)
        self.show_paths_cb.stateChanged.connect(self.update_display_settings)
        self.display_layout.addWidget(self.show_paths_cb)
        
        # 显示热图
        self.show_heatmap_cb = QCheckBox("显示路径热图")
        self.show_heatmap_cb.setChecked(False)
        self.show_heatmap_cb.stateChanged.connect(self.update_display_settings)
        self.display_layout.addWidget(self.show_heatmap_cb)
        
        # 调度算法选择组
        self.algo_group = QGroupBox("调度算法")
        self.algo_layout = QVBoxLayout(self.algo_group)
        self.control_layout.addWidget(self.algo_group)
        
        # 算法选择下拉框
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["CBS冲突解决", "基本调度系统", "自定义算法"])
        self.algo_combo.currentIndexChanged.connect(self.change_algorithm)
        self.algo_layout.addWidget(self.algo_combo)
        
        # 添加状态信息面板
        self.status_group = QGroupBox("系统状态")
        self.status_layout = QGridLayout(self.status_group)
        self.bottom_layout.addWidget(self.status_group)
        
        # 任务统计
        self.status_layout.addWidget(QLabel("任务完成:"), 0, 0)
        self.completed_tasks_label = QLabel("0")
        self.status_layout.addWidget(self.completed_tasks_label, 0, 1)
        
        self.status_layout.addWidget(QLabel("总任务数:"), 0, 2)
        self.total_tasks_label = QLabel(str(len(self.controller.tasks)))
        self.status_layout.addWidget(self.total_tasks_label, 0, 3)
        
        # 冲突统计
        self.status_layout.addWidget(QLabel("冲突检测:"), 1, 0)
        self.conflicts_detected_label = QLabel("0")
        self.status_layout.addWidget(self.conflicts_detected_label, 1, 1)
        
        self.status_layout.addWidget(QLabel("冲突解决:"), 1, 2)
        self.conflicts_resolved_label = QLabel("0")
        self.status_layout.addWidget(self.conflicts_resolved_label, 1, 3)
        
        # 车辆状态
        self.status_layout.addWidget(QLabel("车辆数量:"), 2, 0)
        self.total_vehicles_label = QLabel(str(len(self.controller.vehicles)))
        self.status_layout.addWidget(self.total_vehicles_label, 2, 1)
        
        self.status_layout.addWidget(QLabel("车辆利用率:"), 2, 2)
        self.vehicle_utilization_label = QLabel("0%")
        self.status_layout.addWidget(self.vehicle_utilization_label, 2, 3)
        
        # 路径规划统计
        self.status_layout.addWidget(QLabel("规划次数:"), 3, 0)
        self.planning_count_label = QLabel("0")
        self.status_layout.addWidget(self.planning_count_label, 3, 1)
        
        self.status_layout.addWidget(QLabel("平均规划时间:"), 3, 2)
        self.avg_planning_time_label = QLabel("0ms")
        self.status_layout.addWidget(self.avg_planning_time_label, 3, 3)
        
        # 运行时间
        self.status_layout.addWidget(QLabel("运行时间:"), 4, 0)
        self.run_time_label = QLabel("00:00")
        self.status_layout.addWidget(self.run_time_label, 4, 1)
        
        # 当前调度算法
        self.status_layout.addWidget(QLabel("调度算法:"), 4, 2)
        self.dispatcher_type_label = QLabel(self.controller.dispatcher_type.value)
        self.status_layout.addWidget(self.dispatcher_type_label, 4, 3)
        
        # 添加车辆状态表格
        self.vehicle_status_group = QGroupBox("车辆状态")
        self.vehicle_status_layout = QVBoxLayout(self.vehicle_status_group)
        self.bottom_layout.addWidget(self.vehicle_status_group)
        
        self.vehicle_status_text = QTextEdit()
        self.vehicle_status_text.setReadOnly(True)
        self.vehicle_status_text.setMaximumHeight(120)
        self.vehicle_status_layout.addWidget(self.vehicle_status_text)
        
        # 添加日志面板
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(80)
        self.bottom_layout.addWidget(self.status_text)
        
        # 设置分割比例
        self.splitter.setSizes([600, 200])
        
        # 初始绘制地图
        self.draw_obstacles()
    
    def draw_obstacles(self):
        """绘制障碍物"""
        with self.ui_lock:
            if not hasattr(self.controller, 'obstacles') or not self.controller.obstacles:
                return
                
            # 清理旧的障碍物图形
            if hasattr(self, 'obstacle_item'):
                self.map_view.removeItem(self.obstacle_item)
                
            # 绘制新的障碍物图形
            obstacle_points = self.controller.obstacles
            
            # 提取有效坐标
            x_coords = []
            y_coords = []
            for point in obstacle_points:
                if isinstance(point, (list, tuple)) and len(point) == 2:
                    x, y = point
                    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                        x_coords.append(x)
                        y_coords.append(y)
            
            if x_coords and y_coords:
                self.obstacle_item = pg.ScatterPlotItem(
                    x_coords, y_coords,
                    size=3, pen=None, brush=pg.mkBrush(100, 100, 100, 150)
                )
                self.map_view.addItem(self.obstacle_item)
    
    def toggle_simulation(self):
        """切换模拟开始/停止"""
        if self.sim_thread.running:
            # 停止模拟
            self.sim_thread.stop()
            self.start_btn.setText("开始模拟")
            self.pause_btn.setEnabled(False)
            self.algo_combo.setEnabled(True)
            self.update_status_text("模拟已停止")
        else:
            # 开始模拟
            self.sim_thread.start()
            self.start_btn.setText("停止模拟")
            self.pause_btn.setEnabled(True)
            self.algo_combo.setEnabled(False)
            self.update_status_text("模拟已启动")
    
    def toggle_pause(self):
        """切换暂停/继续"""
        if self.sim_thread.paused:
            # 继续模拟
            self.sim_thread.resume()
            self.pause_btn.setText("暂停")
            self.update_status_text("模拟已继续")
        else:
            # 暂停模拟
            self.sim_thread.pause()
            self.pause_btn.setText("继续")
            self.update_status_text("模拟已暂停")
    
    def change_speed(self, value):
        """改变模拟速度"""
        speed = value / 10.0
        self.sim_thread.set_speed(speed)
        self.speed_label.setText(f"{speed:.1f}x")
    
    def update_display_settings(self):
        """更新显示设置"""
        # 更新障碍物显示
        if hasattr(self, 'obstacle_item'):
            self.obstacle_item.setVisible(self.show_obstacles_cb.isChecked())
        
        # 更新热图显示
        if hasattr(self, 'heatmap_img'):
            self.heatmap_img.setVisible(self.show_heatmap_cb.isChecked())
        elif self.show_heatmap_cb.isChecked():
            # 创建热图
            self.init_heatmap()
            
        # 强制刷新显示
        self.update_ui({})
    
    def change_algorithm(self, index):
        """更改调度算法"""
        # 获取当前选择的算法
        if index == 0:
            selected_type = DispatcherType.CBS
        elif index == 1:
            selected_type = DispatcherType.DISPATCH_SYSTEM
        else:
            selected_type = DispatcherType.CUSTOM
            
        # 提示用户重启系统以应用更改
        QMessageBox.information(
            self, 
            "更改调度算法", 
            "调度算法更改将在下次启动模拟时生效"
        )
        
        # 更新控制器设置
        self.controller.dispatcher_type = selected_type
        self.dispatcher_type_label.setText(selected_type.value)
    
    def init_heatmap(self):
        """初始化热图"""
        # 获取地图尺寸
        map_size = self.controller.config.get('map_size', 200)
        
        # 确保热图数据维度匹配地图尺寸
        self.heatmap_data = np.zeros((map_size, map_size))
        
        # 创建热图图像项
        self.heatmap_img = pg.ImageItem(self.heatmap_data.T)  # 预先设置图像数据
        self.map_view.addItem(self.heatmap_img)
        
        # 设置颜色映射
        pos = np.array([0.0, 0.33, 0.66, 1.0])
        color = np.array([
            [0, 0, 0, 0],
            [0, 0, 255, 50],
            [255, 255, 0, 100],
            [255, 0, 0, 150]
        ])
        cmap = pg.ColorMap(pos, color)
        self.heatmap_img.setLookupTable(cmap.getLookupTable())
        
        # 设置显示范围
        self.heatmap_img.setRect(pg.QtCore.QRectF(0, 0, map_size, map_size))
    
    def update_heatmap(self, paths):
        """更新热图数据"""
        if not self.show_heatmap_cb.isChecked():
            return
            
        if not hasattr(self, 'heatmap_img'):
            self.init_heatmap()
            
        # 衰减现有热图数据
        self.heatmap_data *= 0.98
        
        # 获取地图尺寸 - 修复这里
        map_size = self.controller.config.get('map_size', 200)
        
        # 为每个路径添加热度
        for _, path in paths.items():
            if not path or len(path) < 2:
                continue
                
            for point in path:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < map_size and 0 <= y < map_size:
                    # 确保索引不越界
                    if x < self.heatmap_data.shape[0] and y < self.heatmap_data.shape[1]:
                        self.heatmap_data[x, y] += 0.5
                    
                    # 添加周围点的热度（模糊效果）
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < map_size and 0 <= ny < map_size and 
                                nx < self.heatmap_data.shape[0] and ny < self.heatmap_data.shape[1]):
                                self.heatmap_data[nx, ny] += 0.1
        
        # 更新热图显示
        self.heatmap_img.setImage(self.heatmap_data.T)
        
    def update_ui(self, data):
        """更新UI显示"""
        try:
            with self.ui_lock:
                if not data or not isinstance(data, dict):
                    return
                    
                # Debug output
                print(f"Updating UI with data: vehicles={len(data.get('vehicles', {}))}, paths={len(data.get('paths', {}))}")
                    
                # 更新地图
                self.update_map_display(data)
                
                # 更新热图
                if self.show_heatmap_cb.isChecked() and 'paths' in data:
                    self.update_heatmap(data.get('paths', {}))
                
                # 更新统计信息
                self.update_stats_display(data)
                
                # 更新车辆状态
                self.update_vehicle_status(data)
                
                # 处理应用事件，保持UI响应
                QApplication.processEvents()
        except Exception as e:
            logger.error(f"更新UI时出错: {str(e)}")
            # Print the full traceback for debugging
            import traceback
            traceback.print_exc()
    
    def update_paths_display(self, paths):
        """更新路径显示"""
        try:
            # 清除旧的路径
            for vid, marker_info in self.vehicle_markers.items():
                if marker_info['path_item'] is not None:
                    self.map_view.removeItem(marker_info['path_item'])
                    marker_info['path_item'] = None
            
            # 绘制新的路径
            for vid_str, path in paths.items():
                vid = int(vid_str) if isinstance(vid_str, str) else vid_str
                
                if not path or len(path) < 2:
                    continue
                
                # 确保路径有效
                valid_path = []
                for p in path:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        x, y = p
                        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                            valid_path.append((float(x), float(y)))
                
                if len(valid_path) < 2:
                    continue
                    
                # 提取路径坐标
                x_coords = [p[0] for p in valid_path]
                y_coords = [p[1] for p in valid_path]
                
                # 获取车辆颜色
                vehicle = next((v for v in self.controller.vehicles if v.vehicle_id == vid), None)
                color = getattr(vehicle, 'color', pg.intColor(vid % 10))
                
                # 创建路径线
                path_item = pg.PlotDataItem(
                    x_coords, y_coords,
                    pen=pg.mkPen(color, width=2, style=Qt.DashLine),
                    name=f"Vehicle {vid} Path"
                )
                
                self.map_view.addItem(path_item)
                
                # 更新字典
                if vid in self.vehicle_markers:
                    self.vehicle_markers[vid]['path_item'] = path_item
                    
                print(f"Added path for vehicle {vid}: {len(valid_path)} points, color={color}")
                
        except Exception as e:
            logger.error(f"更新路径显示时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_paths_display(self, paths):
        """更新路径显示"""
        try:
            # 清除旧的路径
            for vid, marker_info in self.vehicle_markers.items():
                if marker_info['path_item']:
                    self.map_view.removeItem(marker_info['path_item'])
                    marker_info['path_item'] = None
            
            # 绘制新的路径
            for vid, path in paths.items():
                if not path or len(path) < 2:
                    continue
                
                # 提取路径坐标
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]
                
                # 获取车辆颜色
                vehicle = next((v for v in self.controller.vehicles if v.vehicle_id == int(vid)), None)
                color = getattr(vehicle, 'color', pg.intColor(int(vid) % 10))
                
                # 创建路径线
                path_item = pg.PlotDataItem(
                    x_coords, y_coords,
                    pen=pg.mkPen(color, width=2, style=Qt.DashLine)
                )
                
                self.map_view.addItem(path_item)
                
                # 更新字典
                if int(vid) in self.vehicle_markers:
                    self.vehicle_markers[int(vid)]['path_item'] = path_item
                
        except Exception as e:
            logger.error(f"更新路径显示时出错: {str(e)}")
    
    def update_stats_display(self, data):
        """更新统计信息显示"""
        try:
            # 从数据更新标签
            if 'tasks_completed' in data:
                self.completed_tasks_label.setText(str(data['tasks_completed']))
            
            if 'total_tasks' in data:
                self.total_tasks_label.setText(str(data['total_tasks']))
                
            if 'conflicts_detected' in data:
                self.conflicts_detected_label.setText(str(data['conflicts_detected']))
                
            if 'conflicts_resolved' in data:
                self.conflicts_resolved_label.setText(str(data['conflicts_resolved']))
            
            # 从统计信息更新其他标签
            if 'stats' in data:
                stats = data['stats']
                
                # 车辆利用率
                if 'vehicle_utilization' in stats:
                    self.vehicle_utilization_label.setText(f"{stats['vehicle_utilization']*100:.1f}%")
                
                # 规划次数
                if 'path_planning_count' in stats:
                    self.planning_count_label.setText(str(stats['path_planning_count']))
                
                # 平均规划时间
                if 'avg_planning_time' in stats:
                    self.avg_planning_time_label.setText(f"{stats['avg_planning_time']*1000:.2f}ms")
                
                # 运行时间
                if 'run_time' in stats:
                    minutes = int(stats['run_time']) // 60
                    seconds = int(stats['run_time']) % 60
                    self.run_time_label.setText(f"{minutes:02d}:{seconds:02d}")
                    
        except Exception as e:
            logger.error(f"更新统计信息显示时出错: {str(e)}")
    
    def update_vehicle_status(self, data):
        """更新车辆状态表格"""
        try:
            # 创建HTML表格
            status_html = "<html><body><table width='100%'>"
            status_html += "<tr><th>车辆ID</th><th>状态</th><th>当前任务</th><th>位置</th></tr>"
            
            for vehicle in self.controller.vehicles:
                # 获取车辆状态
                state_name = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                
                # 获取任务ID
                task_id = vehicle.current_task.task_id if vehicle.current_task else "无任务"
                
                # 获取位置
                pos = vehicle.current_location
                position = f"({pos[0]:.1f}, {pos[1]:.1f})"
                
                # 添加行
                status_html += f"<tr><td>{vehicle.vehicle_id}</td><td>{state_name}</td><td>{task_id}</td><td>{position}</td></tr>"
                
            status_html += "</table></body></html>"
            
            # 更新状态文本
            self.vehicle_status_text.setHtml(status_html)
            
        except Exception as e:
            logger.error(f"更新车辆状态时出错: {str(e)}")
    
    def handle_error(self, error_msg):
        """处理错误信息"""
        self.update_status_text(f"错误: {error_msg}")
    
    def update_status_text(self, message):
        """更新状态文本"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")
        # 滚动到底部
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )
    
    def on_view_changed(self, view):
        """处理视图变化"""
        # 在缩放和平移时调整标记大小等
        pass

def main():
    """主函数"""
    try:
        # 设置全局异常处理器
        sys.excepthook = handle_exception
        
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='露天矿多车协同调度系统')
        parser.add_argument('--duration', type=int, default=120, help='模拟持续时间(秒)')
        parser.add_argument('--vehicles', type=int, default=5, help='车辆数量')
        parser.add_argument('--tasks', type=int, default=10, help='任务数量')
        parser.add_argument('--no-gui', action='store_true', help='不使用图形界面')
        parser.add_argument('--debug', action='store_true', help='启用调试模式')
        parser.add_argument('--dispatcher', type=str, default='cbs', choices=['cbs', 'dispatch_system', 'custom'],
                            help='调度算法类型')
        parser.add_argument('--custom-dispatcher', type=str, help='自定义调度器模块路径 (例如: my_project.my_dispatcher)')
        args = parser.parse_args()
        
        # 设置日志级别
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("调试模式已启用")
        
        # 处理调度器类型
        dispatcher_type = DispatcherType.CBS
        custom_dispatcher = None
        
        if args.dispatcher == 'dispatch_system':
            dispatcher_type = DispatcherType.DISPATCH_SYSTEM
        elif args.dispatcher == 'custom':
            dispatcher_type = DispatcherType.CUSTOM
            
            # 尝试导入自定义调度器
            if args.custom_dispatcher:
                try:
                    module_path, class_name = args.custom_dispatcher.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    dispatcher_class = getattr(module, class_name)
                    custom_dispatcher = dispatcher_class
                    logger.info(f"已加载自定义调度器: {args.custom_dispatcher}")
                except (ImportError, AttributeError, ValueError) as e:
                    logger.error(f"加载自定义调度器失败: {str(e)}")
                    print(f"加载自定义调度器失败: {str(e)}")
                    print("将使用默认CBS调度器")
                    dispatcher_type = DispatcherType.CBS
        
        # 创建配置
        config = {
            'duration': args.duration,
            'num_vehicles': args.vehicles,
            'num_tasks': args.tasks,
            'visualization': not args.no_gui and visualization_available,
            'debug_mode': args.debug
        }
        
        # 创建系统控制器
        controller = SystemController(
            config=config,
            dispatcher_type=dispatcher_type,
            custom_dispatcher=custom_dispatcher
        )
        
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
            print(f"运行命令行模式，调度算法: {dispatcher_type.value}")
            sim_thread = SimulationThread(controller)
            sim_thread.start()
            sim_thread.join()  # 等待模拟结束
    except Exception as e:
        logger.critical(f"程序启动失败: {str(e)}", exc_info=True)
        print(f"\n程序启动失败: {str(e)}")
        sys.exit(1)

def handle_exception(exc_type, exc_value, exc_traceback):
    """全局异常处理函数"""
    # 忽略KeyboardInterrupt异常的处理
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
    # 记录未捕获的异常
    logger.critical("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))
    print("\n程序遇到错误，请查看日志获取详细信息。")

if __name__ == "__main__":
    main()