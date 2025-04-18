"""
露天矿多车协同调度系统 - 增强型集成测试框架 (v2.0)
===============================================================

此模块提供了一个全面的集成测试框架，用于测试和优化露天矿多车协同调度系统，
重点关注：

1. 系统组件间的高效集成
2. 冲突检测与解决能力测试
3. 系统稳定性和错误恢复能力
4. 性能监控与瓶颈分析
5. 多场景测试支持
"""
import os
import sys
import time
import math
import random
import logging
import threading
import traceback
import concurrent.futures
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import numpy as np
import configparser
# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入项目模块
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from utils.geo_tools import GeoUtils
from algorithm.map_service import MapService
from algorithm.path_planner import HybridPathPlanner
from algorithm.dispatch_service import (
    DispatchSystem, ConflictBasedSearch, 
    TransportScheduler, GeoUtils as DGeoUtils
)

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("integration_test")


class IntegrationTestError(Exception):
    """集成测试异常基类"""
    pass


class SystemInitError(IntegrationTestError):
    """系统初始化错误"""
    pass


class TestConfigError(IntegrationTestError):
    """测试配置错误"""
    pass


class TestFramework:
    """露天矿调度系统集成测试框架"""
    
    def __init__(self, config=None, create_visualizer=True):
        """
        初始化测试框架
        
        Args:
            config: 测试配置字典，若为None则使用默认配置
            create_visualizer: 是否创建可视化器
        """
        # 设置随机种子以确保可重现性
        random.seed(42)
        
        self.config = config or self._get_default_config()
        self.components = {}  # 存储系统组件
        self.stats = self._create_stats_container()
        self.test_results = {}
        self.test_start_time = None
        self.running = False
        
        # 创建日志目录
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 文件日志处理器
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # 初始化系统组件
        self._init_system_components()
        
        # 初始化锁
        self.stats_lock = threading.RLock()
        self.system_lock = threading.RLock()
        
        # 初始化可视化器
        self.visualizer = None
        if create_visualizer:
            self.visualizer = SystemVisualizer(self)
            
        logger.info("集成测试框架初始化完成")
    
    def _get_default_config(self):
        """获取默认测试配置"""
        return {
            # 一般测试配置
            'test_duration': 120,  # 测试持续时间（秒）
            'log_interval': 5,     # 日志记录间隔（秒）
            'stats_interval': 2,   # 统计信息更新间隔（秒）
            'visualization': True, # 是否启用可视化
            
            # 地图配置
            'map': {
                'grid_size': 200,
                'obstacle_density': 0.15,
                'add_predefined_obstacles': True
            },
            
            # 车辆配置
            'vehicles': {
                'count': 5,
                'speeds': [5.0, 6.0, 7.0, 8.0, 9.0],
                'capacities': [40, 45, 50, 55, 60],
                'start_at_different_points': True
            },
            
            # 任务配置
            'tasks': {
                'initial_count': 5,
                'generation_rate': 0.2,  # 每秒生成新任务的概率
                'types': ['loading', 'unloading', 'manual'],
                'type_weights': [0.6, 0.3, 0.1],  # 各类型的权重
                'priorities': [1, 2, 3],
                'priority_weights': [0.5, 0.3, 0.2]
            },
            
            # 调度系统配置
            'dispatch': {
                'scheduling_interval': 2.0,
                'conflict_check_interval': 1.0
            },
            
            # 测试场景
            'scenarios': [
                'normal',           # 正常运行
                'high_conflict',    # 高冲突场景
                'path_planning_stress',  # 路径规划压力测试
                'deadlock',         # 死锁场景测试
                'system_resilience'  # 系统恢复能力测试
            ],
            
            # 当前场景
            'current_scenario': 'normal'
        }
    def _update_stats_loop(self):
        """统计数据更新循环"""
        update_interval = self.config['stats_interval']
        
        while self.running:
            try:
                self._update_stats()
                time.sleep(update_interval)
            except Exception as e:
                logger.error(f"更新统计数据出错: {str(e)}")
                time.sleep(1.0)  # 出错时降低更新频率
    def _create_stats_container(self):
        """创建统计信息容器"""
        return {
            'start_time': None,
            'end_time': None,
            'system_uptime': 0,
            
            # 任务统计
            'tasks_created': 0,
            'tasks_assigned': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'task_completion_rate': 0.0,
            'avg_task_completion_time': 0.0,
            'task_completion_times': [],
            
            # 冲突统计
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'conflict_resolution_rate': 0.0,
            
            # 路径规划统计
            'paths_planned': 0,
            'path_planning_failures': 0,
            'path_planning_success_rate': 0.0,
            'avg_path_planning_time': 0.0,
            'path_planning_times': [],
            
            # 系统负载
            'vehicle_utilization': {},
            'loading_point_utilization': {},
            'unloading_point_utilization': {},
            
            # 死锁统计
            'deadlocks_detected': 0,
            'deadlocks_resolved': 0,
            'deadlock_resolution_rate': 0.0,
            
            # 性能统计
            'cpu_usage': [],
            'memory_usage': [],
            
            # 故障恢复
            'system_recoveries': 0,
            'recovery_times': []
        }
    
    def _init_system_components(self):
        """初始化系统组件"""
        try:
            logger.info("开始初始化系统组件")
            
            # 1. 创建GeoUtils和MapService
            self.components['geo_utils'] = GeoUtils()
            self.components['map_service'] = MapService()
            
            # 2. 创建路径规划器
            self.components['path_planner'] = HybridPathPlanner(self.components['map_service'])
            
            # 3. 创建调度系统
            self.components['dispatch'] = DispatchSystem(
                self.components['path_planner'], 
                self.components['map_service']
            )
            
            # 4. 设置路径规划器的dispatch引用
            self.components['path_planner'].dispatch = self.components['dispatch']
            
            # 5. 为路径规划器添加安全保障机制
            self._enhance_path_planner()
            
            # 6. 为调度系统添加性能监控
            self._enhance_dispatch_system()
            
            # 7. 创建预定义障碍物
            if self.config['map']['add_predefined_obstacles']:
                self._create_predefined_obstacles()
            
            logger.info("系统组件初始化完成")
            
        except Exception as e:
            logger.error(f"初始化系统组件时发生错误: {str(e)}")
            traceback.print_exc()
            raise SystemInitError(f"系统初始化失败: {str(e)}")
    
    def _enhance_path_planner(self):
        """增强路径规划器，添加错误恢复机制"""
        planner = self.components['path_planner']
        
        # 保存原始方法
        if not hasattr(planner, '_original_plan_path'):
            planner._original_plan_path = planner.plan_path
        
        # 定义增强版路径规划方法
        def enhanced_plan_path(start, end, vehicle=None):
            """增强版路径规划，带错误恢复"""
            start_time = time.time()
            self.stats['paths_planned'] += 1
            
            try:
                # 尝试使用原始方法
                path = planner._original_plan_path(start, end, vehicle)
                
                # 记录成功时间
                planning_time = time.time() - start_time
                with self.stats_lock:
                    self.stats['path_planning_times'].append(planning_time)
                    self.stats['avg_path_planning_time'] = (
                        sum(self.stats['path_planning_times']) / 
                        len(self.stats['path_planning_times'])
                    )
                
                return path
                
            except Exception as e:
                logger.warning(f"原始路径规划失败: {str(e)}，尝试备选方法")
                self.stats['path_planning_failures'] += 1
                
                try:
                    # 第一备选方案: 使用simple_astar
                    if hasattr(planner, 'simple_astar'):
                        path = planner.simple_astar(start, end, vehicle)
                        if path and len(path) > 1:
                            logger.info("使用simple_astar备选方案成功")
                            return path
                except Exception as e2:
                    logger.warning(f"使用simple_astar备选方案失败: {str(e2)}")
                
                try:
                    # 第二备选方案: 使用_generate_fallback_path
                    if hasattr(planner, '_generate_fallback_path'):
                        path = planner._generate_fallback_path(start, end)
                        if path and len(path) > 1:
                            logger.info("使用_generate_fallback_path备选方案成功")
                            return path
                except Exception as e3:
                    logger.warning(f"使用_generate_fallback_path备选方案失败: {str(e3)}")
                
                # 最终备选方案: 生成简单直线路径
                logger.info("使用简单直线作为最终备选方案")
                return [start, end]
        
        # 应用增强版方法
        planner.plan_path = enhanced_plan_path
        logger.info("路径规划器增强完成，添加了错误恢复机制")
    
    def _enhance_dispatch_system(self):
        """增强调度系统，添加性能监控"""
        dispatch = self.components['dispatch']
        
        # 保存原始方法
        if not hasattr(dispatch, '_original_scheduling_cycle'):
            dispatch._original_scheduling_cycle = dispatch.scheduling_cycle
        
        # 定义增强版调度周期方法
        def enhanced_scheduling_cycle():
            """增强版调度周期，带性能监控"""
            cycle_start = time.time()
            
            try:
                # 运行原始调度周期
                result = dispatch._original_scheduling_cycle()
                
                # 更新调度统计信息
                cycle_time = time.time() - cycle_start
                
                # 检查是否有新完成的任务
                completed_tasks = getattr(dispatch, 'completed_tasks', {})
                current_completed = len(completed_tasks)
                
                if hasattr(self, '_prev_completed_count'):
                    new_completed = current_completed - self._prev_completed_count
                    if new_completed > 0:
                        with self.stats_lock:
                            self.stats['tasks_completed'] += new_completed
                
                self._prev_completed_count = current_completed
                
                # 检查冲突统计
                with self.stats_lock:
                    if hasattr(dispatch, 'performance_metrics'):
                        self.stats['conflicts_detected'] = dispatch.performance_metrics.get('conflict_count', 0)
                        self.stats['conflicts_resolved'] = dispatch.performance_metrics.get('resolved_conflicts', 0)
                        
                        if self.stats['conflicts_detected'] > 0:
                            self.stats['conflict_resolution_rate'] = (
                                self.stats['conflicts_resolved'] / self.stats['conflicts_detected']
                            )
                
                return result
                
            except Exception as e:
                logger.error(f"调度周期执行发生异常: {str(e)}")
                # 记录系统恢复
                self.stats['system_recoveries'] += 1
                recovery_start = time.time()
                
                try:
                    # 简化的恢复逻辑
                    logger.info("尝试系统恢复...")
                    # 重新初始化调度器的状态
                    if hasattr(dispatch, 'running') and not dispatch.running:
                        dispatch.running = True
                    
                    # 记录恢复时间
                    recovery_time = time.time() - recovery_start
                    self.stats['recovery_times'].append(recovery_time)
                    logger.info(f"系统恢复完成，耗时: {recovery_time:.3f}秒")
                    
                except Exception as e2:
                    logger.error(f"系统恢复失败: {str(e2)}")
                
                return None
        
        # 应用增强版方法
        dispatch.scheduling_cycle = enhanced_scheduling_cycle
        logger.info("调度系统增强完成，添加了性能监控和自动恢复机制")
    
    def _create_predefined_obstacles(self):
        """创建预定义障碍物"""
        planner = self.components['path_planner']
        
        # 预定义障碍物多边形
        obstacles = [
            [(20,60), (80,60), (80,70), (20,70)],         # 水平障碍墙1
            [(120,60), (180,60), (180,70), (120,70)],     # 水平障碍墙2
            [(40,30), (60,30), (60,40), (40,40)],         # 小障碍物1
            [(140,30), (160,30), (160,40), (140,40)],     # 小障碍物2
            [(90,100), (110,100), (110,120), (90,120)],   # 中央障碍物
            [(30,20), (50,20), (50,100), (30,100)],       # 垂直障碍墙1
            [(150,20), (170,20), (170,100), (150,100)],   # 垂直障碍墙2
        ]
        
        # 对高冲突场景添加更多障碍物
        if self.config['current_scenario'] == 'high_conflict':
            obstacles.extend([
                [(50,90), (80,90), (80,110), (50,110)],      # 左侧障碍区
                [(120,90), (150,90), (150,110), (120,110)],  # 右侧障碍区
                [(70,140), (130,140), (130,160), (70,160)]   # 上方障碍区
            ])
        
        # 对死锁场景增加特殊障碍物
        if self.config['current_scenario'] == 'deadlock':
            # 创建容易形成死锁的窄通道
            obstacles.extend([
                [(80,30), (90,30), (90,90), (80,90)],     # 中央竖直障碍物左侧
                [(110,30), (120,30), (120,90), (110,90)]  # 中央竖直障碍物右侧
            ])
        
        # 将多边形障碍物转换为点集
        flat_obstacles = []
        for polygon in obstacles:
            min_x = min(p[0] for p in polygon)
            max_x = max(p[0] for p in polygon)
            min_y = min(p[1] for p in polygon)
            max_y = max(p[1] for p in polygon)
            
            for x in range(int(min_x), int(max_x)+1):
                for y in range(int(min_y), int(max_y)+1):
                    if self._point_in_polygon((x, y), polygon):
                        flat_obstacles.append((x, y))
        
        # 更新路径规划器的障碍物集合
        planner.obstacle_grids = set(flat_obstacles)
        logger.info(f"已创建{len(flat_obstacles)}个障碍物点")
    
    def _point_in_polygon(self, point, polygon) -> bool:
        """判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i+1)%n]
            
            if (y > min(p1[1], p2[1]) and y <= max(p1[1], p2[1])) and (x <= max(p1[0], p2[0])):
                if p1[1] != p2[1]:
                    x_intersect = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    if p1[0] == p2[0] or x <= x_intersect:
                        inside = not inside
                        
        return inside
    
    def create_test_vehicles(self):
        """创建测试车辆"""
        dispatch = self.components['dispatch']
        dispatch_config = dispatch._load_config()
        
        # 获取配置
        vehicle_count = self.config['vehicles']['count']
        speeds = self.config['vehicles']['speeds']
        capacities = self.config['vehicles']['capacities']
        start_different = self.config['vehicles']['start_at_different_points']
        
        # 确保有足够的配置值
        while len(speeds) < vehicle_count:
            speeds.append(speeds[-1])
        while len(capacities) < vehicle_count:
            capacities.append(capacities[-1])
        
        # 车辆起始位置
        if start_different:
            # 使用不同的起始位置
            start_positions = []
            
            # 使用停车场
            parking_area = dispatch_config['parking_area']
            for i in range(vehicle_count):
                offset_x = random.uniform(-20, 20)
                offset_y = random.uniform(-20, 20)
                start_positions.append((parking_area[0] + offset_x, parking_area[1] + offset_y))
                
            # 有些车辆从装载点开始
            loading_points = dispatch_config['loading_points']
            for i in range(min(len(loading_points), vehicle_count // 3)):
                start_positions[i] = loading_points[i % len(loading_points)]
        else:
            # 所有车辆从相同位置开始
            parking_area = dispatch_config['parking_area']
            start_positions = [(
                parking_area[0] + random.uniform(-5, 5),
                parking_area[1] + random.uniform(-5, 5)
            ) for _ in range(vehicle_count)]
        
        # 创建车辆
        vehicles = []
        for i in range(vehicle_count):
            config = {
                'current_location': start_positions[i],
                'max_capacity': capacities[i % len(capacities)],
                'max_speed': speeds[i % len(speeds)],
                'base_location': dispatch_config['parking_area'],
                'status': VehicleState.IDLE
            }
            
            # 创建车辆对象
            vehicle = MiningVehicle(
                vehicle_id=i+1,
                map_service=self.components['map_service'],
                config=config
            )
            
            # 确保必要的属性存在
            if not hasattr(vehicle, 'current_path'):
                vehicle.current_path = []
            if not hasattr(vehicle, 'path_index'):
                vehicle.path_index = 0
            
            # 添加到调度系统
            dispatch.vehicles[i+1] = vehicle
            vehicles.append(vehicle)
        
        logger.info(f"已创建{len(vehicles)}辆测试车辆")
        return vehicles
    
    def create_initial_tasks(self):
        """创建初始任务"""
        dispatch = self.components['dispatch']
        dispatch_config = dispatch._load_config()
        
        # 获取配置
        task_count = self.config['tasks']['initial_count']
        task_types = self.config['tasks']['types']
        type_weights = self.config['tasks']['type_weights']
        priorities = self.config['tasks']['priorities']
        priority_weights = self.config['tasks']['priority_weights']
        
        # 匹配权重长度与类型长度
        while len(type_weights) < len(task_types):
            type_weights.append(type_weights[-1])
        while len(priority_weights) < len(priorities):
            priority_weights.append(priority_weights[-1])
        
        # 创建任务
        tasks = []
        for i in range(task_count):
            # 随机选择任务类型
            task_type = random.choices(task_types, weights=type_weights, k=1)[0]
            
            # 根据任务类型设置起点和终点
            if task_type == "loading":
                start_point = random.choice(dispatch_config['loading_points'])
                end_point = dispatch_config['unloading_point']
            elif task_type == "unloading":
                start_point = dispatch_config['unloading_point']
                end_point = dispatch_config['parking_area']
            else:  # manual
                # 随机选择起点和终点
                start_point = (random.uniform(0, 200), random.uniform(0, 200))
                end_point = (random.uniform(0, 200), random.uniform(0, 200))
            
            # 随机选择优先级
            priority = random.choices(priorities, weights=priority_weights, k=1)[0]
            
            # 创建任务
            task = TransportTask(
                task_id=f"INIT-{i+1}",
                start_point=start_point,
                end_point=end_point,
                task_type=task_type,
                priority=priority
            )
            
            # 添加到调度系统
            dispatch.add_task(task)
            tasks.append(task)
            
            # 更新统计数据
            with self.stats_lock:
                self.stats['tasks_created'] += 1
                self.stats['tasks_assigned'] += 1
        
        logger.info(f"已创建{len(tasks)}个初始任务")
        return tasks
    
    def generate_random_task(self):
        """生成随机任务"""
        dispatch = self.components['dispatch']
        dispatch_config = dispatch._load_config()
        
        # 获取配置
        task_types = self.config['tasks']['types']
        type_weights = self.config['tasks']['type_weights']
        priorities = self.config['tasks']['priorities']
        priority_weights = self.config['tasks']['priority_weights']
        
        # 匹配权重长度与类型长度
        while len(type_weights) < len(task_types):
            type_weights.append(type_weights[-1])
        while len(priority_weights) < len(priorities):
            priority_weights.append(priority_weights[-1])
        
        # 随机选择任务类型
        task_type = random.choices(task_types, weights=type_weights, k=1)[0]
        
        # 根据任务类型设置起点和终点
        if task_type == "loading":
            start_point = random.choice(dispatch_config['loading_points'])
            end_point = dispatch_config['unloading_point']
        elif task_type == "unloading":
            start_point = dispatch_config['unloading_point']
            end_point = dispatch_config['parking_area']
        else:  # manual
            # 随机选择起点和终点
            start_point = (random.uniform(0, 200), random.uniform(0, 200))
            end_point = (random.uniform(0, 200), random.uniform(0, 200))
        
        # 随机选择优先级
        priority = random.choices(priorities, weights=priority_weights, k=1)[0]
        
        # 创建任务
        task_id = f"TASK-{int(time.time()*1000 % 100000)}"
        task = TransportTask(
            task_id=task_id,
            start_point=start_point,
            end_point=end_point,
            task_type=task_type,
            priority=priority
        )
        
        # 添加到调度系统
        dispatch.add_task(task)
        
        # 更新统计数据
        with self.stats_lock:
            self.stats['tasks_created'] += 1
            self.stats['tasks_assigned'] += 1
        
        logger.debug(f"已生成随机任务: {task_id} ({task_type})")
        return task
    
    def create_deadlock_tasks(self):
        """创建容易导致死锁的任务组合"""
        dispatch = self.components['dispatch']
        dispatch_config = dispatch._load_config()
        
        # 获取关键坐标
        loading_points = dispatch_config['loading_points']
        unloading_point = dispatch_config['unloading_point']
        
        # 创建相互交叉的任务
        tasks = []
        
        # 任务1: 从左上到右下
        task1 = TransportTask(
            task_id="DEADLOCK-1",
            start_point=(50, 150),
            end_point=(150, 50),
            task_type="manual",
            priority=3
        )
        
        # 任务2: 从右上到左下
        task2 = TransportTask(
            task_id="DEADLOCK-2",
            start_point=(150, 150),
            end_point=(50, 50),
            task_type="manual",
            priority=3
        )
        
        # 任务3: 从下到上通过中央窄道
        task3 = TransportTask(
            task_id="DEADLOCK-3",
            start_point=(95, 20),
            end_point=(95, 180),
            task_type="manual",
            priority=2
        )
        
        # 任务4: 从上到下通过中央窄道
        task4 = TransportTask(
            task_id="DEADLOCK-4",
            start_point=(105, 180),
            end_point=(105, 20),
            task_type="manual",
            priority=2
        )
        
        # 添加到调度系统
        for task in [task1, task2, task3, task4]:
            dispatch.add_task(task)
            tasks.append(task)
            
            # 更新统计数据
            with self.stats_lock:
                self.stats['tasks_created'] += 1
                self.stats['tasks_assigned'] += 1
        
        logger.info(f"已创建{len(tasks)}个死锁测试任务")
        return tasks
    
    def run_test(self, scenario=None):
        """
        运行指定场景的集成测试
        
        Args:
            scenario: 要运行的测试场景名称，若为None则使用配置中的场景
        """
        # 设置场景
        if scenario:
            self.config['current_scenario'] = scenario
        
        current_scenario = self.config['current_scenario']
        logger.info(f"开始运行 '{current_scenario}' 测试场景")
        
        # 重置统计数据
        self.stats = self._create_stats_container()
        self.stats['start_time'] = datetime.now()
        self._prev_completed_count = 0
        
        # 初始化测试
        try:
            # 创建测试车辆
            self.create_test_vehicles()
            
            # 根据场景创建任务
            if current_scenario == 'deadlock':
                self.create_deadlock_tasks()
            else:
                self.create_initial_tasks()
            
            # 启动系统组件
            self._start_system_components()
            
            # 标记测试开始
            self.test_start_time = time.time()
            self.running = True
            
            # 启动统计更新线程
            stats_thread = threading.Thread(target=self._update_stats_loop)
            stats_thread.daemon = True
            stats_thread.start()
            
            # 启动可视化器（如果启用）
            if self.visualizer and self.config['visualization']:
                viz_thread = threading.Thread(target=self.visualizer.start)
                viz_thread.daemon = True
                viz_thread.start()
            
            # 运行测试循环
            self._run_test_loop()
            
            # 等待统计线程和可视化线程完成
            self.running = False
            stats_thread.join(timeout=2)
            
            # 收集最终结果
            self._collect_test_results()
            
            # 输出测试报告
            self._print_test_report()
            
            return self.test_results
            
        except Exception as e:
            logger.error(f"测试运行发生错误: {str(e)}")
            traceback.print_exc()
            self.running = False
            return {"status": "error", "message": str(e)}
    
    def _start_system_components(self):
        """启动系统组件"""
        dispatch = self.components['dispatch']
        
        # 设置调度间隔
        scheduling_interval = self.config['dispatch']['scheduling_interval']
        
        # 启动调度线程
        dispatch_thread = threading.Thread(
            target=dispatch.start_scheduling,
            args=(scheduling_interval,)
        )
        dispatch_thread.daemon = True
        dispatch_thread.start()
        
        logger.info(f"调度系统已启动，调度间隔: {scheduling_interval}秒")
    
    def _run_test_loop(self):
        """运行测试主循环"""
        # 获取测试持续时间
        test_duration = self.config['test_duration']
        log_interval = self.config['log_interval']
        
        # 获取任务生成率
        task_gen_rate = self.config['tasks']['generation_rate']
        
        # 存储上次日志和任务生成的时间
        last_log_time = time.time()
        last_task_gen_time = time.time()
        
        # 测试开始时间
        start_time = time.time()
        
        # 测试循环
        while time.time() - start_time < test_duration and self.running:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 定期打印状态
            if current_time - last_log_time >= log_interval:
                self._log_current_status(elapsed)
                last_log_time = current_time
            
            # 根据场景执行特定操作
            self._execute_scenario_actions(elapsed)
            
            # 随机生成任务
            if current_time - last_task_gen_time >= 1.0:  # 每秒检查一次
                if random.random() < task_gen_rate:
                    self.generate_random_task()
                last_task_gen_time = current_time
            
            # 短暂休眠以减少CPU使用
            time.sleep(0.1)
    
    def _execute_scenario_actions(self, elapsed):
        """根据当前场景执行特定操作"""
        scenario = self.config['current_scenario']
        
        if scenario == 'high_conflict':
            # 高冲突场景：每10秒添加一批交叉任务
            if int(elapsed) % 10 == 0 and int(elapsed) > 0 and elapsed - int(elapsed) < 0.1:
                self._create_crossing_tasks()
                
        elif scenario == 'path_planning_stress':
            # 路径规划压力测试：频繁重规划路径
            if int(elapsed) % 5 == 0 and int(elapsed) > 0 and elapsed - int(elapsed) < 0.1:
                self._stress_path_planning()
                
        elif scenario == 'system_resilience':
            print("  - 系统恢复力建议:")
            print("    - 实现组件状态监控和自动重启机制")
            print("    - 添加数据一致性验证")
            print("    - 考虑采用分布式架构提高容错性")

    def _create_crossing_tasks(self):
        """创建相互交叉的任务，增加冲突概率"""
        dispatch = self.components['dispatch']
        
        # 创建四个角落之间的交叉任务
        corners = [(20, 20), (20, 180), (180, 20), (180, 180)]
        
        for i in range(min(len(corners), 4)):
            start = corners[i]
            end = corners[(i + 2) % len(corners)]  # 对角交叉
            
            task = TransportTask(
                task_id=f"CROSS-{int(time.time()*1000 % 100000)}",
                start_point=start,
                end_point=end,
                task_type="manual",
                priority=random.randint(1, 3)
            )
            
            dispatch.add_task(task)
            
            # 更新统计
            with self.stats_lock:
                self.stats['tasks_created'] += 1
                self.stats['tasks_assigned'] += 1
        
        logger.info(f"已创建{len(corners)}个交叉任务，增加冲突概率")
    
    def _stress_path_planning(self):
        """对路径规划器施加压力"""
        planner = self.components['path_planner']
        vehicles = list(self.components['dispatch'].vehicles.values())
        
        if not vehicles:
            return
        
        # 随机选择部分车辆重新规划路径
        selected_vehicles = random.sample(vehicles, min(3, len(vehicles)))
        
        for vehicle in selected_vehicles:
            if not vehicle.current_task:
                continue
                
            # 强制重新规划路径
            try:
                end_point = vehicle.current_task.end_point
                new_path = planner.plan_path(vehicle.current_location, end_point, vehicle)
                
                if new_path and len(new_path) > 1:
                    vehicle.assign_path(new_path)
                    logger.debug(f"已为车辆 {vehicle.vehicle_id} 重新规划路径")
            except Exception as e:
                logger.warning(f"路径重规划失败: {str(e)}")
    
    def _simulate_system_failure(self):
        """模拟系统故障以测试恢复能力"""
        dispatch = self.components['dispatch']
        
        logger.warning("模拟系统故障...")
        
        # 记录恢复开始时间
        recovery_start = time.time()
        
        try:
            # 1. 模拟调度器故障
            if hasattr(dispatch, 'running'):
                dispatch.running = False
                time.sleep(0.5)  # 暂停运行半秒
                dispatch.running = True
            
            # 2. 清除部分车辆的路径
            vehicles = list(dispatch.vehicles.values())
            if vehicles:
                for vehicle in random.sample(vehicles, min(2, len(vehicles))):
                    vehicle.current_path = []
                    vehicle.path_index = 0
            
            # 3. 模拟障碍物变化
            if hasattr(self.components['path_planner'], 'obstacle_grids'):
                original_obstacles = self.components['path_planner'].obstacle_grids.copy()
                # 随机添加一些临时障碍物
                temp_obstacles = [(random.randint(0, 200), random.randint(0, 200)) 
                                 for _ in range(10)]
                self.components['path_planner'].obstacle_grids.update(temp_obstacles)
                
                # 短暂延迟后恢复原始障碍物
                def restore_obstacles():
                    time.sleep(2.0)
                    self.components['path_planner'].obstacle_grids = original_obstacles
                    logger.info("已恢复原始障碍物")
                
                threading.Thread(target=restore_obstacles, daemon=True).start()
            
            # 记录恢复时间
            recovery_time = time.time() - recovery_start
            with self.stats_lock:
                self.stats['system_recoveries'] += 1
                self.stats['recovery_times'].append(recovery_time)
            
            logger.info(f"系统故障模拟完成，恢复耗时: {recovery_time:.3f}秒")
            
        except Exception as e:
            logger.error(f"故障模拟过程中发生错误: {str(e)}")
    
    def _update_stats_loop(self):
        """统计数据更新循环"""
        update_interval = self.config['stats_interval']
        
        while self.running:
            try:
                self._update_stats()
                time.sleep(update_interval)
            except Exception as e:
                logger.error(f"更新统计数据出错: {str(e)}")
                time.sleep(1.0)  # 出错时降低更新频率
    
    def _update_stats(self):
        """更新统计数据"""
        with self.stats_lock:
            dispatch = self.components['dispatch']
            
            # 更新系统运行时间
            self.stats['system_uptime'] = time.time() - self.test_start_time
            
            # 更新任务统计
            completed_count = len(getattr(dispatch, 'completed_tasks', {}))
            current_active = len(getattr(dispatch, 'active_tasks', {}))
            queued_tasks = len(getattr(dispatch, 'task_queue', []))
            
            # 计算任务完成率
            if self.stats['tasks_assigned'] > 0:
                self.stats['task_completion_rate'] = (
                    self.stats['tasks_completed'] / self.stats['tasks_assigned']
                )
            
            # 更新冲突统计
            if hasattr(dispatch, 'performance_metrics'):
                self.stats['conflicts_detected'] = dispatch.performance_metrics.get('conflict_count', 0)
                self.stats['conflicts_resolved'] = dispatch.performance_metrics.get('resolved_conflicts', 0)
                
                if self.stats['conflicts_detected'] > 0:
                    self.stats['conflict_resolution_rate'] = (
                        self.stats['conflicts_resolved'] / self.stats['conflicts_detected']
                    )
            
            # 更新路径规划统计
            if self.stats['paths_planned'] > 0:
                self.stats['path_planning_success_rate'] = (
                    (self.stats['paths_planned'] - self.stats['path_planning_failures']) / 
                    self.stats['paths_planned']
                )
            
            # 更新车辆利用率
            self.stats['vehicle_utilization'] = self._calculate_vehicle_utilization()
            
            # 更新装卸点利用率
            self.stats['loading_point_utilization'] = self._calculate_loading_point_utilization()
    
    def _calculate_vehicle_utilization(self):
        """计算车辆利用率"""
        dispatch = self.components['dispatch']
        utilization = {}
        
        for vid, vehicle in dispatch.vehicles.items():
            if not hasattr(vehicle, 'state'):
                continue
                
            state_name = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
            
            # 工作状态: EN_ROUTE, UNLOADING, PREPARING
            working_states = ['EN_ROUTE', 'UNLOADING', 'PREPARING']
            is_working = state_name in working_states or state_name == 'VehicleState.EN_ROUTE'
            
            # 有任务但未工作
            has_task = hasattr(vehicle, 'current_task') and vehicle.current_task is not None
            
            # 空闲状态
            is_idle = state_name in ['IDLE'] or state_name == 'VehicleState.IDLE'
            
            utilization[vid] = {
                'state': state_name,
                'working': is_working,
                'has_task': has_task,
                'idle': is_idle
            }
        
        # 计算总体利用率
        if utilization:
            working_count = sum(1 for data in utilization.values() if data['working'])
            total_count = len(utilization)
            overall_utilization = working_count / total_count if total_count > 0 else 0
            utilization['overall'] = overall_utilization
        
        return utilization
    
    def _calculate_loading_point_utilization(self):
        """计算装卸点利用率"""
        dispatch = self.components['dispatch']
        dispatch_config = dispatch._load_config()
        
        loading_points = dispatch_config['loading_points']
        unloading_point = dispatch_config['unloading_point']
        
        # 找出每个点附近的车辆数量
        utilization = {}
        
        # 装载点
        for i, point in enumerate(loading_points):
            point_name = f"loading_point_{i+1}"
            vehicles_at_point = [
                v for v in dispatch.vehicles.values()
                if math.dist(v.current_location, point) < 10.0
            ]
            utilization[point_name] = len(vehicles_at_point)
        
        # 卸载点
        vehicles_at_unloading = [
            v for v in dispatch.vehicles.values()
            if math.dist(v.current_location, unloading_point) < 10.0
        ]
        utilization['unloading_point'] = len(vehicles_at_unloading)
        
        return utilization
    
    def _log_current_status(self, elapsed):
        """记录当前状态"""
        # 格式化时间
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        
        # 获取关键统计数据
        with self.stats_lock:
            completed = self.stats['tasks_completed']
            created = self.stats['tasks_created']
            conflicts = self.stats['conflicts_detected']
            resolved = self.stats['conflicts_resolved']
            planning_failures = self.stats['path_planning_failures']
            
            # 获取车辆状态
            vehicle_states = defaultdict(int)
            if 'vehicle_utilization' in self.stats:
                for vid, data in self.stats['vehicle_utilization'].items():
                    if vid != 'overall' and 'state' in data:
                        vehicle_states[data['state']] += 1
            
            # 计算利用率
            utilization = self.stats['vehicle_utilization'].get('overall', 0) * 100 if 'vehicle_utilization' in self.stats else 0
            
            # 装卸点使用情况
            loading_usage = sum(self.stats['loading_point_utilization'].values()) if 'loading_point_utilization' in self.stats else 0
            unloading_usage = self.stats['loading_point_utilization'].get('unloading_point', 0) if 'loading_point_utilization' in self.stats else 0
        
        # 输出状态日志
        logger.info(
            f"测试进度: {elapsed_str} | "
            f"任务: {completed}/{created} 完成 | "
            f"冲突: {resolved}/{conflicts} 已解决 | "
            f"规划失败: {planning_failures} | "
            f"车辆利用率: {utilization:.1f}%"
        )
        
        # 输出车辆状态分布
        state_str = ", ".join(f"{state}: {count}" for state, count in vehicle_states.items())
        logger.info(f"车辆状态: {state_str}")
        
        # 输出装卸点使用情况
        logger.info(f"装载点使用: {loading_usage}, 卸载点使用: {unloading_usage}")
    
    def _collect_test_results(self):
        """收集测试结果"""
        with self.stats_lock:
            self.stats['end_time'] = datetime.now()
            
            # 计算测试时长
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            
            # 准备测试结果
            self.test_results = {
                'scenario': self.config['current_scenario'],
                'duration': duration,
                'stats': self.stats.copy(),
                'timestamp': datetime.now().isoformat(),
                'status': 'completed'
            }
            
            # 计算关键性能指标
            if self.stats['tasks_assigned'] > 0:
                self.test_results['task_completion_rate'] = (
                    self.stats['tasks_completed'] / self.stats['tasks_assigned']
                )
            else:
                self.test_results['task_completion_rate'] = 0.0
                
            if self.stats['conflicts_detected'] > 0:
                self.test_results['conflict_resolution_rate'] = (
                    self.stats['conflicts_resolved'] / self.stats['conflicts_detected']
                )
            else:
                self.test_results['conflict_resolution_rate'] = 1.0  # 无冲突时为100%
                
            # 计算平均指标
            self.test_results['avg_path_planning_time'] = (
                sum(self.stats['path_planning_times']) / len(self.stats['path_planning_times'])
                if self.stats['path_planning_times'] else 0.0
            )
            
            self.test_results['avg_recovery_time'] = (
                sum(self.stats['recovery_times']) / len(self.stats['recovery_times'])
                if self.stats['recovery_times'] else 0.0
            )
            
            # 系统健康状态评估
            self.test_results['system_health'] = self._assess_system_health()
    
    def _assess_system_health(self):
        """评估系统整体健康状态"""
        health = {
            'status': 'healthy',
            'components': {
                'dispatcher': 'operational',
                'path_planner': 'operational',
                'conflict_resolver': 'operational',
                'task_manager': 'operational'
            },
            'warnings': [],
            'critical_issues': []
        }
        
        # 评估调度器健康状态
        if self.stats['tasks_assigned'] > 10 and self.stats['tasks_completed'] == 0:
            health['components']['dispatcher'] = 'critical'
            health['critical_issues'].append('调度器未能完成任何任务')
            health['status'] = 'critical'
        elif self.stats['task_completion_rate'] < 0.3 and self.stats['tasks_assigned'] > 5:
            health['components']['dispatcher'] = 'degraded'
            health['warnings'].append('调度器任务完成率过低')
            health['status'] = 'degraded'
        
        # 评估路径规划器健康状态
        if self.stats['path_planning_failures'] > self.stats['paths_planned'] * 0.5:
            health['components']['path_planner'] = 'critical'
            health['critical_issues'].append('路径规划失败率过高')
            health['status'] = 'critical'
        elif self.stats['path_planning_failures'] > self.stats['paths_planned'] * 0.2:
            health['components']['path_planner'] = 'degraded'
            health['warnings'].append('路径规划存在较多失败')
            health['status'] = 'degraded'
        
        # 评估冲突解决器健康状态
        if (self.stats['conflicts_detected'] > 10 and 
            self.stats['conflict_resolution_rate'] < 0.5):
            health['components']['conflict_resolver'] = 'critical'
            health['critical_issues'].append('冲突解决能力不足')
            health['status'] = 'critical'
        elif (self.stats['conflicts_detected'] > 5 and 
              self.stats['conflict_resolution_rate'] < 0.8):
            health['components']['conflict_resolver'] = 'degraded'
            health['warnings'].append('部分冲突未能解决')
            health['status'] = 'degraded'
        
        # 评估任务管理器健康状态
        active_tasks = len(getattr(self.components['dispatch'], 'active_tasks', {}))
        queued_tasks = len(getattr(self.components['dispatch'], 'task_queue', []))
        
        if queued_tasks > 20 and self.stats['tasks_completed'] == 0:
            health['components']['task_manager'] = 'critical'
            health['critical_issues'].append('任务队列堆积且无任务完成')
            health['status'] = 'critical'
        elif queued_tasks > 10 and self.stats['task_completion_rate'] < 0.2:
            health['components']['task_manager'] = 'degraded'
            health['warnings'].append('任务队列堆积')
            health['status'] = 'degraded'
        
        return health
    
    def _print_test_report(self):
        """打印测试报告"""
        if not self.test_results:
            logger.warning("没有可用的测试结果")
            return
        
        # 提取关键结果
        scenario = self.test_results['scenario']
        duration = self.test_results['duration']
        stats = self.test_results['stats']
        health = self.test_results['system_health']
        
        # 打印报告头部
        print("\n" + "="*80)
        print(f"露天矿多车协同调度系统 - {scenario.upper()} 场景测试报告")
        print("="*80)
        
        # 打印测试基本信息
        print(f"\n测试持续时间: {int(duration//60)}分 {int(duration%60)}秒")
        print(f"开始时间: {stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结束时间: {stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 打印系统健康状态
        print(f"\n系统健康状态: {health['status'].upper()}")
        print("组件状态:")
        for component, status in health['components'].items():
            print(f"  - {component}: {status}")
        
        if health['warnings']:
            print("\n警告:")
            for warning in health['warnings']:
                print(f"  - {warning}")
                
        if health['critical_issues']:
            print("\n严重问题:")
            for issue in health['critical_issues']:
                print(f"  - {issue}")
        
        # 打印任务统计
        print("\n任务统计:")
        print(f"  - 创建任务数: {stats['tasks_created']}")
        print(f"  - 分配任务数: {stats['tasks_assigned']}")
        print(f"  - 完成任务数: {stats['tasks_completed']}")
        print(f"  - 失败任务数: {stats['tasks_failed']}")
        print(f"  - 任务完成率: {self.test_results['task_completion_rate']*100:.1f}%")
        
        # 打印冲突统计
        print("\n冲突统计:")
        print(f"  - 检测到的冲突: {stats['conflicts_detected']}")
        print(f"  - 解决的冲突: {stats['conflicts_resolved']}")
        print(f"  - 冲突解决率: {self.test_results['conflict_resolution_rate']*100:.1f}%")
        
        # 打印路径规划统计
        print("\n路径规划统计:")
        print(f"  - 规划路径总数: {stats['paths_planned']}")
        print(f"  - 规划失败次数: {stats['path_planning_failures']}")
        print(f"  - 规划成功率: {stats.get('path_planning_success_rate', 0)*100:.1f}%")
        print(f"  - 平均规划时间: {self.test_results['avg_path_planning_time']*1000:.2f}毫秒")
        
        # 打印系统恢复能力
        print("\n系统恢复能力:")
        print(f"  - 系统恢复次数: {stats['system_recoveries']}")
        if stats['recovery_times']:
            print(f"  - 平均恢复时间: {self.test_results['avg_recovery_time']:.2f}秒")
        
        # 打印结论和建议
        print("\n测试结论和建议:")
        if health['status'] == 'healthy':
            print("  系统运行良好，所有组件正常工作。")
        elif health['status'] == 'degraded':
            print("  系统运行正常但存在一些性能问题，需要关注警告信息。")
        else:  # critical
            print("  系统存在严重问题，需要立即解决。")
        
        # 输出详细建议
        self._print_recommendations()
        
        print("\n" + "="*80)
    
    def _print_recommendations(self):
        """根据测试结果打印改进建议"""
        if not self.test_results:
            return
            
        health = self.test_results['system_health']
        stats = self.test_results['stats']
        
        print("\n系统改进建议:")
        
        # 调度器建议
        if health['components']['dispatcher'] != 'operational':
            if self.test_results['task_completion_rate'] < 0.3:
                print("  - 调度算法需要优化以提高任务完成率")
                print("    建议: 改进任务分配策略，考虑车辆负载均衡和位置优化")
        
        # 路径规划建议
        if health['components']['path_planner'] != 'operational':
            if stats['path_planning_failures'] > 5:
                print("  - 路径规划器存在稳定性问题")
                print("    建议: 增强错误处理机制，添加更多备选规划策略")
        
        # 冲突解决建议
        if health['components']['conflict_resolver'] != 'operational':
            if self.test_results['conflict_resolution_rate'] < 0.8:
                print("  - 冲突解决效率不足")
                print("    建议: 优化ConflictBasedSearch算法，改进优先级处理机制")
        
        # 任务管理建议
        if health['components']['task_manager'] != 'operational':
            print("  - 任务管理存在问题")
            print("    建议: 审查任务生命周期管理，确保任务状态正确转换")
        
        # 通用性能建议
        print("  - 性能优化建议:")
        print("    - 考虑使用空间索引加速路径规划和冲突检测")
        print("    - 调整调度周期以平衡响应性和系统负载")
        print("    - 增加关键操作的超时机制")
        
        # 针对特定场景的建议
        scenario = self.test_results['scenario']
        if scenario == 'high_conflict':
            print("  - 高冲突场景建议:")
            print("    - 考虑使用相对速度控制减少车辆交叉")
            print("    - 实现等待行为而非路径重规划")
        elif scenario == 'deadlock':
            print("  - 死锁场景建议:")
            print("    - 添加死锁检测机制")
            print("    - 实现预防性避让策略")
        elif scenario == 'system_resilience':
            print("  - 系统恢复力建议:")
            print("    - 实现组件状态监控和自动重启机制")
            print("    - 添加数据一致性验证")
            print("    - 考虑采用分布式架构提高容错性")

class SystemVisualizer:
    """调度系统可视化器"""
    
    def __init__(self, test_framework):
        """初始化可视化器"""
        self.framework = test_framework
        self.dispatch = test_framework.components['dispatch']
        self.dispatch_config = self.dispatch._load_config()
        
        # 可视化相关属性
        self.fig = None
        self.ax = None
        self.vehicle_plots = {}
        self.path_plots = {}
        self.labels = {}
        self.obstacle_plot = None
        
        # 可视化标志
        self.running = False
        self.update_interval = 0.5  # 更新间隔(秒)
    
    def start(self):
        """启动可视化"""
        try:
            # 导入必要的模块
            import matplotlib.pyplot as plt
            
            # 创建新图形
            plt.ion()  # 开启交互模式
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.ax.set_xlim(-50, 250)
            self.ax.set_ylim(-50, 250)
            self.ax.set_title('露天矿多车协同调度系统 - 实时可视化')
            self.ax.set_xlabel('X坐标')
            self.ax.set_ylabel('Y坐标')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制基础设施
            self._draw_infrastructure()
            
            # 绘制障碍物
            self._draw_obstacles()
            
            # 初始化车辆标记
            self._initialize_vehicle_markers()
            
            # 设置标志
            self.running = True
            
            # 开始可视化循环
            self._visualization_loop()
            
        except ImportError:
            logger.warning("无法导入matplotlib，不能启用可视化")
        except Exception as e:
            logger.error(f"启动可视化时发生错误: {str(e)}")
            traceback.print_exc()
    
    def stop(self):
        """停止可视化"""
        self.running = False
    
    def _draw_infrastructure(self):
        """绘制基础设施"""
        # 绘制装载点
        for i, lp in enumerate(self.dispatch_config['loading_points']):
            self.ax.plot(lp[0], lp[1], 'go', markersize=12, label=f'装载点{i+1}' if i==0 else "")
            self.ax.text(lp[0]+5, lp[1]+5, f'装载点{i+1}', fontsize=10)
            
        # 绘制卸载点
        unload = self.dispatch_config['unloading_point']
        self.ax.plot(unload[0], unload[1], 'rs', markersize=12, label='卸载点')
        self.ax.text(unload[0]+5, unload[1]+5, '卸载点', fontsize=10)
        
        # 绘制停车场
        parking = self.dispatch_config['parking_area']
        self.ax.plot(parking[0], parking[1], 'b^', markersize=12, label='停车场')
        self.ax.text(parking[0]+5, parking[1]+5, '停车场', fontsize=10)
        
        # 添加图例
        self.ax.legend(loc='upper right')
    
    def _draw_obstacles(self):
        """绘制障碍物"""
        if hasattr(self.framework.components['path_planner'], 'obstacle_grids'):
            obstacles = self.framework.components['path_planner'].obstacle_grids
            if obstacles:
                x = [p[0] for p in obstacles]
                y = [p[1] for p in obstacles]
                self.obstacle_plot = self.ax.scatter(
                    x, y, c='gray', s=10, alpha=0.5, marker='s', label='障碍物'
                )
    
    def _initialize_vehicle_markers(self):
        """初始化车辆标记"""
        for i, (vid, vehicle) in enumerate(self.dispatch.vehicles.items()):
            # 使用不同颜色表示不同车辆
            color = plt.cm.tab10(i % 10)
            
            # 创建车辆标记
            vehicle_plot, = self.ax.plot(
                vehicle.current_location[0], 
                vehicle.current_location[1], 
                'o', 
                color=color,
                markersize=10,
                label=f'车辆{vid}'
            )
            
            # 存储车辆标记
            self.vehicle_plots[vid] = {
                'plot': vehicle_plot,
                'color': color,
                'path_plot': None,
                'status_label': None
            }
    
    def _visualization_loop(self):
        """可视化循环"""
        try:
            while self.running and self.framework.running:
                # 更新可视化
                self._update_visualization()
                
                # 处理matplotlib事件
                plt.pause(self.update_interval)
                
        except Exception as e:
            logger.error(f"可视化循环发生错误: {str(e)}")
        finally:
            # 关闭交互模式
            plt.ioff()
    
    def _update_visualization(self):
        """更新可视化内容"""
        try:
            # 获取当前测试时间
            elapsed = time.time() - self.framework.test_start_time
            
            # 更新标题
            scenario = self.framework.config['current_scenario']
            with self.framework.stats_lock:
                completed = self.framework.stats['tasks_completed']
                assigned = self.framework.stats['tasks_assigned']
                conflicts = self.framework.stats['conflicts_detected']
            
            self.ax.set_title(
                f'露天矿多车协同调度系统 - {scenario.upper()} 场景\n'
                f'运行时间: {int(elapsed//60):02d}:{int(elapsed%60):02d} | '
                f'任务: {completed}/{assigned} | '
                f'冲突: {conflicts}'
            )
            
            # 更新车辆位置和路径
            for vid, vehicle in self.dispatch.vehicles.items():
                if vid not in self.vehicle_plots:
                    continue
                    
                # 更新车辆位置
                self.vehicle_plots[vid]['plot'].set_data(
                    vehicle.current_location[0], 
                    vehicle.current_location[1]
                )
                
                # 更新状态标签
                state_name = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                task_id = vehicle.current_task.task_id if vehicle.current_task else "无任务"
                
                label_text = f"{vid}: {state_name}\n{task_id}"
                
                if self.vehicle_plots[vid]['status_label'] is None:
                    # 创建新标签
                    self.vehicle_plots[vid]['status_label'] = self.ax.text(
                        vehicle.current_location[0], 
                        vehicle.current_location[1] + 8,
                        label_text,
                        fontsize=8,
                        color=self.vehicle_plots[vid]['color']
                    )
                else:
                    # 更新现有标签
                    self.vehicle_plots[vid]['status_label'].set_position(
                        (vehicle.current_location[0], vehicle.current_location[1] + 8)
                    )
                    self.vehicle_plots[vid]['status_label'].set_text(label_text)
                
                # 更新路径
                if hasattr(vehicle, 'current_path') and vehicle.current_path:
                    path_x = [p[0] for p in vehicle.current_path]
                    path_y = [p[1] for p in vehicle.current_path]
                    
                    if self.vehicle_plots[vid]['path_plot'] is None:
                        # 创建新路径线
                        path_plot, = self.ax.plot(
                            path_x, path_y, '--', 
                            color=self.vehicle_plots[vid]['color'],
                            alpha=0.5,
                            linewidth=1
                        )
                        self.vehicle_plots[vid]['path_plot'] = path_plot
                    else:
                        # 更新现有路径线
                        self.vehicle_plots[vid]['path_plot'].set_data(path_x, path_y)
                elif self.vehicle_plots[vid]['path_plot'] is not None:
                    # 清除路径
                    self.vehicle_plots[vid]['path_plot'].set_data([], [])
            
            # 刷新图形
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logger.error(f"更新可视化内容时发生错误: {str(e)}")


def run_all_tests():
    """运行所有场景的测试"""
    print("=" * 80)
    print("露天矿多车协同调度系统 - 全场景测试")
    print("=" * 80)
    
    # 创建测试框架
    framework = TestFramework(create_visualizer=False)
    
    # 获取所有场景
    scenarios = framework.config['scenarios']
    
    # 存储每个场景的结果
    results = {}
    
    # 运行每个场景
    for scenario in scenarios:
        print(f"\n正在运行 '{scenario}' 场景...")
        
        # 重新创建测试框架（确保每个场景使用新的实例）
        test = TestFramework(create_visualizer=False)
        test.config['current_scenario'] = scenario
        
        # 调整测试持续时间（减少全场景测试总时间）
        test.config['test_duration'] = 60  # 每个场景1分钟
        
        # 运行测试
        result = test.run_test()
        results[scenario] = result
        
        # 短暂等待，避免系统资源竞争
        time.sleep(1)
    
    # 打印汇总报告
    print("\n" + "=" * 80)
    print("全场景测试汇总报告")
    print("=" * 80)
    
    print("\n场景性能比较:")
    
    # 创建比较表格
    headers = ["场景", "任务完成率", "冲突解决率", "规划成功率", "系统健康状态"]
    row_format = "{:<15} {:<12} {:<12} {:<12} {:<15}"
    
    print(row_format.format(*headers))
    print("-" * 70)
    
    for scenario, result in results.items():
        if not result or isinstance(result, dict) and result.get('status') == 'error':
            print(row_format.format(
                scenario,
                "测试失败",
                "测试失败",
                "测试失败",
                "测试失败"
            ))
            continue
            
        # 提取指标
        task_rate = result.get('task_completion_rate', 0) * 100
        conflict_rate = result.get('conflict_resolution_rate', 0) * 100
        planning_rate = (
            result['stats'].get('path_planning_success_rate', 0) * 100
            if 'stats' in result else 0
        )
        health_status = (
            result['system_health']['status'].upper()
            if 'system_health' in result else "未知"
        )
        
        # 打印行
        print(row_format.format(
            scenario,
            f"{task_rate:.1f}%",
            f"{conflict_rate:.1f}%",
            f"{planning_rate:.1f}%",
            health_status
        ))
    
    # 分析结果，确定最佳和最差场景
    best_scenario = max(
        results.keys(),
        key=lambda s: results[s].get('task_completion_rate', 0)
        if isinstance(results[s], dict) and results[s].get('status') != 'error'
        else 0
    )
    
    worst_scenario = min(
        results.keys(),
        key=lambda s: results[s].get('task_completion_rate', 0)
        if isinstance(results[s], dict) and results[s].get('status') != 'error'
        else 1
    )
    
    print("\n测试结论:")
    print(f"- 最佳表现场景: {best_scenario}")
    print(f"- 最差表现场景: {worst_scenario}")
    
    print("\n关键改进建议:")
    print("1. 集中优化最差场景中暴露的问题")
    print("2. 找出最佳场景的成功因素并应用到其他场景")
    print("3. 平衡各场景性能，特别关注冲突解决和路径规划")
    
    return results


def run_integration_test():
    """
    运行单个集成测试，适合命令行调用
    """
    import argparse
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='露天矿多车协同调度系统集成测试')
    
    parser.add_argument('--scenario', type=str, default='normal',
                      choices=['normal', 'high_conflict', 'path_planning_stress', 
                               'deadlock', 'system_resilience', 'all'],
                      help='测试场景')
    
    parser.add_argument('--duration', type=int, default=120,
                      help='测试持续时间(秒)')
    
    parser.add_argument('--vehicles', type=int, default=5,
                      help='测试车辆数量')
    
    parser.add_argument('--tasks', type=int, default=5,
                      help='初始任务数量')
    
    parser.add_argument('--visual', action='store_true',
                      help='启用可视化')
    
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='日志级别')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))
    
    # 运行所有场景测试
    if args.scenario == 'all':
        results = run_all_tests()
        return
    
    # 创建配置
    config = {
        'test_duration': args.duration,
        'log_interval': 5,
        'stats_interval': 2,
        'visualization': args.visual,
        
        'map': {
            'grid_size': 200,
            'obstacle_density': 0.15,
            'add_predefined_obstacles': True
        },
        
        'vehicles': {
            'count': args.vehicles,
            'speeds': [5.0, 6.0, 7.0, 8.0, 9.0],
            'capacities': [40, 45, 50, 55, 60],
            'start_at_different_points': True
        },
        
        'tasks': {
            'initial_count': args.tasks,
            'generation_rate': 0.2,
            'types': ['loading', 'unloading', 'manual'],
            'type_weights': [0.6, 0.3, 0.1],
            'priorities': [1, 2, 3],
            'priority_weights': [0.5, 0.3, 0.2]
        },
        
        'dispatch': {
            'scheduling_interval': 2.0,
            'conflict_check_interval': 1.0
        },
        
        'scenarios': [
            'normal',
            'high_conflict',
            'path_planning_stress',
            'deadlock', 
            'system_resilience'
        ],
        
        'current_scenario': args.scenario
    }
    
    # 创建并运行测试
    test = TestFramework(config=config, create_visualizer=args.visual)
    result = test.run_test()
    
    # 打印最终状态
    print("\n最终系统状态:")
    dispatch = test.components['dispatch']
    
    print(f"总车辆数: {len(dispatch.vehicles)}")
    
    # 打印车辆状态分布
    vehicle_states = defaultdict(int)
    for vehicle in dispatch.vehicles.values():
        state_name = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
        vehicle_states[state_name] += 1
    
    print("车辆状态分布:")
    for state, count in vehicle_states.items():
        print(f"  {state}: {count}辆")
    
    print(f"\n任务队列长度: {len(getattr(dispatch, 'task_queue', []))}")
    print(f"活动任务数: {len(getattr(dispatch, 'active_tasks', {}))}")
    print(f"已完成任务数: {len(getattr(dispatch, 'completed_tasks', {}))}")
    
    return result


if __name__ == "__main__":
    """主函数: 直接调用时运行集成测试"""
    run_integration_test()

