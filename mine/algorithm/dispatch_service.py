"""
多车协同调度系统核心模块 v4.0
实现功能：
1. 基于时间窗的时空预约表
2. 装卸点优先级调度
3. 充电调度策略
4. CBS冲突避免算法
"""
from __future__ import annotations
import heapq
import threading
import os
import sys
import os
from config.paths import PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import numpy as np
from models.vehicle import MiningVehicle ,VehicleState, TransportStage
from models.task import TransportTask
from algorithm.path_planner import HybridPathPlanner
from algorithm.map_service import MapService
import logging

class TransportScheduler:
    """调度策略抽象层"""
    def __init__(self, config: dict):
        self.loading_points = config['loading_points']
        self.unloading_point = config['unloading_point']
        self.parking_area = config['parking_area']
        self.time_window_size = timedelta(minutes=15)  # 时间窗粒度
        self.max_charging_vehicles = config['max_charging_vehicles']  # 新增配置项        
        # 调度队列
        self.loading_queues = {lp: deque() for lp in self.loading_points}
        self.unloading_queue = deque()
        self.charging_queue = deque()  # 新增充电队列
        
        # 时空预约表
        self.reservation_table = defaultdict(set)  # {(x,y,time_slot): vehicle_ids}
        
    def apply_scheduling_policy(self, vehicles: List[MiningVehicle], tasks: List[TransportTask]) -> Dict:
        """改进后的调度策略"""
        assignments = {}
        vehicle_list = list(vehicles)  # 创建车辆列表副本
        
        # 优先处理手动任务 ▼▼▼
        manual_tasks = [t for t in tasks if t.task_type == "manual"]
        for task in manual_tasks:
            if not vehicle_list:
                break
                
            # 寻找最近空闲车辆
            closest_vehicle = min(
                (v for v in vehicle_list if v.status == VehicleState.IDLE),
                key=lambda v: GeoUtils.calculate_distance(v.current_location, task.start_point),
                default=None
            )
            
            if closest_vehicle:
                assignments[closest_vehicle.vehicle_id] = task
                vehicle_list.remove(closest_vehicle)  # 防止重复分配
                tasks.remove(task)

        # 原有装载任务分配逻辑 ▼▼▼
        for task in tasks:
            if task.task_type == "loading":
                point = min(self.loading_queues, 
                          key=lambda lp: len(self.loading_queues[lp]))
                self.loading_queues[point].append(task)

        for vehicle in vehicle_list:
            if vehicle.status == VehicleState.IDLE and not vehicle.current_task:
                for point in self.loading_queues:
                    if self.loading_queues[point]:
                        task = self.loading_queues[point].popleft()
                        assignments[vehicle.vehicle_id] = task
                        break
        
        # 最后处理卸载任务 ▼▼▼
        unloading_vehicles = [v for v in vehicles if v.status == VehicleState.UNLOADING]
        for vehicle in unloading_vehicles:
            if self.unloading_queue and vehicle.vehicle_id not in assignments:
                task = self.unloading_queue.popleft()
                assignments[vehicle.vehicle_id] = task

        return assignments

class ConflictBasedSearch:
    """CBS冲突避免算法实现"""
    def __init__(self, planner: HybridPathPlanner):
        self.planner = planner
        self.constraints = defaultdict(list)
        
    def find_conflicts(self, paths: Dict[str, List[Tuple]]) -> List[Tuple]:
        """检测路径冲突"""
        conflicts = []
        path_items = list(paths.items())
        
        for i in range(len(path_items)):
            vid1, path1 = path_items[i]
            for j in range(i+1, len(path_items)):
                vid2, path2 = path_items[j]
                for t, ((x1,y1), (x2,y2)) in enumerate(zip(path1, path2)):
                    if (x1,y1) == (x2,y2):
                        conflicts.append( (t, (x1,y1), vid1, vid2) )
        return conflicts
    def _replan_path(self, vehicle: MiningVehicle, max_retries=3):
        """增强路径重规划"""
        for attempt in range(max_retries):
            new_path = self.planner.plan_path(
                vehicle.current_location,
                vehicle.current_task.end_point,
                vehicle
            )
            if new_path:
                vehicle.assign_path(new_path)
                self._update_reservation_table(new_path, vehicle.vehicle_id)
                return new_path
            logging.debug(f"路径重试 {attempt+1}/{max_retries}")
            time.sleep(0.5)
        return None
    def _get_vehicle_priority(self, vehicle_id: str) -> int:
        vehicle = self.planner.dispatch.vehicles[vehicle_id]
        # 修改优先级映射 ▼▼▼
        priorities = {
            VehicleState.UNLOADING: 1,
            VehicleState.PREPARING: 2,
            TransportStage.TRANSPORTING: 3,
            TransportStage.APPROACHING: 4,
            VehicleState.IDLE: 5
        }
        # ▲▲▲
        return priorities.get(vehicle.status if vehicle.status != VehicleState.EN_ROUTE else vehicle.transport_stage, 5)
    def resolve_conflicts(self, paths: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
        """实现基于优先级的冲突解决方案"""
        new_paths = paths.copy()
        
        # 获取所有冲突
        conflicts = self.find_conflicts(paths)
        for conflict in conflicts:
            t, pos, vid1, vid2 = conflict
            
            # 获取车辆优先级
            prio1 = self._get_vehicle_priority(vid1)
            prio2 = self._get_vehicle_priority(vid2)
            
            # 重新规划低优先级车辆路径
            if prio1 < prio2:
                new_path = self._replan_path(vid1, pos, t)
                if new_path:
                    new_paths[vid1] = new_path
            else:
                new_path = self._replan_path(vid2, pos, t)
                if new_path:
                    new_paths[vid2] = new_path
        
        return new_paths

class DispatchSystem:
    """智能调度系统核心"""
    def __init__(self, planner: HybridPathPlanner, map_service: MapService):
        self.planner = planner
        self.map_service = map_service
        self.vehicles: Dict[str, MiningVehicle] = {}
        self.scheduler = TransportScheduler(self._load_config())
        self.cbs = ConflictBasedSearch(planner)
        
        # 任务管理
        self.task_queue = deque()  # 原为 []
        self.active_tasks = {}
        self.lock = threading.RLock()
        self.completed_tasks = {}
        
        # 并发控制
        self.lock = threading.RLock()
        self.vehicle_lock = threading.Lock()
        
    def _load_config(self) -> dict:
        """加载调度配置"""
        return {
            'loading_points': [(-100,50), (0,150), (100,50)],
            'unloading_point': (0,-100),
            'parking_area': (200,200),
            'max_charging_vehicles': 2
        }
    
    # ----------------- 核心调度循环 -----------------
    def scheduling_cycle(self):
        """调度主循环（每30秒触发）"""
        with self.lock:
            self._update_vehicle_states()
            self._dispatch_tasks()  # 触发任务分配
            self._detect_conflicts()
    def _dispatch_tasks(self):
        """任务分配逻辑（补充方法实现）"""
        if not self.task_queue:
            return
            
        assignments = self.scheduler.apply_scheduling_policy(
            list(self.vehicles.values()),
            self.task_queue
        )
        
        # 将分配结果存入激活任务
        with self.vehicle_lock:
            for vid, task in assignments.items():
                vehicle = self.vehicles[vid]
                vehicle.assign_task(task)
                self.active_tasks[task.task_id] = task
                logging.info(f"车辆 {vid} 已分配任务 {task.task_id}")
        
        # 移除已分配任务
        assigned_task_ids = {t.task_id for t in assignments.values()}
        self.task_queue = [t for t in self.task_queue 
                        if t.task_id not in assigned_task_ids]     
    def _update_vehicle_states(self):
        """增强运输阶段追踪"""
        for vid, vehicle in self.vehicles.items():
            # 状态更新保持不变 ▼▼▼（已正确使用新状态系统）
            if vehicle.current_task and vehicle.status != VehicleState.EN_ROUTE:
                vehicle.status = VehicleState.EN_ROUTE
            if vehicle.current_location == self.scheduler.parking_area:
                vehicle.status = VehicleState.IDLE
            elif vehicle.current_location in self.scheduler.loading_points:
                vehicle.status = VehicleState.PREPARING
            elif vehicle.current_location == self.scheduler.unloading_point:
                vehicle.status = VehicleState.UNLOADING
            
            if vehicle.current_task:
                vehicle.status = VehicleState.EN_ROUTE
                if vehicle.current_task.task_type == "loading":
                    vehicle.transport_stage = TransportStage.APPROACHING
                elif vehicle.current_task.task_type == "unloading":
                    vehicle.transport_stage = TransportStage.TRANSPORTING
            if vehicle.current_task and vehicle.path_index >= len(vehicle.current_path)-1:
                completed_task = vehicle.current_task
                self.completed_tasks[completed_task.task_id] = completed_task
                del self.active_tasks[completed_task.task_id]
                vehicle.current_task = None
    def print_ascii_map(self):
        """生成ASCII地图可视化"""
        MAP_SIZE = 40  # 地图尺寸（40x40）
        SCALE = 5      # 坐标缩放因子
        
        # 初始化空地图（修复变量定义位置）
        grid = [['·' for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
        
        # 标注固定设施
        config = self._load_config()
        self._plot_point(grid, config['unloading_point'], 'U', SCALE)
        self._plot_point(grid, config['parking_area'], 'P', SCALE)
        for lp in config['loading_points']:
            self._plot_point(grid, lp, 'L', SCALE)
            
        # 标注车辆位置（根据新状态系统）
        for vehicle in self.vehicles.values():
            symbol = {
                VehicleState.UNLOADING: '▼', 
                VehicleState.PREPARING: '▲',
                TransportStage.TRANSPORTING: '▶',
                TransportStage.APPROACHING: '◀',
                VehicleState.IDLE: '●'
            }.get(vehicle.status if vehicle.status != VehicleState.EN_ROUTE else vehicle.transport_stage, '?')
            self._plot_point(grid, vehicle.current_location, symbol, SCALE)
            
        # 打印地图
        print("\n当前地图布局：")
        for row in grid:
            print(' '.join(row))
    def print_system_status(self):
        """实时系统状态监控"""
        status = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'active_vehicles': len([v for v in self.vehicles.values() 
                                   if v.status != VehicleState.IDLE]),
            'queued_tasks': len(self.task_queue),
            'active_tasks': len(self.active_tasks),
            'charging_queue': len(self.scheduler.charging_queue)  # 修正属性引用
        }
        print("\n系统状态:")
        for k, v in status.items():
            print(f"{k:15}: {v}")
    def add_task(self, task: TransportTask):
        """线程安全的任务添加方法"""
        with self.lock:
            self.task_queue.append(task)  # 保持使用append方法
            logging.info(f"已接收任务 {task.task_id} ({task.task_type})")
    def _detect_conflicts(self):
        """冲突检测方法（补充实现）"""
        all_paths = {vid: v.current_path for v in self.vehicles.values() if v.current_path}
        resolved_paths = self.cbs.resolve_conflicts(all_paths)
        
        with self.vehicle_lock:
            for vid, path in resolved_paths.items():
                if path and vid in self.vehicles:
                    try:
                        self.vehicles[vid].assign_path(path)
                    except ValueError as e:
                        logging.error(f"车辆 {vid} 路径分配失败: {str(e)}")
    def _plot_point(self, grid, point, symbol, scale):
        """坐标转换方法（新增）"""
        MAP_CENTER = len(grid) // 2
        try:
            # 将实际坐标转换为地图索引
            x = int(point[0]/scale) + MAP_CENTER
            y = int(point[1]/scale) + MAP_CENTER
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                grid[x][y] = symbol
        except (TypeError, IndexError) as e:
            logging.warning(f"坐标绘制异常: {str(e)}")
    def dispatch_vehicle_to(self, vehicle_id: str, destination: Tuple[float, float]):
        """直接调度指定车辆到目标位置（新增方法）"""
        with self.vehicle_lock:
            if vehicle_id not in self.vehicles:
                raise ValueError(f"车辆 {vehicle_id} 不存在")
                
            vehicle = self.vehicles[vehicle_id]
            if vehicle.status not in (VehicleState.IDLE, VehicleState.PREPARING):
                raise ValueError(f"车辆 {vehicle_id} 当前状态无法接受新任务")

            # 创建临时运输任务
            manual_task = TransportTask(
                task_id=f"MANUAL-{datetime.now().timestamp()}",
                start_point=vehicle.current_location,
                end_point=destination,
                task_type="manual",
                priority=0  # 最高优先级
            )
            
            # 直接分配任务给车辆
            vehicle.assign_task(manual_task)
            self.active_tasks[manual_task.task_id] = manual_task
            logging.info(f"已手动调度车辆 {vehicle_id} 前往 {destination}")
        
    def process_task_queue(self):
        """修复后的任务激活方法"""
        with self.lock:
            while self.task_queue:
                task = self.task_queue.popleft()
                # 添加任务状态初始化
                task.is_completed = False
                task.assigned_to = None
                self.active_tasks[task.task_id] = task
                logging.debug(f"激活任务 {task.task_id}")
if __name__ == "__main__":
    from algorithm.path_planner import HybridPathPlanner
    from algorithm.map_service import MapService
    from models.vehicle import MiningVehicle
    from models.task import TransportTask
    import logging
    import time

    # 初始化日志和基础服务
    logging.basicConfig(level=logging.INFO)
    map_service = MapService()
    planner = HybridPathPlanner(map_service)
    start_time = time.time()  # 新增时间记录
    # 创建调度系统实例
    dispatch = DispatchSystem(planner, map_service)

    # 初始化测试车辆（不同初始状态）
    test_vehicles = [
        MiningVehicle(
            vehicle_id=1,
            map_service=map_service,  # 新增参数
            config={
                'current_location': (200, 200),
                'max_capacity': 50,
                'max_speed': 8,
                'base_location': (200, 200),
                'status': VehicleState.IDLE
            }
        ),
        MiningVehicle(
            vehicle_id=2,
            map_service=map_service,
            config={
                'current_location': (-100, 50),
                'max_capacity': 50,
                'current_load': 40,
                'max_speed': 6,
                'base_location': (-100, 50)
            }
        ),
        MiningVehicle(
            vehicle_id=3,
            map_service=map_service,
            config={
                'current_location': (150, -80),
                'max_capacity': 50,
                'max_speed': 10,
                'base_location': (150, -80)
            }
        )
    ]


    # 注册车辆到调度系统
    for v in test_vehicles:
        dispatch.vehicles[v.vehicle_id] = v

    # 配置测试任务（包含路径点）
    test_tasks = [
        TransportTask(
            task_id="Load-01",
            start_point=(-100.0, 50.0),
            end_point=(0.0, -100.0),
            task_type="loading",
            waypoints=[(-50, 0), (0, -50)],
            priority=1
        ),
        TransportTask(
            task_id="Unload-01",
            start_point=(0.0, -100.0),
            end_point=(200.0, 200.0),
            task_type="unloading",
            waypoints=[(50, -50), (100, 0)],
            priority=2
        )
    ]

    # 模拟调度循环
    print("=== 系统初始化 ===")
    dispatch.print_system_status()

    for cycle in range(1, 4):
        # 使用线程安全的任务添加方式
        try:
            for task in test_tasks:
                dispatch.add_task(task)
            
            # 执行调度逻辑后打印状态
            dispatch.scheduling_cycle()
            dispatch.print_system_status()
            dispatch.print_ascii_map()  # 更名后的方法
            elapsed = time.time() - start_time
            
            # 显示状态
            print(f"调度耗时: {elapsed:.2f}秒")
            dispatch.print_system_status()
            
            # 模拟时间推进
            time.sleep(2)

            # 输出最终状态
            print("\n=== 最终状态报告 ===")
            print("激活任务:", list(dispatch.active_tasks.keys()))
            print("完成任务:", list(dispatch.completed_tasks.keys()))
        except Exception as e:
            logging.error(f"调度周期{cycle}异常: {str(e)}")
            continue