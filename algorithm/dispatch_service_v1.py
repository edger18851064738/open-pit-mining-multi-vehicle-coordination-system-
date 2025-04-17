"""
多车协同调度系统核心模块 v4.1
实现功能：
1. 基于时间窗的时空预约表
2. 装卸点优先级调度
3. 充电调度策略
4. CBS冲突避免算法
5. QMIX强化学习(可选)
"""
from __future__ import annotations
import heapq
import threading
import os
from config.paths import PROJECT_ROOT
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from algorithm.path_planner import HybridPathPlanner
from algorithm.map_service import MapService
from utils.geo_tools import GeoUtils
import logging
import random
import networkx as nx
import osmnx as ox
class Scheduler:
    """任务调度器"""
    def __init__(self):
        self.parking_area = (200, 200)
        self.loading_points = [(-100, 50), (0, 150), (100, 50)]
        self.unloading_point = (0, -100)
        
        # 预定义路线
        self.predefined_routes = {
            'parking_to_load1': [(200, 200), (150, 150), (100, 100), (-100, 50)],
            'parking_to_load2': [(200, 200), (150, 150), (100, 100), (0, 150)],
            'parking_to_load3': [(200, 200), (150, 150), (100, 100), (100, 50)],
            'load_to_unload1': [(-100, 50), (-50, 0), (0, -100)],
            'load_to_unload2': [(0, 150), (0, 50), (0, -100)],
            'load_to_unload3': [(100, 50), (50, 0), (0, -100)],
            'unload_to_parking': [(0, -100), (100, 0), (200, 200)]
        }
        
    def apply_scheduling_policy(self, vehicles, tasks):
        """应用调度策略"""
        assignments = {}
        for task in tasks:
            for vehicle in vehicles:
                if vehicle.status == VehicleState.IDLE:
                    # 根据任务类型选择预定义路线
                    if task.task_type == "loading":
                        route_key = f"parking_to_load{random.randint(1,3)}"
                    else:
                        route_key = "unload_to_parking"
                    
                    # 分配任务和路线
                    assignments[vehicle.vehicle_id] = {
                        'task': task,
                        'route': self.predefined_routes[route_key]
                    }
                    break
        return assignments

class DispatchService:
    """智能调度服务 v4.1（简化优化版）"""
    
    def __init__(self, planner: HybridPathPlanner, map_service: MapService, use_qmix: bool = False):
        self.planner = planner
        self.map_service = map_service
        self.vehicles: Dict[int, MiningVehicle] = {}
        self.task_queue = []
        self.vehicle_queue = deque()  # 新增车辆队列
        self.assigned_tasks: Dict[str, TransportTask] = {}
        self.max_retries = 3
        self.use_qmix = use_qmix
        
        # 初始化调度器
        self.scheduler = Scheduler()
        
        # QMIX相关初始化
        if self.use_qmix:
            self._init_qmix_network()
            self.episode_buffer = []
            
        # 并发控制
        self.lock = threading.Lock()
        self.reservation_lock = threading.Lock()
        self.reservation_table = defaultdict(set)
        self.failed_tasks = {}
        self.active_tasks = {}  # 存储当前活动的任务
        self.completed_tasks = {}  # 存储已完成的任务
        
    def _segments_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """检测两条线段是否相交
        
        参数:
            p1, p2: 第一条线段的起点和终点
            p3, p4: 第二条线段的起点和终点
            
        返回:
            bool: 如果线段相交返回True，否则返回False
        """
        def ccw(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
            
        # 检查线段是否相交
        d1 = ccw(p1, p2, p3)
        d2 = ccw(p1, p2, p4)
        d3 = ccw(p3, p4, p1)
        d4 = ccw(p3, p4, p2)
        
        # 线段相交的条件
        if ((d1 * d2 < 0) and (d3 * d4 < 0)):
            return True
            
        # 检查共线情况
        if d1 == 0 and self._on_segment(p1, p2, p3):
            return True
        if d2 == 0 and self._on_segment(p1, p2, p4):
            return True
        if d3 == 0 and self._on_segment(p3, p4, p1):
            return True
        if d4 == 0 and self._on_segment(p3, p4, p2):
            return True
            
        return False
        
    def _on_segment(self, p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
        """检查点r是否在线段pq上"""
        if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
            return True
        return False
        
    def _detect_conflicts(self):
        """增强版冲突检测方法，考虑时间窗和路径段相交"""
        all_paths = {vid: v.current_path for v in self.vehicles.values() if v.current_path}
        
        # 检查路径段相交和时间窗重叠
        for vid1, path1 in all_paths.items():
            for vid2, path2 in all_paths.items():
                if vid1 >= vid2:
                    continue
                
                # 检查路径段相交
                for i in range(len(path1)-1):
                    for j in range(len(path2)-1):
                        if self._segments_intersect(path1[i], path1[i+1], path2[j], path2[j+1]):
                            logging.warning(f"检测到车辆 {vid1} 和 {vid2} 路径段相交冲突")
                            return True
                
                # 检查时间窗重叠
                if vid1 in self.reservation_table and vid2 in self.reservation_table:
                    for seg1 in self.reservation_table[vid1]:
                        for seg2 in self.reservation_table[vid2]:
                            if seg1 == seg2:
                                logging.warning(f"检测到车辆 {vid1} 和 {vid2} 时间窗冲突")
                                return True
        
        return False
        
    def generate_new_task_for_vehicle(self, vehicle: MiningVehicle):
        """为车辆生成新任务
        
        参数:
            vehicle: 需要分配任务的车辆对象
        """
        if not isinstance(vehicle, MiningVehicle):
            logging.error(f"无效的车辆类型: {type(vehicle)}")
            return
            
        # 验证任务类型
        if not hasattr(vehicle, 'current_task') or not isinstance(vehicle.current_task, (TransportTask, type(None))):
            logging.error(f"无效的任务类型: {type(vehicle.current_task) if hasattr(vehicle, 'current_task') else 'None'}")
            return
            
        # 生成任务类型
        task_type = random.choice(["loading", "unloading"])
        loading_points = [(-100, 50), (0, 150), (100, 50)]
        unloading_point = (0, -100)
        
        # 设置起点和终点
        start = random.choice(loading_points) if task_type == "loading" else unloading_point
        end = unloading_point if task_type == "loading" else random.choice(loading_points)
        
        # 创建新任务
        task_id = f"TASK-{len(self.task_queue) + 1}"
        task = TransportTask(
            task_id=task_id,
            start_point=start,
            end_point=end,
            task_type=task_type,
            priority=random.randint(1, 3)
        )
        
        # 添加任务到队列
        self.add_task(task)
        logging.info(f"为车辆 {vehicle.vehicle_id} 生成新任务 {task_id} ({task_type})")
        
        # 分配任务给车辆
        vehicle.current_task = task
        vehicle.status = VehicleState.EN_ROUTE
        
    def update(self, vehicles: List[MiningVehicle], tasks: List[TransportTask]):
        """更新车辆状态和任务分配
        
        参数:
            vehicles: 车辆列表
            tasks: 任务列表
        """
        with self.lock:
            # 将空闲车辆加入队列
            for vehicle in vehicles:
                if vehicle.state == VehicleState.IDLE and vehicle not in self.vehicle_queue:
                    self.vehicle_queue.append(vehicle)
                    
            self._update_vehicle_states()
            
            # 按队列顺序分配任务
            while self.vehicle_queue and tasks:
                vehicle = self.vehicle_queue.popleft()
                if vehicle.state == VehicleState.IDLE:
                    self._assign_task_to_vehicle(vehicle, tasks.pop(0))
            
            self._dispatch_tasks()
            
    def scheduling_cycle(self):
        """调度主循环（每30秒触发）"""
        with self.lock:
            self._update_vehicle_states()
            self._dispatch_tasks()  # 触发任务分配
            self._detect_conflicts()
            
    def _dispatch_tasks(self):
        """任务分配逻辑"""
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
        
    def process_task_queue(self):
        """处理任务队列，将任务从队列移动到活动任务列表"""
        with self.lock:
            while self.task_queue:
                task = self.task_queue.pop(0)
                if isinstance(task, TransportTask):
                    task.is_completed = False
                    task.assigned_to = None
                    self.active_tasks[task.task_id] = task
                    logging.debug(f"激活任务 {task.task_id}")
                else:
                    logging.error(f"无效的任务类型: {type(task)}")
        
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
                if completed_task.task_id in self.active_tasks:
                    del self.active_tasks[completed_task.task_id]
                else:
                    logging.warning(f"尝试删除不存在的任务ID: {completed_task.task_id}")
                vehicle.current_task = None

    def _init_qmix_network(self):
        """初始化QMIX网络结构"""
        import torch
        import torch.nn as nn
        
        # 定义智能体网络
        class AgentNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                return self.fc3(x)
                
        # 定义混合网络
        class MixingNetwork(nn.Module):
            def __init__(self, n_agents, state_dim, mixing_hidden_dim):
                super().__init__()
                self.n_agents = n_agents
                self.state_dim = state_dim
                
                # 超网络生成混合权重
                self.hyper_w1 = nn.Linear(state_dim, n_agents * mixing_hidden_dim)
                self.hyper_b1 = nn.Linear(state_dim, mixing_hidden_dim)
                self.hyper_w2 = nn.Linear(state_dim, mixing_hidden_dim)
                self.hyper_b2 = nn.Linear(state_dim, 1)
                
            def forward(self, agent_qs, states):
                # 第一层
                w1 = torch.abs(self.hyper_w1(states))
                b1 = self.hyper_b1(states)
                w1 = w1.view(-1, self.n_agents, self.mixing_hidden_dim)
                b1 = b1.view(-1, 1, self.mixing_hidden_dim)
                
                # 第二层
                w2 = torch.abs(self.hyper_w2(states))
                b2 = self.hyper_b2(states)
                w2 = w2.view(-1, self.mixing_hidden_dim, 1)
                b2 = b2.view(-1, 1, 1)
                
                # 前向传播
                hidden = torch.bmm(agent_qs, w1) + b1
                hidden = torch.relu(hidden)
                y = torch.bmm(hidden, w2) + b2
                
                return y
                
        # 初始化网络
        self.agent_network = AgentNetwork(input_dim=10, hidden_dim=64, output_dim=5)
        self.mixing_network = MixingNetwork(n_agents=7, state_dim=20, mixing_hidden_dim=32)
        self.target_agent_network = AgentNetwork(input_dim=10, hidden_dim=64, output_dim=5)
        self.target_mixing_network = MixingNetwork(n_agents=7, state_dim=20, mixing_hidden_dim=32)
        
        # 优化器
        self.optimizer = torch.optim.RMSprop(params=list(self.agent_network.parameters()) + 
                                           list(self.mixing_network.parameters()), lr=0.0005)

    def get_vehicles(self) -> List[MiningVehicle]:
        """获取当前所有车辆列表"""
        with self.lock:
            return self.vehicles.copy() if hasattr(self.vehicles, 'copy') else list(self.vehicles)
            
    def get_tasks(self) -> List[TransportTask]:
        """获取当前所有任务列表"""
        with self.lock:
            return list(self.task_queue) + list(self.assigned_tasks.values())
            
    def register_vehicles(self, vehicles: List[MiningVehicle]):
        """注册车辆并初始化监控"""
        with self.lock:
            # 新增运行时属性检查
            if not all(isinstance(v.map_service, MapService) for v in vehicles):
                raise TypeError("车辆地图服务类型错误")
            
            # 新增车辆容量校验
            if any(v.max_capacity <= 0 for v in vehicles):
                raise ValueError("车辆最大载重必须大于0")
            if not all(hasattr(v, 'map_service') for v in vehicles):
                raise ValueError("车辆实例缺少map_service属性")
            if not all(v.vehicle_id for v in vehicles):
                raise ValueError("车辆ID不能为空")
                
            # 确保正确存储车辆对象
            self.vehicles = vehicles.copy() if hasattr(vehicles, 'copy') else list(vehicles)
            logging.info(f"注册车辆列表: {[v.vehicle_id for v in self.vehicles]}")
            self._init_vehicle_monitoring()
            return True

    def assign_next_task(self) -> Optional[TransportTask]:
        """动态优先级任务分配（带冲突检测和路径规划）"""
        with self.lock:
            if not self.task_queue:
                return None

            _, _, _, task = heapq.heappop(self.task_queue)
            vehicle = self._select_optimal_vehicle(task)
            
            if vehicle:
                # 在分配任务前先规划路径
                try:
                    path = self.planner.optimize_path(
                        vehicle.current_location,
                        task.start_point,
                        vehicle
                    )
                    # 检查路径是否有效
                    if len(path) > 0 and self._safe_assign_task(task, vehicle, path):
                        self.assigned_tasks[task.task_id] = task
                        task.assign_to_vehicle(vehicle, path)
                        task.status = 'assigned'
                        return task
                except Exception as e:
                    logging.warning(f"路径规划失败: {str(e)}")
            
            self._handle_failed_assignment(task)
            return task if self.task_queue else None
    def _handle_task_completion(self, vehicle: MiningVehicle):
        """任务完成处理"""
        if vehicle.current_task.task_type == "unloading":
            # 卸载完成后安排充电
            vehicle.should_charge = True
            self._route_to_parking(vehicle)
        elif vehicle.current_task.task_type == "loading":
            # 装载完成后安排运输
            unloading_path = self.planner.plan_path(
                vehicle.current_location,
                self.scheduler.unloading_point,
                vehicle
            )
            vehicle.assign_path(unloading_path)

    def _select_optimal_vehicle(self, task: TransportTask) -> Optional[MiningVehicle]:
        """基于Q-learning的车辆选择算法"""
        candidates = []
        for v in self.vehicles:
            try:
                # 统一坐标格式处理
                current_loc = (
                    v.current_location[0] + self.map_service.config.getfloat('MAP', 'virtual_origin_x'),
                    v.current_location[1] + self.map_service.config.getfloat('MAP', 'virtual_origin_y')
                ) if self.map_service.config.get('MAP', 'data_type') == 'virtual' else v.current_location
                
                valid_conditions = [
                    v.status == 'idle',
                    v.current_load + task.total_weight <= v.max_capacity,
                    v.max_speed >= 1.0,
                    self.map_service.validate_coordinates(current_loc)
                ]
                
                if all(valid_conditions):
                    # 计算Q值
                    q_value = self._calculate_q_value(v, task)
                    candidates.append((v, q_value))
                    logging.debug(f"候选车辆 {v.vehicle_id} | Q值:{q_value:.2f} | 位置:{current_loc}")
            except Exception as e:
                logging.error(f"车辆状态异常 [{v.vehicle_id}]: {str(e)}")
        
        # 选择Q值最高的车辆
        return max(candidates, key=lambda x: x[1])[0] if candidates else None
        
    def _calculate_q_value(self, vehicle: MiningVehicle, task: TransportTask) -> float:
        """计算车辆-任务匹配的Q值"""
        if not self.use_qmix:
            # 基线规则方法
            distance = GeoUtils.haversine(vehicle.current_location, task.start_point)
            capacity_ratio = vehicle.remaining_capacity / task.total_weight
            battery_level = vehicle.get_battery_status() / 100
            
            path = self.planner.plan_path(vehicle.current_location, task.start_point, vehicle)
            path_length = len(path) if path else 1000
            conflict_risk = self._calculate_conflict_risk(vehicle, task)
            
            return (
                0.3 * (1 / (distance + 1)) + 
                0.2 * capacity_ratio + 
                0.2 * battery_level + 
                0.2 * (1 / (path_length + 1)) + 
                0.1 * (1 - conflict_risk)
            )
        else:
            # QMIX神经网络方法
            state = self._get_agent_state(vehicle, task)
            with torch.no_grad():
                q_values = self.agent_network(torch.FloatTensor(state))
            return q_values.max().item()
            
    
        
    def _update_target_network(self):
        """更新目标网络参数"""
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())
        
    def _get_agent_state(self, vehicle: MiningVehicle, task: TransportTask) -> List[float]:
        """获取智能体状态向量"""
        distance = GeoUtils.haversine(vehicle.current_location, task.start_point)
        capacity_ratio = vehicle.remaining_capacity / task.total_weight
        battery_level = vehicle.get_battery_status() / 100
        path_length = len(self.planner.plan_path(vehicle.current_location, task.start_point, vehicle)) if vehicle.current_path else 0
        conflict_risk = self._calculate_conflict_risk(vehicle, task)
        
        return [
            distance / 1000,  # 归一化距离
            capacity_ratio,
            battery_level,
            1 / (path_length + 1),
            1 - conflict_risk,
            vehicle.current_load / vehicle.max_capacity,
            vehicle.current_speed / vehicle.max_speed,
            task.priority / 10,
            len(self.task_queue) / 20,
            len(self.assigned_tasks) / 7
        ]
        
    def _calculate_reward(self, vehicle: MiningVehicle, task: TransportTask) -> float:
        """计算即时奖励"""
        # 任务完成奖励
        completion_reward = 10.0 if task.is_completed else 0.0
        
        # 效率奖励
        distance = GeoUtils.haversine(vehicle.current_location, task.start_point)
        efficiency_reward = 1 / (distance + 1)
        
        # 冲突惩罚
        conflict_penalty = -self._calculate_conflict_risk(vehicle, task)
        
        # 电池惩罚
        battery_penalty = -0.1 * (1 - vehicle.get_battery_status() / 100)
        
        return completion_reward + efficiency_reward + conflict_penalty + battery_penalty

    def _calculate_match_score(self, vehicle: MiningVehicle, task: TransportTask) -> float:
        """车辆-任务匹配度算法"""
        # 类型兼容处理
        vehicle_coord = (vehicle.current_location.x, vehicle.current_location.y) \
            if hasattr(vehicle.current_location, 'x') else vehicle.current_location
        task_coord = task.start_point if isinstance(task.start_point, tuple) else \
            (task.start_point.x, task.start_point.y)
        
        distance_score = 1 / (GeoUtils.haversine(vehicle_coord, task_coord) + 1)
        capacity_score = vehicle.max_capacity / task.total_weight
        conflict_score = self._calculate_conflict_risk(vehicle, task)
        return 0.4 * distance_score + 0.3 * capacity_score + 0.3 * conflict_score

    def _calculate_conflict_risk(self, vehicle: MiningVehicle, task: TransportTask) -> float:
        """冲突风险预测算法（新增重试机制）"""
        for retry in range(3):
            try:
                # 确保传递坐标元组（兼容Node类型）
                vehicle_coord = (vehicle.current_location.x, vehicle.current_location.y) \
                    if hasattr(vehicle.current_location, 'x') else vehicle.current_location
                task_coord = task.end_point if isinstance(task.end_point, tuple) else \
                    (task.end_point.x, task.end_point.y)
                
                test_path = self.planner.optimize_path(
                    start=vehicle_coord,  # 传递坐标元组
                    end=task_coord,       # 传递坐标元组
                    vehicle=vehicle
                )
                if test_path:
                    return 1 / (self._detect_path_conflicts(test_path) + 1)
            except PathOptimizationError as e:
                logging.debug(f"路径规划重试中 [{retry+1}/3] | 原因: {str(e)}")
                continue
        return 0

    def _safe_assign_task(self, task: TransportTask, vehicle: MiningVehicle) -> bool:
        try:
            est_departure = datetime.now()
            
            # 新增调试日志
            logging.debug(f"开始分配任务 {task.task_id} → 车辆 {vehicle.vehicle_id}")
            logging.debug(f"车辆当前位置: {vehicle.current_location} 负载: {vehicle.current_load}/{vehicle.max_capacity}")

            # 修正坐标转换逻辑
            vehicle_coord = (vehicle.current_location.x, vehicle.current_location.y) \
                if hasattr(vehicle.current_location, 'x') else vehicle.current_location
            task_end_coord = task.end_point if isinstance(task.end_point, tuple) else \
                (task.end_point.x, task.end_point.y)
    
            # 增加可达性预检查
            if not self.map_service.is_reachable(vehicle_coord):
                raise PathOptimizationError(f"车辆当前位置不可达: {vehicle_coord}")
            if not self.map_service.is_reachable(task_end_coord):
                raise PathOptimizationError(f"任务终点不可达: {task_end_coord}")
    
            # 修正地形硬度比较逻辑
            end_hardness = self.map_service.get_terrain_hardness(task_end_coord)
            if end_hardness < vehicle.min_hardness:  # 修改比较方向
                raise PathOptimizationError(f"终点地形硬度过低: {end_hardness:.1f} < {vehicle.min_hardness}")

            optimized_path = self.planner.optimize_path(
                start=vehicle_coord,
                end=task_end_coord,
                vehicle=vehicle
            )
            
            # 新增路径详情日志
            logging.info(f"路径规划成功 | 任务:{task.task_id} 长度:{len(optimized_path)} 节点:{optimized_path[:3]}...{optimized_path[-3:]}")
            logging.debug(f"完整路径: {optimized_path}")

            # 合并重复的路径检查逻辑
            if not optimized_path or len(optimized_path) < 2:
                raise PathOptimizationError(f"无效路径长度: {len(optimized_path)}")
                
            # 修复时间有效性检查（原逻辑有重复异常抛出）
            planning_time = (datetime.now() - est_departure).total_seconds()
            if planning_time > 10:
                raise PathOptimizationError(f"规划超时: {planning_time:.2f}s")
    
            # 统一路径验证（保持父类方法调用）
            if not self.map_service.validate_path(optimized_path):
                invalid_points = [p for p in optimized_path if not self.map_service.is_reachable(p)]
                raise PathOptimizationError(f"路径包含{len(invalid_points)}个无效点")

            # 添加时间有效性检查
            if (datetime.now() - est_departure).seconds > 10:
                raise PathOptimizationError("路径规划超时")
                raise PathOptimizationError("空路径")
            valid_nodes = self.map_service.road_network.nodes
            if any(point not in valid_nodes for point in optimized_path):
                raise PathOptimizationError("路径包含无效节点")
            
            if not optimized_path or len(optimized_path) < 2:
                raise PathOptimizationError(f"无效路径长度: {len(optimized_path)}")
            if not self.map_service.validate_path(optimized_path):
                raise PathOptimizationError("路径包含障碍节点")

            self._update_reservation_table(optimized_path, vehicle.vehicle_id, est_departure)
            
            # 新增分配成功日志（添加负载变化信息）
            logging.info(f"任务分配成功 | 车辆:{vehicle.vehicle_id} "
                       f"新负载:{vehicle.current_load + task.total_weight}/{vehicle.max_capacity}kg "
                       f"耗时:{planning_time:.2f}s")
            
            vehicle.register_task_assignment(task)
            task.assign_to_vehicle(vehicle, optimized_path)
            return True
            
        except (PathOptimizationError, TaskValidationError) as e:
            # 增强错误日志（添加车辆状态）
            logging.error(f"分配失败 | 车辆:{vehicle.vehicle_id} 状态:{vehicle.status} "
                        f"电池:{vehicle.get_battery_status()}% 错误:{str(e)}")
            return False

    def register_vehicles(self, vehicles: List[MiningVehicle]):
        """注册车辆并初始化监控"""
        with self.lock:
            # 新增车辆注册详情日志
            logging.info(f"注册车辆列表: {[v.vehicle_id for v in vehicles]}")
            for v in vehicles:
                logging.debug(f"车辆配置详情 | ID:{v.vehicle_id} "
                            f"最大载重:{v.max_capacity} 最低硬度:{v.min_hardness}")
            
            self.vehicles = vehicles.copy()

    def monitor_system_health(self) -> Dict:
        """实时系统健康监测"""
        return {
            'timestamp': datetime.now().isoformat(),
            'vehicles': self._get_vehicle_status(),
            'tasks': {
                'pending': len(self.task_queue),
                'active': len(self.assigned_tasks),
                'failed': len(self.failed_tasks)
            },
            'conflicts': sum(self._detect_path_conflicts(t.assigned_path) 
                           for t in self.assigned_tasks.values())
        }

    def _get_vehicle_status(self) -> Dict:
        """车辆状态跟踪"""
        return {
            v.vehicle_id: {
                'position': v.current_location,
                'speed': v.current_speed,
                'load': f"{v.current_load}/{v.max_capacity}",
                'status': v.status,
                'battery': v.get_battery_status(),
                'sensors': v.get_sensor_readings()
            } for v in self.vehicles
        }

    def _detect_path_conflicts(self, path: List[Tuple]) -> int:
        """路径冲突检测"""
        conflicts = 0
        with self.reservation_lock:
            for v in self.vehicles:
                if v.current_path:
                    # 检查路径段是否交叉
                    for i in range(len(path)-1):
                        for j in range(len(v.current_path)-1):
                            if self._segments_intersect(path[i], path[i+1], 
                                                     v.current_path[j], v.current_path[j+1]):
                                conflicts += 1
                                # 添加调试日志
                                logging.debug(f"检测到路径冲突 | 车辆:{v.vehicle_id} "
                                            f"路径1:{path[i]}->{path[i+1]} "
                                            f"路径2:{v.current_path[j]}->{v.current_path[j+1]}")
        return conflicts

    def _handle_failed_assignment(self, task: TransportTask):
        """失败任务处理"""
        self.failed_tasks[task.task_id] = self.failed_tasks.get(task.task_id, 0) + 1
        if self.failed_tasks[task.task_id] <= self.max_retries:
            self._requeue_task(task)
        else:
            logging.error(f"任务永久失败 [{task.task_id}]")
            self._cleanup_failed_task(task)

    def _requeue_task(self, task: TransportTask):
        """智能重排队"""
        task.priority += 1
        task.deadline += timedelta(minutes=15)
        heapq.heappush(self.task_queue, (-task.priority, datetime.now(), task.task_id, task))

    def add_task(self, task: TransportTask):
        """线程安全任务添加"""
        with self.lock:
            heapq.heappush(self.task_queue, (-task.priority, datetime.now(), task.task_id, task))

    def _init_vehicle_monitoring(self):
        """车辆状态监控初始化"""
        for v in self.vehicles:
            v.register_status_callback(self._handle_vehicle_status_change)

    def _handle_vehicle_status_change(self, vehicle: MiningVehicle):
        """车辆状态变更响应"""
        if vehicle.status == 'emergency':
            affected_task = next((t for t in self.assigned_tasks.values() 
                                if t.assigned_vehicle == vehicle), None)
            if affected_task:
                self._handle_emergency(affected_task, vehicle)

    def _handle_emergency(self, task: TransportTask, vehicle: MiningVehicle):
        """紧急情况处理"""
        logging.critical(f"车辆紧急状态 [{vehicle.vehicle_id}]")
        self._cleanup_failed_task(task)
        vehicle.perform_emergency_stop()
        
        if task.retry_count < task.max_retries:
            self.add_task(task)
            logging.info(f"任务重新排队 [{task.task_id}]")

    def _cleanup_failed_task(self, task: TransportTask):
        """清理失败任务"""
        if task.task_id in self.assigned_tasks:
            del self.assigned_tasks[task.task_id]
        if task.assigned_vehicle:
            task.assigned_vehicle.abort_current_task()

    def _find_candidate_vehicles(self, task: TransportTask) -> List[MiningVehicle]:
        """增强型候选车辆筛选（添加调试日志和容错机制）"""
        candidates = []
        for vehicle in self.available_vehicles:
            try:
                # 添加路径可达性验证
                route = self.planner.plan_route(
                    vehicle.current_location,
                    task.start_point,
                    vehicle_type=vehicle.vehicle_type
                )
                if route['error'] is None:
                    candidates.append(vehicle)
                    logging.debug(f"候选车辆 {vehicle.vehicle_id} | 当前位置: {vehicle.current_location} | 剩余容量: {vehicle.remaining_capacity}")
            except Exception as e:
                logging.warning(f"车辆 {vehicle.vehicle_id} 验证异常: {str(e)}")
        
        # 新增空候选处理机制
        if not candidates:
            logging.warning("无候选车辆，尝试放宽筛选条件...")
            return self._fallback_candidate_search(task)
            
        return candidates

    def _fallback_candidate_search(self, task: TransportTask) -> List[MiningVehicle]:
        """回退机制：当无候选车辆时"""
        # 1. 检查车辆状态是否误判
        available = [v for v in self.vehicles 
                   if v.status == 'idle' and v.remaining_capacity >= task.total_weight]
        
        # 2. 放宽路径要求
        if available:
            logging.info("尝试放宽路径约束寻找候选车辆")
            try:
                return [v for v in available 
                      if self.planner.validate_rough_path(v.current_location, task.start_point)]
            except:
                return available[:1]  # 至少返回一个
        
        # 3. 返回容量最大的三台车辆
        return sorted(self.vehicles, 
                    key=lambda x: x.remaining_capacity, 
                    reverse=True)[:3]

if __name__ == "__main__":
    """调度系统集成测试模块"""
    logging.basicConfig(level=logging.DEBUG,
                      format='%(asctime)s | %(levelname)-8s | %(message)s',
                      datefmt='%H:%M:%S')
    
    # 模拟地图服务
    # 修改测试地图服务部分
    # 在测试代码部分修改地图服务初始化
    # 在测试地图服务类中修复方法定义位置问题
    class TestMapService(MapService):
        def __init__(self):
            super().__init__()
            self.road_network = nx.grid_2d_graph(10, 10)
            self._obstacle_nodes = {(3,3), (6,6), (8,2)}
            
            # 节点属性初始化
            for node in self.road_network.nodes():
                self.road_network.nodes[node].update({
                    'x': node[0], 
                    'y': node[1],
                    'hardness': 5.0,
                    'grade': 2.0
                })
            
            # 添加地形硬度获取方法（实现父类抽象方法）
        def get_terrain_hardness(self, point: tuple) -> float:
            if point not in self.road_network.nodes:
                raise ValueError(f"无效坐标点: {point}")
            return self.road_network.nodes[point]['hardness']

        def is_reachable(self, point: tuple) -> bool:
            """检查节点是否可达（重写父类方法）"""
            return (
                point in self.road_network.nodes 
                and point not in self._obstacle_nodes
            )
    
        # 修正路径验证方法
        def validate_path(self, path):
            """统一路径验证逻辑（重写父类方法）"""
            return (
                len(path) >= 2 and
                all(self.is_reachable(p) for p in path) and
                nx.has_path(self.road_network, path[0], path[-1])
            )

        # 修正坐标生成函数
        def generate_valid_point(self, max_attempts=100) -> Tuple[int, int]:
            """生成虚拟坐标系下的网格坐标点"""
            if not hasattr(self, 'road_network'):
                raise AttributeError("路网数据未初始化")
                
            valid_nodes = [n for n in self.road_network.nodes if self.is_reachable(n)]
            if not valid_nodes:
                raise PathOptimizationError("地图中没有可用节点")
                
            return random.choice(valid_nodes)  # 直接返回网格坐标节点
        
    # 初始化测试组件
    test_map_service = TestMapService()
    test_planner = HybridPathPlanner(test_map_service)
    test_service = DispatchService(test_planner, test_map_service)
    
    test_vehicles = [
        MiningVehicle(
            vehicle_id=i,
            map_service=test_map_service,
            config={
                'max_capacity': 50,      # 确保键名正确
                'min_hardness': 3.0,     # 新增配置项
                'current_load': 10 + i*5,
                'max_speed': 5.0 - i*0.5,
                'fuel_capacity': 100.0,  # 新增燃油容量配置
                'position': (0, 0),
                'fuel_capacity': 100.0,
                'steering_angle': 30,
                'current_location': (0, 0)  # 新增初始位置
            }
        ) for i in range(1,4)
    ]

    # 车辆模拟补丁
    for v in test_vehicles:
        # 状态回调注册
        def status_callback(vehicle: MiningVehicle):
            def wrapper(func):
                vehicle.status_callback = func
            return wrapper
        v.register_status_callback = status_callback(v)
        
        # 任务接受能力判断
        def can_accept_task(self, task):
            return self.status == 'idle' and self.current_load + task.total_weight <= self.max_capacity
        v.can_accept_task = can_accept_task.__get__(v)
        
        v.status = 'idle'

    # 注册车辆
    test_service.register_vehicles(test_vehicles)

    # 修改测试任务生成逻辑（确保终点可达）
    def generate_valid_point():
        """生成有效坐标点（避开障碍）"""
        valid_nodes = [
            n for n in test_map_service.road_network.nodes()
            if test_map_service.is_reachable(n)
        ]
        return random.choice(valid_nodes)
        point = (
            random.choice([0, 2, 4, 6, 8]),
            random.choice([0, 2, 4, 6, 8])
        )
        if not test_map_service.road_network.has_node(point):
            raise ValueError(f"无效测试坐标: {point}")
        return point
    
    def generate_valid_task():
        """生成有效可达任务"""
        max_retry = 5
        for _ in range(max_retry):
            start = generate_valid_point()
            end = generate_valid_point()
            if nx.has_path(test_map_service.road_network, start, end):
                return TransportTask(
                    task_id=f"TASK-{random.randint(1000,9999)}",
                    start_point=start,
                    end_point=end,
                    waypoints=[],
                    priority=random.choice([1,2,3]),
                    total_weight=random.randint(20,40)
                )
        raise RuntimeError("无法生成有效任务")
    
        # 替换原有任务生成循环
        for i in range(10):
            new_task = generate_valid_task()
            test_service.add_task(new_task)

        # 运行测试
        try:
            print("🚚 开始调度系统压力测试（Ctrl+C停止）")
            cycle = 0
            while True:
                cycle += 1
                assigned = test_service.assign_next_task()
                if assigned:
                    logging.info(f"分配成功 | 任务:{assigned.task_id} → 车辆:{assigned.assigned_vehicle.vehicle_id}")
                
                if cycle % 5 == 0:
                    status = test_service.monitor_system_health()
                    print(f"\n=== 周期 {cycle} ===")
                    print(f"活跃任务: {status['tasks']['active']} | 待处理: {status['tasks']['pending']}")
                    print(f"路径冲突: {status['conflicts']}次")
                    
                threading.Event().wait(0.5)
                
        except KeyboardInterrupt:
            print("\n测试正常终止，最终状态：")
            print(f"成功分配任务: {len(test_service.assigned_tasks)}个")
            print(f"失败任务: {len(test_service.failed_tasks)}个")
