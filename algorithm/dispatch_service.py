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
import traceback
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

# 设置更详细的日志格式
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
                (v for v in vehicle_list if v.state == VehicleState.IDLE),
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
            if vehicle.state == VehicleState.IDLE and not vehicle.current_task:
                for point in self.loading_queues:
                    if self.loading_queues[point]:
                        task = self.loading_queues[point].popleft()
                        assignments[vehicle.vehicle_id] = task
                        break
        
        # 最后处理卸载任务 ▼▼▼
        unloading_vehicles = [v for v in vehicles if v.state == VehicleState.UNLOADING]
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
        # 确保初始化reservation_table属性
        self.reservation_table = {}
        
    def find_conflicts(self, paths: Dict[str, List[Tuple]]) -> List[Tuple]:
        """检测路径冲突"""
        conflicts = []
        path_items = list(paths.items())
        
        # 增加调试日志
        logging.debug(f"检查路径冲突: {len(path_items)}条路径")
        
        for i in range(len(path_items)):
            vid1, path1 = path_items[i]
            for j in range(i+1, len(path_items)):
                vid2, path2 = path_items[j]
                
                # 确保路径长度一致性调试
                min_len = min(len(path1), len(path2))
                if min_len == 0:
                    logging.debug(f"跳过空路径: 车辆{vid1}({len(path1)}点) 或 车辆{vid2}({len(path2)}点)")
                    continue
                
                logging.debug(f"比较车辆{vid1}和{vid2}的路径: {min_len}个点")
                
                for t in range(min_len):
                    point1 = path1[t]
                    point2 = path2[t]
                    if point1 == point2:
                        logging.debug(f"发现冲突: 时间{t}, 位置{point1}, 车辆{vid1}和{vid2}")
                        conflicts.append((t, point1, vid1, vid2))
        
        logging.info(f"共发现{len(conflicts)}个冲突点")
        return conflicts
        
    def _replan_path(self, vehicle_id, pos=None, t=None, max_retries=3):
        """增强路径重规划"""
        # 调试信息
        logging.debug(f"开始为车辆{vehicle_id}重新规划路径: pos={pos}, t={t}")
        
        try:
            # 确保我们有一个vehicle对象
            if not hasattr(self.planner, 'dispatch'):
                logging.error("规划器未初始化dispatch属性")
                return None
                
            if vehicle_id not in self.planner.dispatch.vehicles:
                logging.error(f"找不到车辆ID: {vehicle_id}")
                return None
                
            vehicle = self.planner.dispatch.vehicles[vehicle_id]
            
            # 调试信息
            logging.debug(f"车辆{vehicle_id}当前位置: {vehicle.current_location}")
            logging.debug(f"车辆{vehicle_id}当前任务: {vehicle.current_task.task_id if vehicle.current_task else 'None'}")
            
            for attempt in range(max_retries):
                # 根据冲突位置或任务终点规划新路径
                if vehicle.current_task and hasattr(vehicle.current_task, 'end_point'):
                    end_point = vehicle.current_task.end_point
                    logging.debug(f"使用任务终点: {end_point}")
                elif pos is not None:
                    end_point = pos
                    logging.debug(f"使用冲突位置: {pos}")
                else:
                    logging.warning(f"无法为车辆{vehicle_id}规划路径: 既无任务终点也无冲突位置")
                    return None
                    
                try:
                    logging.debug(f"尝试规划路径: {vehicle.current_location} -> {end_point}")
                    new_path = self.planner.plan_path(
                        vehicle.current_location,
                        end_point,
                        vehicle
                    )
                    
                    if new_path and len(new_path) > 0:
                        logging.debug(f"成功规划路径: {len(new_path)}个点")
                        vehicle.assign_path(new_path)
                        self._update_reservation_table(new_path, vehicle_id)
                        return new_path
                    else:
                        logging.warning(f"路径规划返回空路径")
                except Exception as e:
                    logging.error(f"路径规划异常: {str(e)}")
                    
                logging.debug(f"路径重试 {attempt+1}/{max_retries}")
                import time
                time.sleep(0.5)
            
            logging.warning(f"车辆{vehicle_id}路径规划达到最大重试次数")
            return None
            
        except Exception as e:
            logging.error(f"重规划路径发生异常: {str(e)}")
            traceback.print_exc()
            return None
            
    def _get_vehicle_priority(self, vehicle_id: str) -> int:
        try:
            if not hasattr(self.planner, 'dispatch'):
                logging.error("规划器未初始化dispatch属性")
                return 5  # 返回默认优先级
                
            if vehicle_id not in self.planner.dispatch.vehicles:
                logging.error(f"找不到车辆ID: {vehicle_id}")
                return 5  # 返回默认优先级
                
            vehicle = self.planner.dispatch.vehicles[vehicle_id]
            
            # 修改优先级映射 ▼▼▼
            priorities = {
                VehicleState.UNLOADING: 1,
                VehicleState.PREPARING: 2,
                TransportStage.TRANSPORTING: 3,
                TransportStage.APPROACHING: 4,
                VehicleState.IDLE: 5
            }
            
            current_state = vehicle.state
            transport_stage = vehicle.transport_stage if hasattr(vehicle, 'transport_stage') else None
            
            if current_state == VehicleState.EN_ROUTE and transport_stage:
                priority = priorities.get(transport_stage, 5)
            else:
                priority = priorities.get(current_state, 5)
                
            logging.debug(f"车辆{vehicle_id}优先级: {priority} (状态:{current_state}, 阶段:{transport_stage})")
            return priority
            
        except Exception as e:
            logging.error(f"获取车辆优先级异常: {str(e)}")
            return 5  # 返回默认优先级
            
    def resolve_conflicts(self, paths: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
        """实现基于优先级的冲突解决方案"""
        logging.debug(f"开始解决冲突: 有{len(paths)}条路径")
        
        # 防御性编程: 如果路径为空则直接返回
        if not paths:
            logging.warning("无路径数据，跳过冲突检测")
            return paths
            
        try:
            new_paths = paths.copy()
            
            # 获取所有冲突
            conflicts = self.find_conflicts(paths)
            logging.debug(f"找到{len(conflicts)}个冲突点")
            
            for i, conflict in enumerate(conflicts):
                logging.debug(f"处理冲突 {i+1}/{len(conflicts)}")
                t, pos, vid1, vid2 = conflict
                
                logging.debug(f"冲突详情: 时间={t}, 位置={pos}, 车辆1={vid1}, 车辆2={vid2}")
                
                # 获取车辆优先级
                prio1 = self._get_vehicle_priority(vid1)
                prio2 = self._get_vehicle_priority(vid2)
                
                logging.debug(f"车辆优先级: {vid1}={prio1}, {vid2}={prio2}")
                
                # 重新规划低优先级车辆路径
                if prio1 < prio2:
                    logging.debug(f"重规划车辆{vid1}路径(高优先级)")
                    new_path = self._replan_path(vehicle_id=vid1, pos=pos, t=t)
                    if new_path:
                        new_paths[vid1] = new_path
                    else:
                        logging.warning(f"车辆{vid1}路径重规划失败")
                else:
                    logging.debug(f"重规划车辆{vid2}路径(低优先级)")
                    new_path = self._replan_path(vehicle_id=vid2, pos=pos, t=t)
                    if new_path:
                        new_paths[vid2] = new_path
                    else:
                        logging.warning(f"车辆{vid2}路径重规划失败")
            
            logging.debug(f"冲突解决完成, 返回{len(new_paths)}条路径")
            return new_paths
            
        except Exception as e:
            logging.error(f"解决冲突过程发生异常: {str(e)}")
            traceback.print_exc()
            # 出错时返回原始路径
            return paths
            
    def _update_reservation_table(self, path, vehicle_id):
        """更新路径预约表"""
        try:
            logging.debug(f"更新车辆{vehicle_id}的路径预约表, 路径长度: {len(path)}")
            
            # 为路径的每个点添加预约
            for i in range(len(path) - 1):
                segment = (path[i], path[i+1])
                self.reservation_table[segment] = vehicle_id
                
            logging.debug(f"预约表更新完成, 当前有{len(self.reservation_table)}个预约")
        except Exception as e:
            logging.error(f"更新预约表异常: {str(e)}")

class DispatchSystem:
    """智能调度系统核心"""
    def __init__(self, planner: HybridPathPlanner, map_service: MapService):
        self.planner = planner
        self.map_service = map_service
        self.vehicles: Dict[str, MiningVehicle] = {}
        self.scheduler = TransportScheduler(self._load_config())
        
        # 关键部分: 设置planner.dispatch引用
        self.planner.dispatch = self
        
        self.cbs = ConflictBasedSearch(planner)
        
        # 任务管理
        self.task_queue = deque()  # 原为 []
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # 并发控制
        self.lock = threading.RLock()
        self.vehicle_lock = threading.Lock()
        
        logging.info("调度系统初始化完成")
        
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
        logging.info("开始调度周期")
        try:
            with self.lock:
                logging.debug("状态更新阶段")
                self._update_vehicle_states()
                
                logging.debug("任务分配阶段")
                self._dispatch_tasks()
                
                logging.debug("冲突检测阶段")
                self._detect_conflicts()
                
            logging.info("调度周期成功完成")
        except Exception as e:
            logging.error(f"调度周期执行异常: {str(e)}")
            traceback.print_exc()
            
    def _dispatch_tasks(self):
        """任务分配逻辑（补充方法实现）"""
        if not self.task_queue:
            logging.debug("任务队列为空，跳过任务分配")
            return
            
        logging.debug(f"任务分配开始, 队列中有{len(self.task_queue)}个任务")
        assignments = self.scheduler.apply_scheduling_policy(
            list(self.vehicles.values()),
            self.task_queue
        )
        
        logging.debug(f"调度策略返回{len(assignments)}个任务分配")
        
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
        logging.debug(f"任务分配结束, 队列中剩余{len(self.task_queue)}个任务")
        
    def _update_vehicle_states(self):
        """增强运输阶段追踪"""
        logging.debug(f"开始更新{len(self.vehicles)}辆车的状态")
        
        for vid, vehicle in self.vehicles.items():
            logging.debug(f"车辆{vid}当前状态: {vehicle.state}")
            
            # 状态更新保持不变 ▼▼▼（使用state属性代替status）
            if vehicle.current_task and vehicle.state != VehicleState.EN_ROUTE:
                vehicle.state = VehicleState.EN_ROUTE
                logging.debug(f"车辆{vid}状态更新为EN_ROUTE")
                
            if vehicle.current_location == self.scheduler.parking_area:
                vehicle.state = VehicleState.IDLE
                logging.debug(f"车辆{vid}位于停车场，状态更新为IDLE")
            elif vehicle.current_location in self.scheduler.loading_points:
                vehicle.state = VehicleState.PREPARING
                logging.debug(f"车辆{vid}位于装载点，状态更新为PREPARING")
            elif vehicle.current_location == self.scheduler.unloading_point:
                vehicle.state = VehicleState.UNLOADING
                logging.debug(f"车辆{vid}位于卸载点，状态更新为UNLOADING")
            
            if vehicle.current_task:
                vehicle.state = VehicleState.EN_ROUTE
                if vehicle.current_task.task_type == "loading":
                    vehicle.transport_stage = TransportStage.APPROACHING
                    logging.debug(f"车辆{vid}运输阶段更新为APPROACHING")
                elif vehicle.current_task.task_type == "unloading":
                    vehicle.transport_stage = TransportStage.TRANSPORTING
                    logging.debug(f"车辆{vid}运输阶段更新为TRANSPORTING")
                    
            # 检查任务完成情况
            if vehicle.current_task and hasattr(vehicle, 'path_index') and hasattr(vehicle, 'current_path'):
                if vehicle.path_index >= len(vehicle.current_path)-1:
                    completed_task = vehicle.current_task
                    logging.info(f"车辆{vid}完成任务{completed_task.task_id}")
                    self.completed_tasks[completed_task.task_id] = completed_task
                    if completed_task.task_id in self.active_tasks:
                        del self.active_tasks[completed_task.task_id]
                    vehicle.current_task = None
                    
        logging.debug("车辆状态更新完成")
        
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
            
        # 标注车辆位置（根据新状态系统，使用state属性替代status）
        for vehicle in self.vehicles.values():
            symbol = {
                VehicleState.UNLOADING: '▼', 
                VehicleState.PREPARING: '▲',
                TransportStage.TRANSPORTING: '▶',
                TransportStage.APPROACHING: '◀',
                VehicleState.IDLE: '●'
            }.get(vehicle.state if vehicle.state != VehicleState.EN_ROUTE else vehicle.transport_stage, '?')
            self._plot_point(grid, vehicle.current_location, symbol, SCALE)
            
        # 打印地图
        print("\n当前地图布局：")
        for row in grid:
            print(' '.join(row))
    def print_system_status(self):
        """实时系统状态监控（更新使用state而非status属性）"""
        status = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'active_vehicles': len([v for v in self.vehicles.values() 
                                   if v.state != VehicleState.IDLE]),
            'queued_tasks': len(self.task_queue),
            'active_tasks': len(self.active_tasks),
            'charging_queue': len(self.scheduler.charging_queue)
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
        logging.debug("开始冲突检测")
        
        # 收集所有车辆的路径
        all_paths = {}
        for vid, vehicle in self.vehicles.items():
            if hasattr(vehicle, 'current_path') and vehicle.current_path:
                logging.debug(f"车辆{vid}有路径，长度: {len(vehicle.current_path)}")
                all_paths[str(vid)] = vehicle.current_path  # 确保键是字符串
            else:
                logging.debug(f"车辆{vid}没有路径")
        
        logging.debug(f"收集到{len(all_paths)}条路径")
        
        if not all_paths:
            logging.debug("没有可检测的路径，跳过冲突检测")
            return
            
        try:
            # 冲突解决
            resolved_paths = self.cbs.resolve_conflicts(all_paths)
            logging.debug(f"冲突解决返回{len(resolved_paths)}条路径")
            
            # 更新车辆路径
            with self.vehicle_lock:
                for vid_str, path in resolved_paths.items():
                    try:
                        vid = int(vid_str) if vid_str.isdigit() else vid_str
                        if path and vid in self.vehicles:
                            logging.debug(f"为车辆{vid}分配新路径，长度: {len(path)}")
                            self.vehicles[vid].assign_path(path)
                        else:
                            logging.debug(f"跳过车辆{vid}的路径分配: 路径为空或车辆不存在")
                    except ValueError as e:
                        logging.error(f"车辆 {vid} 路径分配失败: {str(e)}")
                        
            logging.debug("冲突检测完成")
        except Exception as e:
            logging.error(f"冲突检测异常: {str(e)}")
            traceback.print_exc()
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
            if vehicle.state not in (VehicleState.IDLE, VehicleState.PREPARING):
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

# 添加GeoUtils类的简单实现（防止导入问题）
class GeoUtils:
    @staticmethod
    def calculate_distance(point1, point2):
        """计算两点间的距离"""
        from math import sqrt
        return sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

if __name__ == "__main__":
    # 导入和初始化顺序对引用关系很重要
    from algorithm.path_planner import HybridPathPlanner
    from algorithm.map_service import MapService
    from models.vehicle import MiningVehicle
    from models.task import TransportTask
    import logging
    import time

    # 初始化日志和基础服务
    logging.basicConfig(level=logging.INFO,
                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 设置环境变量以解决NumExpr警告
    import os
    os.environ["NUMEXPR_MAX_THREADS"] = "16"
    
    # 记录调试信息
    logging.info("开始初始化系统组件")
    
    map_service = MapService()
    planner = HybridPathPlanner(map_service)
    
    # 修补planner.plan_path方法以防止错误
    def safe_plan_path(start, end, vehicle=None):
        logging.debug(f"安全路径规划: {start} -> {end}")
        # 生成简单直线路径的备选方案
        try:
            # 尝试使用原始方法
            return planner.original_plan_path(start, end, vehicle)
        except Exception as e:
            logging.warning(f"原始路径规划失败: {str(e)}, 使用备选方案")
            # 简单直线路径作为备选
            return [start, end]
    
    # 保存原始方法并替换
    if not hasattr(planner, 'original_plan_path'):
        planner.original_plan_path = planner.plan_path
        planner.plan_path = safe_plan_path
    
    start_time = time.time()  # 新增时间记录
    # 创建调度系统实例
    dispatch = DispatchSystem(planner, map_service)
    
    # 确保planner有dispatch引用
    planner.dispatch = dispatch

    # 初始化测试车辆（不同初始状态）
    logging.info("初始化测试车辆")
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

    # 添加初始化检查和调试信息
    for v in test_vehicles:
        if not hasattr(v, 'current_path'):
            v.current_path = []
        if not hasattr(v, 'path_index'):
            v.path_index = 0
        logging.debug(f"初始化车辆{v.vehicle_id}完成: 位置={v.current_location}, 状态={v.state}")

    # 注册车辆到调度系统
    for v in test_vehicles:
        dispatch.vehicles[v.vehicle_id] = v
    logging.info(f"已注册{len(test_vehicles)}辆车到调度系统")

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
    logging.info(f"已创建{len(test_tasks)}个测试任务")

    # 模拟调度循环
    print("=== 系统初始化 ===")
    dispatch.print_system_status()

    for cycle in range(1, 4):
        try:
            logging.info(f"开始调度周期 {cycle}")
            for task in test_tasks:
                dispatch.add_task(task)
            
            # 执行调度逻辑后打印状态
            dispatch.scheduling_cycle()
            dispatch.print_system_status()
            dispatch.print_ascii_map()
            elapsed = time.time() - start_time
            
            # 显示状态
            print(f"调度耗时: {elapsed:.2f}秒")
            dispatch.print_system_status()
            
            # 模拟时间推进
            logging.info(f"调度周期 {cycle} 完成，等待2秒...")
            time.sleep(2)

            # 输出最终状态
            if cycle == 3:
                print("\n=== 最终状态报告 ===")
                print("激活任务:", list(dispatch.active_tasks.keys()))
                print("完成任务:", list(dispatch.completed_tasks.keys()))
        except Exception as e:
            logging.error(f"调度周期{cycle}异常: {str(e)}")
            traceback.print_exc()
            continue