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
import time
import traceback
import math
import random
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
import configparser
# 设置更详细的日志格式
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

"""
修改后的TransportScheduler类实现
"""
class TransportScheduler:
    """增强型调度策略实现"""
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
        """改进后的任务分配策略，加入随机性和距离考量"""
        assignments = {}
        vehicle_list = list(vehicles)  # 创建车辆列表副本
        
        # 先按状态分类车辆
        idle_vehicles = [v for v in vehicle_list if v.state == VehicleState.IDLE and not v.current_task]
        unloading_vehicles = [v for v in vehicle_list if v.state == VehicleState.UNLOADING and not v.current_task]
        
        # 优先处理手动任务
        manual_tasks = [t for t in tasks if t.task_type == "manual"]
        for task in manual_tasks:
            if not idle_vehicles:
                break
                
            # 寻找最近空闲车辆
            closest_vehicle = min(
                idle_vehicles,
                key=lambda v: GeoUtils.calculate_distance(v.current_location, task.start_point),
                default=None
            )
            
            if closest_vehicle:
                assignments[closest_vehicle.vehicle_id] = task
                idle_vehicles.remove(closest_vehicle)  # 从可用车辆中移除
                vehicle_list.remove(closest_vehicle)  # 防止重复分配
        
        # 随机处理装载任务
        loading_tasks = [t for t in tasks if t.task_type == "loading" and t not in assignments.values()]
        random.shuffle(loading_tasks)  # 随机打乱任务顺序
        
        for task in loading_tasks:
            if not idle_vehicles:
                # 没有空闲车辆时将任务加入对应装载点队列
                best_queue = min(self.loading_queues, key=lambda lp: len(self.loading_queues[lp]))
                self.loading_queues[best_queue].append(task)
                continue
                
            # 计算每辆车到任务起点的距离
            distances = [(v, GeoUtils.calculate_distance(v.current_location, task.start_point)) 
                         for v in idle_vehicles]
            # 基于距离和随机因素选择车辆(80%选择最近车辆，20%随机选择)
            if random.random() < 0.8:
                chosen_vehicle, _ = min(distances, key=lambda x: x[1])
            else:
                chosen_vehicle = random.choice(idle_vehicles)
                
            assignments[chosen_vehicle.vehicle_id] = task
            idle_vehicles.remove(chosen_vehicle)
            vehicle_list.remove(chosen_vehicle)
        
        # 处理卸载任务
        unloading_tasks = [t for t in tasks if t.task_type == "unloading" and t not in assignments.values()]
        for task in unloading_tasks:
            if not unloading_vehicles:
                self.unloading_queue.append(task)
                continue
                
            # 选择负载最高的车辆优先卸载
            chosen_vehicle = max(unloading_vehicles, 
                                key=lambda v: getattr(v, 'current_load', 0))
            assignments[chosen_vehicle.vehicle_id] = task
            unloading_vehicles.remove(chosen_vehicle)
            vehicle_list.remove(chosen_vehicle)
        
        # 处理未分配的任务
        remaining_tasks = [t for t in tasks if t not in assignments.values() 
                          and t.task_type not in ["manual", "loading", "unloading"]]
        
        for task in remaining_tasks:
            if not vehicle_list:
                # 根据任务类型加入不同队列
                if hasattr(task, 'task_type'):
                    if task.task_type == "charging":
                        self.charging_queue.append(task)
                    else:
                        # 默认加入装载队列
                        best_queue = min(self.loading_queues, key=lambda lp: len(self.loading_queues[lp]))
                        self.loading_queues[best_queue].append(task)
                continue
                
            # 随机选择剩余车辆
            chosen_vehicle = random.choice(vehicle_list)
            assignments[chosen_vehicle.vehicle_id] = task
            vehicle_list.remove(chosen_vehicle)
            
        return assignments

"""
增强型冲突检测与解决算法
"""
class ConflictBasedSearch:
    """CBS冲突避免算法增强实现"""
    def __init__(self, planner: HybridPathPlanner):
        self.planner = planner
        self.constraints = defaultdict(list)
        self.reservation_table = {}
        self.reservation_lock = threading.RLock()
        self.conflict_cache = {}  # 缓存最近检测到的冲突
        
    def find_conflicts(self, paths: Dict[str, List[Tuple]]) -> List[Tuple]:
        """增强版路径冲突检测，同时考虑节点冲突和边冲突"""
        conflicts = []
        path_items = list(paths.items())
        
        logging.debug(f"检查路径冲突: {len(path_items)}条路径")
        
        # 1. 检测节点冲突(同一位置冲突)
        for i in range(len(path_items)):
            vid1, path1 = path_items[i]
            for j in range(i+1, len(path_items)):
                vid2, path2 = path_items[j]
                
                min_len = min(len(path1), len(path2))
                if min_len <= 1:
                    continue
                
                # 检测相同时间点的位置冲突
                for t in range(min_len):
                    if path1[t] == path2[t]:
                        logging.debug(f"发现位置冲突: 时间{t}, 位置{path1[t]}, 车辆{vid1}和{vid2}")
                        conflicts.append(("node", t, path1[t], vid1, vid2))
        
        # 2. 检测边冲突(路径段交叉)
        for i in range(len(path_items)):
            vid1, path1 = path_items[i]
            for j in range(i+1, len(path_items)):
                vid2, path2 = path_items[j]
                
                min_len = min(len(path1), len(path2))
                if min_len <= 1:
                    continue
                
                # 检测路径段交叉
                for t in range(min_len-1):
                    # 路径段A: path1[t] -> path1[t+1]
                    # 路径段B: path2[t] -> path2[t+1]
                    if self._segments_intersect(path1[t], path1[t+1], path2[t], path2[t+1]):
                        mid_point = self._get_intersection_point(
                            path1[t], path1[t+1], path2[t], path2[t+1]
                        )
                        logging.debug(f"发现路径交叉: 时间{t}, 位置{mid_point}, 车辆{vid1}和{vid2}")
                        conflicts.append(("edge", t, mid_point, vid1, vid2))
        
        logging.info(f"共发现{len(conflicts)}个冲突点")
        return conflicts
        
    def _segments_intersect(self, p1, p2, p3, p4):
        """判断两线段是否相交"""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # 共线
            return 1 if val > 0 else 2  # 顺时针/逆时针
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # 一般情况
        if o1 != o2 and o3 != o4:
            return True
        
        # 特殊情况 - 共线且重叠
        if o1 == 0 and self._on_segment(p1, p3, p2): return True
        if o2 == 0 and self._on_segment(p1, p4, p2): return True
        if o3 == 0 and self._on_segment(p3, p1, p4): return True
        if o4 == 0 and self._on_segment(p3, p2, p4): return True
        
        return False
    
    def _on_segment(self, p, q, r):
        """判断点q是否在线段pr上"""
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    def _get_intersection_point(self, p1, p2, p3, p4):
        """计算两线段的交点坐标"""
        # 简单情况：取两线段中点
        return ((p1[0] + p2[0] + p3[0] + p4[0]) / 4, 
                (p1[1] + p2[1] + p3[1] + p4[1]) / 4)
        
    # 增强CBS冲突解决算法中的重规划方法，提高错误处理能力
    def _replan_path(self, vehicle_id, pos=None, t=None, max_retries=3):
        """增强路径重规划，支持多级后备方案，提高错误恢复能力"""
        logging.debug(f"开始为车辆{vehicle_id}重新规划路径: pos={pos}, t={t}")
        
        try:
            # 确保有vehicle对象
            if not hasattr(self.planner, 'dispatch'):
                logging.error("规划器未初始化dispatch属性")
                return None
                
            if vehicle_id not in self.planner.dispatch.vehicles:
                logging.error(f"找不到车辆ID: {vehicle_id}")
                return None
                
            vehicle = self.planner.dispatch.vehicles[vehicle_id]
            logging.debug(f"车辆{vehicle_id}当前位置: {vehicle.current_location}")
            
            # 确保车辆位置是有效的
            if not hasattr(vehicle, 'current_location') or vehicle.current_location is None:
                logging.error(f"车辆{vehicle_id}位置无效")
                return None
            
            # 确定目标位置
            end_point = None
            if vehicle.current_task and hasattr(vehicle.current_task, 'end_point'):
                end_point = vehicle.current_task.end_point
            elif pos is not None:
                end_point = pos
            else:
                logging.warning(f"无法为车辆{vehicle_id}规划路径: 无目标位置")
                return None
            
            # 确保起点和终点是有效的元组
            start_point = vehicle.current_location
            if not isinstance(start_point, tuple) and hasattr(start_point, '__getitem__'):
                start_point = (start_point[0], start_point[1])
            if not isinstance(end_point, tuple) and hasattr(end_point, '__getitem__'):
                end_point = (end_point[0], end_point[1])
            
            # 多级规划策略
            for attempt in range(max_retries):
                # 1. 尝试标准规划
                try:
                    logging.debug(f"尝试标准规划(尝试{attempt+1}): {start_point} -> {end_point}")
                    new_path = self.planner.plan_path(start_point, end_point, vehicle)
                    
                    if new_path and len(new_path) > 0:
                        logging.debug(f"成功规划路径: {len(new_path)}个点")
                        vehicle.assign_path(new_path)
                        self._update_reservation_table(new_path, vehicle_id)
                        return new_path
                except Exception as e:
                    logging.warning(f"标准规划失败: {str(e)}")
                
                # 2. 尝试添加中间点绕行
                try:
                    # 在当前位置和目标之间添加偏移点
                    mid_x = (start_point[0] + end_point[0]) / 2
                    mid_y = (start_point[1] + end_point[1]) / 2
                    # 添加随机偏移以避免再次冲突
                    offset = 20 * (random.random() - 0.5)
                    mid_point = (mid_x + offset, mid_y + offset)
                    
                    logging.debug(f"尝试中间点绕行(尝试{attempt+1}): 经过{mid_point}")
                    # 先规划到中间点
                    path1 = self.planner.plan_path(start_point, mid_point, vehicle)
                    # 再从中间点到终点
                    path2 = self.planner.plan_path(mid_point, end_point, vehicle)
                    
                    if path1 and path2 and len(path1) > 0 and len(path2) > 0:
                        # 合并路径(去掉重复的中间点)
                        new_path = path1 + path2[1:]
                        logging.debug(f"成功规划绕行路径: {len(new_path)}个点")
                        vehicle.assign_path(new_path)
                        self._update_reservation_table(new_path, vehicle_id)
                        return new_path
                except Exception as e:
                    logging.warning(f"绕行规划失败: {str(e)}")
                
                # 3. 尝试延迟策略(等待一段时间)
                if attempt == max_retries - 1:
                    logging.debug("尝试延迟策略: 原地等待")
                    try:
                        # 构造一个原地等待路径(重复当前位置)
                        wait_path = [start_point] * 5  # 生成5个相同点表示等待
                        
                        # 添加直线路径作为备用
                        dx = end_point[0] - start_point[0]
                        dy = end_point[1] - start_point[1]
                        for i in range(1, 5):
                            wait_path.append((
                                start_point[0] + dx * i / 5,
                                start_point[1] + dy * i / 5
                            ))
                        wait_path.append(end_point)
                        
                        logging.debug(f"采用延迟路径: {len(wait_path)}个点")
                        vehicle.assign_path(wait_path)
                        self._update_reservation_table(wait_path, vehicle_id)
                        return wait_path
                    except Exception as e:
                        logging.warning(f"延迟策略失败: {str(e)}")
                        # 最终备用方案：简单直线路径
                        simple_path = [start_point, end_point]
                        vehicle.assign_path(simple_path)
                        self._update_reservation_table(simple_path, vehicle_id)
                        return simple_path
                
                logging.debug(f"路径重试 {attempt+1}/{max_retries}")
                time.sleep(0.2)  # 短暂等待后重试
            
            logging.warning(f"车辆{vehicle_id}路径规划达到最大重试次数")
            # 所有方法都失败，使用直线路径
            simple_path = [start_point, end_point]
            vehicle.assign_path(simple_path)
            self._update_reservation_table(simple_path, vehicle_id)
            return simple_path
            
        except Exception as e:
            logging.error(f"重规划路径发生异常: {str(e)}")
            try:
                # 紧急恢复：总是确保返回一个简单的路径
                start_point = vehicle.current_location
                simple_path = [start_point, end_point or (start_point[0] + 100, start_point[1] + 100)]
                vehicle.assign_path(simple_path)
                return simple_path
            except:
                logging.error("无法进行紧急恢复")
                return None
            
    def _get_vehicle_priority(self, vehicle_id: str) -> int:
        """获取车辆优先级，考虑多种因素"""
        try:
            if not hasattr(self.planner, 'dispatch'):
                return 5  # 默认优先级
                
            if vehicle_id not in self.planner.dispatch.vehicles:
                return 5  # 默认优先级
                
            vehicle = self.planner.dispatch.vehicles[vehicle_id]
            
            # 综合优先级计算
            base_priority = {
                VehicleState.UNLOADING: 1,  # 最高优先级
                VehicleState.PREPARING: 2,
                TransportStage.TRANSPORTING: 2,  # 已装载的车辆优先
                TransportStage.APPROACHING: 3,
                VehicleState.IDLE: 4
            }
            
            current_state = vehicle.state
            transport_stage = getattr(vehicle, 'transport_stage', None)
            
            # 基础优先级
            if current_state == VehicleState.EN_ROUTE and transport_stage:
                priority = base_priority.get(transport_stage, 3)
            else:
                priority = base_priority.get(current_state, 3)
            
            # 任务优先级加权
            if hasattr(vehicle, 'current_task') and vehicle.current_task:
                task_priority = getattr(vehicle.current_task, 'priority', 1)
                # 任务优先级影响(0-1)，值越小优先级越高
                priority_modifier = max(0, 1 - task_priority/10)
                priority = max(1, priority - priority_modifier)
            
            # 车辆负载加权
            if getattr(vehicle, 'current_load', 0) > 0:
                # 负载车辆优先级提高
                priority = max(1, priority - 0.5)
                
            logging.debug(f"车辆{vehicle_id}优先级: {priority:.1f} (状态:{current_state}, 阶段:{transport_stage})")
            return priority
            
        except Exception as e:
            logging.error(f"获取车辆优先级异常: {str(e)}")
            return 5  # 返回默认优先级
            
    def resolve_conflicts(self, paths: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
        """优化的冲突解决方案，采用多策略方法"""
        if not paths:
            return paths
            
        try:
            new_paths = paths.copy()
            
            # 获取所有冲突
            conflicts = self.find_conflicts(paths)
            
            if not conflicts:
                return new_paths
                
            # 按冲突时间排序处理
            conflicts.sort(key=lambda x: x[1])
            
            for i, conflict in enumerate(conflicts):
                conflict_type, t, pos, vid1, vid2 = conflict
                logging.debug(f"处理冲突 {i+1}/{len(conflicts)}: 类型={conflict_type}, 时间={t}")
                
                # 获取车辆优先级
                prio1 = self._get_vehicle_priority(vid1)
                prio2 = self._get_vehicle_priority(vid2)
                
                logging.debug(f"车辆优先级: {vid1}={prio1:.1f}, {vid2}={prio2:.1f}")
                
                # 确定冲突解决策略
                if abs(prio1 - prio2) < 0.5:
                    # 优先级相近时，随机选择一辆车重规划
                    vehicle_to_replan = vid1 if random.random() < 0.5 else vid2
                    logging.debug(f"优先级相近，随机选择车辆{vehicle_to_replan}进行重规划")
                elif prio1 < prio2:
                    # 低数字优先级更高
                    vehicle_to_replan = vid2
                    logging.debug(f"车辆{vid1}优先级更高，为车辆{vid2}重规划")
                else:
                    vehicle_to_replan = vid1
                    logging.debug(f"车辆{vid2}优先级更高，为车辆{vid1}重规划")
                
                # 执行路径重规划
                new_path = self._replan_path(vehicle_id=vehicle_to_replan, pos=pos, t=t)
                if new_path:
                    new_paths[vehicle_to_replan] = new_path
                    # 缓存已解决的冲突，避免重复处理
                    conflict_key = (vid1, vid2, t)
                    self.conflict_cache[conflict_key] = time.time()
                else:
                    logging.warning(f"车辆{vehicle_to_replan}路径重规划失败，冲突未解决")
            
            # 清理过期冲突缓存(超过30秒)
            current_time = time.time()
            expired_keys = [k for k, v in self.conflict_cache.items() if current_time - v > 30]
            for k in expired_keys:
                del self.conflict_cache[k]
                
            return new_paths
            
        except Exception as e:
            logging.error(f"解决冲突过程发生异常: {str(e)}")
            traceback.print_exc()
            # 出错时返回原始路径
            return paths
            
    def _update_reservation_table(self, path, vehicle_id):
        """更新路径预约表，增加时间维度"""
        try:
            with self.reservation_lock:
                logging.debug(f"更新车辆{vehicle_id}的路径预约表, 路径长度: {len(path)}")
                
                # 清除此车辆之前的预约
                segments_to_remove = [k for k, v in self.reservation_table.items() if v == vehicle_id]
                for k in segments_to_remove:
                    del self.reservation_table[k]
                
                # 为路径的每个点添加预约
                for i in range(len(path) - 1):
                    segment = (path[i], path[i+1])
                    self.reservation_table[segment] = vehicle_id
                    
                    # 同时预约关键节点(更长时间)
                    for j in range(3):  # 添加3个时间单位的节点预约
                        if i+j < len(path):
                            node_key = (path[i+j], i+j)
                            self.reservation_table[node_key] = vehicle_id
                
                logging.debug(f"预约表更新完成, 当前有{len(self.reservation_table)}个预约")
        except Exception as e:
            logging.error(f"更新预约表异常: {str(e)}")

"""
修改后的DispatchSystem类实现
"""
class DispatchSystem:
    """智能调度系统核心"""
    def __init__(self, planner: HybridPathPlanner, map_service: MapService):
        self.planner = planner
        self.map_service = map_service
        self.vehicles: Dict[str, MiningVehicle] = {}
        self.scheduler = TransportScheduler(self._load_config())
        
        # 改进：先设置好planner的dispatch引用
        self.planner.dispatch = self
        
        # 改进：确保从属关系正确初始化
        self.cbs = ConflictBasedSearch(planner)
        
        # 任务管理
        self.task_queue = deque()
        self.active_tasks = {}
        self.completed_tasks = {}
        
        # 并发控制
        self.lock = threading.RLock()
        self.vehicle_lock = threading.Lock()
        
        # 改进：添加运行状态标志
        self.running = True
        self.last_scheduling_time = time.time()
        
        # 系统性能指标
        self.performance_metrics = {
            'conflict_count': 0,
            'resolved_conflicts': 0,
            'avg_waiting_time': 0,
            'completed_tasks_count': 0,
            'failed_tasks_count': 0
        }
        
        logging.info("调度系统初始化完成")
        
    def _load_config(self) -> dict:
        """加载调度配置"""
        try:
            # 尝试从配置文件加载
            config_path = os.path.join(PROJECT_ROOT, 'config.ini')
            if os.path.exists(config_path):
                config = configparser.ConfigParser()
                config.read(config_path)
                return {
                    'loading_points': eval(config.get('DISPATCH', 'loading_points', 
                                                    fallback="[(-100,50), (0,150), (100,50)]")),
                    'unloading_point': eval(config.get('DISPATCH', 'unloading_point', 
                                                    fallback="(0,-100)")),
                    'parking_area': eval(config.get('DISPATCH', 'parking_area', 
                                                    fallback="(200,200)")),
                    'max_charging_vehicles': int(config.get('DISPATCH', 'max_charging_vehicles', 
                                                        fallback="2"))
                }
        except Exception as e:
            logging.warning(f"加载配置失败: {str(e)}，使用默认配置")
            
        # 默认配置
        return {
            'loading_points': [(-100,50), (0,150), (100,50)],
            'unloading_point': (0,-100),
            'parking_area': (200,200),
            'max_charging_vehicles': 2
        }
    
    # ----------------- 核心调度循环 -----------------
    def start_scheduling(self, interval=10):
        """启动调度循环"""
        self.running = True
        
        try:
            while self.running:
                start_time = time.time()
                self.scheduling_cycle()
                
                # 计算实际所需周期时间
                elapsed = time.time() - start_time
                wait_time = max(0, interval - elapsed)
                
                if wait_time > 0:
                    logging.debug(f"调度周期完成，等待{wait_time:.1f}秒")
                    time.sleep(wait_time)
                else:
                    logging.warning(f"调度周期耗时过长: {elapsed:.1f}秒，无法维持{interval}秒间隔")
        except KeyboardInterrupt:
            logging.info("接收到中断信号，停止调度")
            self.running = False
        except Exception as e:
            logging.error(f"调度循环异常: {str(e)}")
            traceback.print_exc()
            self.running = False
            
    def stop_scheduling(self):
        """停止调度循环"""
        self.running = False
        logging.info("调度系统已停止")
        
    def scheduling_cycle(self):
        """改进的调度主循环"""
        logging.info("开始调度周期")
        cycle_start = time.time()
        
        try:
            with self.lock:
                # 1. 状态更新阶段
                logging.debug("状态更新阶段")
                self._update_vehicle_states()
                
                # 2. 任务处理阶段 
                logging.debug("任务分配阶段")
                self._dispatch_tasks()
                
                # 3. 路径规划阶段
                logging.debug("路径规划阶段")
                self._plan_paths()
                
                # 4. 冲突检测阶段
                logging.debug("冲突检测阶段")
                conflict_count = self._detect_conflicts()
                
                # 更新性能指标
                self.performance_metrics['conflict_count'] += conflict_count
                
                # 5. 车辆移动模拟 (仅在模拟模式下)
                logging.debug("车辆移动阶段")
                self._move_vehicles()
                
                # 6. 完成任务检查
                self._check_completed_tasks()
                
            # 7. 打印系统状态 (不需要锁)
            if time.time() - self.last_scheduling_time > 5:  # 每5秒打印一次状态
                self.print_system_status()
                self.last_scheduling_time = time.time()
                
            cycle_time = time.time() - cycle_start
            logging.info(f"调度周期成功完成，耗时: {cycle_time:.3f}秒")
            
        except Exception as e:
            logging.error(f"调度周期执行异常: {str(e)}")
            traceback.print_exc()
            
    # 修改 _dispatch_tasks 方法
    def _dispatch_tasks(self):
        """增强的任务分配逻辑"""
        # 确保仅选择真正空闲的车辆
        available_vehicles = [v for v in self.vehicles.values() 
                            if v.state == VehicleState.IDLE and not v.current_task]
        
        if not available_vehicles and not self.task_queue:
            logging.debug("无可用车辆或任务队列为空，跳过任务分配")
            return
            
        logging.debug(f"任务分配开始, 队列中有{len(self.task_queue)}个任务, {len(available_vehicles)}辆车可用")
        
        # 用调度策略为每辆空闲车辆分配任务
        assignments = self.scheduler.apply_scheduling_policy(
            list(available_vehicles),  # 仅传入可用车辆
            list(self.task_queue)
        )
        
        # 将分配结果存入激活任务
        with self.vehicle_lock:
            for vid, task in assignments.items():
                if vid in self.vehicles:
                    vehicle = self.vehicles[vid]
                    # 再次检查车辆状态和任务情况
                    if vehicle.state == VehicleState.IDLE and not vehicle.current_task:
                        try:
                            vehicle.assign_task(task)
                            self.active_tasks[task.task_id] = task
                            logging.info(f"车辆 {vid} 已分配任务 {task.task_id}")
                            
                            # 从任务队列中移除已分配任务
                            if task in self.task_queue:
                                self.task_queue.remove(task)
                        except Exception as e:
                            logging.error(f"分配任务给车辆 {vid} 失败: {str(e)}")
                    else:
                        logging.warning(f"车辆 {vid} 状态已变更或已有任务，取消分配")
            
    def _generate_random_task(self):
        """生成随机任务"""
        config = self._load_config()
        task_types = ["loading", "unloading"]
        task_type = random.choice(task_types)
        
        if task_type == "loading":
            # 随机选择一个装载点
            start_point = random.choice(config['loading_points'])
            end_point = config['unloading_point']
        else:
            # 卸载任务
            start_point = config['unloading_point']
            end_point = config['parking_area']
            
        task_id = f"{task_type.upper()}-{int(time.time())}"
        
        task = TransportTask(
            task_id=task_id,
            start_point=start_point,
            end_point=end_point,
            task_type=task_type,
            waypoints=[],  # 可以自动生成中间点
            priority=random.randint(1, 3)  # 随机优先级
        )
        
        self.task_queue.append(task)
        logging.info(f"生成随机任务: {task_id} ({task_type})")
        
    def _plan_paths(self):
        """为所有有任务但无路径的车辆规划路径 - 增强型错误处理"""
        with self.vehicle_lock:
            vehicles_need_path = [v for v in self.vehicles.values() 
                                if v.current_task and (not hasattr(v, 'current_path') or 
                                                    not v.current_path)]
            
            if not vehicles_need_path:
                return
                
            logging.debug(f"为{len(vehicles_need_path)}辆车规划路径")
            
            for vehicle in vehicles_need_path:
                try:
                    # 获取任务起终点
                    start = vehicle.current_location
                    end = vehicle.current_task.end_point
                    
                    # 确保坐标是有效的元组
                    if not isinstance(start, tuple) and hasattr(start, '__getitem__'):
                        start = (start[0], start[1])
                    if not isinstance(end, tuple) and hasattr(end, '__getitem__'):
                        end = (end[0], end[1])
                    
                    # 检查坐标有效性
                    if (not isinstance(start, tuple) or len(start) < 2 or 
                        not isinstance(end, tuple) or len(end) < 2):
                        logging.error(f"车辆{vehicle.vehicle_id}坐标无效: start={start}, end={end}")
                        continue
                    
                    logging.debug(f"车辆{vehicle.vehicle_id}路径规划: {start} -> {end}")
                    
                    # 调用路径规划器 - 捕获所有可能的异常
                    try:
                        path = self.planner.plan_path(start, end, vehicle)
                    except Exception as e:
                        logging.error(f"车辆{vehicle.vehicle_id}路径规划异常: {str(e)}")
                        # 生成备用路径
                        path = [start, end]
                    
                    # 验证路径有效性
                    if path and len(path) > 0:
                        # 防止路径中包含None或非法点
                        valid_path = []
                        for point in path:
                            if point is None:
                                continue
                            # 确保每个点是元组且有x,y坐标
                            if hasattr(point, 'as_tuple'):
                                valid_path.append(point.as_tuple())
                            elif isinstance(point, tuple) and len(point) >= 2:
                                valid_path.append(point)
                            elif hasattr(point, '__getitem__') and len(point) >= 2:
                                valid_path.append((point[0], point[1]))
                        
                        # 确保至少包含起点和终点
                        if len(valid_path) < 2:
                            valid_path = [start, end]
                        
                        vehicle.assign_path(valid_path)
                        logging.debug(f"车辆{vehicle.vehicle_id}路径规划成功: {len(valid_path)}个点")
                    else:
                        logging.warning(f"车辆{vehicle.vehicle_id}路径规划失败: 返回空路径")
                        # 分配简单直线路径作为备选
                        vehicle.assign_path([start, end])
                except Exception as e:
                    logging.error(f"车辆{vehicle.vehicle_id}路径规划过程异常: {str(e)}")
        
    def _update_vehicle_states(self):
        """增强的车辆状态更新，确保状态一致性"""
        logging.debug(f"开始更新{len(self.vehicles)}辆车的状态")
        
        for vid, vehicle in self.vehicles.items():
            prev_state = vehicle.state
            
            # 1. 基于位置更新车辆状态
            self._update_vehicle_state_by_location(vehicle)
            
            # 2. 基于任务更新车辆运输阶段
            self._update_vehicle_transport_stage(vehicle)
            
            # 3. 检查任务完成情况
            self._check_vehicle_task_completion(vehicle)
            
            # 记录状态变化
            if prev_state != vehicle.state:
                logging.debug(f"车辆{vid}状态从{prev_state}变为{vehicle.state}")
        
        logging.debug("车辆状态更新完成")
        
    def _update_vehicle_state_by_location(self, vehicle):
        """基于位置更新车辆状态"""
        config = self._load_config()
        
        # 检查车辆是否在关键位置
        if self._is_at_location(vehicle.current_location, config['parking_area']):
            vehicle.state = VehicleState.IDLE
        elif any(self._is_at_location(vehicle.current_location, lp) for lp in config['loading_points']):
            vehicle.state = VehicleState.PREPARING
        elif self._is_at_location(vehicle.current_location, config['unloading_point']):
            vehicle.state = VehicleState.UNLOADING
        elif vehicle.current_task:
            vehicle.state = VehicleState.EN_ROUTE
    
    def _is_at_location(self, current_loc, target_loc, threshold=5.0):
        """判断车辆是否到达目标位置"""
        dx = current_loc[0] - target_loc[0]
        dy = current_loc[1] - target_loc[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance <= threshold
        
    def _update_vehicle_transport_stage(self, vehicle):
        """更新车辆运输阶段"""
        if vehicle.current_task and vehicle.state == VehicleState.EN_ROUTE:
            if vehicle.current_task.task_type == "loading":
                vehicle.transport_stage = TransportStage.APPROACHING
            elif vehicle.current_task.task_type == "unloading":
                vehicle.transport_stage = TransportStage.TRANSPORTING
            else:
                vehicle.transport_stage = TransportStage.RETURNING
                
    def _check_vehicle_task_completion(self, vehicle):
        """检查车辆任务完成情况"""
        if not vehicle.current_task:
            return
            
        # 判断是否到达终点
        if hasattr(vehicle, 'current_path') and vehicle.current_path and hasattr(vehicle, 'path_index'):
            if vehicle.path_index >= len(vehicle.current_path) - 1:
                # 到达路径终点，标记任务完成
                completed_task = vehicle.current_task
                logging.info(f"车辆{vehicle.vehicle_id}完成任务{completed_task.task_id}")
                
                # 更新任务状态
                self.completed_tasks[completed_task.task_id] = completed_task
                if completed_task.task_id in self.active_tasks:
                    del self.active_tasks[completed_task.task_id]
                
                # 更新车辆状态
                vehicle.current_task = None
                
                # 更新性能指标
                self.performance_metrics['completed_tasks_count'] += 1
                
    def _check_completed_tasks(self):
        """检查任务完成状态，更新系统状态"""
        # 清理已完成但未更新的任务
        tasks_to_remove = []
        
        for task_id, task in self.active_tasks.items():
            # 任务已完成但未移除
            if hasattr(task, 'is_completed') and task.is_completed:
                tasks_to_remove.append(task_id)
                self.completed_tasks[task_id] = task
                
        # 批量移除已完成任务
        for task_id in tasks_to_remove:
            del self.active_tasks[task_id]
        
        if tasks_to_remove:
            logging.debug(f"已清理{len(tasks_to_remove)}个已完成任务")
            
    def _move_vehicles(self):
        """模拟车辆移动"""
        with self.vehicle_lock:
            for vid, vehicle in self.vehicles.items():
                if vehicle.state == VehicleState.EN_ROUTE and hasattr(vehicle, 'current_path'):
                    if vehicle.current_path and hasattr(vehicle, 'path_index'):
                        # 确保索引在有效范围内
                        if vehicle.path_index < len(vehicle.current_path) - 1:
                            vehicle.path_index += 1
                            vehicle.current_location = vehicle.current_path[vehicle.path_index]
                            logging.debug(f"车辆{vid}移动到: {vehicle.current_location}")
        
    def _detect_conflicts(self):
        """增强版冲突检测与解决"""
        logging.debug("开始冲突检测")
        
        # 收集所有车辆的路径
        all_paths = {}
        for vid, vehicle in self.vehicles.items():
            if vehicle.current_path and vehicle.state == VehicleState.EN_ROUTE:
                # 只考虑当前位置之后的路径段
                if hasattr(vehicle, 'path_index'):
                    remaining_path = vehicle.current_path[vehicle.path_index:]
                    if len(remaining_path) > 1:
                        all_paths[str(vid)] = remaining_path
        
        if not all_paths:
            logging.debug("没有可检测的路径，跳过冲突检测")
            return 0
            
        try:
            # 冲突解决
            conflict_start = time.time()
            resolved_paths = self.cbs.resolve_conflicts(all_paths)
            conflict_time = time.time() - conflict_start
            
            if conflict_time > 0.5:
                logging.warning(f"冲突解决耗时较长: {conflict_time:.2f}秒")
            
            # 统计冲突数量
            conflict_count = sum(1 for vid in resolved_paths if resolved_paths[vid] != all_paths.get(vid, []))
            
            # 更新车辆路径
            if conflict_count > 0:
                with self.vehicle_lock:
                    for vid_str, path in resolved_paths.items():
                        try:
                            vid = vid_str if not vid_str.isdigit() else int(vid_str)
                            if path and vid in self.vehicles:
                                # 保留当前位置
                                current_pos = self.vehicles[vid].current_location
                                # 确保新路径从当前位置开始
                                if path[0] != current_pos:
                                    path = [current_pos] + path
                                logging.debug(f"为车辆{vid}分配新路径，长度: {len(path)}")
                                self.vehicles[vid].assign_path(path)
                                self.vehicles[vid].path_index = 0  # 重置路径索引
                        except Exception as e:
                            logging.error(f"车辆 {vid} 路径分配失败: {str(e)}")
            
            logging.debug(f"冲突检测完成，发现并解决了{conflict_count}个冲突")
            return conflict_count
            
        except Exception as e:
            logging.error(f"冲突检测异常: {str(e)}")
            traceback.print_exc()
            return 0
            
    def add_task(self, task: TransportTask):
        """线程安全的任务添加方法"""
        with self.lock:
            self.task_queue.append(task)
            logging.info(f"已接收任务 {task.task_id} ({task.task_type})")
            
    def add_vehicle(self, vehicle: MiningVehicle):
        """添加车辆到调度系统"""
        with self.vehicle_lock:
            self.vehicles[vehicle.vehicle_id] = vehicle
            logging.info(f"已添加车辆 {vehicle.vehicle_id}")
            
    def print_ascii_map(self):
        """生成ASCII地图可视化"""
        MAP_SIZE = 40  # 地图尺寸（40x40）
        SCALE = 10      # 坐标缩放因子
        
        # 初始化空地图
        grid = [['·' for _ in range(MAP_SIZE)] for _ in range(MAP_SIZE)]
        
        # 标注固定设施
        config = self._load_config()
        self._plot_point(grid, config['unloading_point'], 'U', SCALE)
        self._plot_point(grid, config['parking_area'], 'P', SCALE)
        for i, lp in enumerate(config['loading_points']):
            self._plot_point(grid, lp, f'L{i+1}', SCALE)
            
        # 标注车辆位置
        vehicle_symbols = {
            VehicleState.UNLOADING: '▼', 
            VehicleState.PREPARING: '▲',
            TransportStage.TRANSPORTING: '▶',
            TransportStage.APPROACHING: '◀',
            VehicleState.IDLE: '●',
            VehicleState.EN_ROUTE: '►'
        }
        
        for vid, vehicle in self.vehicles.items():
            # 确定车辆符号
            if vehicle.state == VehicleState.EN_ROUTE and hasattr(vehicle, 'transport_stage'):
                symbol = vehicle_symbols.get(vehicle.transport_stage, '►')
            else:
                symbol = vehicle_symbols.get(vehicle.state, '?')
                
            # 添加车辆ID
            vehicle_label = f"{symbol}{vid}"
            self._plot_vehicle(grid, vehicle.current_location, vehicle_label, SCALE)
            
        # 打印地图
        print("\n当前地图布局：")
        for row in grid:
            print(' '.join(row))
            
    def _plot_point(self, grid, point, symbol, scale):
        """坐标转换方法"""
        MAP_CENTER = len(grid) // 2
        try:
            # 将实际坐标转换为地图索引
            x = int(point[0]/scale) + MAP_CENTER
            y = int(point[1]/scale) + MAP_CENTER
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                grid[y][x] = symbol
        except (TypeError, IndexError) as e:
            logging.warning(f"坐标绘制异常: {str(e)}")
            
    def _plot_vehicle(self, grid, point, symbol, scale):
        """绘制带ID的车辆"""
        MAP_CENTER = len(grid) // 2
        try:
            # 将实际坐标转换为地图索引
            x = int(point[0]/scale) + MAP_CENTER
            y = int(point[1]/scale) + MAP_CENTER
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
                grid[y][x] = symbol
        except (TypeError, IndexError) as e:
            logging.warning(f"车辆坐标绘制异常: {str(e)}")
            
    def print_system_status(self):
        """实时系统状态监控"""
        config = self._load_config()
        active_vehicles = len([v for v in self.vehicles.values() 
                             if v.state != VehicleState.IDLE])
        
        # 计算每个装载点的排队车辆
        loading_status = {}
        for i, lp in enumerate(config['loading_points']):
            vehicles_at_point = [v for v in self.vehicles.values() 
                              if self._is_at_location(v.current_location, lp)]
            loading_status[f'装载点{i+1}'] = len(vehicles_at_point)
            
        # 计算卸载点的排队车辆
        unloading_vehicles = len([v for v in self.vehicles.values() 
                               if self._is_at_location(v.current_location, config['unloading_point'])])
        
        # 计算各种状态的车辆数量
        vehicle_states = {}
        for state in VehicleState:
            vehicle_states[state.name] = len([v for v in self.vehicles.values() if v.state == state])
        
        # 格式化输出
        print("\n系统状态 (时间: {})".format(datetime.now().strftime("%H:%M:%S")))
        print("─" * 40)
        print(f"活动车辆: {active_vehicles}/{len(self.vehicles)}")
        print(f"排队任务: {len(self.task_queue)}  活动任务: {len(self.active_tasks)}  已完成: {len(self.completed_tasks)}")
        
        print("\n车辆状态分布:")
        for state, count in vehicle_states.items():
            if count > 0:
                print(f"  {state}: {count}辆")
                
        print("\n装载点状态:")
        for name, count in loading_status.items():
            print(f"  {name}: {count}辆")
        print(f"  卸载点: {unloading_vehicles}辆")
        
        print("\n性能指标:")
        print(f"  已解决冲突: {self.performance_metrics['resolved_conflicts']}")
        print(f"  完成任务数: {self.performance_metrics['completed_tasks_count']}")
        print("─" * 40)
        
    def dispatch_vehicle_to(self, vehicle_id: str, destination: Tuple[float, float]):
        """直接调度指定车辆到目标位置"""
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
            
            # 立即规划路径
            path = self.planner.plan_path(vehicle.current_location, destination, vehicle)
            if path:
                vehicle.assign_path(path)
                logging.debug(f"已为车辆{vehicle_id}规划手动路径: {len(path)}个点")

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
# 在 path_planner.py 中增强 safe_plan_path 方法
    def safe_plan_path(start, end, vehicle=None):
        logging.debug(f"安全路径规划: {start} -> {end}")
        try:
            # 确保起点和终点是有效的元组
            if hasattr(start, 'as_tuple'):
                start = start.as_tuple()
            if hasattr(end, 'as_tuple'):
                end = end.as_tuple()
                
            # 尝试使用原始方法
            try:
                return planner.original_plan_path(start, end, vehicle)
            except Exception as e:
                logging.warning(f"原始路径规划失败: {str(e)}, 使用备选方案")
                
            # 直接返回简单直线路径
            return [start, end]
        except Exception as e:
            logging.error(f"安全路径规划完全失败: {str(e)}")
            # 确保返回有效路径，即使出现严重错误
            if isinstance(start, tuple) and isinstance(end, tuple):
                return [start, end]
            else:
                # 如果连起点终点都不是元组，返回默认值
                return [(0, 0), (100, 100)]
    
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