import time
import logging
import threading
import heapq
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

class ConflictBasedSearch:
    """
    冲突基于搜索(CBS)算法实现
    
    CBS是一种用于多智能体路径规划的算法，它通过以下步骤工作：
    1. 为每个智能体独立规划路径
    2. 检测路径之间的冲突
    3. 添加约束并重新规划来解决冲突
    4. 递归地应用这个过程，直到找到无冲突的解决方案
    
    该算法适用于矿山环境中的多车辆协同调度场景。
    """
    
    def __init__(self, path_planner):
        """
        初始化CBS算法
        
        Args:
            path_planner: 路径规划器对象，用于单车路径规划
        """
        self.planner = path_planner
        self.constraints = defaultdict(list)  # 车辆的路径约束
        self.reservation_table = {}  # 时空点预约表
        self.reservation_lock = threading.RLock()
        self.conflict_cache = {}  # 缓存最近检测到的冲突
        self.stats = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'replanning_count': 0,
            'total_resolution_time': 0
        }
        
    def find_conflicts(self, paths: Dict[str, List[Tuple]]) -> List[Dict]:
        """
        检测路径之间的冲突
        
        Args:
            paths: 车辆ID到路径的映射字典
            
        Returns:
            List[Dict]: 冲突信息列表，每个冲突包含类型、位置、涉及车辆等信息
        """
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
                    if self._points_close_enough(path1[t], path2[t]):
                        logging.debug(f"发现位置冲突: 时间{t}, 位置{path1[t]}, 车辆{vid1}和{vid2}")
                        conflicts.append({
                            "type": "vertex",
                            "time": t,
                            "location": path1[t],
                            "agent1": vid1,
                            "agent2": vid2
                        })
                        
                        # 更新统计
                        self.stats['conflicts_detected'] += 1
                        break  # 只记录这两辆车的第一个冲突
        
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
                        conflicts.append({
                            "type": "edge",
                            "time": t,
                            "location": mid_point,
                            "agent1": vid1,
                            "agent2": vid2
                        })
                        
                        # 更新统计
                        self.stats['conflicts_detected'] += 1
                        break  # 只记录这两辆车的第一个冲突
        
        logging.info(f"共发现{len(conflicts)}个冲突点")
        return conflicts
        
    def _points_close_enough(self, p1, p2, threshold=3.0):
        """判断两点是否足够接近（视为冲突）"""
        if p1 == p2:
            return True
            
        try:
            import math
            return math.dist(p1, p2) < threshold
        except (TypeError, ValueError):
            # 如果无法计算距离，比较坐标
            try:
                dx = abs(p1[0] - p2[0])
                dy = abs(p1[1] - p2[1])
                return dx*dx + dy*dy < threshold*threshold
            except:
                # 如果还是失败，假设不冲突
                return False
        
    def _segments_intersect(self, p1, p2, p3, p4):
        """判断两线段是否相交"""
        def orientation(p, q, r):
            try:
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if abs(val) < 1e-9:
                    return 0  # 共线
                return 1 if val > 0 else 2  # 顺时针/逆时针
            except:
                return 0  # 出错时默认共线
        
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
        try:
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                    q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        except:
            return False
    
    def _get_intersection_point(self, p1, p2, p3, p4):
        """计算两线段的交点坐标"""
        try:
            # 精确计算交点 (线性代数方法)
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            if abs(denom) < 1e-9:
                # 线段平行，取中点
                return ((p1[0] + p2[0] + p3[0] + p4[0]) / 4, 
                        (p1[1] + p2[1] + p3[1] + p4[1]) / 4)
                
            ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            
            return (x, y)
        except:
            # 计算失败，使用简单方法
            return ((p1[0] + p2[0] + p3[0] + p4[0]) / 4, 
                    (p1[1] + p2[1] + p3[1] + p4[1]) / 4)
    
    def _get_vehicle_priority(self, vehicle_id: str) -> float:
        """
        获取车辆优先级，值越小优先级越高
        
        考虑因素:
        - 任务类型 (卸载>装载>空驶)
        - 载重情况 (满载>空载)
        - 任务优先级
        """
        try:
            if not hasattr(self.planner, 'dispatch'):
                return 5.0  # 默认优先级
                
            if vehicle_id not in self.planner.dispatch.vehicles:
                return 5.0  # 默认优先级
                
            vehicle = self.planner.dispatch.vehicles[vehicle_id]
            
            # 基础优先级 (基于状态)
            priority = 5.0
            
            # 检查车辆状态
            if hasattr(vehicle, 'state'):
                state_name = vehicle.state.name if hasattr(vehicle.state, 'name') else str(vehicle.state)
                
                # 状态优先级
                if 'UNLOAD' in state_name:
                    priority = 1.0  # 卸载状态最高优先级
                elif 'PREPAR' in state_name:
                    priority = 2.0  # 准备状态
                elif 'EN_ROUTE' in state_name or 'ENROUTE' in state_name:
                    priority = 3.0  # 在途状态
                elif 'IDLE' in state_name:
                    priority = 4.0  # 空闲状态
            
            # 检查运输阶段
            if hasattr(vehicle, 'transport_stage') and vehicle.transport_stage:
                stage_name = vehicle.transport_stage.name if hasattr(vehicle.transport_stage, 'name') else str(vehicle.transport_stage)
                
                if 'TRANSPORT' in stage_name:
                    priority = min(priority, 2.0)  # 运输阶段优先级提高
            
            # 考虑载重情况
            if hasattr(vehicle, 'current_load') and hasattr(vehicle, 'max_capacity'):
                load_ratio = vehicle.current_load / vehicle.max_capacity
                if load_ratio > 0.5:  # 载重超过50%
                    priority -= 0.5  # 优先级提高
            
            # 考虑任务优先级
            if hasattr(vehicle, 'current_task') and vehicle.current_task:
                if hasattr(vehicle.current_task, 'priority'):
                    task_priority = vehicle.current_task.priority
                    # 将任务优先级转换为调整值 (任务优先级高，数值小)
                    priority_adjustment = max(0, 1 - task_priority/5.0)
                    priority -= priority_adjustment
            
            # 确保优先级在合理范围内
            return max(1.0, min(5.0, priority))
            
        except Exception as e:
            logging.error(f"计算车辆优先级出错: {str(e)}")
            return 5.0  # 出错时返回最低优先级
    
    def resolve_conflicts(self, paths: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
        """
        解决路径冲突
        
        使用优先级和重规划策略解决冲突
        
        Args:
            paths: 车辆ID到路径的映射字典
            
        Returns:
            Dict[str, List[Tuple]]: 无冲突的路径映射
        """
        if not paths:
            return paths
            
        try:
            start_time = time.time()
            new_paths = paths.copy()
            
            # 获取所有冲突
            conflicts = self.find_conflicts(paths)
            
            if not conflicts:
                return new_paths
                
            # 按冲突时间排序处理
            conflicts.sort(key=lambda x: x["time"])
            
            # 处理每个冲突
            for i, conflict in enumerate(conflicts):
                conflict_type = conflict["type"]
                conflict_time = conflict["time"]
                conflict_location = conflict["location"]
                vehicle1 = conflict["agent1"]
                vehicle2 = conflict["agent2"]
                
                logging.debug(f"处理冲突 {i+1}/{len(conflicts)}: 类型={conflict_type}, 时间={conflict_time}")
                
                # 获取车辆优先级 (数字越小优先级越高)
                prio1 = self._get_vehicle_priority(vehicle1)
                prio2 = self._get_vehicle_priority(vehicle2)
                
                logging.debug(f"车辆优先级: {vehicle1}={prio1:.1f}, {vehicle2}={prio2:.1f}")
                
                # 确定冲突解决策略
                if abs(prio1 - prio2) < 0.5:
                    # 优先级相近时，随机选择一辆车重规划
                    import random
                    vehicle_to_replan = vehicle1 if random.random() < 0.5 else vehicle2
                    logging.debug(f"优先级相近，随机选择车辆{vehicle_to_replan}进行重规划")
                elif prio1 < prio2:
                    # 低数字优先级更高，保留优先级高的车辆路径
                    vehicle_to_replan = vehicle2
                    logging.debug(f"车辆{vehicle1}优先级更高，为车辆{vehicle2}重规划")
                else:
                    vehicle_to_replan = vehicle1
                    logging.debug(f"车辆{vehicle2}优先级更高，为车辆{vehicle1}重规划")
                
                # 执行路径重规划
                new_path = self._replan_path(vehicle_to_replan, conflict_location, conflict_time)
                if new_path:
                    new_paths[vehicle_to_replan] = new_path
                    self.stats['conflicts_resolved'] += 1
                    
                    # 缓存已解决的冲突，避免重复处理
                    conflict_key = (vehicle1, vehicle2, conflict_time)
                    self.conflict_cache[conflict_key] = time.time()
                else:
                    logging.warning(f"车辆{vehicle_to_replan}路径重规划失败，冲突未解决")
            
            # 清理过期冲突缓存(超过30秒)
            current_time = time.time()
            expired_keys = [k for k, v in self.conflict_cache.items() if current_time - v > 30]
            for k in expired_keys:
                del self.conflict_cache[k]
            
            # 更新统计信息
            self.stats['total_resolution_time'] += (time.time() - start_time)
                
            return new_paths
            
        except Exception as e:
            logging.error(f"解决冲突过程发生异常: {str(e)}")
            # 出错时返回原始路径
            return paths