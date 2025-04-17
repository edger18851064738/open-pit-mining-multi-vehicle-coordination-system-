import sys
import os
import math
import time
import heapq
import logging
import threading
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Set, Callable
from functools import lru_cache
import matplotlib.pyplot as plt
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError
from algorithm.map_service import MapService

# 常量定义
MAX_CACHE_SIZE = 1000  # 缓存最大条目数
CACHE_EXPIRY = 600     # 缓存过期时间(秒)
EPSILON = 1e-5         # 浮点数比较精度

# 使用缓冲池优化Node对象创建，减少内存分配
class NodePool:
    """节点对象池，减少内存分配"""
    def __init__(self, max_size=10000):
        self.pool = []
        self.max_size = max_size
        self.in_use = set()
        self.lock = threading.RLock()
        
    def get_node(self, x, y, t=0):
        """从池中获取节点或创建新节点"""
        with self.lock:
            # 查找现有节点
            for node in self.pool:
                if node not in self.in_use:
                    node.x = x
                    node.y = y
                    node.t = t
                    self.in_use.add(node)
                    return node
            
            # 创建新节点
            node = Node(x, y, t)
            if len(self.pool) < self.max_size:
                self.pool.append(node)
            self.in_use.add(node)
            return node
            
    def release_node(self, node):
        """释放节点回池"""
        with self.lock:
            if node in self.in_use:
                self.in_use.remove(node)
                
    def release_all(self):
        """释放所有节点"""
        with self.lock:
            self.in_use.clear()

# 全局节点池
global_node_pool = NodePool()

class Node:
    """优化版三维路径节点（含时间维度）"""
    __slots__ = ('x', 'y', 't')  # 使用__slots__减少内存占用
    
    def __init__(self, x: int, y: int, t: int = 0):
        self.x = x
        self.y = y
        self.t = t
        
    def __eq__(self, other):
        if other is None:
            return False
        return self.x == other.x and self.y == other.y
        
    def __hash__(self):
        # 使用预计算的哈希值提高性能
        return hash((self.x, self.y))
    
    def __lt__(self, other):
        # 当f值相等时，按坐标排序
        return (self.x, self.y) < (other.x, other.y)
        
    def distance_to(self, other):
        """优化的距离计算，避免开方操作"""
        return abs(self.x - other.x) + abs(self.y - other.y)
        
    def euclidean_to(self, other):
        """欧式距离计算，仅在需要时使用"""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)
        
    def as_tuple(self):
        """返回坐标元组"""
        return (self.x, self.y)

class PriorityQueue:
    """优化的优先级队列实现"""
    def __init__(self):
        self.elements = []
        self.entry_finder = {}  # 映射项目到条目
        self.counter = 0        # 用于打破平局
        
    def push(self, item, priority):
        """添加新项或更新现有项的优先级"""
        if item in self.entry_finder:
            self.remove(item)
        self.counter += 1
        entry = [priority, self.counter, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.elements, entry)
        
    def remove(self, item):
        """标记现有项为已移除"""
        entry = self.entry_finder.pop(item)
        entry[-1] = None  # 避免内存泄漏
        
    def pop(self):
        """移除并返回最小优先级的项"""
        while self.elements:
            priority, count, item = heapq.heappop(self.elements)
            if item is not None:
                del self.entry_finder[item]
                return item
        raise KeyError('从空队列弹出')
        
    def empty(self):
        """检查队列是否为空"""
        return not self.entry_finder

class SpatialIndex:
    """空间索引结构，加速空间查询"""
    def __init__(self, cell_size=10):
        self.cell_size = cell_size
        self.grid = defaultdict(set)
        
    def add_point(self, point):
        """添加点到索引"""
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        self.grid[(cell_x, cell_y)].add(point)
        
    def add_points(self, points):
        """批量添加点"""
        for point in points:
            self.add_point(point)
            
    def query_point(self, point, radius=0):
        """查询点附近的点"""
        result = set()
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        
        # 计算需要检查的单元格范围
        cell_radius = max(1, int(radius // self.cell_size) + 1)
        
        # 检查相邻单元格
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.grid:
                    for p in self.grid[cell_key]:
                        if radius == 0 or math.dist(point, p) <= radius:
                            result.add(p)
                            
        return result
        
    def clear(self):
        """清空索引"""
        self.grid.clear()

class PathCache:
    """高性能路径缓存系统"""
    def __init__(self, max_size=MAX_CACHE_SIZE, expiry=CACHE_EXPIRY):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.expiry = expiry
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key):
        """获取缓存项"""
        with self.lock:
            now = time.time()
            if key in self.cache:
                # 检查是否过期
                if now - self.timestamps[key] <= self.expiry:
                    # 更新时间戳
                    self.timestamps[key] = now
                    self.hits += 1
                    return self.cache[key].copy()  # 返回副本避免修改缓存
                else:
                    # 过期删除
                    del self.cache[key]
                    del self.timestamps[key]
            
            self.misses += 1
            return None
            
    def put(self, key, value):
        """添加缓存项"""
        with self.lock:
            now = time.time()
            
            # 检查容量
            if len(self.cache) >= self.max_size:
                # 删除最旧的项
                oldest_key = min(self.timestamps, key=self.timestamps.get)
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                
            # 添加新项
            self.cache[key] = value.copy()  # 存储副本避免外部修改
            self.timestamps[key] = now
            
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            
    def get_stats(self):
        """获取缓存统计信息"""
        with self.lock:
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses
            }

class HybridPathPlanner:
    """高性能矿山场景路径规划器"""
    
    def __init__(self, map_service: MapService):
        # 基础属性
        self.map_service = map_service
        self.dispatch = None  # 由DispatchSystem设置
        
        # 系统状态
        self.load_time = time.time()
        self.planning_count = 0
        self.total_planning_time = 0
        
        # 空间数据
        self.obstacle_index = SpatialIndex(cell_size=20)  # 障碍物空间索引
        self.obstacle_grids = set()                       # 障碍点集合
        self.haul_roads = set()                          # 运输道路集合
        self.dynamic_obstacles = set()                   # 动态障碍物集合
        
        # 预约系统
        self.reservation_table = {}                      # 时间窗口预约表
        self.reservation_lock = threading.RLock()        # 预约表锁
        
        # 缓存系统
        self.path_cache = PathCache(max_size=1000, expiry=600)  # 路径缓存
        self.terrain_cache = {}                               # 地形数据缓存
        
        # 算法参数
        self.conflict_check_interval = 0.5  # 冲突检测间隔(秒)
        
        # 加载配置
        try:
            self._load_mine_config()
        except Exception as e:
            logging.warning(f"配置加载失败，使用默认值: {str(e)}")
            self.grid_size = 20.0
            self.max_grade = 15.0
            self.min_turn_radius = 15.0
            
        # 预计算方向
        self.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角线
        ]
        
        # 优化性能的预计算值
        self.diagonal_cost = 1.414  # √2
            
    def _load_mine_config(self):
        """加载配置（使用简化逻辑避免依赖问题）"""
        # 使用默认值
        self.grid_size = 20.0
        self.max_grade = 15.0
        self.min_turn_radius = 15.0
        
        # 尝试从配置文件加载
        try:
            import configparser
            config_path = os.path.join(PROJECT_ROOT, 'config.ini')
            if os.path.exists(config_path):
                config = configparser.ConfigParser()
                config.read(config_path)
                if 'MAP' in config:
                    self.grid_size = config.getfloat('MAP', 'grid_size', fallback=20.0)
                    self.max_grade = config.getfloat('MAP', 'max_grade', fallback=15.0)
                    self.min_turn_radius = config.getfloat('MAP', 'min_turn_radius', fallback=15.0)
        except Exception as e:
            logging.warning(f"配置文件加载失败: {str(e)}")
            
    def plan_path(self, start: Tuple, end: Tuple, vehicle=None) -> List[Tuple]:
        """高性能路径规划入口方法"""
        # 性能计数
        start_time = time.time()
        self.planning_count += 1
        
        try:
            # 验证输入
            if not start or not end:
                return []
                
            # 检查起点和终点是否相同
            if math.isclose(start[0], end[0], abs_tol=EPSILON) and math.isclose(start[1], end[1], abs_tol=EPSILON):
                return [start]
                
            # 缓存键生成 (带车辆特征，如果适用)
            cache_key = None
            if vehicle:
                # 包含车辆特性的缓存键
                cache_key = (
                    "path",
                    start,
                    end,
                    getattr(vehicle, 'turning_radius', 0), 
                    getattr(vehicle, 'min_hardness', 0),
                    getattr(vehicle, 'current_load', 0)
                )
            else:
                # 基础缓存键
                cache_key = ("base_path", start, end)
                
            # 检查缓存
            cached_path = self.path_cache.get(cache_key)
            if cached_path:
                logging.debug(f"使用缓存路径: {start} -> {end}")
                return cached_path
                
            # 执行路径规划（根据是否有车辆参数选择算法）
            path = []
            if vehicle:
                path = self._optimized_astar(start, end, vehicle)
            else:
                path = self._fast_astar(start, end)
                
            # 检查结果有效性
            if not path or len(path) < 2:
                logging.warning(f"路径规划失败，使用直线路径: {start} -> {end}")
                path = [start, end]
                
            # 路径优化（平滑处理）
            if len(path) > 2:
                path = self._optimized_smooth(path)
                
            # 缓存结果
            self.path_cache.put(cache_key, path)
            
            # 记录性能指标
            elapsed = time.time() - start_time
            self.total_planning_time += elapsed
            
            if elapsed > 0.1:  # 记录较慢的规划
                logging.debug(f"路径规划耗时较长: {elapsed:.3f}秒 ({start} -> {end})")
                
            return path
            
        except Exception as e:
            logging.error(f"路径规划异常: {str(e)}")
            elapsed = time.time() - start_time
            self.total_planning_time += elapsed
            # 出错时使用简单直线路径
            return [start, end]
            
    def _fast_astar(self, start: Tuple, end: Tuple) -> List[Tuple]:
        """优化的A*算法 - 不考虑车辆约束时的快速版本"""
        # 创建起点和终点节点
        start_node = global_node_pool.get_node(*start)
        end_node = global_node_pool.get_node(*end)
        
        # 检查终点是否为障碍物
        if self._is_obstacle_fast(end):
            logging.warning(f"终点{end}是障碍物，无法规划路径")
            global_node_pool.release_node(start_node)
            global_node_pool.release_node(end_node)
            return []
            
        # 优化的开放列表（使用自定义优先队列）
        open_queue = PriorityQueue()
        open_queue.push(start_node, 0)
        
        # 使用集合加速查找
        closed_set = set()
        
        # 距离和估计成本字典
        g_score = {start_node: 0}
        f_score = {start_node: self._fast_heuristic(start_node, end_node)}
        
        # 父节点字典 - 使用字典加速查找
        came_from = {}
        
        # A*主循环 - 使用更高效的循环判断
        while not open_queue.empty():
            current = open_queue.pop()
            
            # 到达目标点
            if self._is_same_point(current, end_node):
                path = self._reconstruct_path(came_from, current)
                # 释放节点回池
                global_node_pool.release_all()
                return path
                
            # 标记为已访问
            closed_set.add(current)
            
            # 邻居遍历 - 使用预计算方向数组
            for dx, dy in self.directions:
                # 快速创建邻居节点
                nx, ny = current.x + dx, current.y + dy
                neighbor = global_node_pool.get_node(nx, ny)
                
                # 优化的跳过逻辑
                if neighbor in closed_set:
                    global_node_pool.release_node(neighbor)  # 释放未使用的节点
                    continue
                    
                # 快速障碍物检查
                if self._is_obstacle_fast((nx, ny)):
                    global_node_pool.release_node(neighbor)
                    continue
                    
                # 直线检查优化 - 仅在必要时执行
                if dx != 0 and dy != 0 and self._is_diagonal_blocked(current, neighbor):
                    global_node_pool.release_node(neighbor)
                    continue
                
                # 移动成本计算优化
                move_cost = 1.0 if dx == 0 or dy == 0 else self.diagonal_cost
                
                # 快速运输道路检查
                if (nx, ny) in self.haul_roads:
                    move_cost *= 0.6
                    
                # 计算新路径成本
                tentative_g = g_score[current] + move_cost
                
                # 优化的路径更新逻辑
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # 更新路径信息
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_value = tentative_g + self._fast_heuristic(neighbor, end_node)
                    f_score[neighbor] = f_value  # 更新f值缓存
                    
                    # 更新队列
                    open_queue.push(neighbor, f_value)
                else:
                    # 不需要此节点
                    global_node_pool.release_node(neighbor)
                    
        # 路径未找到
        global_node_pool.release_all()
        return []
        
    def _optimized_astar(self, start: Tuple, end: Tuple, vehicle) -> List[Tuple]:
        """优化的A*算法 - 考虑车辆约束的完整版本"""
        # 创建起点和终点节点
        start_node = global_node_pool.get_node(*start)
        end_node = global_node_pool.get_node(*end)
        
        # 获取车辆属性 - 预先提取避免重复访问
        turning_radius = getattr(vehicle, 'turning_radius', 10.0)
        min_hardness = getattr(vehicle, 'min_hardness', 2.5)
        current_load = getattr(vehicle, 'current_load', 0)
        
        # 检查终点是否为障碍物
        if self._is_obstacle_fast(end):
            logging.warning(f"终点{end}是障碍物，无法规划路径")
            global_node_pool.release_all()
            return []
            
        # 高效的优先队列和数据结构
        open_queue = PriorityQueue()
        open_queue.push(start_node, 0)
        closed_set = set()
        g_score = {start_node: 0}
        f_score = {start_node: self._fast_heuristic(start_node, end_node)}
        came_from = {}
        
        # 算法主循环优化
        while not open_queue.empty():
            current = open_queue.pop()
            
            # 快速目标检查
            if self._is_same_point(current, end_node):
                path = self._reconstruct_path(came_from, current)
                global_node_pool.release_all()
                return path
                
            # 避免重复处理
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # 遍历所有可能方向
            for dx, dy in self.directions:
                nx, ny = current.x + dx, current.y + dy
                neighbor = global_node_pool.get_node(nx, ny)
                
                # 优化的筛选逻辑（先进行快速检查）
                if neighbor in closed_set:
                    global_node_pool.release_node(neighbor)
                    continue
                    
                if self._is_obstacle_fast((nx, ny)):
                    global_node_pool.release_node(neighbor)
                    continue
                    
                if dx != 0 and dy != 0 and self._is_diagonal_blocked(current, neighbor):
                    global_node_pool.release_node(neighbor)
                    continue
                
                # 车辆特定约束检查（仅在通过基本检查后执行）
                terrain_hardness = self._get_cached_terrain_hardness(nx, ny)
                if terrain_hardness < min_hardness:
                    global_node_pool.release_node(neighbor)
                    continue
                    
                # 转弯半径检查（仅在必要时）
                if hasattr(vehicle, 'last_position') and vehicle.last_position:
                    if not self._check_turn_radius(current, neighbor, vehicle):
                        global_node_pool.release_node(neighbor)
                        continue
                
                # 计算移动成本（考虑车辆特性）
                move_cost = self._calculate_vehicle_move_cost(
                    current, neighbor, 
                    terrain_hardness, current_load,
                    dx, dy
                )
                
                # 路径更新
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_value = tentative_g + self._enhanced_heuristic(neighbor, end_node, vehicle)
                    f_score[neighbor] = f_value
                    open_queue.push(neighbor, f_value)
                else:
                    global_node_pool.release_node(neighbor)
                    
        # 未找到路径
        global_node_pool.release_all()
        return []
        
    def _is_same_point(self, a, b, tolerance=EPSILON):
        """优化的点相等判断"""
        return abs(a.x - b.x) < tolerance and abs(a.y - b.y) < tolerance
        
    def _is_obstacle_fast(self, point: Tuple) -> bool:
        """高性能障碍物检查"""
        return point in self.obstacle_grids or point in self.dynamic_obstacles
        
    def _is_diagonal_blocked(self, current: Node, neighbor: Node) -> bool:
        """检查对角线移动是否被阻塞"""
        # 直接检查两个直角路径是否都被阻塞
        corner1 = (current.x, neighbor.y)
        corner2 = (neighbor.x, current.y)
        return self._is_obstacle_fast(corner1) or self._is_obstacle_fast(corner2)
        
    def _fast_heuristic(self, a: Node, b: Node) -> float:
        """快速启发式函数 - 使用曼哈顿+对角线距离"""
        dx = abs(b.x - a.x)
        dy = abs(b.y - a.y)
        return (dx + dy) + (self.diagonal_cost - 2) * min(dx, dy)
        
    def _enhanced_heuristic(self, a: Node, b: Node, vehicle) -> float:
        """增强型启发式函数 - 考虑车辆因素"""
        # 基础启发式值
        base_h = self._fast_heuristic(a, b)
        
        # 考虑载重
        load_factor = 1.0 + getattr(vehicle, 'current_load', 0) / 50000
        
        # 考虑地形 - 使用简单缓存避免重复计算
        terrain_factor = 1.0
        terrain_key = (a.x, a.y)
        if terrain_key in self.terrain_cache:
            terrain_hardness = self.terrain_cache[terrain_key]
            min_hardness = getattr(vehicle, 'min_hardness', 2.5)
            terrain_factor = max(1.0, min_hardness / terrain_hardness)
        
        # 考虑运输道路 - 快速查找
        road_factor = 0.8 if (a.x, a.y) in self.haul_roads else 1.0
        
        return base_h * load_factor * terrain_factor * road_factor
        
    def _get_cached_terrain_hardness(self, x: int, y: int) -> float:
        """使用缓存获取地形硬度"""
        terrain_key = (x, y)
        if terrain_key not in self.terrain_cache:
            # 从地图服务获取硬度并缓存
            try:
                hardness = self.map_service.get_terrain_hardness(x, y)
                self.terrain_cache[terrain_key] = hardness
                return hardness
            except:
                # 使用默认值
                self.terrain_cache[terrain_key] = 3.0
                return 3.0
        return self.terrain_cache[terrain_key]
        
    def _calculate_vehicle_move_cost(self, current: Node, neighbor: Node, 
                                    terrain_hardness: float, current_load: float,
                                    dx: int, dy: int) -> float:
        """优化的车辆移动成本计算"""
        # 基础移动成本 - 使用预计算值
        base_cost = 1.0 if dx == 0 or dy == 0 else self.diagonal_cost
        
        # 载重影响 - 使用简化计算
        load_factor = 1.0 + current_load / 100.0
        
        # 地形影响 - 使用预先获取的硬度
        terrain_factor = max(1.0, 3.0 / terrain_hardness)
        
        # 运输道路优惠 - 快速查找
        road_discount = 0.6 if (neighbor.x, neighbor.y) in self.haul_roads else 1.0
        
        # 返回综合成本
        return base_cost * load_factor * terrain_factor * road_discount
        
    def _check_turn_radius(self, current: Node, neighbor: Node, vehicle) -> bool:
        """优化的转弯半径检查"""
        if not hasattr(vehicle, 'last_position') or not vehicle.last_position:
            return True
            
        # 计算角度变化
        if not hasattr(vehicle, 'steering_angle') or vehicle.steering_angle <= 0:
            return True
            
        # 计算向量
        v1 = (current.x - vehicle.last_position.x, current.y - vehicle.last_position.y)
        v2 = (neighbor.x - current.x, neighbor.y - current.y)
        
        # 快速检查 - 如果两个向量方向相同，不需要转弯
        if v1[0] == v2[0] and v1[1] == v2[1]:
            return True
            
        # 计算转弯半径 - 使用简化计算
        try:
            angle_rad = math.acos(
                (v1[0]*v2[0] + v1[1]*v2[1]) / 
                (math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2))
            )
            turn_radius = 1 / math.sin(angle_rad/2) if angle_rad > 0 else float('inf')
            return turn_radius >= self.min_turn_radius
        except:
            # 计算出错默认通过
            return True
        
    def _reconstruct_path(self, came_from: Dict, current: Node) -> List[Tuple]:
        """重建路径 - 优化的实现"""
        path = [(current.x, current.y)]
        while current in came_from:
            current = came_from[current]
            path.append((current.x, current.y))
        return list(reversed(path))
        
    def _optimized_smooth(self, path: List[Tuple]) -> List[Tuple]:
        """优化的路径平滑算法"""
        if len(path) <= 2:
            return path
            
        # 使用道格拉斯-普克算法进行平滑
        result = self._douglas_peucker(path, 2.0)
        
        # 如果需要，可以进一步减少点数
        if len(result) > 20:
            # 基于距离的采样
            smoothed = [result[0]]  # 保留起点
            distance_threshold = 10.0
            last_point = result[0]
            
            for i in range(1, len(result) - 1):
                curr_dist = math.dist(last_point, result[i])
                if curr_dist >= distance_threshold:
                    smoothed.append(result[i])
                    last_point = result[i]
                    
            smoothed.append(result[-1])  # 保留终点
            return smoothed
            
        return result
        
    def _douglas_peucker(self, points: List[Tuple], epsilon: float) -> List[Tuple]:
        """道格拉斯-普克算法实现"""
        if len(points) <= 2:
            return points
            
        # 找到最远点
        dmax = 0
        index = 0
        start, end = points[0], points[-1]
        
        for i in range(1, len(points) - 1):
            d = self._perpendicular_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d
                
        if dmax > epsilon:
            # 递归处理
            rec1 = self._douglas_peucker(points[:index+1], epsilon)
            rec2 = self._douglas_peucker(points[index:], epsilon)
            
            # 合并结果，避免重复点
            return rec1[:-1] + rec2
        else:
            return [points[0], points[-1]]
            
    def _perpendicular_distance(self, point: Tuple, line_start: Tuple, line_end: Tuple) -> float:
        """计算点到线段的垂直距离"""
        if line_start == line_end:
            return math.dist(point, line_start)
            
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 线性代数方法，计算点到直线距离
        num = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
        den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        return num / den if den > 0 else 0
        
    def mark_obstacle_area(self, polygons: List[List[Tuple]]):
        """批量标记障碍物区域 - 优化实现"""
        # 清除旧索引
        self.obstacle_index.clear()
        self.obstacle_grids.clear()
        
        # 处理每个多边形
        for polygon in polygons:
            # 快速计算边界框
            min_x = min(p[0] for p in polygon)
            max_x = max(p[0] for p in polygon)
            min_y = min(p[1] for p in polygon)
            max_y = max(p[1] for p in polygon)
            
            # 检查边界框内点
            for x in range(int(min_x), int(max_x+1)):
                for y in range(int(min_y), int(max_y+1)):
                    point = (x, y)
                    if self._point_in_polygon(point, polygon):
                        self.obstacle_grids.add(point)
                        
        # 更新空间索引
        self.obstacle_index.add_points(self.obstacle_grids)
        logging.debug(f"已标记{len(self.obstacle_grids)}个障碍物点")
        
    def _point_in_polygon(self, point: Tuple, polygon: List[Tuple]) -> bool:
        """射线法判断点是否在多边形内 - 优化实现"""
        x, y = point
        n = len(polygon)
        inside = False
        
        # 首先快速检查点是否在多边形顶点上
        if point in polygon:
            return True
            
        # 使用射线法检查
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i+1)%n]
            
            # 快速分支判断，减少计算量
            if ((p1[1] > y) != (p2[1] > y)) and (x < (p2[0]-p1[0])*(y-p1[1])/(p2[1]-p1[1]) + p1[0]):
                inside = not inside
                
        return inside
        
    def add_dynamic_obstacle(self, point: Tuple, radius: float = 5.0):
        """添加动态障碍物"""
        x, y = point
        # 使用预计算的圆形模板加速
        radius_int = int(radius)
        for dx in range(-radius_int, radius_int + 1):
            for dy in range(-radius_int, radius_int + 1):
                if dx*dx + dy*dy <= radius*radius:
                    self.dynamic_obstacles.add((x + dx, y + dy))
                    
    def remove_dynamic_obstacle(self, point: Tuple, radius: float = 5.0):
        """移除动态障碍物"""
        # 使用空间索引加速查询
        to_remove = set()
        x, y = point
        radius_int = int(radius)
        
        for dx in range(-radius_int, radius_int + 1):
            for dy in range(-radius_int, radius_int + 1):
                check_point = (x + dx, y + dy)
                if dx*dx + dy*dy <= radius*radius and check_point in self.dynamic_obstacles:
                    to_remove.add(check_point)
                    
        self.dynamic_obstacles -= to_remove
        
    def check_path_conflict(self, path: List[Tuple], vehicle_id: str) -> bool:
        """检查路径冲突 - 优化实现"""
        if not path or len(path) < 2:
            return False
            
        try:
            with self.reservation_lock:
                # 首先检查路径点是否在障碍物中（批量快速检查）
                for point in path:
                    if point in self.obstacle_grids or point in self.dynamic_obstacles:
                        return True
                        
                # 检查路径段冲突 - 使用线段相交测试
                for i in range(len(path) - 1):
                    segment = (path[i], path[i+1])
                    
                    # 检查线段预约
                    if segment in self.reservation_table and self.reservation_table[segment] != vehicle_id:
                        return True
                        
                    # 使用Bresenham算法检查线段上的点
                    line_points = self._bresenham_line(segment[0], segment[1])
                    for point in line_points:
                        if point in self.obstacle_grids or point in self.dynamic_obstacles:
                            return True
                            
                return False
        except Exception as e:
            logging.error(f"路径冲突检查异常: {str(e)}")
            return False
            
    def _bresenham_line(self, start: Tuple, end: Tuple) -> List[Tuple]:
        """优化的Bresenham直线算法"""
        # 使用内置GeoUtils实现
        return GeoUtils.bresenham_line(start, end)
        
    def reserve_path(self, path: List[Tuple], vehicle_id: str):
        """为路径预约时间窗口"""
        if not path or len(path) < 2:
            return
            
        with self.reservation_lock:
            # 清除该车辆之前的预约
            self.clear_path_reservation(vehicle_id)
            
            # 添加新预约
            for i in range(len(path) - 1):
                segment = (path[i], path[i+1])
                self.reservation_table[segment] = vehicle_id
                
    def clear_path_reservation(self, vehicle_id: str):
        """清除车辆的路径预约"""
        with self.reservation_lock:
            segments_to_remove = [s for s, v in self.reservation_table.items() if v == vehicle_id]
            for segment in segments_to_remove:
                if segment in self.reservation_table:
                    del self.reservation_table[segment]
                    
    def clear_caches(self):
        """清除所有缓存数据"""
        self.path_cache.clear()
        self.terrain_cache.clear()
        global_node_pool.release_all()
        
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        runtime = time.time() - self.load_time
        avg_planning_time = self.total_planning_time / self.planning_count if self.planning_count > 0 else 0
        
        return {
            'runtime': f"{runtime:.1f}秒",
            'planning_count': self.planning_count,
            'avg_planning_time': f"{avg_planning_time*1000:.2f}毫秒",
            'cache_stats': self.path_cache.get_stats(),
            'obstacle_count': len(self.obstacle_grids),
            'dynamic_obstacles': len(self.dynamic_obstacles),
            'reservations': len(self.reservation_table)
        }
        
    def visualize_path(self, path: List[Tuple], title: str = None):
        """路径可视化调试工具"""
        plt.figure(figsize=(10, 10))
        
        # 绘制障碍物
        obstacles_x = [p[0] for p in self.obstacle_grids]
        obstacles_y = [p[1] for p in self.obstacle_grids]
        plt.scatter(obstacles_x, obstacles_y, c='gray', s=5, alpha=0.5, label='障碍物')
        
        # 绘制动态障碍物
        if self.dynamic_obstacles:
            dyn_obstacles_x = [p[0] for p in self.dynamic_obstacles]
            dyn_obstacles_y = [p[1] for p in self.dynamic_obstacles]
            plt.scatter(dyn_obstacles_x, dyn_obstacles_y, c='red', s=10, alpha=0.7, label='动态障碍物')
        
        # 绘制运输道路
        if self.haul_roads:
            roads_x = [p[0] for p in self.haul_roads]
            roads_y = [p[1] for p in self.haul_roads]
            plt.scatter(roads_x, roads_y, c='green', s=5, alpha=0.3, label='运输道路')
            
        # 绘制路径
        if path and len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='路径')
            
            # 标记起点和终点
            plt.plot(path_x[0], path_y[0], 'go', markersize=10, label='起点')
            plt.plot(path_x[-1], path_y[-1], 'ro', markersize=10, label='终点')
            
        # 设置图形属性
        if title:
            plt.title(title)
        else:
            plt.title(f"路径规划 ({len(path)}个点)")
            
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.show()
        
    def release_resources(self):
        """释放所有资源"""
        self.path_cache.clear()
        self.terrain_cache.clear()
        global_node_pool.release_all()
        self.obstacle_grids.clear()
        self.dynamic_obstacles.clear()
        self.haul_roads.clear()
        self.reservation_table.clear()
        logging.info("已释放所有路径规划资源")