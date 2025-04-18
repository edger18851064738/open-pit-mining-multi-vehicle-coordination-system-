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
from matplotlib import rcParams
import queue
rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 使用系统默认中文字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
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
    """改进的节点池实现，保证线程安全和对象一致性"""
    
    def __init__(self, max_size=10000):
        self.available = []  # 可用节点池
        self.max_size = max_size
        self.lock = threading.RLock()
        
    def get_node(self, x, y, t=0):
        """获取节点，优先从池中获取"""
        with self.lock:
            if self.available:
                # 创建新节点而不是修改现有节点
                # 这避免了修改后哈希值变化的问题
                node = Node(x, y, t)
                return node
            else:
                # 创建新节点
                node = Node(x, y, t)
                if len(self.available) < self.max_size:
                    self.available.append(None)  # 仅占位，不复用对象
                return node
                
    def release_node(self, node):
        """释放节点，实际上不再复用"""
        # 此方法只为兼容性保留，实际不再复用节点
        pass
        
    def release_all(self):
        """释放所有节点"""
        with self.lock:
            self.available.clear()
# 全局节点池
global_node_pool = NodePool()

class Node:
    """改进的路径节点类，增强稳定性和错误处理"""
    __slots__ = ('x', 'y', 't', '_hash')
    
    def __init__(self, x, y, t=0):
        try:
            self.x = float(x)
            self.y = float(y)
            self.t = float(t)
            # 预计算哈希值，确保不变性
            self._hash = hash((round(self.x, 6), round(self.y, 6)))
        except (TypeError, ValueError) as e:
            logging.error(f"创建Node失败: {str(e)}, 使用默认值(0,0,0)")
            self.x = 0.0
            self.y = 0.0
            self.t = 0.0
            self._hash = hash((0, 0))
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        # 使用更精确的比较
        return (abs(self.x - other.x) < 1e-9 and 
                abs(self.y - other.y) < 1e-9)
    
    def __hash__(self):
        # 返回预计算的哈希值，确保不变性
        return self._hash
    
    def __repr__(self):
        return f"Node({self.x:.6f}, {self.y:.6f}, {self.t:.1f})"
    
    def as_tuple(self):
        """获取坐标元组"""
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
        self._in_original_call = False  # 标记是否是原始方法调用
        
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
      
    def plan_with_timeout(self, start_point, end_point, vehicle=None, timeout=5.0):
        """带超时的路径规划函数，使用安全的A*算法"""
        # 使用队列安全地传递结果
        result_queue = queue.Queue()
        
        def planning_worker():
            try:
                # 直接使用安全的A*算法
                path_result = self.simple_astar(start_point, end_point, vehicle)
                result_queue.put(path_result)
            except Exception as e:
                # 捕获异常并放入队列
                logging.error(f"路径规划出错: {str(e)}")
                result_queue.put(Exception(f"路径规划出错: {str(e)}"))
        
        # 启动规划线程
        planning_thread = threading.Thread(target=planning_worker, daemon=True)
        start_time = time.time()
        planning_thread.start()
        
        # 等待线程完成或超时
        planning_thread.join(timeout)
        
        # 计算耗时
        elapsed = time.time() - start_time
        
        # 获取结果
        if planning_thread.is_alive():
            # 超时情况，生成避开障碍物的路径
            logging.warning(f"路径规划超时 ({timeout}秒)，生成备用路径")
            path = self._generate_fallback_path(start_point, end_point)
            return path, elapsed
        else:
            try:
                # 检查队列中的结果
                result = result_queue.get(block=False)
                if isinstance(result, Exception):
                    # 出错，生成备用路径
                    logging.warning(f"路径规划失败: {str(result)}，生成备用路径")
                    path = self._generate_fallback_path(start_point, end_point)
                    return path, elapsed
                elif not result or len(result) < 2:
                    # 无效结果，生成备用路径
                    logging.warning(f"无效路径，生成备用路径")
                    path = self._generate_fallback_path(start_point, end_point)
                    return path, elapsed
                else:
                    # 规划成功
                    logging.info(f"路径规划成功！路径包含 {len(result)} 个点，用时 {elapsed:.3f} 秒")
                    return result, elapsed
            except queue.Empty:
                # 队列为空，生成备用路径
                logging.warning(f"路径规划过程异常终止，生成备用路径")
                path = self._generate_fallback_path(start_point, end_point)
                return path, elapsed

    def _generate_fallback_path(self, start, end):
        """Generate a reliable fallback path, ensuring start and end are valid tuples"""
        try:
            # Ensure start and end are tuples with valid coordinates
            start = tuple(start) if isinstance(start, (list, np.ndarray)) else start
            end = tuple(end) if isinstance(end, (list, np.ndarray)) else end
            
            if hasattr(start, 'as_tuple'):
                start = start.as_tuple()
            if hasattr(end, 'as_tuple'):
                end = end.as_tuple()
                
            # Make sure we have valid coordinates
            if not isinstance(start, tuple) or len(start) < 2:
                start = (0, 0)
            if not isinstance(end, tuple) or len(end) < 2:
                end = (100, 100)
                
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            # Check for obstacles on direct path
            line_points = []
            try:
                line_points = self._bresenham_line(start, end)
            except Exception as e:
                logging.warning(f"Bresenham line calculation failed: {str(e)}")
                
            obstacles_on_path = []
            for p in line_points:
                if self._is_obstacle_fast(p):
                    obstacles_on_path.append(p)
                    if len(obstacles_on_path) >= 3:  # Only need to check a few points
                        break
            
            if not obstacles_on_path:
                # No obstacles, use direct path
                return [start, end]
            
            # Create a path with waypoints to navigate around obstacles
            path = [start]
            
            # Decide detour strategy based on path direction
            if abs(dx) > abs(dy):
                # Horizontal path is longer, use vertical detour
                offset = max(30, abs(dx) * 0.2)  # At least 30 units
                
                # Generate two midpoints for smoother navigation
                mid1 = (start[0] + dx * 0.3, start[1] + offset)
                mid2 = (start[0] + dx * 0.7, start[1] + offset)
                
                # Check if midpoints are obstacles
                if self._is_obstacle_fast(mid1) or self._is_obstacle_fast(mid2):
                    # Try opposite direction
                    mid1 = (start[0] + dx * 0.3, start[1] - offset)
                    mid2 = (start[0] + dx * 0.7, start[1] - offset)
            else:
                # Vertical path is longer, use horizontal detour
                offset = max(30, abs(dy) * 0.2)  # At least 30 units
                
                # Generate two midpoints for smoother navigation
                mid1 = (start[0] + offset, start[1] + dy * 0.3)
                mid2 = (start[0] + offset, start[1] + dy * 0.7)
                
                # Check if midpoints are obstacles
                if self._is_obstacle_fast(mid1) or self._is_obstacle_fast(mid2):
                    # Try opposite direction
                    mid1 = (start[0] - offset, start[1] + dy * 0.3)
                    mid2 = (start[0] - offset, start[1] + dy * 0.7)
            
            # Final check for midpoints
            if self._is_obstacle_fast(mid1) or self._is_obstacle_fast(mid2):
                # Both directions have obstacles, use a single random midpoint with larger offset
                larger_offset = max(50, max(abs(dx), abs(dy)) * 0.4)
                mid_point = (
                    (start[0] + end[0]) / 2 + random.uniform(-larger_offset, larger_offset),
                    (start[1] + end[1]) / 2 + random.uniform(-larger_offset, larger_offset)
                )
                path.append(mid_point)
            else:
                # Use the two midpoints
                path.extend([mid1, mid2])
            
            # Always include end point
            path.append(end)
            return path
        except Exception as e:
            logging.error(f"Fallback path generation error: {str(e)}")
            # Absolute last resort - direct connection
            return [start if isinstance(start, tuple) else (0, 0), 
                    end if isinstance(end, tuple) else (100, 100)]
    def original_plan_path(self, start: Tuple, end: Tuple, vehicle=None) -> List[Tuple]:
        """原始路径规划方法，避免递归调用"""
        try:
            # 确保起点和终点是元组
            if hasattr(start, 'as_tuple'):
                start = start.as_tuple()
            if hasattr(end, 'as_tuple'):
                end = end.as_tuple()
            
            # 执行路径规划（根据是否有车辆参数选择算法）
            path = []
            if vehicle:
                path = self._optimized_astar(start, end, vehicle)
            else:
                path = self._fast_astar(start, end)
                
            # 检查结果有效性
            if not path or len(path) < 2:
                logging.warning(f"路径规划失败，使用备用路径: {start} -> {end}")
                path = self._generate_fallback_path(start, end)
                
            # 路径优化（平滑处理）
            if len(path) > 2:
                try:
                    path = self._optimized_smooth(path)
                except Exception as e:
                    logging.warning(f"路径平滑失败: {str(e)}")
                    
            return path
        except Exception as e:
            logging.error(f"原始路径规划出错: {str(e)}")
            # 使用备用路径
            return self._generate_fallback_path(start, end)
    def simple_astar(self, start: Tuple, end: Tuple, vehicle=None) -> List[Tuple]:
        """Simple and robust A* algorithm that doesn't depend on Node class"""
        try:
            # Ensure start and end are tuples with numeric coordinates
            start_x, start_y = float(start[0]), float(start[1])
            end_x, end_y = float(end[0]), float(end[1])
            
            # Check if start or end point is an obstacle
            if self._is_obstacle_fast(start):
                logging.warning(f"Start point {start} is an obstacle, adjusting...")
                # Try to adjust start point
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        new_start = (start_x + dx, start_y + dy)
                        if not self._is_obstacle_fast(new_start):
                            start_x, start_y = new_start
                            break
            
            if self._is_obstacle_fast(end):
                logging.warning(f"End point {end} is an obstacle, adjusting...")
                # Try to adjust end point
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        new_end = (end_x + dx, end_y + dy)
                        if not self._is_obstacle_fast(new_end):
                            end_x, end_y = new_end
                            break
            
            # Heuristic function: Manhattan distance
            def heuristic(x, y):
                return abs(x - end_x) + abs(y - end_y)
            
            # Priority queue (min-heap) with (f-value, coordinates)
            open_list = []
            heapq.heappush(open_list, (heuristic(start_x, start_y), (start_x, start_y)))
            
            # Track visited points and parent relationships
            closed_set = set()
            came_from = {}
            g_score = {(start_x, start_y): 0}
            
            # Movement directions: cardinal and diagonal
            directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # Cardinal
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal
            ]
            
            # Main search loop
            while open_list:
                # Get point with lowest f-value
                _, current = heapq.heappop(open_list)
                
                # Goal check: if we're close enough to the goal
                if abs(current[0] - end_x) < 3 and abs(current[1] - end_y) < 3:
                    # Reconstruct path
                    path = [end]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    path.reverse()
                    # Make sure end point is included
                    if path[-1] != end:
                        path.append(end)
                    return path
                
                # Mark as visited
                closed_set.add(current)
                
                # Check all neighbors
                for dx, dy in directions:
                    # Calculate neighbor coordinates
                    nx, ny = current[0] + dx, current[1] + dy
                    neighbor = (nx, ny)
                    
                    # Skip if already visited
                    if neighbor in closed_set:
                        continue
                    
                    # Skip obstacles
                    if self._is_obstacle_fast(neighbor):
                        continue
                    
                    # Calculate movement cost (higher for diagonal moves)
                    move_cost = 1.4 if dx != 0 and dy != 0 else 1.0
                    tentative_g = g_score[current] + move_cost
                    
                    # Update path if it's better
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_value = tentative_g + heuristic(nx, ny)
                        
                        # Add to open list for exploration
                        heapq.heappush(open_list, (f_value, neighbor))
            
            # If no path found, generate a path that avoids obstacles
            logging.warning(f"No path found from {start} to {end}, using zig-zag path")
            
            # Generate a multi-point path to navigate around obstacles
            path = [start]
            
            # Check for obstacles on direct path
            line_points = self._bresenham_line(start, end)
            obstacles_on_path = [p for p in line_points if self._is_obstacle_fast(p)]
            
            if obstacles_on_path:
                # Add mid-points to navigate around obstacles
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                
                # Choose detour direction based on predominant movement
                if abs(dx) > abs(dy):
                    # Horizontal movement predominant, try vertical detour
                    mid1 = (start[0] + dx/3, start[1] + 40 if dy >= 0 else start[1] - 40)
                    mid2 = (start[0] + 2*dx/3, start[1] + 40 if dy >= 0 else start[1] - 40)
                else:
                    # Vertical movement predominant, try horizontal detour
                    mid1 = (start[0] + 40 if dx >= 0 else start[0] - 40, start[1] + dy/3)
                    mid2 = (start[0] + 40 if dx >= 0 else start[0] - 40, start[1] + 2*dy/3)
                
                # Check if midpoints are valid
                if not self._is_obstacle_fast(mid1) and not self._is_obstacle_fast(mid2):
                    path.extend([mid1, mid2])
                else:
                    # Try opposite direction
                    if abs(dx) > abs(dy):
                        mid1 = (start[0] + dx/3, start[1] - 40 if dy >= 0 else start[1] + 40)
                        mid2 = (start[0] + 2*dx/3, start[1] - 40 if dy >= 0 else start[1] + 40)
                    else:
                        mid1 = (start[0] - 40 if dx >= 0 else start[0] + 40, start[1] + dy/3)
                        mid2 = (start[0] - 40 if dx >= 0 else start[0] + 40, start[1] + 2*dy/3)
                        
                    # Check again
                    if not self._is_obstacle_fast(mid1) and not self._is_obstacle_fast(mid2):
                        path.extend([mid1, mid2])
                    else:
                        # Last resort: use a random midpoint with larger offset
                        mid_x = (start[0] + end[0]) / 2 + random.uniform(-50, 50)
                        mid_y = (start[1] + end[1]) / 2 + random.uniform(-50, 50)
                        path.append((mid_x, mid_y))
            
            # Always include the end point
            path.append(end)
            return path
        except Exception as e:
            logging.error(f"simple_astar error: {str(e)}")
            # Return direct path as last resort
            return [start, end]
    
    def plan_path(self, start: Tuple, end: Tuple, vehicle=None) -> List[Tuple]:
        """Enhanced high-performance path planning entry method with better error recovery"""
        # Performance counting
        start_time = time.time()
        self.planning_count += 1
        
        try:
            # Validate input
            if not start or not end:
                logging.warning("Invalid start or end point, returning empty path")
                return []
                
            # Make sure we're working with tuples
            if hasattr(start, 'as_tuple'):
                start = start.as_tuple()
            if hasattr(end, 'as_tuple'):
                end = end.as_tuple()
                
            # Check if start and end points are the same
            if math.isclose(start[0], end[0], abs_tol=EPSILON) and math.isclose(start[1], end[1], abs_tol=EPSILON):
                return [start]
                
            # Create cache key based on vehicle features if available
            cache_key = None
            if vehicle:
                # Cache key with vehicle properties
                cache_key = (
                    "path",
                    start,
                    end,
                    getattr(vehicle, 'turning_radius', 0), 
                    getattr(vehicle, 'min_hardness', 0),
                    getattr(vehicle, 'current_load', 0)
                )
            else:
                # Basic cache key
                cache_key = ("base_path", start, end)
                    
            # Check cache
            cached_path = self.path_cache.get(cache_key)
            if cached_path:
                logging.debug(f"Using cached path: {start} -> {end}")
                return cached_path
                
            # Try different path planning strategies
            path = []
            
            # Strategy 1: Try optimized vehicle-aware A* planning
            try:
                if vehicle:
                    path = self._optimized_astar(start, end, vehicle)
                else:
                    path = self._fast_astar(start, end)
            except Exception as e:
                logging.warning(f"Node-based path planning failed: {str(e)}")
                path = []
            
            # Strategy 2: Try simple A* without Node objects
            if not path or len(path) < 2:
                try:
                    logging.info(f"Using simple_astar as fallback: {start} -> {end}")
                    path = self.simple_astar(start, end, vehicle)
                except Exception as e:
                    logging.warning(f"Simple A* fallback failed: {str(e)}")
                    path = []
                    
            # Strategy 3: Use fallback path generation
            if not path or len(path) < 2:
                logging.info(f"Using fallback path generation: {start} -> {end}")
                path = self._generate_fallback_path(start, end)
                    
            # Path smoothing (if we have enough points)
            if len(path) > 2:
                try:
                    path = self._optimized_smooth(path)
                except Exception as e:
                    logging.warning(f"Path smoothing failed: {str(e)}")
                    
            # Cache result
            self.path_cache.put(cache_key, path)
            
            # Record performance metrics
            elapsed = time.time() - start_time
            self.total_planning_time += elapsed
            
            if elapsed > 0.1:  # Log slow planning
                logging.debug(f"Path planning took relatively long: {elapsed:.3f}s ({start} -> {end})")
                
            return path
                
        except Exception as e:
            logging.error(f"Path planning exception: {str(e)}")
            elapsed = time.time() - start_time
            self.total_planning_time += elapsed
            # Return simple direct path on error
            return self._generate_fallback_path(start, end)

    def _fast_astar(self, start: Tuple, end: Tuple) -> List[Tuple]:
        """优化的A*算法 - 不考虑车辆约束时的快速版本"""
        try:
            # 安全创建起点和终点节点
            start_node = self.safe_get_node(*start)
            end_node = self.safe_get_node(*end)
            
            # 检查终点是否为障碍物
            if self._is_obstacle_fast(end):
                logging.warning(f"终点{end}是障碍物，无法规划路径")
                global_node_pool.release_node(start_node)
                global_node_pool.release_node(end_node)
                # 返回备用路径
                return self._generate_fallback_path(start, end)
                
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
            
            # 限制搜索迭代次数，防止无限循环
            max_iterations = 10000
            iteration = 0
            
            # A*主循环 - 使用更高效的循环判断
            while not open_queue.empty() and iteration < max_iterations:
                iteration += 1
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
                    neighbor = self.safe_get_node(nx, ny)
                    
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
            
            # 路径未找到或达到最大迭代次数
            if iteration >= max_iterations:
                logging.warning(f"A*搜索达到最大迭代次数 {max_iterations}，使用备用路径")
                
            global_node_pool.release_all()
            # 返回备用路径
            return self._generate_fallback_path(start, end)
        except Exception as e:
            logging.error(f"_fast_astar出错: {str(e)}")
            return self._generate_fallback_path(start, end)
    def safe_get_node(self, x, y, t=0):
        """Safe method to create Node objects with error handling"""
        try:
            # Handle tuple/list input
            if isinstance(x, (tuple, list)) and len(x) >= 2:
                t = y if isinstance(y, (int, float)) else 0
                y = x[1]
                x = x[0]
                    
            # Ensure coordinates are numeric
            x = float(x) if x is not None else 0.0
            y = float(y) if y is not None else 0.0
            t = float(t) if t is not None else 0.0
                
            return global_node_pool.get_node(x, y, t)
        except Exception as e:
            logging.error(f"Node creation error: {str(e)}, using default Node(0,0,0)")
            # Return default node on error
            return Node(0, 0, 0)  
    def _optimized_astar(self, start: Tuple, end: Tuple, vehicle) -> List[Tuple]:
        """完全重写的A*算法，避免Node对象复用问题"""
        # 转换为元组确保一致性
        if hasattr(start, 'as_tuple'):
            start = start.as_tuple()
        if hasattr(end, 'as_tuple'):
            end = end.as_tuple()
            
        # 直接使用元组，不依赖Node对象
        open_set = []  # 使用堆队列
        heapq.heappush(open_set, (0, start))
        
        came_from = {}  # 路径记录
        g_score = {start: 0}  # 起点到当前点的代价
        f_score = {start: self._euclidean_distance(start, end)}  # 估计总代价
        
        closed_set = set()  # 已访问集合
        
        # 如果对象太多，限制迭代次数
        max_iterations = 10000
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # 获取f值最小的点
            _, current = heapq.heappop(open_set)
            
            # 到达目标（允许一定误差）
            if self._is_close_to(current, end, tolerance=3.0):
                # 重建路径
                path = self._reconstruct_tuple_path(came_from, current, end)
                return path
                
            # 标记为已访问
            closed_set.add(current)
            
            # 遍历相邻点
            for dx, dy in self.directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                # 跳过已访问点和障碍物
                if neighbor in closed_set or self._is_obstacle_fast(neighbor):
                    continue
                    
                # 对角线移动时检查拐角
                if dx != 0 and dy != 0:
                    # 检查两个相邻直角点是否都是障碍物
                    if (self._is_obstacle_fast((current[0], ny)) or 
                        self._is_obstacle_fast((nx, current[1]))):
                        continue
                
                # 计算移动代价
                move_cost = math.sqrt(dx*dx + dy*dy)  # 欧几里得距离
                
                # 考虑车辆特性
                if vehicle:
                    terrain_hardness = self._get_cached_terrain_hardness(nx, ny)
                    # 检查最小硬度
                    min_hardness = getattr(vehicle, 'min_hardness', 2.5)
                    if terrain_hardness < min_hardness:
                        continue
                        
                    # 调整移动成本
                    current_load = getattr(vehicle, 'current_load', 0)
                    move_cost *= (1.0 + current_load / 100.0)  # 负载增加成本
                    move_cost *= max(1.0, 3.0 / terrain_hardness)  # 地形影响
                
                # 路径更新
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # 记录最佳路径
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # 更新f值并加入开放集
                    f_value = tentative_g + self._euclidean_distance(neighbor, end)
                    
                    # 安全地添加到堆中
                    for i, (_, p) in enumerate(open_set):
                        if p == neighbor:
                            open_set[i] = (f_value, neighbor)  # 更新
                            heapq.heapify(open_set)
                            break
                    else:
                        heapq.heappush(open_set, (f_value, neighbor))  # 新增
        
        # 未找到路径，使用备选方案
        logging.warning(f"A*算法未找到路径 ({iterations}次迭代), 使用备选路径")
        return self._generate_fallback_path(start, end)
    def _euclidean_distance(self, p1, p2):
        """计算两点间欧几里得距离"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
    def _is_close_to(self, p1, p2, tolerance=1.0):
        """检查两点是否足够接近"""
        return self._euclidean_distance(p1, p2) <= tolerance
    def _reconstruct_tuple_path(self, came_from, current, end):
        """从came_from字典重建路径，使用元组而非Node对象"""
        path = [end]  # 从终点开始
        
        # 跟踪直到起点
        while current in came_from:
            path.append(current)
            current = came_from[current]
        
        path.append(current)  # 添加起点
        path.reverse()  # 反转为起点到终点
        
        return path
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
        
    def _reconstruct_path(self, came_from: Dict, current) -> List[Tuple]:
        """More robust path reconstruction that handles potentially invalid nodes"""
        try:
            # Start with current node
            if hasattr(current, 'x') and hasattr(current, 'y'):
                path = [(current.x, current.y)]
            else:
                # Handle case where current is not a Node object
                path = [current if isinstance(current, tuple) else (0, 0)]
            
            # Backtrack through came_from map
            while current in came_from:
                current = came_from[current]
                
                # Handle each node or coordinate appropriately
                if hasattr(current, 'x') and hasattr(current, 'y'):
                    path.append((current.x, current.y))
                elif isinstance(current, tuple) and len(current) >= 2:
                    path.append(current)
                else:
                    logging.warning(f"Invalid node in path reconstruction: {current}")
                    # Skip invalid nodes
                    continue
                    
            # Return reversed path (from start to end)
            return list(reversed(path))
        except Exception as e:
            logging.error(f"Path reconstruction error: {str(e)}")
            # Return whatever we have so far
            return list(reversed(path)) if path else []


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

if __name__ == "__main__":
    """
    HybridPathPlanner 优化版测试模块
    在 150x150 地图上测试两个装载点、一个卸载点和一个停车点之间的路径规划
    """
    import os
    import sys
    import time
    import matplotlib.pyplot as plt
    import numpy as np
    import random
    
    # 确保路径正确
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, PROJECT_ROOT)
    
    from algorithm.map_service import MapService
    from utils.geo_tools import GeoUtils
    from utils.path_tools import PathProcessor
    
    # 设置随机种子，保证结果可重现
    random.seed(42)
    
    # 配置日志级别减少输出
    logging.basicConfig(level=logging.INFO)
    
    print("===============================================")
    print("   HybridPathPlanner 路径规划测试 (150x150 地图)")
    print("===============================================")
    
    # 辅助函数：创建简单地图
    def create_simple_map(size=150, obstacle_density=0.03):  # 减少障碍物密度
        """创建简单的测试地图"""
        # 模拟配置
        config_path = os.path.join(PROJECT_ROOT, 'config.ini')
        if os.path.exists(config_path):
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)
            # 修改地图尺寸
            if 'MAP' not in config:
                config['MAP'] = {}
            config['MAP']['grid_size'] = str(size)
            config['MAP']['obstacle_density'] = str(obstacle_density)
            # 保存回文件
            with open(config_path, 'w') as f:
                config.write(f)
            print(f"已更新配置文件: {config_path}")
        
        # 创建地图服务
        try:
            geo_utils = GeoUtils()
            map_service = MapService(PathProcessor(geo_utils))
            print("地图服务初始化成功")
            return map_service
        except Exception as e:
            print(f"地图服务初始化失败: {str(e)}")
            # 如果创建Map服务失败，创建一个简单的模拟对象
            class SimpleMapService:
                def __init__(self):
                    self.geo_utils = GeoUtils()
                    self.grid_size = size
                    
                def get_terrain_hardness(self, x, y):
                    # 简化版永远返回中等硬度
                    return 3.0
                    
            return SimpleMapService()
    
    # 辅助函数：生成障碍物
    def generate_obstacles(planner, num_obstacles=8, obstacle_size=3, point_pairs=None):
        """生成随机障碍物，确保测试路径上有障碍物阻挡直线路径"""
        obstacles = set()
        # 添加一些随机障碍物
        for _ in range(num_obstacles):
            center_x = random.randint(10, 140)
            center_y = random.randint(10, 140)
            # 生成障碍物周围的点
            for dx in range(-obstacle_size, obstacle_size + 1):
                for dy in range(-obstacle_size, obstacle_size + 1):
                    if dx**2 + dy**2 <= obstacle_size**2:  # 圆形障碍物
                        obstacles.add((center_x + dx, center_y + dy))
        
        # 在每对测试点的直线路径上放置障碍物
        if point_pairs:
            for start_name, end_name in point_pairs:
                start_point = test_points[start_name]
                end_point = test_points[end_name]
                
                # 计算直线中点附近的位置
                mid_x = (start_point[0] + end_point[0]) / 2
                mid_y = (start_point[1] + end_point[1]) / 2
                
                # 在中点位置放置障碍物
                for dx in range(-obstacle_size, obstacle_size + 1):
                    for dy in range(-obstacle_size, obstacle_size + 1):
                        if dx**2 + dy**2 <= obstacle_size**2:  # 圆形障碍物
                            obstacles.add((int(mid_x + dx), int(mid_y + dy)))
        
        # 避免在测试点上生成障碍物
        for point in test_points.values():
            # 清除测试点周围10个单位的障碍物
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    if dx**2 + dy**2 <= 100:  # 10的平方
                        obstacles.discard((point[0] + dx, point[1] + dy))
        
        # 设置到路径规划器
        planner.obstacle_grids = obstacles
        print(f"已生成 {len(obstacles)} 个障碍物点")
        
        # 预填充地形硬度缓存
        for x in range(0, 150, 5):  # 每5个单位取样
            for y in range(0, 150, 5):
                planner.terrain_cache[(x, y)] = 3.0  # 使用默认硬度
                
        return obstacles
    
    # 优化A*搜索的参数
    def optimize_planner(planner):
        """优化路径规划器的参数"""
        # 设置更大的搜索步长
        planner.directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 对角线
        ]
        
        # 限制点数
        original_fast_astar = planner._fast_astar
        
        def limited_fast_astar(start, end):
            # 计算起点终点距离
            dist = math.dist(start, end)
            # 设置最大点数限制
            max_nodes = min(1000, int(dist * 10))
            
            # 计数变量
            nodes_expanded = [0]
            
            # 覆盖内部函数
            original_is_same_point = planner._is_same_point
            
            def counted_is_same_point(a, b, tolerance=EPSILON):
                nodes_expanded[0] += 1
                if nodes_expanded[0] > max_nodes:
                    # 如果展开太多节点，提前结束
                    return True
                return original_is_same_point(a, b, tolerance)
            
            # 替换函数
            planner._is_same_point = counted_is_same_point
            
            try:
                return original_fast_astar(start, end)
            finally:
                # 恢复原始函数
                planner._is_same_point = original_is_same_point
        
        # 替换函数
        planner._fast_astar = limited_fast_astar
        
        # 类似地优化优化版A*
        original_optimized_astar = planner._optimized_astar
        
        def limited_optimized_astar(start, end, vehicle):
            # 计算起点终点距离
            dist = math.dist(start, end)
            # 设置最大点数限制
            max_nodes = min(1000, int(dist * 10))
            
            # 计数变量
            nodes_expanded = [0]
            
            # 覆盖内部函数
            original_is_same_point = planner._is_same_point
            
            def counted_is_same_point(a, b, tolerance=EPSILON):
                nodes_expanded[0] += 1
                if nodes_expanded[0] > max_nodes:
                    # 如果展开太多节点，提前结束
                    return True
                return original_is_same_point(a, b, tolerance)
            
            # 替换函数
            planner._is_same_point = counted_is_same_point
            
            try:
                return original_optimized_astar(start, end, vehicle)
            finally:
                # 恢复原始函数
                planner._is_same_point = original_is_same_point
        
        # 替换函数
        planner._optimized_astar = limited_optimized_astar
    
    # 绘制地图和测试点
    def plot_map_and_points():
        plt.figure(figsize=(12, 12))
        plt.xlim(0, 150)
        plt.ylim(0, 150)
        plt.grid(True, alpha=0.3)
        
        # 绘制障碍物（如果有）
        if hasattr(planner, 'obstacle_grids') and planner.obstacle_grids:
            obstacle_x = [p[0] for p in planner.obstacle_grids]
            obstacle_y = [p[1] for p in planner.obstacle_grids]
            plt.scatter(obstacle_x, obstacle_y, c='gray', s=10, alpha=0.5, label='障碍物')
        
        # 绘制测试点
        colors = ['green', 'green', 'red', 'blue']
        markers = ['o', 'o', 's', '^']  # 不同形状：圆形、圆形、方形、三角形
        for i, (name, coords) in enumerate(test_points.items()):
            plt.scatter(coords[0], coords[1], c=colors[i], s=150, 
                       marker=markers[i], edgecolors='black', zorder=100)
            plt.text(coords[0] + 5, coords[1] + 5, name, fontsize=12, weight='bold', zorder=100)
            
        plt.title("路径规划测试地图 (150x150)", fontsize=16)
        plt.xlabel("X 坐标", fontsize=14)
        plt.ylabel("Y 坐标", fontsize=14)

# 定义测试点位置（在 150x150 的地图范围内）
    test_points = {
        "装载点1": (30, 120),
        "装载点2": (120, 30),
        "卸载点": (75, 75),
        "停车点": (120, 120)
    }
    # 创建一个简单的车辆对象用于测试
    class TestVehicle:
        def __init__(self, load=0):
            self.vehicle_id = "test"
            self.turning_radius = 10.0
            self.min_hardness = 2.5
            self.current_load = load
            self.max_speed = 5.0
            self.path_index = 0
            self.current_path = []
            self.last_position = None

    # 两种车辆情况：空载和满载
    test_vehicles = {
        "空载车辆": TestVehicle(load=0),
        "满载车辆": TestVehicle(load=50000)
    }


    # 只测试重要的点对路径，减少计算量
    selected_point_pairs = [
        ("装载点1", "卸载点"),
        ("装载点2", "卸载点"),
        ("卸载点", "停车点"),
        ("停车点", "装载点1"),
        ("停车点", "装载点2"),
        ("装载点1", "装载点2")
    ]
    
    # 显示所有要测试的路径
    print(f"\n步骤3: 将测试以下 {len(selected_point_pairs)} 条路径:")
    for start_name, end_name in selected_point_pairs:
        print(f"  {start_name} → {end_name}")
    
    # 创建测试用的地图服务
    print("\n步骤1: 初始化地图服务...")
    map_service = create_simple_map(size=150, obstacle_density=0.03)
    
    # 创建路径规划器
    print("\n步骤2: 创建路径规划器...")
    planner = HybridPathPlanner(map_service)
    
    # 添加一些障碍物，确保每对测试点的直线路径上有障碍物
    obstacles = generate_obstacles(planner, num_obstacles=8, obstacle_size=5, point_pairs=selected_point_pairs)
    
    # 优化规划器
    optimize_planner(planner)
    
    # 绘制所有路径
    plot_map_and_points()
    
    # 使用不同颜色绘制路径
    cmap = plt.cm.rainbow
    colors = [cmap(i / len(selected_point_pairs)) for i in range(len(selected_point_pairs))]
    
    # 路径列表，用于绘制每个路径
    all_paths = {}
    
    # 验证每对点之间有障碍物
    print("\n步骤3: 验证路径阻塞情况...")
    for start_name, end_name in selected_point_pairs:
        start_point = test_points[start_name]
        end_point = test_points[end_name]
        has_obstacle = False
        
        # 使用 bresenham 算法检查直线路径上是否有障碍物
        line_points = GeoUtils.bresenham_line(start_point, end_point)
        for point in line_points:
            if point in obstacles:
                has_obstacle = True
                break
                
        print(f"  {start_name} → {end_name}: {'已阻塞✓' if has_obstacle else '未阻塞✗'}")
    
    # 规划并记录所有路径
    print("\n步骤4: 开始路径规划测试...")
    for i, (start_name, end_name) in enumerate(selected_point_pairs):
        start_point = test_points[start_name]
        end_point = test_points[end_name]
        
        print(f"\n规划路径: {start_name} → {end_name}")
        
        # 对于装载点到卸载点的路径，使用满载车辆
        if "装载点" in start_name and end_name == "卸载点":
            test_vehicle = test_vehicles["满载车辆"]
            print("  使用满载车辆")
        else:
            test_vehicle = test_vehicles["空载车辆"]
            print("  使用空载车辆")
        
        # 使用改进的带超时规划方法
        path, elapsed = planner.plan_with_timeout(start_point, end_point, test_vehicle, timeout=5.0)
        
        print(f"  路径规划完成: {len(path)}个点, 耗时: {elapsed:.3f}秒")
        
        # 验证路径是否规避了障碍物
        if len(path) > 2:
            print("  生成了非直线路径，正在验证...")
            is_direct = math.isclose(
                math.dist(start_point, end_point),
                sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1)),
                abs_tol=10.0
            )
            print(f"  验证结果: {'直线路径✗' if is_direct else '规避障碍物✓'}")
        
        # 绘制路径
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, '-', color=colors[i], linewidth=2.5, alpha=0.7, 
                label=f"{start_name} → {end_name}")
        
        # 添加起点和终点标记
        plt.plot(path_x[0], path_y[0], 'o', color=colors[i], markersize=8)
        plt.plot(path_x[-1], path_y[-1], 'x', color=colors[i], markersize=8)
        
        # 保存路径
        all_paths[(start_name, end_name)] = path
            
    # 显示图例
    plt.legend(loc='upper right')
    
    # 计算路径统计信息
    if all_paths:
        path_lengths = {}
        for (start_name, end_name), path in all_paths.items():
            length = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))
            path_lengths[(start_name, end_name)] = length
        
        # 显示路径长度统计
        print("\n步骤5: 路径长度统计:")
        for (start_name, end_name), length in sorted(path_lengths.items(), key=lambda x: x[1]):
            print(f"  {start_name} → {end_name}: {length:.2f} 单位")
    
    # 保存图片和显示
    plt.tight_layout()
    save_path = os.path.join(PROJECT_ROOT, "path_planning_test.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n图像已保存到: {save_path}")
    print("\n正在显示路径规划结果图...")
    plt.show()
    
    print("\n路径规划测试完成！")
    print("===============================================")