import sys
import os
import math
import time
import heapq
import logging
import threading
import random
from typing import List, Tuple, Dict, Optional, Union, Set, Callable
import numpy as np
from collections import deque 
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError
from algorithm.map_service import MapService

# 常量定义
MAX_ITERATIONS = 10000     # A*搜索的最大迭代次数
DEFAULT_TIMEOUT = 5.0      # 默认超时时间(秒)
EPSILON = 1e-6             # 浮点数比较精度
CACHE_SIZE = 1000          # 缓存大小
CACHE_EXPIRY = 600         # 缓存过期时间(秒)

class PathPlanningError(Exception):
    """路径规划错误基类"""
    pass

class TimeoutError(PathPlanningError):
    """超时错误"""
    pass

class NoPathFoundError(PathPlanningError):
    """无法找到路径错误"""
    pass

class SpatialIndex:
    """空间索引结构，用于加速空间查询"""
    def __init__(self, cell_size=10):
        self.cell_size = cell_size
        self.grid = {}
        
    def add_point(self, point):
        """添加点到索引"""
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        cell_key = (cell_x, cell_y)
        
        if cell_key not in self.grid:
            self.grid[cell_key] = set()
        self.grid[cell_key].add(point)
            
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
    def __init__(self, max_size=CACHE_SIZE, expiry=CACHE_EXPIRY):
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
    """
    高性能混合型路径规划器 - 优化版
    
    提供了多种路径规划策略，适合不同场景:
    1. A*算法 - 精确路径规划
    2. 备选路径生成 - 当A*失败时提供替代方案
    3. 超时控制 - 确保规划过程不会无限阻塞
    
    支持障碍物检测、路径平滑和性能统计等功能。
    """
    
    def __init__(self, map_service: MapService):
        """
        初始化路径规划器
        
        Args:
            map_service: 地图服务对象，提供地形和障碍物信息
        """
        # 基础组件
        self.map_service = map_service
        self.dispatch = None  # 由DispatchSystem设置
        
        # 性能监控
        self.load_time = time.time()
        self.planning_count = 0
        self.total_planning_time = 0
        self.success_count = 0
        self.failure_count = 0
        
        # 地图尺寸 - 确保有这个属性
        self.map_size = getattr(map_service, 'grid_size', 200)
        
        # 空间数据
        self.obstacle_index = SpatialIndex(cell_size=20)  # 障碍物空间索引
        self.obstacle_grids = set()                       # 障碍点集合
        
        # 预约系统
        self.reservation_table = {}                      # 路径段预约表
        self.reservation_lock = threading.RLock()        # 预约表锁
        
        # 缓存系统
        self.path_cache = PathCache(max_size=CACHE_SIZE, expiry=CACHE_EXPIRY)
        
        # 加载地图配置
        self._load_map_config()
            
        # 方向数组用于A*搜索 (8个方向)
        self.directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),   # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
        
        # 预计算的移动成本
        self.move_costs = {
            (0, 1): 1.0, (1, 0): 1.0, (0, -1): 1.0, (-1, 0): 1.0,  # 直线移动成本
            (1, 1): 1.414, (1, -1): 1.414, (-1, 1): 1.414, (-1, -1): 1.414  # 对角线移动成本
        }
        
        # 初始化空间索引
        self._init_spatial_index()
        
        logging.info("路径规划器初始化完成")
            
    def _load_map_config(self):
        """加载地图配置"""
        try:
            # 尝试读取配置文件
            self.grid_size = 20.0  # 默认网格大小
            self.max_grade = 15.0  # 最大坡度
            self.min_turn_radius = 15.0  # 最小转弯半径
            
            # 如果MapService提供了配置，优先使用
            if hasattr(self.map_service, 'config'):
                config = self.map_service.config
                if hasattr(config, 'grid_size'):
                    self.grid_size = float(config.grid_size)
                if hasattr(config, 'max_grade'):
                    self.max_grade = float(config.max_grade)
                if hasattr(config, 'min_turn_radius'):
                    self.min_turn_radius = float(config.min_turn_radius)
                    
            logging.debug(f"加载地图配置: 网格大小={self.grid_size}, 最大坡度={self.max_grade}, 最小转弯半径={self.min_turn_radius}")
            
        except Exception as e:
            logging.warning(f"加载地图配置失败，使用默认值: {str(e)}")
            self.grid_size = 20.0
            self.max_grade = 15.0
            self.min_turn_radius = 15.0
      
    def plan_path(self, start, end, vehicle=None):
        """
        路径规划主入口方法
        
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            vehicle: 可选的车辆对象，用于考虑车辆特性
            
        Returns:
            List[Tuple[float, float]]: 规划的路径点列表
        """
        # 性能计数
        start_time = time.time()
        self.planning_count += 1
        
        try:
            # 标准化输入坐标
            start = self._validate_point(start)
            end = self._validate_point(end)
            
            # 检查起点和终点是否相同
            if self._points_equal(start, end):
                return [start]
                
            # 创建缓存键
            cache_key = self._create_cache_key(start, end, vehicle)
                    
            # 检查缓存
            cached_path = self.path_cache.get(cache_key)
            if cached_path:
                logging.debug(f"使用缓存路径: {start} -> {end}")
                return cached_path
                
            # 尝试A*算法
            path = self._astar(start, end, vehicle)
            
            # 若A*失败，使用备选路径
            if not path or len(path) < 2:
                logging.debug(f"A*路径规划失败，使用备选路径: {start} -> {end}")
                path = self._generate_fallback_path(start, end)
                self.failure_count += 1
            else:
                self.success_count += 1
                
            # 路径平滑（如果点数足够）
            if len(path) > 3:
                try:
                    path = self._smooth_path(path)
                except Exception as e:
                    logging.warning(f"路径平滑失败: {str(e)}")
                    
            # 缓存结果
            self.path_cache.put(cache_key, path)
            
            # 记录性能指标
            elapsed = time.time() - start_time
            self.total_planning_time += elapsed
            
            if elapsed > 0.1:  # 记录较慢的规划
                logging.debug(f"路径规划耗时较长: {elapsed:.3f}秒 ({start} -> {end})")
                
            return path
        except Exception as e:
            logging.error(f"备选路径生成失败: {str(e)}")
            # 最简单的后备方案 - 直接连接起点和终点
            return [start, end]
    
    def _init_spatial_index(self):
        """初始化空间索引结构"""
        self.obstacle_grid_size = 10  # 网格尺寸
        self.obstacle_index_grid = {}
        
        # 建立网格索引
        if hasattr(self, 'obstacle_grids') and self.obstacle_grids:
            for obs_x, obs_y in self.obstacle_grids:
                # 计算网格坐标
                grid_x = obs_x // self.obstacle_grid_size
                grid_y = obs_y // self.obstacle_grid_size
                grid_key = (grid_x, grid_y)
                
                # 添加到网格
                if grid_key not in self.obstacle_index_grid:
                    self.obstacle_index_grid[grid_key] = set()
                
                self.obstacle_index_grid[grid_key].add((obs_x, obs_y))
            
            logging.info(f"空间索引初始化完成: {len(self.obstacle_index_grid)}个网格，{len(self.obstacle_grids)}个障碍点")
    def _bresenham_line(self, start, end):
        """
        使用Bresenham算法生成线段上的所有点
        
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            
        Returns:
            List[Tuple[int, int]]: 线段上的所有点
        """
        x1, y1 = int(round(start[0])), int(round(start[1]))
        x2, y2 = int(round(end[0])), int(round(end[1]))
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        
        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
        return points
    
    def _smooth_path(self, path):
        """平滑路径并确保不穿过障碍物（改进版）"""
        if len(path) <= 2:
            return path
        
        # 使用道格拉斯-普克算法进行平滑
        result = self._douglas_peucker(path, 2.0)
        
        # 检查平滑后路径的每个相邻点之间是否穿过障碍物
        safe_path = [result[0]]  # 添加起点
        
        for i in range(1, len(result)):
            prev = safe_path[-1]
            curr = result[i]
            
            # 检查连线是否穿过障碍物
            line_points = self._bresenham_line(prev, curr)
            has_obstacle = False
            
            for p in line_points:
                if self._is_obstacle_fast(p):
                    has_obstacle = True
                    break
            
            if has_obstacle:
                # 如果连线穿过障碍物，找出原始路径中的中间点添加到安全路径中
                orig_idx_prev = path.index(prev)
                orig_idx_curr = path.index(curr) if curr in path else len(path) - 1
                
                # 添加原始路径中的点来避开障碍物
                for j in range(orig_idx_prev + 1, orig_idx_curr + 1):
                    if j < len(path):
                        safe_path.append(path[j])
            else:
                # 如果连线没有穿过障碍物，直接添加当前点
                safe_path.append(curr)
        
        # 确保终点添加到路径中
        if safe_path[-1] != path[-1]:
            safe_path.append(path[-1])
        
        return safe_path
    
    def _douglas_peucker(self, points, epsilon):
        """
        道格拉斯-普克算法实现
        
        用于简化路径，保留关键点
        """
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
            
    def _perpendicular_distance(self, point, line_start, line_end):
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
    
    def mark_obstacle_area(self, polygons):
        """
        批量标记障碍物区域
        
        Args:
            polygons: 多边形顶点列表的列表
        """
        # 清除旧索引
        self.obstacle_index.clear()
        self.obstacle_grids.clear()
        
        # 处理每个多边形
        for polygon in polygons:
            # 计算边界框
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
        self._init_spatial_index()  # 更新网格索引
        logging.debug(f"已标记{len(self.obstacle_grids)}个障碍物点")
        
    def _point_in_polygon(self, point, polygon):
        """射线法判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        
        # 快速检查点是否在多边形顶点上
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

    def _is_obstacle_fast(self, point):
        """强化版快速障碍物检测 - 大幅增强避障范围"""
        # 转换为整数坐标
        x, y = int(round(point[0])), int(round(point[1]))

        # 直接检查点是否在障碍物集合中
        if (x, y) in self.obstacle_grids:
            return True
            
        # 多检查一下周围的点，大幅扩大障碍物的"影响范围"
        for dx in range(-3, 4):  # 从(-2,3)扩大到(-3,4)
            for dy in range(-3, 4):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.obstacle_grids:
                    # 使用距离加权 - 越近的障碍物影响越大
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance < 2.0:  # 从1.5增加到2.0，扩大影响范围
                        return True
        return False

    def _count_nearby_obstacles(self, point, radius=3):
        """计算点附近障碍物数量 - 增加半径以提前检测障碍物"""
        x, y = point
        count = 0
        weight_sum = 0.0  # 加权计数
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = int(x + dx), int(y + dy)
                check_point = (nx, ny)
                
                # 确保点在地图范围内
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self._is_obstacle_fast(check_point):
                        # 距离加权 - 越近的障碍物权重越大
                        distance = math.sqrt(dx*dx + dy*dy)
                        weight = 1.0 / max(0.5, distance)  # 避免除零
                        weight_sum += weight
                        count += 1
        
        # 返回加权后的数量，使近距离障碍物影响更大
        return count * (1.0 + weight_sum/10.0)

    def _is_obstacle(self, point):
        """检查点是否为障碍物（改进版）"""
        # 转换为整数坐标（因为障碍物通常以整数坐标存储）
        x, y = int(round(point[0])), int(round(point[1]))
        int_point = (x, y)
        
        # 基本检查
        if int_point in self.obstacle_grids:
            return True
        
        # 使用空间索引进行检查
        if hasattr(self, 'obstacle_index') and self.obstacle_index.grid:
            nearby_points = self.obstacle_index.query_point(int_point, radius=1)  # 检查周围1个单位的点
            if nearby_points:
                return True
        
        # 使用地图服务进行检查
        try:
            if hasattr(self.map_service, 'is_obstacle') and callable(getattr(self.map_service, 'is_obstacle')):
                return self.map_service.is_obstacle(point)
        except Exception as e:
            logging.debug(f"地图服务障碍物检查出错: {str(e)}")
        
        return False
    
    def _get_directions(self, current, goal, use_jps=False):
        """根据当前情况获取搜索方向"""
        # 基础方向：8个方向
        all_directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),   # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
        
        if not use_jps:
            return all_directions
        
        # 在Jump Point Search中，我们可以根据当前点和目标点的关系优先选择某些方向
        dx, dy = goal[0] - current[0], goal[1] - current[1]
        
        # 确定主方向
        primary_directions = []
        
        if dx > 0:
            primary_directions.append((1, 0))
        elif dx < 0:
            primary_directions.append((-1, 0))
            
        if dy > 0:
            primary_directions.append((0, 1))
        elif dy < 0:
            primary_directions.append((0, -1))
        
        # 添加对角线方向（如果适用）
        if dx != 0 and dy != 0:
            diag_x = 1 if dx > 0 else -1
            diag_y = 1 if dy > 0 else -1
            primary_directions.append((diag_x, diag_y))
        
        # 先检查主方向，再检查其他方向
        return primary_directions + [d for d in all_directions if d not in primary_directions]

    def _calculate_move_cost(self, direction, current, neighbor, vehicle=None, came_from=None):
        """计算移动成本，考虑各种因素"""
        dx, dy = direction
        
        # 基础移动成本
        base_cost = self.move_costs.get(direction, 1.0)
        
        # 计算地形成本
        terrain_cost = 1.0
        try:
            if hasattr(self.map_service, 'get_terrain_hardness'):
                hardness = self.map_service.get_terrain_hardness(neighbor[0], neighbor[1])
                min_hardness = getattr(vehicle, 'min_hardness', 0) if vehicle else 0
                
                if hardness < min_hardness:
                    return float('inf')  # 无法通过的地形
                    
                terrain_cost = max(1.0, 3.0 / hardness)
        except:
            pass
        
        # 路径平滑性成本（避免频繁转向）
        smoothness_cost = 1.0
        if vehicle and hasattr(vehicle, 'turning_radius') and vehicle.turning_radius > 0:
            # 检查是否需要转弯
            if came_from and current in came_from:
                prev = came_from[current]
                prev_dx, prev_dy = current[0] - prev[0], current[1] - prev[1]
                angle_change = self._calculate_angle_change((prev_dx, prev_dy), direction)
                
                if angle_change > 90:  # 急转弯
                    smoothness_cost = 1.5
                elif angle_change > 45:  # 中等转弯
                    smoothness_cost = 1.2
        
        # 拥挤区域成本 - 如果太接近障碍物，增加成本
        crowding_cost = 1.0
        obstacles_nearby = self._count_nearby_obstacles(neighbor, radius=3)
        if obstacles_nearby > 0:
            crowding_cost = 1.0 + min(0.5, obstacles_nearby * 0.1)  # 最多增加50%成本
        
        # 综合成本
        return base_cost * terrain_cost * smoothness_cost * crowding_cost

    def _astar(self, start, end, vehicle=None):
        """改进的A*路径规划算法实现"""
        try:
            # 添加更多日志便于调试
            logging.debug(f"开始A*搜索: 从 {start} 到 {end}")
            
            # 调整起点和终点避开障碍物
            start_is_obstacle = self._is_obstacle_fast(start)
            end_is_obstacle = self._is_obstacle_fast(end)
            
            if start_is_obstacle:
                adjusted_start = self._find_nearest_non_obstacle(start, max_radius=10)
                if adjusted_start:
                    logging.debug(f"起点调整为: {adjusted_start}")
                    start = adjusted_start
                else:
                    logging.warning("无法找到非障碍物起点")
                    return None
            
            if end_is_obstacle:
                adjusted_end = self._find_nearest_non_obstacle(end, max_radius=10)
                if adjusted_end:
                    logging.debug(f"终点调整为: {adjusted_end}")
                    end = adjusted_end
                else:
                    logging.warning("无法找到非障碍物终点")
                    return None
            
            # 更好的启发式函数 - 特别适用于迷宫
            def heuristic(pos):
                dx, dy = abs(pos[0] - end[0]), abs(pos[1] - end[1])
                
                # 基本曼哈顿距离
                d_manhattan = dx + dy
                
                # 欧几里得距离
                d_euclidean = math.sqrt(dx*dx + dy*dy)
                
                # 切比雪夫距离 - 对角线移动有利
                d_chebyshev = max(dx, dy)
                
                # 迷宫因子 - 避免估计过于乐观
                maze_factor = 1.2
                
                # 组合启发式，根据迷宫的特性调整权重
                return 0.3 * d_manhattan + 0.4 * d_euclidean + 0.3 * d_chebyshev * maze_factor
            
            # 初始化开放集和闭合集
            # 使用优先队列和集合组合提高效率
            open_set = []
            open_set_hash = set()  # 加速成员检查
            closed_set = set()
            
            # 初始f值，g值和节点
            start_f = heuristic(start)
            heapq.heappush(open_set, (start_f, 0, start))
            open_set_hash.add(start)
            
            # 路径追踪
            came_from = {}
            g_score = {start: 0}
            f_score = {start: start_f}
            
            # 搜索参数
            iterations = 0
            max_iterations = MAX_ITERATIONS * 2  # 增加最大迭代次数
            expanded_nodes = 0
            
            # 调试信息
            debug_intervals = max_iterations // 10  # 每隔一定迭代次数记录一次调试信息
            
            # 主搜索循环
            while open_set and iterations < max_iterations:
                iterations += 1
                
                # 获取f值最低的节点
                current_f, current_g, current = heapq.heappop(open_set)
                open_set_hash.remove(current)
                
                # 定期输出调试信息
                if iterations % debug_intervals == 0:
                    logging.debug(f"A*迭代次数: {iterations}, 开放集大小: {len(open_set)}, 闭合集大小: {len(closed_set)}")
                    logging.debug(f"当前节点: {current}, 当前f值: {current_f}, 距离终点: {math.dist(current, end)}")
                
                # 检查是否到达目标点
                if self._close_enough(current, end, threshold=2.0):
                    # 重建路径
                    path = self._reconstruct_path(came_from, current, end)
                    logging.info(f"A*搜索成功! 迭代次数: {iterations}, 展开节点: {expanded_nodes}, 路径长度: {len(path)}")
                    return path
                
                # 添加到闭合集
                closed_set.add(current)
                expanded_nodes += 1
                
                # 调查所有可能的移动方向
                directions = [
                    (0, 1), (1, 0), (0, -1), (-1, 0),  # 垂直和水平方向
                    (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线方向
                ]
                
                # 遍历所有邻居节点
                for dx, dy in directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # 检查边界
                    if not (0 <= neighbor[0] < self.map_size and 0 <= neighbor[1] < self.map_size):
                        continue
                    
                    # 跳过已经在闭合集中的节点
                    if neighbor in closed_set:
                        continue
                    
                    # 检查障碍物
                    if self._is_obstacle_fast(neighbor):
                        continue
                    
                    # 检查对角线移动时的拐角通行性
                    if dx != 0 and dy != 0:
                        # 检查两个相邻的直线方向是否可通行
                        if self._is_obstacle_fast((current[0], current[1] + dy)) or \
                        self._is_obstacle_fast((current[0] + dx, current[1])):
                            continue
                    
                    # 计算新的g值 - 考虑移动成本
                    move_cost = math.sqrt(dx*dx + dy*dy)  # 直线距离
                    tentative_g = g_score[current] + move_cost
                    
                    # 如果找到了更好的路径
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        # 记录最佳路径
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor)
                        
                        # 如果节点不在开放集中，添加它
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
                
            # 搜索失败
            if iterations >= max_iterations:
                logging.warning(f"A*搜索达到最大迭代次数 ({max_iterations})，搜索失败")
            else:
                logging.warning(f"A*搜索失败，开放集为空")
            
            # 尝试退回到广度优先搜索
            logging.info("尝试使用广度优先搜索作为后备方案...")
            path = self._bfs_pathfinding(start, end)
            if path and len(path) > 1:
                logging.info(f"广度优先搜索成功找到路径，长度: {len(path)}")
                return path
                
            return None
        
        except Exception as e:
            logging.error(f"A*算法执行出错: {str(e)}", exc_info=True)
            return None
    def _bfs_pathfinding(self, start, end):
        """
        广度优先搜索路径规划 - 作为A*的后备方案
        不考虑启发式函数，纯粹寻找最短路径
        """
        from collections import deque  # 添加导入
        
        logging.debug(f"开始BFS搜索: 从 {start} 到 {end}")
        
        # 初始化队列和访问集合
        queue = deque([(start, [start])])  # (当前点, 路径)
        visited = set([start])
        
        # 设置最大迭代次数
        iterations = 0
        max_iterations = MAX_ITERATIONS * 3  # BFS可能需要更多迭代
        
        # 主搜索循环
        while queue and iterations < max_iterations:
            iterations += 1
            
            # 获取当前节点和到达该节点的路径
            current, path = queue.popleft()
            
            # 定期输出调试信息
            if iterations % 1000 == 0:
                logging.debug(f"BFS迭代次数: {iterations}, 队列大小: {len(queue)}, 当前点: {current}")
            
            # 检查是否到达目标
            if self._close_enough(current, end):
                logging.info(f"BFS搜索成功! 迭代次数: {iterations}, 路径长度: {len(path)}")
                # 确保终点在路径中
                if not self._points_equal(path[-1], end):
                    path.append(end)
                return path
            
            # 检查所有可能方向
            directions = [
                (0, 1), (1, 0), (0, -1), (-1, 0),  # 上下左右
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
            ]
            
            # 检查所有邻居
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查边界
                if not (0 <= neighbor[0] < self.map_size and 0 <= neighbor[1] < self.map_size):
                    continue
                
                # 跳过已访问的节点
                if neighbor in visited:
                    continue
                
                # 检查障碍物
                if self._is_obstacle_fast(neighbor):
                    continue
                
                # 检查对角线移动
                if dx != 0 and dy != 0:
                    if self._is_obstacle_fast((current[0], current[1] + dy)) or \
                    self._is_obstacle_fast((current[0] + dx, current[1])):
                        continue
                
                # 添加到队列和已访问集合
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))
                visited.add(neighbor)
        
        # 搜索失败
        if iterations >= max_iterations:
            logging.warning(f"BFS搜索达到最大迭代次数 ({max_iterations})，搜索失败")
        else:
            logging.warning("BFS搜索失败，队列为空")
        
        return None
    def _jump_point_search(self, current, direction, end):
        """Jump Point Search优化 - 在指定方向上跳跃搜索"""
        dx, dy = direction
        nx, ny = current[0] + dx, current[1] + dy
        
        # 如果超出边界或是障碍物，返回None
        if not (0 <= nx < self.map_size and 0 <= ny < self.map_size) or self._is_obstacle_fast((nx, ny)):
            return None
        
        # 如果到达终点，返回当前点
        if (nx, ny) == end or self._close_enough((nx, ny), end):
            return (nx, ny)
        
        # 检查是否是强制邻居（有障碍物导致必须转向）
        has_forced_neighbor = False
        
        # 对角线移动
        if dx != 0 and dy != 0:
            # 检查两个相邻的直线方向
            if self._is_obstacle_fast((current[0], ny)) and not self._is_obstacle_fast((nx, current[1])):
                has_forced_neighbor = True
            elif not self._is_obstacle_fast((current[0], ny)) and self._is_obstacle_fast((nx, current[1])):
                has_forced_neighbor = True
        
        # 水平移动
        elif dx != 0:
            # 检查上下方向
            if ((ny+1) < self.map_size and self._is_obstacle_fast((nx, ny+1)) and not self._is_obstacle_fast((current[0], ny+1))) or \
               ((ny-1) >= 0 and self._is_obstacle_fast((nx, ny-1)) and not self._is_obstacle_fast((current[0], ny-1))):
                has_forced_neighbor = True
        
        # 垂直移动
        elif dy != 0:
            # 检查左右方向
            if ((nx+1) < self.map_size and self._is_obstacle_fast((nx+1, ny)) and not self._is_obstacle_fast((nx+1, current[1]))) or \
               ((nx-1) >= 0 and self._is_obstacle_fast((nx-1, ny)) and not self._is_obstacle_fast((nx-1, current[1]))):
                has_forced_neighbor = True
        
        # 如果有强制邻居，返回当前跳点
        if has_forced_neighbor:
            return (nx, ny)
        
        # 递归继续在相同方向跳跃搜索
        return self._jump_point_search((nx, ny), direction, end)

    def _find_nearest_non_obstacle(self, point, max_radius=5):
        """
        寻找最近的非障碍点
        
        Args:
            point: 原始坐标点
            max_radius: 最大搜索半径
            
        Returns:
            Tuple 或 None: 找到的非障碍点，或None表示未找到
        """
        # 搜索从内向外扩张
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # 只检查当前圈
                    if max(abs(dx), abs(dy)) != r:
                        continue
                        
                    candidate = (point[0] + dx, point[1] + dy)
                    if not self._is_obstacle_fast(candidate):
                        return candidate
        
        return None  # 未找到合适点
    
    def _close_enough(self, current, end, threshold=3.0):
        """检查当前点是否足够接近终点"""
        return math.dist(current, end) <= threshold
    
    def _calculate_angle_change(self, dir1, dir2):
        """计算两个方向向量之间的角度变化(度)"""
        # 计算两个向量的点积
        dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1]
        
        # 计算向量长度
        len1 = math.sqrt(dir1[0]**2 + dir1[1]**2)
        len2 = math.sqrt(dir2[0]**2 + dir2[1]**2)
        
        # 计算夹角的余弦值
        if len1 * len2 == 0:
            return 0
            
        cos_angle = dot_product / (len1 * len2)
        
        # 防止浮点误差导致cos_angle超出[-1,1]范围
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # 转换为角度
        angle = math.degrees(math.acos(cos_angle))
        return angle
    
    def _reconstruct_path(self, came_from, current, end):
        """
        重建路径
        
        Args:
            came_from: 父节点字典
            current: 当前节点(终点)
            end: 原始终点坐标
            
        Returns:
            List[Tuple]: 完整路径
        """
        # 创建路径
        path = [current]
        
        # 从终点回溯到起点
        while current in came_from:
            current = came_from[current]
            path.append(current)
            
        # 反转路径，从起点到终点
        path.reverse()
        
        # 确保终点在路径中
        if not self._points_equal(path[-1], end):
            path.append(end)
            
        return path
    
    def _generate_fallback_path(self, start, end):
        """
        生成备选路径，确保避开障碍物
        """
        try:
            # 标准化坐标
            start = self._validate_point(start)
            end = self._validate_point(end)
                
            # 直线距离和方向
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # 检查直线路径上的障碍物
            line_points = self._bresenham_line(start, end)
            has_obstacles = False
            
            for point in line_points:
                if self._is_obstacle_fast(point):
                    has_obstacles = True
                    break
            
            # 如果直线路径无障碍，直接返回
            if not has_obstacles:
                return [start, end]
            
            # 有障碍物，尝试更复杂的绕行路径
            
            # 尝试找到多个中间点绕行
            path = [start]
            
            # 尝试几种不同的绕行方案
            detour_attempts = [
                # 尝试上方绕行
                (start[0] + dx/3, start[1] + abs(dx)/2),
                (start[0] + 2*dx/3, start[1] + abs(dx)/2),
                
                # 尝试下方绕行
                (start[0] + dx/3, start[1] - abs(dx)/2),
                (start[0] + 2*dx/3, start[1] - abs(dx)/2),
                
                # 尝试左侧绕行
                (start[0] - abs(dy)/2, start[1] + dy/3),
                (start[0] - abs(dy)/2, start[1] + 2*dy/3),
                
                # 尝试右侧绕行
                (start[0] + abs(dy)/2, start[1] + dy/3),
                (start[0] + abs(dy)/2, start[1] + 2*dy/3),
            ]
            
            # 尝试各种绕行路径
            for i in range(0, len(detour_attempts), 2):
                mid1 = detour_attempts[i]
                mid2 = detour_attempts[i+1]
                
                # 验证中间点不在障碍物上
                if not self._is_obstacle_fast(mid1) and not self._is_obstacle_fast(mid2):
                    # 验证连接线不穿过障碍物
                    line1 = self._bresenham_line(start, mid1)
                    line2 = self._bresenham_line(mid1, mid2)
                    line3 = self._bresenham_line(mid2, end)
                    
                    if (not any(self._is_obstacle_fast(p) for p in line1) and
                        not any(self._is_obstacle_fast(p) for p in line2) and
                        not any(self._is_obstacle_fast(p) for p in line3)):
                        # 找到有效路径
                        return [start, mid1, mid2, end]
            
            # 所有尝试都失败，使用网格搜索寻找简单路径
            # 这是最后的备选方案，不会太高效但确保能找到路径
            grid_size = 20
            best_path = None
            min_length = float('inf')
            
            # 生成一组网格点
            for dx in range(-5, 6, 2):
                for dy in range(-5, 6, 2):
                    # 在起点和终点间寻找一个中点
                    mid_point = (
                        (start[0] + end[0]) / 2 + dx * grid_size,
                        (start[1] + end[1]) / 2 + dy * grid_size
                    )
                    
                    # 检查中点是否在障碍物上
                    if self._is_obstacle_fast(mid_point):
                        continue
                        
                    # 检查从起点到中点的路径
                    line1 = self._bresenham_line(start, mid_point)
                    if any(self._is_obstacle_fast(p) for p in line1):
                        continue
                        
                    # 检查从中点到终点的路径
                    line2 = self._bresenham_line(mid_point, end)
                    if any(self._is_obstacle_fast(p) for p in line2):
                        continue
                    
                    # 找到一条可行路径
                    path_length = len(line1) + len(line2)
                    if path_length < min_length:
                        min_length = path_length
                        best_path = [start, mid_point, end]
            
            # 如果找到了可行路径，返回
            if best_path:
                return best_path
                
            # 最终方案：大范围绕行
            perimeter_size = max(abs(dx), abs(dy)) * 2
            corners = [
                (start[0] - perimeter_size, start[1] - perimeter_size),
                (start[0] - perimeter_size, end[1] + perimeter_size),
                (end[0] + perimeter_size, end[1] + perimeter_size),
                (end[0] + perimeter_size, start[1] - perimeter_size)
            ]
            
            # 找到至少两个不在障碍物上的角点
            valid_corners = [c for c in corners if not self._is_obstacle_fast(c)]
            
            if len(valid_corners) >= 2:
                # 选择第一个和最后一个有效角点
                return [start, valid_corners[0], valid_corners[-1], end]
                
            # 所有尝试都失败，只能返回直线路径
            logging.warning(f"无法找到避开障碍物的路径 {start} -> {end}，返回直线路径")
            return [start, end]
            
        except Exception as e:
            logging.error(f"备选路径生成失败: {str(e)}")
            # 最简单的后备方案 - 直接连接起点和终点
            return [start, end]

    def reserve_path(self, path, vehicle_id):
        """
        为路径预约时间窗口
        
        Args:
            path: 路径点列表
            vehicle_id: 车辆ID
        """
        if not path or len(path) < 2:
            return
            
        with self.reservation_lock:
            # 清除该车辆之前的预约
            self.clear_path_reservation(vehicle_id)
            
            # 添加新预约
            for i in range(len(path) - 1):
                segment = (path[i], path[i+1])
                self.reservation_table[segment] = vehicle_id
                
    def clear_path_reservation(self, vehicle_id):
        """清除车辆的路径预约"""
        with self.reservation_lock:
            segments_to_remove = [s for s, v in self.reservation_table.items() if v == vehicle_id]
            for segment in segments_to_remove:
                if segment in self.reservation_table:
                    del self.reservation_table[segment]
                    
    def check_path_conflict(self, path, vehicle_id):
        """
        检查路径是否与其他车辆预约冲突
        
        Args:
            path: 路径点列表
            vehicle_id: 车辆ID
            
        Returns:
            bool: 是否存在冲突
        """
        if not path or len(path) < 2:
            return False
            
        with self.reservation_lock:
            # 检查每个路径段
            for i in range(len(path) - 1):
                segment = (path[i], path[i+1])
                
                # 检查该段是否已被其他车辆预约
                if segment in self.reservation_table and self.reservation_table[segment] != vehicle_id:
                    return True
                    
                # 检查反向段
                reverse_segment = (path[i+1], path[i])
                if reverse_segment in self.reservation_table and self.reservation_table[reverse_segment] != vehicle_id:
                    return True
                    
                # 检查交叉点
                for other_segment, other_id in self.reservation_table.items():
                    if other_id != vehicle_id:
                        if self._segments_intersect(segment[0], segment[1], other_segment[0], other_segment[1]):
                            return True
                            
            return False
            
    def _segments_intersect(self, p1, p2, p3, p4):
        """检查两线段是否相交"""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if abs(val) < EPSILON:
                return 0  # 共线
            return 1 if val > 0 else 2  # 顺时针/逆时针
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # 一般情况
        if o1 != o2 and o3 != o4:
            return True
            
        # 特殊情况
        if o1 == 0 and self._on_segment(p1, p3, p2): return True
        if o2 == 0 and self._on_segment(p1, p4, p2): return True
        if o3 == 0 and self._on_segment(p3, p1, p4): return True
        if o4 == 0 and self._on_segment(p3, p2, p4): return True
        
        return False
    
    def _on_segment(self, p, q, r):
        """检查点q是否在线段pr上"""
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
    
    def clear_caches(self):
        """清除所有缓存数据"""
        self.path_cache.clear()
        
    def get_performance_stats(self):
        """
        获取性能统计
        
        Returns:
            Dict: 性能统计信息
        """
        runtime = time.time() - self.load_time
        avg_planning_time = self.total_planning_time / self.planning_count if self.planning_count > 0 else 0
        success_rate = self.success_count / self.planning_count if self.planning_count > 0 else 0
        
        return {
            'runtime': f"{runtime:.1f}秒",
            'planning_count': self.planning_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': f"{success_rate*100:.1f}%",
            'avg_planning_time': f"{avg_planning_time*1000:.2f}毫秒",
            'cache_stats': self.path_cache.get_stats(),
            'obstacle_count': len(self.obstacle_grids),
            'reservations': len(self.reservation_table)
        }

    def plan_with_timeout(self, start, end, vehicle=None, timeout=DEFAULT_TIMEOUT):
        """
        带超时控制的路径规划方法
        
        Args:
            start: 起点坐标
            end: 终点坐标
            vehicle: 可选的车辆对象
            timeout: 超时时间(秒)
            
        Returns:
            Tuple[List[Tuple], float]: (路径, 耗时)
        """
        import queue
        import threading
        
        # 使用队列安全地传递结果
        result_queue = queue.Queue()
        
        def planning_worker():
            try:
                # 执行路径规划
                path_result = self.plan_path(start, end, vehicle)
                result_queue.put(path_result)
            except Exception as e:
                # 捕获异常并放入队列
                logging.error(f"路径规划线程出错: {str(e)}")
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
            # 超时情况，生成备用路径
            logging.warning(f"路径规划超时 ({timeout}秒)，生成备用路径")
            path = self._generate_fallback_path(start, end)
            return path, elapsed
        else:
            try:
                # 检查队列中的结果
                result = result_queue.get(block=False)
                if isinstance(result, Exception):
                    # 出错，生成备用路径
                    logging.warning(f"路径规划失败: {str(result)}，生成备用路径")
                    path = self._generate_fallback_path(start, end)
                    return path, elapsed
                elif not result or len(result) < 2:
                    # 无效结果，生成备用路径
                    logging.warning(f"无效路径，生成备用路径")
                    path = self._generate_fallback_path(start, end)
                    return path, elapsed
                else:
                    # 规划成功
                    logging.info(f"路径规划成功！路径包含 {len(result)} 个点，用时 {elapsed:.3f} 秒")
                    return result, elapsed
            except queue.Empty:
                # 队列为空，生成备用路径
                logging.warning(f"路径规划过程异常终止，生成备用路径")
                path = self._generate_fallback_path(start, end)
                return path, elapsed
    
    def _create_cache_key(self, start, end, vehicle):
        """创建缓存键"""
        if vehicle:
            # 包含车辆属性的缓存键
            return (
                "path",
                start,
                end,
                getattr(vehicle, 'turning_radius', 0), 
                getattr(vehicle, 'min_hardness', 0),
                getattr(vehicle, 'current_load', 0)
            )
        else:
            # 基本缓存键
            return ("base_path", start, end)
            
    def _validate_point(self, point):
        """验证并标准化坐标点"""
        if isinstance(point, tuple) and len(point) >= 2:
            return (float(point[0]), float(point[1]))
        elif hasattr(point, 'as_tuple'):
            return point.as_tuple()
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            return (float(point.x), float(point.y))
        elif isinstance(point, (list, np.ndarray)) and len(point) >= 2:
            return (float(point[0]), float(point[1]))
        else:
            # 无效点警告
            logging.warning(f"无效的坐标点: {point}，使用(0,0)")
            return (0.0, 0.0)
    def _points_equal(self, p1, p2, tolerance=EPSILON):
        """检查两点是否相等(考虑浮点误差)"""
        return (abs(p1[0] - p2[0]) < tolerance and 
                abs(p1[1] - p2[1]) < tolerance)