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
                if self._is_obstacle(p):
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
    
    def _astar(self, start, end, vehicle=None):
        """
        A*路径规划算法实现
        
        Args:
            start: 起点坐标
            end: 终点坐标
            vehicle: 可选的车辆对象
            
        Returns:
            List[Tuple[float, float]]: 规划的路径点列表
        """
        try:
            # 检查起点或终点是否为障碍物
            start_is_obstacle = self._is_obstacle(start)
            end_is_obstacle = self._is_obstacle(end)
            
            # 尝试调整起点
            if start_is_obstacle:
                logging.warning(f"起点 {start} 是障碍物，尝试调整...")
                adjusted_start = self._find_nearest_non_obstacle(start)
                if adjusted_start:
                    start = adjusted_start
                else:
                    # 无法调整，放弃A*
                    logging.warning("无法找到调整后的起点，使用备选路径")
                    return None
            
            # 尝试调整终点
            if end_is_obstacle:
                logging.warning(f"终点 {end} 是障碍物，尝试调整...")
                adjusted_end = self._find_nearest_non_obstacle(end)
                if adjusted_end:
                    end = adjusted_end
                else:
                    # 无法调整，放弃A*
                    logging.warning("无法找到调整后的终点，使用备选路径")
                    return None
            
            # 启发式函数：曼哈顿距离+欧几里得距离的加权
            def heuristic(x, y):
                dx, dy = abs(x - end[0]), abs(y - end[1])
                # 曼哈顿距离和欧几里得距离的加权组合
                return 0.5 * (dx + dy) + 0.5 * math.sqrt(dx*dx + dy*dy)
            
            # 优先队列（最小堆）与访问集合
            open_list = []
            heapq.heappush(open_list, (heuristic(start[0], start[1]), start))
            closed_set = set()
            came_from = {}
            g_score = {start: 0}
            
            # 主搜索循环
            iterations = 0
            max_iterations = MAX_ITERATIONS
            
            # 设置车辆特定的搜索参数
            if vehicle:
                # 如果是满载车辆，降低最大迭代次数（避免过度搜索）
                if hasattr(vehicle, 'current_load') and vehicle.current_load > 0.8 * getattr(vehicle, 'max_capacity', 50):
                    max_iterations = int(MAX_ITERATIONS * 0.8)
                    
                # 如果是空车，可以更激进地搜索
                if hasattr(vehicle, 'current_load') and vehicle.current_load < 0.2 * getattr(vehicle, 'max_capacity', 50):
                    max_iterations = int(MAX_ITERATIONS * 1.2)
            
            while open_list and iterations < max_iterations:
                iterations += 1
                
                # 获取f值最低的点
                _, current = heapq.heappop(open_list)
                
                # 目标检查：如果已经足够接近目标
                if self._close_enough(current, end):
                    # 重建路径
                    path = self._reconstruct_path(came_from, current, end)
                    logging.debug(f"A*搜索成功，共{iterations}次迭代，路径长度={len(path)}")
                    return path
                
                # 标记为已访问
                closed_set.add(current)
                
                # 检查所有邻居
                for dx, dy in self.directions:
                    # 计算邻居坐标
                    nx, ny = current[0] + dx, current[1] + dy
                    neighbor = (nx, ny)
                    
                    # 跳过已访问
                    if neighbor in closed_set:
                        continue
                    
                    # 跳过障碍物
                    if self._is_obstacle(neighbor):
                        continue
                    
                    # 对角线移动时检查拐角阻塞
                    if dx != 0 and dy != 0:
                        if self._is_obstacle((current[0], ny)) or self._is_obstacle((nx, current[1])):
                            continue
                    
                    # 计算移动成本
                    move_cost = self.move_costs.get((dx, dy), 1.0)
                    
                    # 考虑车辆特性
                    if vehicle:
                        # 载重影响
                        if hasattr(vehicle, 'current_load') and hasattr(vehicle, 'max_capacity'):
                            load_ratio = vehicle.current_load / vehicle.max_capacity
                            move_cost *= (1.0 + load_ratio * 0.5)  # 满载时成本增加50%
                            
                        # 最小转弯半径考虑
                        if hasattr(vehicle, 'turning_radius') and vehicle.turning_radius > 0:
                            if current in came_from:
                                prev = came_from[current]
                                # 检查转弯角度
                                prev_dx, prev_dy = current[0] - prev[0], current[1] - prev[1]
                                angle_change = self._calculate_angle_change((prev_dx, prev_dy), (dx, dy))
                                if angle_change > 90:  # 急转弯
                                    move_cost *= 1.5  # 急转弯成本增加
                    
                    # 地形影响（如果地图服务提供）
                    try:
                        if hasattr(self.map_service, 'get_terrain_hardness'):
                            hardness = self.map_service.get_terrain_hardness(nx, ny)
                            if hardness < getattr(vehicle, 'min_hardness', 0):
                                continue  # 地形太软，无法通过
                            move_cost *= max(1.0, 3.0 / hardness)  # 地形软度增加成本
                    except:
                        pass  # 忽略地形检查错误
                    
                    # 路径更新
                    tentative_g = g_score[current] + move_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_value = tentative_g + heuristic(nx, ny)
                        
                        # 添加到开放列表
                        heapq.heappush(open_list, (f_value, neighbor))
            
            # 搜索结束但未找到路径
            if iterations >= max_iterations:
                logging.warning(f"A*搜索达到最大迭代次数 {max_iterations}，使用备选路径")
            else:
                logging.warning(f"A*搜索未找到从 {start} 到 {end} 的路径，使用备选路径")
            
            return None
            
        except Exception as e:
            logging.error(f"A*算法执行错误: {str(e)}")
            return None
    
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
                    if not self._is_obstacle(candidate):
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
                if self._is_obstacle(point):
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
                if not self._is_obstacle(mid1) and not self._is_obstacle(mid2):
                    # 验证连接线不穿过障碍物
                    line1 = self._bresenham_line(start, mid1)
                    line2 = self._bresenham_line(mid1, mid2)
                    line3 = self._bresenham_line(mid2, end)
                    
                    if (not any(self._is_obstacle(p) for p in line1) and
                        not any(self._is_obstacle(p) for p in line2) and
                        not any(self._is_obstacle(p) for p in line3)):
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
                    if self._is_obstacle(mid_point):
                        continue
                        
                    # 检查从起点到中点的路径
                    line1 = self._bresenham_line(start, mid_point)
                    if any(self._is_obstacle(p) for p in line1):
                        continue
                        
                    # 检查从中点到终点的路径
                    line2 = self._bresenham_line(mid_point, end)
                    if any(self._is_obstacle(p) for p in line2):
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
            valid_corners = [c for c in corners if not self._is_obstacle(c)]
            
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