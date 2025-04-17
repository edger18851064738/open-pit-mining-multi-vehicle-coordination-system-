import sys
import os
import math
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import heapq
import logging
from typing import List, Tuple, Dict, Optional,Union
import networkx as nx
from utils.geo_tools import GeoUtils
from config.settings import MapConfig, PathConfig
from algorithm.map_service import MapService
from utils.path_tools import PathOptimizationError
from config.settings import AppConfig  # 新增关键导入
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import time
import threading

import numpy as np
# 移除重复定义的PathOptimizationError，使用utils.path_tools中的定义

class Node:
    """三维路径节点（含时间维度）"""
    __slots__ = ('x', 'y', 't')
    
    def __init__(self, x: int, y: int, t: int = 0):
        self.x = x
        self.y = y
        self.t = t
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
        
    def __hash__(self):
        return hash((self.x, self.y))
    
    # 新增比较运算符
    def __lt__(self, other):
        # 当f值相等时，按坐标排序
        return (self.x, self.y) < (other.x, other.y)
class MiningVehicle:
    """矿用运输车辆实体（简化测试版）"""
    def __init__(self, vehicle_id: str, config: Dict):
        self.vehicle_id = vehicle_id
        # 确保从配置获取min_hardness，并设置默认值
        self.min_hardness = config.get('min_hardness', 2.5)  # 新增默认值
        self.max_load = config.get('max_load', 50)
        self.current_load = config.get('current_load', 0)  # 已存在的正确属性
        self.speed = config.get('speed', 5)
        self.steering_angle = config.get('steering_angle', 30)
        self.last_position = None
        self.turning_radius = config.get('turning_radius', 10.0)  # 新增转向半径属性

class HybridPathPlanner:
    """矿山场景专用路径规划器"""
    
    def __init__(self, map_service):
        self.obstacle_grids = set()
        self.map_service = map_service
        self.haul_roads = set()  # 新增运输道路集合
        self.reservation_table = {}  # 时间窗口预约表
        self.dynamic_obstacles = set()  # 动态障碍物集合
        self.conflict_check_interval = 0.5  # 冲突检测间隔(秒)
        self.path_cache = {}  # 添加路径缓存字典
        
        try:
            self._load_mine_config()
        except Exception as e:  # 添加配置加载异常处理
            logging.warning(f"配置加载失败，使用默认值: {str(e)}")
            self.grid_size = 20.0
            self.max_grade = 15.0
            self.min_turn_radius = 15.0
    def _is_goal_reached(self, current, goal, threshold=1.0):
        """判断是否到达目标点（欧氏距离）"""
        return math.hypot(current[0]-goal[0], current[1]-goal[1]) <= threshold
        
    def _check_path_conflict(self, path: List[Tuple], vehicle_id: str) -> bool:
        """增强版路径冲突检测
        
        参数:
            path: 待检测路径
            vehicle_id: 当前车辆ID
            
        返回:
            bool: 是否检测到冲突
        """
        with self.reservation_lock:
            # 检查静态障碍物
            for point in path:
                if point in self.obstacle_grids or point in self.dynamic_obstacles:
                    return True
                    
            # 检查路径线段是否穿过障碍物
            for i in range(len(path)-1):
                segment = (path[i], path[i+1])
                # 使用更精确的Bresenham算法检测直线路径上的每个点
                points = GeoUtils.bresenham_line(segment[0], segment[1])
                for point in points:
                    if point in self.obstacle_grids:
                        return True
                    
            # 检查时间窗口预约
            for i in range(len(path)-1):
                segment = (path[i], path[i+1])
                if segment in self.reservation_table and self.reservation_table[segment] != vehicle_id:
                    return True
                    
        return False
        
    def _reserve_path_segment(self, segment: Tuple[Tuple, Tuple], vehicle_id: str):
        """预约路径段的时间窗口"""
        with self.reservation_lock:
            self.reservation_table[segment] = vehicle_id
            
    def _clear_path_reservation(self, vehicle_id: str):
        """清除车辆的路径预约"""
        with self.reservation_lock:
            segments = [seg for seg, vid in self.reservation_table.items() if vid == vehicle_id]
            for seg in segments:
                del self.reservation_table[seg]        
    def _load_mine_config(self):
        """从配置加载参数（增强容错）"""
        config = AppConfig.load()
        # 添加空值检查
        if config and hasattr(config, 'map'):
            self.grid_size = config.map.GRID_SIZE
            self.max_grade = config.map.MAX_GRADE
            self.min_turn_radius = config.map.MIN_TURN_RADIUS
        else:  # 添加默认值保障
            self.grid_size = 20.0
            self.max_grade = 15.0
            self.min_turn_radius = 15.0

    def optimize_path(self, start: Tuple[float, float], 
                    end: Tuple[float, float],
                    vehicle: MiningVehicle) -> List[Tuple[float, float]]:
        """整合调用关系的修正版（添加路径缓存机制）"""
        # 生成缓存键（使用起点和终点坐标作为键）
        cache_key = (start, end)
        
        # 检查缓存中是否已有该路径
        if cache_key in self.path_cache:
            logging.debug(f"使用缓存路径: {start} -> {end}")
            return self.path_cache[cache_key]
            
        # 调用plan_path获取基础路径
        base_path = self.plan_path(start, end)
        
        # 应用RS曲线优化
        optimized_path = self._apply_rs_curve(base_path, vehicle.turning_radius)
        
        # 坐标转换（保持原有逻辑）
        try:
            # 尝试获取虚拟坐标原点
            if hasattr(self.map_service, 'config') and hasattr(self.map_service.config, 'get'):
                data_type = self.map_service.config.get('MAP', 'data_type', fallback='')
                if data_type == 'virtual':
                    origin_str = self.map_service.config.get('MAP', 'virtual_origin', fallback='0,0')
                    origin_x, origin_y = map(float, origin_str.split(','))
                    optimized_path = [(p[0]+origin_x, p[1]+origin_y) for p in optimized_path]
        except Exception as e:
            logging.warning(f"坐标转换失败，使用原始坐标: {str(e)}")
        
        # 将结果存入缓存
        self.path_cache[cache_key] = optimized_path
        logging.debug(f"路径已缓存: {start} -> {end}")
            
        return optimized_path

    def _apply_rs_curve(self, path: List[Tuple], radius: float) -> List[Tuple]:
        """批量应用RS曲线"""
        if len(path) < 2:
            return path
            
        rs_path = []
        for i in range(len(path)-1):
            segment = self._generate_rs_path(path[i], path[i+1], radius)
            rs_path.extend(segment[:-1])
        rs_path.append(path[-1])
        return rs_path
    def _calculate_move_cost(self, start, end, load, hardness):
        """动态移动成本计算"""
        # 将Node对象的访问方式从下标改为属性访问
        distance = math.hypot(end.x - start.x, end.y - start.y)
        # 使用Node的属性获取坐标
        terrain_cost = self.map_service.get_terrain_hardness(end.x, end.y)
        return distance * (1 + load/50000) * max(1, terrain_cost/hardness)

    def _is_line_through_obstacle(self, start, end):
        """检测两点之间的直线是否穿过障碍物"""
        # 使用Bresenham算法生成直线上的所有点
        points = GeoUtils.bresenham_line(start, end)
        # 检查每个点是否在障碍物集合中
        for point in points:
            if point in self.obstacle_grids:
                return True
        return False
        
    def _generate_rs_path(self, current, end, radius):
        """RS曲线生成（修复坐标类型）"""
        # 转换current为tuple类型
        if isinstance(current, Node):
            current = (current.x, current.y)
        path = [current]
        
        # 修复Node对象访问方式
        if isinstance(end, Node):
            end_point = (end.x, end.y)
        else:
            end_point = end  # 保留原有逻辑
        
        dx = end_point[0] - current[0]
        dy = end_point[1] - current[1]
        
        # 检查直接路径是否穿过障碍物
        if not self._is_line_through_obstacle(current, end_point):
            # 直接路径无障碍物，使用直线
            path.append(end_point)
            return path
            
        # 需要绕过障碍物，生成RS曲线
        mid_point = (
            current[0] + dx/2 - dy/radius,
            current[1] + dy/2 + dx/radius
        )
        path.append(mid_point)
        path.append(end_point)
        return path
    def plan_path(self, start: Tuple, end: Tuple) -> List[Tuple]:
        """虚拟坐标转换的核心路径规划方法（添加路径缓存机制）"""
        # 生成缓存键（基础路径规划）
        base_cache_key = ("base_path", start, end)
        
        # 检查缓存中是否已有该基础路径
        if base_cache_key in self.path_cache:
            logging.debug(f"使用缓存基础路径: {start} -> {end}")
            return self.path_cache[base_cache_key]
            
        # 移除对config的依赖，直接使用默认值
        try:
            # 尝试获取虚拟坐标原点
            if hasattr(self.map_service, 'config') and hasattr(self.map_service.config, 'get'):
                data_type = self.map_service.config.get('MAP', 'data_type', fallback='')
                if data_type == 'virtual':
                    origin_str = self.map_service.config.get('MAP', 'virtual_origin', fallback='0,0')
                    origin_x, origin_y = map(float, origin_str.split(','))
                    start = (start[0] - origin_x, start[1] - origin_y)
                    end = (end[0] - origin_x, end[1] - origin_y)
        except Exception as e:
            logging.warning(f"坐标转换失败，使用原始坐标: {str(e)}")

        # 调用核心算法（保持原有逻辑）
        try:
            raw_path = self._mine_astar(Node(*start), Node(*end), MiningVehicle("dummy", {}))
            result = self._smooth_path(raw_path)
            
            # 将结果存入缓存
            self.path_cache[base_cache_key] = result
            logging.debug(f"基础路径已缓存: {start} -> {end}")
            
            return result
        except PathOptimizationError as e:
            logging.error(f"路径规划失败: {str(e)}")
            return []
    def _mine_astar(self, start: Node, end: Node, vehicle: MiningVehicle):
        """优化后的A*算法实现"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        closed_set = set()
        
        # 八方向移动：上下左右 + 对角线
        directions = [(-1,0), (1,0), (0,-1), (0,1),
                     (-1,-1), (-1,1), (1,-1), (1,1)]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end:
                return self._build_path(came_from, current)
                
            if current in closed_set:
                continue
            closed_set.add(current)
            
            for dx, dy in directions:
                neighbor = Node(current.x + dx, current.y + dy)
                
                # 跳过已处理的节点
                if neighbor in closed_set:
                    continue
                    
                # 增强障碍物检查
                if (neighbor.x, neighbor.y) in self.obstacle_grids:
                    continue
                    
                # 检查当前节点到邻居节点之间是否有障碍物
                if not self._is_straight_line((current.x, current.y), (neighbor.x, neighbor.y)):
                    continue
                    
                if not self._check_vehicle_constraints(neighbor, vehicle):
                    continue

                # 预计算地形硬度和移动成本
                terrain_hardness = self.map_service.get_terrain_hardness(neighbor.x, neighbor.y)
                move_cost = self._calculate_move_cost(current, neighbor, vehicle.current_load, terrain_hardness)
                
                # 对角线移动额外成本
                if abs(dx) + abs(dy) == 2:
                    move_cost *= 1.4
                
                # 运输道路优先
                if (neighbor.x, neighbor.y) in self.haul_roads:
                    move_cost *= 0.6
                    
                tentative_g = g_score[current] + move_cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        raise PathOptimizationError("无可行路径")
# 在_mine_astar方法后添加以下方法
    def _smooth_path(self, raw_path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """路径平滑处理（简化版）"""
        if len(raw_path) < 3:
            return raw_path
            
        smoothed = [raw_path[0]]
        for point in raw_path[1:-1]:
            # 简单过滤锯齿状路径
            if not self._is_straight_line(smoothed[-1], raw_path[raw_path.index(point)+1]):
                smoothed.append(point)
        smoothed.append(raw_path[-1])
        return smoothed

    def _is_straight_line(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """判断两点间是否无障碍"""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        steps = int(max(abs(dx), abs(dy)))
        
        for i in range(1, steps+1):
            x = a[0] + round(i*dx/steps)
            y = a[1] + round(i*dy/steps)
            if (x, y) in self.obstacle_grids:
                return False
        return True
    def _calculate_cost(self, current: Node, neighbor: Node) -> float:
        """地形感知的移动成本计算"""
        base_cost = 1.0
        if self._is_haul_road(neighbor.x, neighbor.y):
            return base_cost * 0.6
        return base_cost * self._get_terrain_resistance(neighbor.x, neighbor.y)

    def _heuristic(self, a: Union[Tuple, Node], b: Union[Tuple, Node]) -> float:
        """优化后的启发式函数，使用对角线距离(Diagonal distance)"""
        def get_x(point): return point.x if isinstance(point, Node) else point[0]
        def get_y(point): return point.y if isinstance(point, Node) else point[1]
        dx = abs(get_x(a) - get_x(b))
        dy = abs(get_y(a) - get_y(b))
        return (dx + dy) + (1.414 - 2) * min(dx, dy)
    def mark_haul_road(self, polygon):
        """标记运输道路区域"""
        min_x = min(p[0] for p in polygon)
        max_x = max(p[0] for p in polygon)
        min_y = min(p[1] for p in polygon)
        max_y = max(p[1] for p in polygon)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                self.haul_roads.add((x, y))
        logging.debug(f"标记运输道路区域: {polygon} -> {max_x-min_x+1}x{max_y-min_y+1} 网格")
    def mark_obstacle_area(self, polygons: List[List[Tuple[int, int]]]):
        """批量标记障碍物区域"""
        for polygon in polygons:
            min_x = min(p[0] for p in polygon)
            max_x = max(p[0] for p in polygon)
            min_y = min(p[1] for p in polygon)
            max_y = max(p[1] for p in polygon)
            
            for x in range(min_x, max_x+1):
                for y in range(min_y, max_y+1):
                    if self._point_in_polygon((x,y), polygon):
                        self.obstacle_grids.add((x,y))

    def _point_in_polygon(self, point, polygon) -> bool:
        """射线法判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i+1)%n]
            
            if (x == p1[0] and y == p1[1]) or (x == p2[0] and y == p2[1]):
                return True
                
            if ((p1[1] > y) != (p2[1] > y)):
                xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                if x == xinters:
                    return True
                if x < xinters:
                    inside = not inside
                    
        return inside

    def _validate_terrain(self, node: Node, vehicle: MiningVehicle) -> bool:
        """地形综合验证（增加调试日志）"""
        grade = self._get_terrain_grade(node.x, node.y)
        hardness = self._get_terrain_hardness(node.x, node.y)
        turn_valid = self._validate_turn_radius(node, vehicle)
        
        if grade > self.max_grade:
            logging.debug(f"节点({node.x},{node.y})坡度{grade}%超过限制{self.max_grade}%")
        if hardness < vehicle.min_hardness:
            logging.debug(f"节点({node.x},{node.y})硬度{hardness:.1f}低于要求{vehicle.min_hardness}")
        if not turn_valid:
            logging.debug(f"节点({node.x},{node.y})转弯半径验证失败")
            
        return grade <= self.max_grade and hardness >= vehicle.min_hardness and turn_valid
        
    def _validate_turn_radius(self, node: Node, vehicle: MiningVehicle) -> bool:
        """统一坐标访问方式"""
        if not vehicle.last_position or vehicle.steering_angle <= 0:
            return True
            
        # 统一使用Node属性访问
        dx = node.x - vehicle.last_position.x
        dy = node.y - vehicle.last_position.y
        
        if dx == 0 and dy == 0:
            return True
            
        try:
            turn_radius = math.hypot(dx, dy) / (2 * math.sin(math.radians(vehicle.steering_angle)))
            return turn_radius >= self.min_turn_radius
        except (ZeroDivisionError, AttributeError):
            return False

    def _build_path(self, came_from: dict, current: Node) -> List[Tuple[int, int]]:
        """重构路径并添加时间维度"""
        path = []
        while current in came_from:
            path.append((current.x, current.y))
            current = came_from[current]
        return list(reversed(path))

    def _check_vehicle_constraints(self, node: Node, vehicle: MiningVehicle) -> bool:
        """车辆约束检查"""
        return (node.x, node.y, node.t) not in self.reservation_table

    # 简化版地图服务方法
    def _is_haul_road(self, x: int, y: int) -> bool:
        return (x + y) % 10 == 0  # 虚拟运输道路模式

    def _get_terrain_resistance(self, x: int, y: int) -> float:
        return 0.8 + (x % 3 + y % 2) * 0.2

    def _get_terrain_grade(self, x: int, y: int) -> float:
        return abs(x - y) % 20

    def _get_terrain_hardness(self, x: int, y: int) -> float:
        return 3.0 - (x % 3 + y % 2) * 0.3

    # 在HybridPathPlanner类中添加资源释放方法
    def release_resources(self):
        """释放FORTRAN底层资源"""
        if hasattr(self, '_fortran_handle'):
            try:
                self._fortran_handle.cleanup()  # 调用FORTRAN的清理函数
                del self._fortran_handle
            except Exception as e:
                logging.error(f"资源释放失败: {str(e)}")
    def _debug_visualize(self, path):
        """路径调试可视化"""
        plt.figure(figsize=(10, 10))
        plt.scatter(*zip(*self.obstacle_grids), c='gray', s=10, label='Obstacles')
        if path:
            plt.plot(*zip(*path), 'r-', linewidth=2, label='Path')
        plt.legend()
        plt.show()
if __name__ == "__main__":
    # 配置调试日志
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rcParams['font.sans-serif'] = ['SimHei']  # Windows系统字体
    rcParams['axes.unicode_minus'] = False 
    
    # 初始化规划器（关键修复部分）
    class MockMapService:
        def __init__(self):
            self.config = {  # 新增config属性
                'MAP': {
                    'data_type': 'virtual',
                    'virtual_origin': '50,50',
                    'virtual_range': '100,100'
                }
            }
        def validate_coordinates(self, coord):
            """模拟坐标验证（始终返回True）"""
            return True
        
        def get_terrain_data(self, x, y):
            return {"grade": abs(x-y)%20, "hardness": 3.0 - (x%3 + y%2)*0.3}
        # 新增地形硬度获取方法
        def get_terrain_hardness(self, x, y):
            return self.get_terrain_data(x, y)["hardness"]
    planner = HybridPathPlanner(MockMapService())

    # 测试用例配置
    test_cases = [
        {
            "name": "基础测试",
            "obstacles": [
                [(10,45), (90,45), (90,55), (10,55)],
                [(40,20), (60,20), (60,30), (40,30)],
                [(40,70), (60,70), (60,80), (40,80)],
                [(25,20), (35,20), (35,80), (25,80)],
                [(65,20), (75,20), (75,80), (65,80)],
                [(45,30), (55,30), (55,40), (45,40)],
                [(45,60), (55,60), (55,70), (45,70)],
                [(20,25), (30,25), (30,35), (20,35)],
                [(70,65), (80,65), (80,75), (70,75)]
            ],
            "start": (10, 10),
            "end": (60, 60),
            "radius": 10.0
        },
        {
            "name": "复杂迷宫测试",
            "obstacles": [
                [(20,20), (80,20), (80,80), (20,80)],  # 外框
                [(30,30), (70,30), (70,70), (30,70)],  # 内框
                [(40,40), (60,40), (60,60), (40,60)],  # 核心区
                [(25,25), (35,25), (35,75), (25,75)],  # 左侧障碍
                [(65,25), (75,25), (75,75), (65,75)]   # 右侧障碍
            ],
            "start": (10, 10),
            "end": (90, 90),
            "radius": 15.0
        },
        {
            "name": "小半径转弯测试",
            "obstacles": [
                [(30,30), (70,30), (70,70), (30,70)]
            ],
            "start": (10, 50),
            "end": (90, 50),
            "radius": 5.0
        }
    ]

    # 执行测试用例
    for case in test_cases:
        print(f"\n=== 测试用例: {case['name']} ===")
        print(f"起点: {case['start']}, 终点: {case['end']}, 转向半径: {case['radius']}")
        
        # 清除之前的障碍物
        planner.obstacle_grids.clear()
        planner.mark_obstacle_area(case['obstacles'])
        
        # 创建测试车辆
        vehicle = MiningVehicle("test_vehicle", {
            'turning_radius': case['radius'],
            'min_hardness': 2.0
        })
        
        # 验证起点终点可达性
        if case['start'] in planner.obstacle_grids:
            raise PathOptimizationError("起点位于障碍物内")
        if case['end'] in planner.obstacle_grids:
            raise PathOptimizationError("终点位于障碍物内")
        
        # 执行路径规划并计时
        start_time = time.time()
        try:
            path = planner.optimize_path(case['start'], case['end'], vehicle)
            elapsed = time.time() - start_time
            print(f"路径规划成功! 耗时: {elapsed:.3f}秒")
            print(f"路径长度: {len(path)}个点")
            
            # 可视化结果
            planner._debug_visualize(path)
        except PathOptimizationError as e:
            print(f"路径规划失败: {str(e)}")
            planner._debug_visualize([])