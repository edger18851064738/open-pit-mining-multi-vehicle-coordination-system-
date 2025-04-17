import sys
import os
import math
import random
import threading
import time
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import heapq
import logging
from typing import List, Tuple, Dict, Optional, Union
import networkx as nx
import matplotlib.pyplot as plt 
from matplotlib import rcParams
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError
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
    
    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

class HybridPathPlanner:
    """矿山场景专用路径规划器（基础版本）"""
    
    def __init__(self, map_service):
        self.obstacle_grids = set()
        self.map_service = map_service
        self.haul_roads = set()
        self.reservation_table = {}
        self.dynamic_obstacles = set()
        self.conflict_check_interval = 0.5
        self.reservation_lock = threading.Lock()
        
        # 优化后的配置
        self.grid_size = 10.0  # 更精细的网格划分
        self.max_grade = 12.0  # 更严格的坡度限制
        self.min_turn_radius = 12.0  # 更小的转弯半径

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

    def optimize_path(self, start: Tuple[float, float], 
                    end: Tuple[float, float],
                    vehicle: MiningVehicle) -> List[Tuple[float, float]]:
        """路径规划入口方法"""
        base_path = self.plan_path(start, end)
        optimized_path = self._apply_rs_curve(base_path, vehicle.turning_radius)
        return optimized_path

    def plan_path(self, start: Tuple, end: Tuple) -> List[Tuple]:
        """基础路径规划"""
        try:
            raw_path = self._mine_astar(Node(*start), Node(*end), MiningVehicle("dummy", {}))
            return self._smooth_path(raw_path)
        except Exception as e:
            logging.error(f"路径规划失败: {str(e)}")
            return []

    def _mine_astar(self, start: Node, end: Node, vehicle: MiningVehicle):
        """优化的A*算法实现"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        
        # 扩展搜索方向，增加45度方向
        directions = [(-1,0), (1,0), (0,-1), (0,1), 
                     (-1,-1), (-1,1), (1,-1), (1,1),
                     (-2,-1), (-2,1), (2,-1), (2,1),
                     (-1,-2), (-1,2), (1,-2), (1,2)]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end:
                return self._build_path(came_from, current)
                
            for dx, dy in directions:
                neighbor = Node(current.x + dx, current.y + dy)
                
                if (neighbor.x, neighbor.y) in self.obstacle_grids:
                    continue
                    
                if not self._is_straight_line((current.x, current.y), (neighbor.x, neighbor.y)):
                    continue
                    
                # 改进的成本计算，考虑地形和载重
                move_cost = self._calculate_move_cost(current, neighbor, vehicle.current_load, vehicle.hardness)
                tentative_g = g_score[current] + move_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        raise Exception("无可行路径")

    def _heuristic(self, a: Union[Tuple, Node], b: Union[Tuple, Node]) -> float:
        """改进的启发式函数，考虑对角线距离"""
        dx = abs(a.x - b.x) if isinstance(a, Node) else abs(a[0] - b[0])
        dy = abs(a.y - b.y) if isinstance(a, Node) else abs(a[1] - b[1])
        return max(dx, dy) + 0.414 * min(dx, dy)  # 对角线距离启发式
        
    def _calculate_move_cost(self, current: Node, neighbor: Node, load: float, hardness: float) -> float:
        """计算移动成本
        
        参数:
            current: 当前节点
            neighbor: 相邻节点
            load: 车辆当前载重
            hardness: 地形硬度
            
        返回:
            移动成本值
        """
        # 基础移动成本
        base_cost = 1.0
        
        # 考虑地形硬度影响
        terrain_factor = 1.0 + (hardness - 2.5) * 0.2
        
        # 考虑载重影响
        load_factor = 1.0 + load * 0.01
        
        # 考虑坡度变化
        current_grade = self.map_service.get_terrain_data(current.x, current.y)['grade']
        neighbor_grade = self.map_service.get_terrain_data(neighbor.x, neighbor.y)['grade']
        grade_change = abs(current_grade - neighbor_grade)
        grade_factor = 1.0 + grade_change * 0.05
        
        # 对角线移动成本稍高
        dx = abs(neighbor.x - current.x)
        dy = abs(neighbor.y - current.y)
        if dx > 0 and dy > 0:
            base_cost *= 1.414  # 对角线距离系数
            
        return base_cost * terrain_factor * load_factor * grade_factor

    def _is_straight_line(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """直线检查"""
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        steps = max(abs(dx), abs(dy))
        
        for i in range(1, steps+1):
            x = a[0] + round(i*dx/steps)
            y = a[1] + round(i*dy/steps)
            if (x, y) in self.obstacle_grids:
                return False
        return True

    def _build_path(self, came_from: dict, current: Node) -> List[Tuple[int, int]]:
        """重构路径"""
        path = []
        while current in came_from:
            path.append((current.x, current.y))
            current = came_from[current]
        return list(reversed(path))

    def _smooth_path(self, raw_path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """路径平滑"""
        if len(raw_path) < 3:
            return raw_path
            
        smoothed = [raw_path[0]]
        for point in raw_path[1:-1]:
            if not self._is_straight_line(smoothed[-1], raw_path[raw_path.index(point)+1]):
                smoothed.append(point)
        smoothed.append(raw_path[-1])
        return smoothed

    def _apply_rs_curve(self, path: List[Tuple], radius: float) -> List[Tuple]:
        """RS曲线应用"""
        if len(path) < 2:
            return path
            
        rs_path = []
        for i in range(len(path)-1):
            segment = self._generate_rs_path(path[i], path[i+1], radius)
            rs_path.extend(segment[:-1])
        rs_path.append(path[-1])
        return rs_path

    def _generate_rs_path(self, current, end, radius):
        """生成RS曲线"""
        if isinstance(current, Node):
            current = (current.x, current.y)
        path = [current]
        
        if isinstance(end, Node):
            end_point = (end.x, end.y)
        else:
            end_point = end
            
        dx = end_point[0] - current[0]
        dy = end_point[1] - current[1]
        
        if not self._is_line_through_obstacle(current, end_point):
            path.append(end_point)
            return path
            
        mid_point = (
            current[0] + dx/2 - dy/radius,
            current[1] + dy/2 + dx/radius
        )
        path.append(mid_point)
        path.append(end_point)
        return path

    def _is_line_through_obstacle(self, start, end):
        """直线障碍检查"""
        points = self._bresenham_line(start, end)
        return any(point in self.obstacle_grids for point in points)

    def _bresenham_line(self, start, end):
        """Bresenham直线算法"""
        x1, y1 = start
        x2, y2 = end
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        sx = -1 if x1 > x2 else 1
        sy = -1 if y1 > y2 else 1
        
        if dx > dy:
            err = dx / 2.0
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))
        return points

class OptimizedHybridPathPlanner(HybridPathPlanner):
    """优化后的路径规划器"""
    
    MAX_ITERATIONS = 5000  # 最大迭代次数限制
    DIRECTIONS = [(-1,0), (1,0), (0,-1), (0,1),  # 四方向移动
                 (-1,-1), (-1,1), (1,-1), (1,1)]  # 对角线移动
    
    def __init__(self, map_service):
        super().__init__(map_service)
        self.reservation_lock = threading.Lock()  # 添加缺失的锁
        self.node_cache = {}  # 节点缓存优化
        
    def _heuristic(self, a: Union[Tuple, Node], b: Union[Tuple, Node]) -> float:
        """优化后的启发函数 - 使用对角线距离和坡度因素"""
        dx = abs(a.x - b.x) if isinstance(a, Node) else abs(a[0] - b[0])
        dy = abs(a.y - b.y) if isinstance(a, Node) else abs(a[1] - b[1])
        
        # 考虑坡度因素
        if isinstance(a, Node) and isinstance(b, Node):
            grade_factor = abs(self.map_service.get_terrain_data(a.x, a.y)['grade'] - 
                             self.map_service.get_terrain_data(b.x, b.y)['grade']) * 0.1
        else:
            grade_factor = 0
            
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy) + grade_factor
    
    def _mine_astar(self, start: Node, end: Node, vehicle: MiningVehicle):
        """优化后的A*算法实现"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        iteration = 0
        visited = set()
        
        while open_set and iteration < self.MAX_ITERATIONS:
            iteration += 1
            current = heapq.heappop(open_set)[1]
            
            if current in visited:
                continue
            visited.add(current)
            
            if current == end:
                return self._build_path(came_from, current)
                
            for dx, dy in self.DIRECTIONS:
                # 使用节点缓存优化
                neighbor_key = (current.x + dx, current.y + dy)
                if neighbor_key in self.node_cache:
                    neighbor = self.node_cache[neighbor_key]
                else:
                    neighbor = Node(*neighbor_key)
                    self.node_cache[neighbor_key] = neighbor
                
                # 快速障碍物检查
                if (neighbor.x, neighbor.y) in self.obstacle_grids:
                    continue
                    
                # 优化后的直线检查
                if not self._optimized_line_check((current.x, current.y), (neighbor.x, neighbor.y)):
                    continue
                    
                if not self._check_vehicle_constraints(neighbor, vehicle):
                    continue

                # 动态成本计算
                move_cost = self._calculate_move_cost(current, neighbor, vehicle.current_load, 
                                                   self.map_service.get_terrain_hardness(neighbor.x, neighbor.y))
                
                tentative_g = g_score[current] + move_cost
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        raise PathOptimizationError(f"路径规划超过最大迭代次数 {self.MAX_ITERATIONS}")
    
    def _check_vehicle_constraints(self, node: Node, vehicle: MiningVehicle) -> bool:
        """车辆约束检查"""
        # 简化版约束检查 - 仅检查是否被预留
        with self.reservation_lock:
            return (node.x, node.y) not in self.reservation_table
            
    def _calculate_move_cost(self, current: Node, neighbor: Node, load: float, hardness: float) -> float:
        """计算移动成本
        
        参数:
            current: 当前节点
            neighbor: 相邻节点
            load: 车辆当前载重
            hardness: 地形硬度
            
        返回:
            移动成本值
        """
        # 基础移动成本
        base_cost = 1.0
        
        # 考虑地形硬度影响
        terrain_factor = 1.0 + (hardness - 2.5) * 0.2
        
        # 考虑载重影响
        load_factor = 1.0 + load * 0.01
        
        # 考虑坡度变化
        current_grade = self.map_service.get_terrain_data(current.x, current.y)['grade']
        neighbor_grade = self.map_service.get_terrain_data(neighbor.x, neighbor.y)['grade']
        grade_change = abs(current_grade - neighbor_grade)
        grade_factor = 1.0 + grade_change * 0.05
        
        # 对角线移动成本稍高
        dx = abs(neighbor.x - current.x)
        dy = abs(neighbor.y - current.y)
        if dx > 0 and dy > 0:
            base_cost *= 1.414  # 对角线距离系数
            
        return base_cost * terrain_factor * load_factor * grade_factor
            
    def _optimized_line_check(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """优化后的直线无障碍检查"""
        # 快速检查端点
        if a in self.obstacle_grids or b in self.obstacle_grids:
            return False
            
        # 使用Bresenham算法优化
        line_points = GeoUtils.bresenham_line(a, b)
        return not any(point in self.obstacle_grids for point in line_points)


class MockMapService:
    """模拟地图服务"""
    def __init__(self):
        self.config = {
            'MAP': {
                'data_type': 'virtual',
                'virtual_origin': '50,50',
                'virtual_range': '100,100'
            }
        }
    
    def validate_coordinates(self, coord):
        return True
    
    def get_terrain_data(self, x, y):
        return {"grade": abs(x-y)%20, "hardness": 3.0 - (x%3 + y%2)*0.3}
    
    def get_terrain_hardness(self, x, y):
        return self.get_terrain_data(x, y)["hardness"]

class MiningVehicle:
    """简化版矿车类"""
    def __init__(self, vehicle_id, map_service, config=None):
        self.vehicle_id = vehicle_id
        self.map_service = map_service
        config = config or {}
        self.min_hardness = config.get('min_hardness', 2.5)
        self.max_load = config.get('max_load', 50)
        self.speed = config.get('speed', 5)
        self.steering_angle = config.get('steering_angle', 30)
        self.current_load = config.get('current_load', 0)
        self.turning_radius = config.get('turning_radius', 10.0)
        self.last_position = None


# 优化后的测试主程序
def run_optimized_test():
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

    # 固定障碍物配置
    obstacle_polygons = [
        # 横向障碍带
        [(10,45), (90,45), (90,55), (10,55)],  # 中部水平障碍
        [(40,20), (60,20), (60,30), (40,30)],   # 下部矩形障碍
        [(40,70), (60,70), (60,80), (40,80)],   # 上部矩形障碍
        
        # 纵向障碍带
        [(25,20), (35,20), (35,80), (25,80)],   # 左侧纵向
        [(65,20), (75,20), (75,80), (65,80)],   # 右侧纵向
        
        # 迷宫式障碍
        [(45,30), (55,30), (55,40), (45,40)],   # 中部障碍1
        [(45,60), (55,60), (55,70), (45,70)],   # 中部障碍2
        [(20,25), (30,25), (30,35), (20,35)],   # 左下迷宫
        [(70,65), (80,65), (80,75), (70,75)],   # 右上迷宫
    ]

    # 标记障碍物（不再保留任何通道）
    planner.mark_obstacle_area(obstacle_polygons)

    # 定义关键点位置（四角分布）
    start_point = (5, 5)  # 起始点位置，避开障碍物
    parking_point = (85, 30)  # 调整后的停车场位置，避开障碍物
    load_points = [
        (10, 10),  # 左下角装载点1
        (10, 90),  # 左上角装载点2
        (90, 10),  # 右下角装载点3
    ]
    unload_point = (90, 90)  # 右上角卸载点

    # 验证关键点可达性
    key_points = [parking_point, unload_point] + load_points
    for point in key_points:
        if point in planner.obstacle_grids:
            raise PathOptimizationError(f"关键点{point}位于障碍物内")

    # 车辆配置
    test_vehicle = MiningVehicle("XTR-1000", {
        'min_hardness': 2.5,
        'max_load': 50,
        'speed': 5,
        'steering_angle': 30,
        'current_load': 35
    })
    test_vehicle.last_position = Node(start_point[0], start_point[1])  # 保持Node类型一致性

    # 执行路径规划并可视化
    try:

            # 规划所有路径
            paths = []
            
            # 规划装载点到停车场的路径
            for i, load_point in enumerate(load_points):
                path = planner.optimize_path(load_point, parking_point, test_vehicle)
                paths.append((f"装载点{i+1}到停车场", path, 'blue'))
                print(f"装载点{i+1}到停车场路径点数量: {len(path)}")
            
            # 规划停车场到卸载点的路径
            unload_path = planner.optimize_path(parking_point, unload_point, test_vehicle)
            paths.append(("停车场到卸载点", unload_path, 'green'))
            print(f"停车场到卸载点路径点数量: {len(unload_path)}")
            
            # 绘制所有路径
            plt.figure(figsize=(15, 15))
            plt.rc('font', size=12)
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            plt.title("多路径规划演示", fontsize=16, pad=20)
            plt.scatter(*zip(*planner.obstacle_grids), c='#2F4F4F', s=60, marker='s', label='障碍物')
            
            # 绘制每条路径
            for title, path, color in paths:
                plt.plot(*zip(*path), '-', linewidth=2.5, label=title, color=color, alpha=0.7)
            
            # 标记所有关键点
            plt.scatter(parking_point[0], parking_point[1], c='yellow', s=300, 
                    edgecolors='black', marker='s', label='停车场')
            plt.scatter(unload_point[0], unload_point[1], c='magenta', s=300,
                    edgecolors='black', marker='*', label='卸载点')
            
            # 标记所有装载点
            for i, load_point in enumerate(load_points):
                plt.scatter(load_point[0], load_point[1], c='lime', s=200,
                          edgecolors='black', marker=f'${i+1}$', label=f'装载点{i+1}')
            
            plt.xlabel("X坐标 (米)")
            plt.ylabel("Y坐标 (米)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
            
    except PathOptimizationError as e:
            logging.error(f"路径规划失败：{str(e)}")

if __name__ == "__main__":
    run_optimized_test()