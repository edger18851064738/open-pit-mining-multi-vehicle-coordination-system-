"""
混合路径规划器 - 整合强化A*算法
实现高效的露天矿多车路径规划
"""

import sys
import os
import math
import time
import heapq
import logging
import threading
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError
from algorithm.map_service import MapService
from algorithm.reinforced_path_planner import ReinforcedAStar, PathCache, DEFAULT_TIMEOUT
EPSILON = 0.0001
import matplotlib.pyplot as plt
# 设置中文字体，可以选择系统中已安装的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False 
class HybridPathPlanner:
    """混合路径规划器 - 整合强化A*算法"""
    
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
        
        # 地图尺寸
        self.map_size = getattr(map_service, 'grid_size', 200)
        
        # 空间数据
        self.obstacle_grids = set()  # 障碍点集合
        
        # 预约系统
        self.reservation_table = {}  # 路径段预约表
        self.reservation_lock = threading.RLock()  # 预约表锁
        
        # 缓存系统
        self.path_cache = PathCache()
        
        # 强化A*路径规划器
        self.reinforced_astar = ReinforcedAStar(
            obstacle_grids=self.obstacle_grids,
            map_size=self.map_size
        )
        
        # 加载地图配置
        self._load_map_config()
        
        # 加载已知障碍物
        self._load_obstacles_from_map()
        
        # 方向数组 (8个方向)
        self.directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),   # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
        
        logging.info("混合路径规划器初始化成功")
    
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
    
    def _load_obstacles_from_map(self):
        """从地图服务加载障碍物数据"""
        try:
            # 检查地图服务是否有障碍物信息
            if hasattr(self.map_service, '_obstacle_nodes'):
                # 转换节点障碍物为网格坐标
                for node in self.map_service._obstacle_nodes:
                    if hasattr(self.map_service.road_network, 'nodes'):
                        node_data = self.map_service.road_network.nodes.get(node)
                        if node_data and 'x' in node_data and 'y' in node_data:
                            self.obstacle_grids.add((int(node_data['x']), int(node_data['y'])))
                            
            # 或者直接使用地图服务的obstacle_grids
            elif hasattr(self.map_service, 'obstacle_grids'):
                self.obstacle_grids = set(self.map_service.obstacle_grids)
                
            # 将障碍物同步到强化A*规划器
            self.reinforced_astar.obstacle_grids = self.obstacle_grids
                
            logging.info(f"从地图加载了 {len(self.obstacle_grids)} 个障碍点")
                
        except Exception as e:
            logging.warning(f"从地图加载障碍物失败: {str(e)}")
    
    def plan_path(self, start, end, vehicle=None, force_replan=False):
        """
        路径规划主入口方法
        
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            vehicle: 可选的车辆对象，用于考虑车辆特性
            force_replan: 是否强制重新规划路径（忽略缓存）
            
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
                    
            # 检查缓存（如果不强制重新规划）
            cached_path = None
            if not force_replan:
                cached_path = self.path_cache.get(cache_key)
                
            if cached_path:
                logging.debug(f"使用缓存路径: {start} → {end}")
                return cached_path
                
            # 使用强化A*算法规划路径
            path = self.reinforced_astar.pathfind(start, end)
            
            # 若路径规划失败，使用备选路径
            if not path or len(path) < 2:
                logging.debug(f"路径规划失败，使用备选路径: {start} → {end}")
                path = self._generate_fallback_path(start, end)
                self.failure_count += 1
            else:
                self.success_count += 1
                    
            # 确保路径有效 - 至少包含起点和终点
            if not path or len(path) < 2:
                path = [start, end]
                    
            # 缓存结果
            self.path_cache.put(cache_key, path)
            
            # 记录性能指标
            elapsed = time.time() - start_time
            self.total_planning_time += elapsed
            
            if elapsed > 0.1:  # 记录较慢的规划
                logging.debug(f"路径规划耗时较长: {elapsed:.3f}秒 ({start} → {end})")
                
            # 记录路径长度用于调试
            logging.info(f"规划成功: 路径长度 {len(path)}, 耗时 {elapsed:.3f}秒")
                
            return path
        except Exception as e:
            logging.error(f"路径规划失败: {str(e)}")
            # 最简单的后备方案 - 直接连接起点和终点
            self.failure_count += 1
            return [start, end]
    
    def _generate_fallback_path(self, start, end):
        """生成备选路径，确保避开障碍物"""
        try:
            # 标准化坐标
            start = self._validate_point(start)
            end = self._validate_point(end)
                
            # 直线距离和方向
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # 检查直线路径上的障碍物
            has_obstacles = any(self.reinforced_astar.is_obstacle(p) 
                               for p in self.reinforced_astar.get_line_points(start, end))
            
            # 如果直线路径无障碍，直接返回
            if not has_obstacles:
                return [start, end]
            
            # 有障碍物，尝试更复杂的绕行路径
            # 计算中点位置并添加偏移
            midpoint = (start[0] + dx/2, start[1] + dy/2)
            offset_distance = max(20.0, distance/5)
            
            # 尝试不同的偏移方向
            offset_directions = [
                (offset_distance, offset_distance),    # 右上
                (-offset_distance, offset_distance),   # 左上
                (offset_distance, -offset_distance),   # 右下
                (-offset_distance, -offset_distance),  # 左下
            ]
            
            for offset_x, offset_y in offset_directions:
                # 计算偏移中点
                detour_point = (midpoint[0] + offset_x, midpoint[1] + offset_y)
                
                # 确保点在地图范围内
                detour_point = (
                    max(0, min(self.map_size, detour_point[0])),
                    max(0, min(self.map_size, detour_point[1]))
                )
                
                # 检查绕行点是否安全
                if not self.reinforced_astar.is_obstacle(detour_point):
                    # 检查路径段是否安全
                    path1_safe = not any(self.reinforced_astar.is_obstacle(p) 
                                      for p in self.reinforced_astar.get_line_points(start, detour_point))
                    path2_safe = not any(self.reinforced_astar.is_obstacle(p) 
                                      for p in self.reinforced_astar.get_line_points(detour_point, end))
                    
                    if path1_safe and path2_safe:
                        return [start, detour_point, end]
            
            # 所有尝试都失败，返回直接路径
            logging.warning(f"无法找到避开障碍物的备选路径 {start} → {end}")
            return [start, end]
            
        except Exception as e:
            logging.error(f"备选路径生成失败: {str(e)}")
            # 最简单的后备方案 - 直接连接起点和终点
            return [start, end]
    
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
        elif isinstance(point, (list)) and len(point) >= 2:
            return (float(point[0]), float(point[1]))
        elif hasattr(point, '__getitem__') and len(point) >= 2:
            return (float(point[0]), float(point[1]))
        else:
            # 无效点警告
            logging.warning(f"无效的坐标点: {point}，使用(0,0)")
            return (0.0, 0.0)
    
    def _points_equal(self, p1, p2, tolerance=EPSILON):
        """检查两点是否相等(考虑浮点误差)"""
        return (abs(p1[0] - p2[0]) < tolerance and 
                abs(p1[1] - p2[1]) < tolerance)
    
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
        
        # 获取强化学习相关统计
        rl_stats = {
            'episodes_trained': self.reinforced_astar.episodes_trained,
            'successful_paths': self.reinforced_astar.successful_paths,
            'q_values_states': len(self.reinforced_astar.q_values),
            'experience_buffer_size': len(self.reinforced_astar.experience_buffer)
        }
        
        return {
            'runtime': f"{runtime:.1f}秒",
            'planning_count': self.planning_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': f"{success_rate*100:.1f}%",
            'avg_planning_time': f"{avg_planning_time*1000:.2f}毫秒",
            'cache_stats': self.path_cache.get_stats(),
            'obstacle_count': len(self.obstacle_grids),
            'reservations': len(self.reservation_table),
            'reinforced_learning': rl_stats
        }