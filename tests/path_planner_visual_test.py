#!/usr/bin/env python3
"""
HybridPathPlanner 增强可视化测试工具

此脚本用于测试和可视化路径规划器的路径规划功能，通过动态可视化展示：
1. 路径规划过程和冲突检测
2. 多车协同运动 
3. 地图障碍物和关键点
4. A*搜索过程分析

为定位 optimized_path_planner 中的问题提供直观界面。
"""

import os
import sys
import time
import math
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Polygon, Arrow
from matplotlib.colors import to_rgba
import matplotlib.gridspec as gridspec
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
import threading
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 导入项目模块
try:
    from algorithm.optimized_path_planner import HybridPathPlanner
    from algorithm.map_service import MapService
    from algorithm.cbs import ConflictBasedSearch
    from utils.geo_tools import GeoUtils
    from models.vehicle import MiningVehicle, VehicleState
    logging.info("成功导入所需模块")
except ImportError as e:
    logging.error(f"导入模块失败: {str(e)}")
    sys.exit(1)
# 在导入模块后添加以下代码，设置中文字体支持
def setup_chinese_font():
    """设置中文字体支持"""
    import matplotlib as mpl
    try:
        # 尝试使用系统中文字体
        fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
        for font in fonts:
            try:
                mpl.rcParams['font.sans-serif'] = [font] + mpl.rcParams['font.sans-serif']
                mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                # 测试字体是否可用
                from matplotlib.font_manager import FontProperties
                FontProperties(family=font)
                logging.info(f"成功设置中文字体: {font}")
                return True
            except:
                continue
                
        # 如果系统字体都不可用，尝试使用Matplotlib内置字体
        mpl.rcParams['font.sans-serif'] = ['DejaVu Sans'] + mpl.rcParams['font.sans-serif']
        logging.warning("未找到合适的中文字体，使用默认字体，中文可能显示为方块")
        return False
    except Exception as e:
        logging.warning(f"设置中文字体时出错: {str(e)}")
        return False

# 在__init__方法中调用此函数
class PathPlannerVisualizer:
    """路径规划器高级可视化测试工具"""
    
    def __init__(self, map_size=200, num_vehicles=5, num_test_points=6, debug_mode=False):
        """初始化可视化环境"""
        # 设置地图尺寸
        self.map_size = map_size
        self.num_vehicles = num_vehicles
        self.num_test_points = num_test_points
        self.debug_mode = debug_mode
        
        # 创建地图服务和路径规划器
        self.geo_utils = GeoUtils()
        self.map_service = MapService()
        self.planner = HybridPathPlanner(self.map_service)
        
        # 初始化冲突检测器
        self.cbs = ConflictBasedSearch(self.planner)
        
        # 初始化测试对象
        self.vehicles = self._create_test_vehicles()
        self.test_points = self._create_test_points()
        self.obstacles = self._create_obstacles()
        
        # 将障碍物应用到规划器
        self.planner.obstacle_grids = set(self.obstacles)
        
        # Mock dispatch 对象 (路径规划器需要)
        class MockDispatch:
            def __init__(self):
                self.vehicles = {}
        
        mock_dispatch = MockDispatch()
        for vehicle in self.vehicles:
            mock_dispatch.vehicles[vehicle.vehicle_id] = vehicle
        self.planner.dispatch = mock_dispatch
        
        # 路径规划测试参数
        self.current_test_idx = 0
        self.test_pairs = []
        self._generate_test_pairs()
        
        # 动画控制
        self.animation_speed = 1.0
        self.show_path = True
        self.pause = False
        self.step_mode = False
        
        # 当前活动车辆和路径
        self.active_vehicles = []
        self.vehicle_paths = {}
        self.vehicle_path_progress = {}
        
        # A*调试数据
        self.debug_data = {
            'visited_nodes': [],
            'current_path': [],
            'open_set': [],
            'closed_set': [],
            'current_node': None
        }
        
        # 性能数据收集
        self.stats = {
            'planning_times': [],
            'path_lengths': [],
            'conflicts': [],
            'visited_nodes_count': []
        }
        
        # 执行历史记录
        self.history = []
        
        # 视图控制
        self.view_mode = "normal"  # normal, debug, heatmap
        
        # 设置可视化
        self.setup_visualization()
        
        logging.info("可视化环境初始化完成")
    
    def _create_test_vehicles(self) -> List[MiningVehicle]:
        """创建测试车辆"""
        vehicles = []
        
        # 车辆起始位置（均匀分布在地图边缘）
        positions = []
        
        # 左边缘
        for i in range(self.num_vehicles // 4 + 1):
            y = self.map_size * (i + 1) / (self.num_vehicles // 4 + 2)
            positions.append((20, y))
            
        # 右边缘
        for i in range(self.num_vehicles // 4 + 1):
            y = self.map_size * (i + 1) / (self.num_vehicles // 4 + 2)
            positions.append((self.map_size - 20, y))
            
        # 上边缘
        for i in range(self.num_vehicles // 4 + 1):
            x = self.map_size * (i + 1) / (self.num_vehicles // 4 + 2)
            positions.append((x, self.map_size - 20))
            
        # 下边缘
        for i in range(self.num_vehicles // 4 + 1):
            x = self.map_size * (i + 1) / (self.num_vehicles // 4 + 2)
            positions.append((x, 20))
        
        # 确保有足够的起始位置
        while len(positions) < self.num_vehicles:
            positions.append((random.randint(20, self.map_size-20), 
                            random.randint(20, self.map_size-20)))
            
        # 随机颜色
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_vehicles))
        
        # 创建车辆
        for i in range(self.num_vehicles):
            config = {
                'current_location': positions[i],
                'max_capacity': 50,
                'max_speed': random.uniform(5.0, 8.0),
                'min_hardness': 2.5,
                'turning_radius': 10.0,
                'base_location': (100, 100)
            }
            
            vehicle = MiningVehicle(
                vehicle_id=i+1,
                map_service=self.map_service,
                config=config
            )
            
            # 添加颜色属性
            vehicle.color = colors[i]
            
            # 确保必要属性存在
            if not hasattr(vehicle, 'current_path'):
                vehicle.current_path = []
            if not hasattr(vehicle, 'path_index'):
                vehicle.path_index = 0
            
            vehicles.append(vehicle)
            
        logging.info(f"创建了 {len(vehicles)} 辆测试车辆")
        return vehicles
    
    def _create_test_points(self) -> Dict[str, Tuple[float, float]]:
        """创建测试点位置"""
        # 关键固定点位
        points = {
            "中心": (self.map_size // 2, self.map_size // 2),
            "左上": (30, self.map_size - 30),
            "右上": (self.map_size - 30, self.map_size - 30),
            "左下": (30, 30),
            "右下": (self.map_size - 30, 30),
        }
        
        # 添加随机测试点
        for i in range(1, self.num_test_points - 5 + 1):
            valid_point = False
            attempts = 0
            while not valid_point and attempts < 20:
                attempts += 1
                x = random.randint(30, self.map_size - 30)
                y = random.randint(30, self.map_size - 30)
                
                # 确保点不是障碍物且离其他点有一定距离
                if not self._is_obstacle_area(x, y):
                    # 检查与其他点的距离
                    min_dist = min(
                        [math.dist((x, y), p) for p in points.values()],
                        default=float('inf')
                    )
                    if min_dist > 30:  # 最小距离阈值
                        valid_point = True
                        points[f"测试点{i}"] = (x, y)
            
            # 如果找不到合适点，就用随机点
            if attempts >= 20 and f"测试点{i}" not in points:
                points[f"测试点{i}"] = (
                    random.randint(30, self.map_size - 30),
                    random.randint(30, self.map_size - 30)
                )
        
        logging.info(f"创建了 {len(points)} 个测试点")
        return points
  
    def _is_obstacle_area(self, x, y, margin=10):
        """检查点是否在障碍物区域（包括边缘）"""
        # 障碍物区域
        obstacle_areas = [
            (80, 30, 120, 80),    # 中下方障碍物
            (30, 80, 80, 120),    # 左中障碍物
            (120, 80, 170, 120),  # 右中障碍物
            (80, 120, 120, 170)   # 中上方障碍物
        ]
        
        for area in obstacle_areas:
            x_min, y_min, x_max, y_max = area
            if (x_min - margin <= x <= x_max + margin and 
                y_min - margin <= y <= y_max + margin):
                return True
                
        # 也检查多边形障碍物
        for polygon in self._create_polygon_obstacles():
            if self._point_in_polygon((x, y), polygon):
                return True
        
        return False
    
    def _create_obstacles(self) -> List[Tuple[int, int]]:
        """创建障碍物"""
        obstacles = []
        
        # 矩形障碍物区域
        obstacle_areas = [
            (80, 30, 120, 80),    # 中下方障碍物
            (30, 80, 80, 120),    # 左中障碍物
            (120, 80, 170, 120),  # 右中障碍物
            (80, 120, 120, 170)   # 中上方障碍物
        ]
        
        # 生成矩形障碍点
        for area in obstacle_areas:
            x_min, y_min, x_max, y_max = area
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    obstacles.append((x, y))
        
        # 添加多边形障碍物点
        for polygon in self._create_polygon_obstacles():
            # 找出多边形的边界框
            x_values = [p[0] for p in polygon]
            y_values = [p[1] for p in polygon]
            min_x, max_x = min(x_values), max(x_values)
            min_y, max_y = min(y_values), max(y_values)
            
            # 检查边界框内的每个点
            for x in range(int(min_x), int(max_x) + 1):
                for y in range(int(min_y), int(max_y) + 1):
                    if self._point_in_polygon((x, y), polygon):
                        obstacles.append((x, y))
        
        logging.info(f"创建了 {len(obstacles)} 个障碍点")
        return obstacles
    
    def _create_polygon_obstacles(self) -> List[List[Tuple[float, float]]]:
        """创建多边形障碍物"""
        polygons = [
            # 不规则多边形障碍物
            [(10, 150), (40, 170), (50, 140), (30, 130)],  # 左上不规则形状
            [(160, 40), (180, 60), (190, 40), (170, 20)],  # 右下不规则形状
            
            # 三角形障碍物
            [(100, 140), (120, 160), (80, 160)],  # 上方三角形
            
            # 细长障碍物（模拟墙）
            [(60, 50), (70, 50), (70, 100), (60, 100)]  # 垂直墙
        ]
        return polygons
        
    def _point_in_polygon(self, point, polygon) -> bool:
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
    
    def _generate_test_pairs(self):
        """生成测试点对"""
        point_names = list(self.test_points.keys())
        
        # 确保每个点都被访问到
        for i in range(len(point_names)):
            next_idx = (i + 1) % len(point_names)
            self.test_pairs.append((point_names[i], point_names[next_idx]))
            
        # 添加一些长距离交叉路径
        self.test_pairs.append(("左上", "右下"))
        self.test_pairs.append(("右上", "左下"))
        
        # 添加一些随机的点对
        for _ in range(self.num_vehicles * 2):
            start_idx = random.randint(0, len(point_names) - 1)
            end_idx = start_idx
            while end_idx == start_idx:
                end_idx = random.randint(0, len(point_names) - 1)
            self.test_pairs.append((point_names[start_idx], point_names[end_idx]))
            
        logging.info(f"生成了 {len(self.test_pairs)} 个测试点对")
    
    def setup_visualization(self):
        """设置可视化环境"""
        # 创建图形和布局
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1])
        
        # 主地图视图
        self.ax_main = self.fig.add_subplot(gs[0, 0])
        self.ax_main.set_xlim(0, self.map_size)
        self.ax_main.set_ylim(0, self.map_size)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('路径规划器测试可视化')
        self.ax_main.set_xlabel('X坐标')
        self.ax_main.set_ylabel('Y坐标')
        self.ax_main.grid(True, linestyle='--', alpha=0.7)
        
        # 状态面板
        self.ax_status = self.fig.add_subplot(gs[0, 1])
        self.ax_status.set_title('系统状态')
        self.ax_status.axis('off')
        
        # 调试视图
        self.ax_debug = self.fig.add_subplot(gs[1, 0])
        self.ax_debug.set_title('路径规划调试')
        self.ax_debug.set_xlabel('X坐标')
        self.ax_debug.set_ylabel('Y坐标')
        self.ax_debug.grid(True, linestyle='--', alpha=0.7)
        
        # 统计面板 
        self.ax_stats = self.fig.add_subplot(gs[1, 1])
        self.ax_stats.set_title('性能统计')
        self.ax_stats.axis('off')
        
        # 绘制背景和障碍物
        self._draw_map_background(self.ax_main)
        self._draw_map_background(self.ax_debug)
        
        # 创建车辆标记
        self._initialize_vehicle_markers()
        
        # 创建调试视图元素
        self._initialize_debug_view()
        
        # 创建状态文本
        self.status_text = self.ax_status.text(
            0.05, 0.95, '',
            transform=self.ax_status.transAxes,
            verticalalignment='top',
            fontsize=10
        )
        
        # 创建统计面板文本
        self.stats_text = self.ax_stats.text(
            0.05, 0.95, '',
            transform=self.ax_stats.transAxes,
            verticalalignment='top',
            fontsize=10
        )
        
        # 设置紧凑布局
        self.fig.tight_layout()
        
        # 设置键盘事件处理
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_event)
    
    def _draw_map_background(self, ax):
        """绘制地图背景和障碍物"""
        # 绘制测试点
        for name, point in self.test_points.items():
            ax.plot(point[0], point[1], 'o', markersize=10)
            ax.text(point[0]+5, point[1]+5, name, fontsize=8)
        
        # 绘制矩形障碍物区域
        obstacle_areas = [
            (80, 30, 120, 80),    # 中下方障碍物
            (30, 80, 80, 120),    # 左中障碍物
            (120, 80, 170, 120),  # 右中障碍物
            (80, 120, 120, 170)   # 中上方障碍物
        ]
        
        for area in obstacle_areas:
            x_min, y_min, x_max, y_max = area
            width, height = x_max - x_min, y_max - y_min
            rect = Rectangle((x_min, y_min), width, height, 
                            facecolor='gray', alpha=0.5)
            ax.add_patch(rect)
        
        # 绘制多边形障碍物
        for polygon_vertices in self._create_polygon_obstacles():
            poly = Polygon(polygon_vertices, facecolor='gray', alpha=0.5)
            ax.add_patch(poly)
    
    def _initialize_vehicle_markers(self):
        """初始化车辆标记"""
        self.vehicle_markers = []
        self.vehicle_labels = []
        self.vehicle_path_lines = []  # 新增一个列表存储路径线对象
        
        for vehicle in self.vehicles:
            # 车辆标记
            marker = Circle(vehicle.current_location, radius=5, 
                        color=vehicle.color, label=f'车辆{vehicle.vehicle_id}')
            self.ax_main.add_patch(marker)
            self.vehicle_markers.append(marker)
            
            # 车辆标签
            label = self.ax_main.text(
                vehicle.current_location[0]+5, 
                vehicle.current_location[1]+5, 
                f'{vehicle.vehicle_id}', 
                color=vehicle.color, 
                fontweight='bold'
            )
            self.vehicle_labels.append(label)
            
            # 车辆路径
            path_line, = self.ax_main.plot(
                [], [], '-', 
                color=vehicle.color, 
                alpha=0.7, 
                linewidth=2
            )
            self.vehicle_path_lines.append(path_line)  # 添加到线条列表
        
        # 添加图例
        self.ax_main.legend(loc='upper right')
    
    def _initialize_debug_view(self):
        """初始化调试视图元素"""
        # A*搜索可视化
        self.debug_elements = {
            'open_set': self.ax_debug.scatter([], [], c='green', marker='o', s=30, alpha=0.3, label='Open Set'),
            'closed_set': self.ax_debug.scatter([], [], c='red', marker='x', s=30, alpha=0.3, label='Closed Set'),
            'current_path': self.ax_debug.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Current Path')[0],
            'current_node': self.ax_debug.scatter([], [], c='yellow', marker='*', s=100, label='Current Node'),
            'start_point': self.ax_debug.scatter([], [], c='green', marker='s', s=100, label='Start'),
            'end_point': self.ax_debug.scatter([], [], c='red', marker='s', s=100, label='Goal')
        }
        
        # 添加图例
        self.ax_debug.legend(loc='upper right')
        
        # 设置与主视图相同的范围
        self.ax_debug.set_xlim(self.ax_main.get_xlim())
        self.ax_debug.set_ylim(self.ax_main.get_ylim())
    
    def _on_key_event(self, event):
        """键盘事件处理"""
        if event.key == ' ':  # 空格键
            self.pause = not self.pause
            if self.pause:
                logging.info("动画已暂停")
            else:
                logging.info("动画已继续")
        elif event.key == 'p':  # 显示/隐藏路径
            self.show_path = not self.show_path
            logging.info(f"路径显示: {'开启' if self.show_path else '关闭'}")
        elif event.key == '+' or event.key == '=':  # 加快速度
            self.animation_speed = min(5.0, self.animation_speed * 1.5)
            logging.info(f"动画速度: {self.animation_speed:.1f}x")
        elif event.key == '-':  # 减慢速度
            self.animation_speed = max(0.1, self.animation_speed / 1.5)
            logging.info(f"动画速度: {self.animation_speed:.1f}x")
        elif event.key == 'n':  # 下一步
            if self.step_mode:
                self.pause = False
                self._update_frame(None)
                self.pause = True
                logging.info("执行一步")
        elif event.key == 's':  # 切换步进模式
            self.step_mode = not self.step_mode
            if self.step_mode:
                self.pause = True
            logging.info(f"步进模式: {'开启' if self.step_mode else '关闭'}")
        elif event.key == 'd':  # 切换调试视图
            self.view_mode = "debug" if self.view_mode != "debug" else "normal"
            logging.info(f"视图模式: {self.view_mode}")
        elif event.key == 'h':  # 切换热图视图
            self.view_mode = "heatmap" if self.view_mode != "heatmap" else "normal"
            logging.info(f"视图模式: {self.view_mode}")
        elif event.key == 'c':  # 检测冲突
            self._check_path_conflicts()
            logging.info("执行冲突检测")
        elif event.key == 'r':  # 解决冲突
            self._resolve_path_conflicts()
            logging.info("执行冲突解决")
    
    def update_frame(self, frame_num):
        """动画更新函数"""
        if self.pause:
            return self.vehicle_markers + self.vehicle_labels + self.vehicle_path_lines
        
        # 更新车辆位置和路径
        self._update_vehicles()
        
        # 更新状态文本
        self._update_status_display()
        
        # 更新统计信息
        self._update_stats_display()
        
        # 更新调试视图
        if self.view_mode == "debug":
            self._update_debug_view()
        elif self.view_mode == "heatmap":
            self._update_heatmap_view()
        
        # 如果没有活动车辆，启动新测试
        if len(self.active_vehicles) == 0 and self.current_test_idx < len(self.test_pairs):
            self._start_new_test()
        
        # 修改这里：使用vehicle_path_lines而不是vehicle_paths
        return self.vehicle_markers + self.vehicle_labels + self.vehicle_path_lines
    
    def _update_vehicles(self):
        """更新车辆位置"""
        if not self.active_vehicles:
            return
            
        for vehicle in list(self.active_vehicles):
            path = self.vehicle_paths.get(vehicle, [])
            
            if not path or len(path) < 2:
                self.active_vehicles.remove(vehicle)
                continue
                
            # 获取当前路径进度
            progress = self.vehicle_path_progress.get(vehicle, 0)
            
            # 检查是否到达终点
            if progress >= len(path) - 1:
                # 车辆到达终点
                self.active_vehicles.remove(vehicle)
                continue
                
            # 更新路径进度
            progress += self.simulation_speed * 0.1  # 根据模拟速度调整步长
            progress = min(progress, len(path) - 1)  # 确保不超过路径长度
            self.vehicle_path_progress[vehicle] = progress
            
            # 计算当前位置
            progress_int = int(progress)
            progress_frac = progress - progress_int
            
            if progress_int < len(path) - 1:
                current = path[progress_int]
                next_point = path[progress_int + 1]
                
                # 检查当前路径段是否穿过障碍物
                if self._is_path_segment_blocked(current, next_point):
                    # 如果路径段穿过障碍物，尝试重新规划路径
                    new_path = self._replan_path_for_vehicle(vehicle, vehicle.current_location, path[-1])
                    if new_path and len(new_path) > 1:
                        self.vehicle_paths[vehicle] = new_path
                        self.vehicle_path_progress[vehicle] = 0
                        continue
                
                # 插值计算当前位置
                vehicle.current_location = (
                    current[0] + (next_point[0] - current[0]) * progress_frac,
                    current[1] + (next_point[1] - current[1]) * progress_frac
                )
            else:
                # 到达终点
                vehicle.current_location = path[-1]
        
        # 检查是否所有车辆都完成测试
        if not self.active_vehicles:
            # 记录测试持续时间
            test_duration = time.time() - self.test_stats.get('start_time', time.time())
            self.test_stats['test_durations'].append(test_duration)
            
            # 移动到下一个测试
            self.current_test_info['current_idx'] += 1
            
            # 更新热图
            if self.show_heatmap:
                self._update_heatmap_from_paths()

    def _is_path_segment_blocked(self, start, end):
        """检查路径段是否穿过障碍物"""
        # 使用Bresenham算法获取路径段上的所有点
        points = self._bresenham_line(start, end)
        
        # 检查这些点是否有障碍物
        for point in points:
            if self._is_obstacle(point):
                return True
        
        return False

    def _bresenham_line(self, start, end):
        """Bresenham算法获取线段上的所有点"""
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

    def _is_obstacle(self, point):
        """检查点是否为障碍物"""
        # 转为整数坐标
        x, y = int(round(point[0])), int(round(point[1]))
        
        return (x, y) in self.obstacles

    def _replan_path_for_vehicle(self, vehicle, current_pos, destination):
        """为车辆重新规划路径，避开障碍物"""
        # 尝试使用规划器生成新路径
        try:
            new_path = self.planner.plan_path(current_pos, destination, vehicle)
            if new_path and len(new_path) > 1:
                return new_path
        except Exception as e:
            logging.warning(f"重规划路径失败: {str(e)}")
        
        # 如果规划失败，尝试更智能的备选路径
        return self._smart_fallback_path(current_pos, destination)

    def _smart_fallback_path(self, start, end):
        """生成更智能的备选路径，确保不穿过障碍物"""
        # 先尝试直接连接
        if not self._is_path_segment_blocked(start, end):
            return [start, end]
        
        # 如果直连被阻挡，尝试找中间点
        # 1. 计算起点到终点的方向
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        # 2. 尝试不同方向的偏移点
        for angle_offset in [0, 45, -45, 90, -90, 135, -135, 180]:
            # 转换为弧度
            angle = math.atan2(dy, dx) + math.radians(angle_offset)
            # 计算偏移距离 (与原始距离相关)
            offset_dist = dist * 0.5  # 使用原始距离的一半
            
            # 计算中间点
            mid_x = start[0] + math.cos(angle) * offset_dist
            mid_y = start[1] + math.sin(angle) * offset_dist
            mid_point = (mid_x, mid_y)
            
            # 检查路径段是否可行
            if (not self._is_path_segment_blocked(start, mid_point) and 
                not self._is_path_segment_blocked(mid_point, end) and
                not self._is_obstacle(mid_point)):
                return [start, mid_point, end]
        
        # 如果所有常规方向都不行，尝试使用多个中间点
        # 这里使用四个点，形成一个绕行路径
        mid_points = []
        angles = [45, 135, -135, -45]  # 大致形成一个矩形路径
        
        for angle_offset in angles:
            angle = math.radians(angle_offset)
            # 使用较大偏移确保绕开障碍物
            offset_dist = dist * 0.7
            
            mid_x = start[0] + math.cos(angle) * offset_dist
            mid_y = start[1] + math.sin(angle) * offset_dist
            mid_point = (mid_x, mid_y)
            
            # 确保中间点不是障碍物
            if not self._is_obstacle(mid_point):
                mid_points.append(mid_point)
        
        if mid_points:
            # 构建完整路径 [起点, 中间点1, 中间点2, ..., 终点]
            path = [start] + mid_points + [end]
            
            # 检查新路径中每段是否有障碍物，移除导致障碍的点
            valid_path = [start]
            for i in range(1, len(path)):
                if not self._is_path_segment_blocked(valid_path[-1], path[i]):
                    valid_path.append(path[i])
            
            # 确保终点在路径中
            if valid_path[-1] != end:
                valid_path.append(end)
            
            return valid_path
        
        # 最后手段：随机尝试多个中间点
        for _ in range(10):  # 尝试10次
            # 随机偏移
            mid_x = start[0] + random.uniform(-dist, dist)
            mid_y = start[1] + random.uniform(-dist, dist)
            mid_point = (mid_x, mid_y)
            
            if (not self._is_obstacle(mid_point) and
                not self._is_path_segment_blocked(start, mid_point) and
                not self._is_path_segment_blocked(mid_point, end)):
                return [start, mid_point, end]
        
        # 实在找不到路径，返回直连路径，但记录警告
        logging.warning(f"无法找到避开障碍物的路径: {start} -> {end}")
        return [start, end]
    
    def _update_status_display(self):
        """更新状态显示"""
        active_count = len(self.active_vehicles)
        completed_tests = self.current_test_idx
        total_tests = len(self.test_pairs)
        
        # 状态信息
        status_text = [
            f"测试状态:",
            f"活动车辆: {active_count}/{self.num_vehicles}",
            f"完成测试: {completed_tests}/{total_tests} ({completed_tests/total_tests*100:.1f}%)",
            f"动画速度: {self.animation_speed:.1f}x",
            f"视图模式: {self.view_mode}",
            f"步进模式: {'开启' if self.step_mode else '关闭'}",
            f"路径显示: {'开启' if self.show_path else '关闭'}",
            f"\n当前活动:",
        ]
        
        # 添加活动车辆信息
        for vehicle in self.active_vehicles:
            task_info = f"{vehicle.current_task.task_id}" if hasattr(vehicle, 'current_task') and vehicle.current_task else "无任务"
            progress = self.vehicle_path_progress.get(vehicle, 0)
            path_len = len(self.vehicle_paths.get(vehicle, []))
            if path_len > 0:
                progress_pct = min(100, progress / path_len * 100)
            else:
                progress_pct = 0
                
            status_text.append(
                f"车辆{vehicle.vehicle_id}: {task_info} - 进度: {progress}/{path_len} ({progress_pct:.1f}%)"
            )
        
        # 更新状态文本
        self.status_text.set_text('\n'.join(status_text))
    
    def _update_stats_display(self):
        """更新统计信息显示"""
        # 计算统计数据
        avg_planning_time = np.mean(self.stats['planning_times']) if self.stats['planning_times'] else 0
        avg_path_length = np.mean(self.stats['path_lengths']) if self.stats['path_lengths'] else 0
        conflict_count = len(self.stats['conflicts'])
        total_tests = max(1, self.current_test_idx)
        
        # 统计信息文本
        stats_text = [
            f"性能统计:",
            f"平均规划时间: {avg_planning_time:.2f}毫秒",
            f"平均路径长度: {avg_path_length:.1f}点",
            f"检测到的冲突: {conflict_count}个",
            f"规划成功率: {100 * len(self.stats['path_lengths']) / total_tests:.1f}%",
            f"\n路径规划指标:",
            f"最短路径: {min(self.stats['path_lengths']) if self.stats['path_lengths'] else 0}点",
            f"最长路径: {max(self.stats['path_lengths']) if self.stats['path_lengths'] else 0}点",
            f"最快规划: {min(self.stats['planning_times']) * 1000:.2f}毫秒" if self.stats['planning_times'] else "最快规划: N/A",
            f"最慢规划: {max(self.stats['planning_times']) * 1000:.2f}毫秒" if self.stats['planning_times'] else "最慢规划: N/A",
        ]
        
        # 更新统计文本
        self.stats_text.set_text('\n'.join(stats_text))

    def _update_debug_view(self):
        """更新调试视图，显示A*搜索过程"""
        # 更新A*搜索可视化元素
        if 'current_search' in self.debug_data:
            search_data = self.debug_data['current_search']
            
            # 更新开放集
            if 'open_set' in search_data and search_data['open_set']:
                open_x = [p[0] for p in search_data['open_set']]
                open_y = [p[1] for p in search_data['open_set']]
                self.debug_elements['open_set'].set_offsets(np.column_stack([open_x, open_y]))
            else:
                self.debug_elements['open_set'].set_offsets(np.zeros((0, 2)))
                
            # 更新关闭集
            if 'closed_set' in search_data and search_data['closed_set']:
                closed_x = [p[0] for p in search_data['closed_set']]
                closed_y = [p[1] for p in search_data['closed_set']]
                self.debug_elements['closed_set'].set_offsets(np.column_stack([closed_x, closed_y]))
            else:
                self.debug_elements['closed_set'].set_offsets(np.zeros((0, 2)))
                
            # 更新当前路径
            if 'current_path' in search_data and search_data['current_path']:
                path_x = [p[0] for p in search_data['current_path']]
                path_y = [p[1] for p in search_data['current_path']]
                self.debug_elements['current_path'].set_data(path_x, path_y)
            else:
                self.debug_elements['current_path'].set_data([], [])
                
            # 更新当前节点
            if 'current_node' in search_data and search_data['current_node']:
                self.debug_elements['current_node'].set_offsets([search_data['current_node']])
            else:
                self.debug_elements['current_node'].set_offsets(np.zeros((0, 2)))
                
            # 更新起点和终点
            if 'start' in search_data:
                self.debug_elements['start_point'].set_offsets([search_data['start']])
            if 'goal' in search_data:
                self.debug_elements['end_point'].set_offsets([search_data['goal']])
                
        # 设置调试视图标题
        if hasattr(self, 'current_debug_test') and self.current_debug_test:
            start_name, end_name = self.current_debug_test
            self.ax_debug.set_title(f'路径规划调试: {start_name} → {end_name}')

    def _update_heatmap_view(self):
        """更新热图视图，显示路径密度和冲突点"""
        # 如果是第一次进入热图模式，需要创建热图
        if not hasattr(self, 'heatmap_data'):
            self.heatmap_data = np.zeros((self.map_size, self.map_size))
            
        # 为活动车辆的路径添加密度
        for vehicle in self.active_vehicles:
            if vehicle in self.vehicle_paths:
                path = self.vehicle_paths[vehicle]
                for x, y in path:
                    # 确保坐标在有效范围内
                    if 0 <= int(x) < self.map_size and 0 <= int(y) < self.map_size:
                        self.heatmap_data[int(y), int(x)] += 0.5
        
        # 为冲突点添加高密度
        for conflict in self.stats['conflicts']:
            if 'location' in conflict:
                x, y = conflict['location']
                # 确保坐标在有效范围内
                if 0 <= int(x) < self.map_size and 0 <= int(y) < self.map_size:
                    self.heatmap_data[int(y), int(x)] += 5.0
        
        # 应用衰减以便路径随时间淡出
        self.heatmap_data *= 0.99
        
        # 更新热图显示
        if not hasattr(self, 'heatmap'):
            self.heatmap = self.ax_debug.imshow(
                self.heatmap_data, 
                cmap='hot', 
                interpolation='bilinear',
                alpha=0.7,
                extent=[0, self.map_size, 0, self.map_size],
                origin='lower'
            )
            self.ax_debug.set_title('路径密度热图 (明亮区域代表高使用率)')
        else:
            self.heatmap.set_data(self.heatmap_data)

    def _check_path_conflicts(self):
        """检测活动车辆之间的路径冲突"""
        # 收集当前活动的车辆路径
        active_paths = {}
        for vehicle in self.active_vehicles:
            if vehicle in self.vehicle_paths:
                # 使用字符串ID作为键以满足CBS的要求
                active_paths[str(vehicle.vehicle_id)] = self.vehicle_paths[vehicle]
        
        if len(active_paths) < 2:
            logging.info("检测冲突需要至少两条活动路径")
            return
            
        # 调用CBS进行冲突检测
        conflicts = self.cbs.find_conflicts(active_paths)
        
        # 记录冲突
        if conflicts:
            self.stats['conflicts'].extend(conflicts)
            
            # 在主视图中标记冲突点
            for conflict in conflicts:
                location = conflict['location']
                conflict_time = conflict['time']
                
                # 创建冲突标记
                conflict_marker = Circle(
                    location, radius=3, 
                    facecolor='red', edgecolor='yellow', 
                    alpha=0.7, zorder=10
                )
                self.ax_main.add_patch(conflict_marker)
                
                # 添加冲突信息标签
                conflict_text = self.ax_main.text(
                    location[0] + 5, location[1] + 5,
                    f"冲突: t={conflict_time}",
                    color='red', fontweight='bold', fontsize=8,
                    alpha=0.8, zorder=10
                )
                
                # 设置标记淡出
                def fade_out_marker(marker, text, fade_time=5.0):
                    """让标记随时间淡出"""
                    start_time = time.time()
                    
                    def update_alpha():
                        elapsed = time.time() - start_time
                        if elapsed >= fade_time:
                            # 移除标记
                            marker.remove()
                            text.remove()
                            return False
                        
                        # 更新透明度
                        alpha = 1.0 - (elapsed / fade_time)
                        marker.set_alpha(alpha)
                        text.set_alpha(alpha)
                        return True
                    
                    # 创建临时定时器
                    timer = self.fig.canvas.new_timer(interval=100)
                    timer.add_callback(update_alpha)
                    timer.start()
                
                # 启动淡出效果
                fade_out_marker(conflict_marker, conflict_text)
            
            logging.info(f"检测到 {len(conflicts)} 个路径冲突")
        else:
            logging.info("未检测到路径冲突")

    def _resolve_path_conflicts(self):
        """使用CBS解决活动车辆之间的路径冲突"""
        # 收集当前活动的车辆路径
        active_paths = {}
        for vehicle in self.active_vehicles:
            if vehicle in self.vehicle_paths:
                # 使用字符串ID作为键以满足CBS的要求
                active_paths[str(vehicle.vehicle_id)] = self.vehicle_paths[vehicle]
        
        if len(active_paths) < 2:
            logging.info("解决冲突需要至少两条活动路径")
            return
            
        # 调用CBS进行冲突解决
        start_time = time.time()
        resolved_paths = self.cbs.resolve_conflicts(active_paths)
        resolution_time = time.time() - start_time
        
        # 记录性能数据
        self.stats['planning_times'].append(resolution_time)
        
        changed_count = 0
        if resolved_paths:
            # 更新车辆路径
            for vid_str, new_path in resolved_paths.items():
                if vid_str in active_paths and new_path != active_paths[vid_str]:
                    vid = int(vid_str)
                    changed_count += 1
                    
                    # 找到对应的车辆
                    for vehicle in self.vehicles:
                        if vehicle.vehicle_id == vid:
                            # 更新路径
                            self.vehicle_paths[vehicle] = new_path
                            # 重置路径进度
                            self.vehicle_path_progress[vehicle] = 0
                            break
            
            logging.info(f"CBS解决了 {changed_count} 条路径冲突，耗时: {resolution_time:.3f}秒")

    def _start_new_test(self):
        """启动新的测试"""
        if self.current_test_idx >= len(self.test_pairs):
            logging.info("所有测试已完成")
            # 显示最终统计信息
            self._show_final_stats()
            return
            
        # 获取测试点对
        start_name, end_name = self.test_pairs[self.current_test_idx]
        start_point = self.test_points[start_name]
        end_point = self.test_points[end_name]
        
        # 更新标题
        self.ax_main.set_title(f'测试 {self.current_test_idx+1}/{len(self.test_pairs)}: {start_name} → {end_name}')
        
        # 保存当前测试信息用于调试视图
        self.current_debug_test = (start_name, end_name)
        
        # 选择一辆空闲车辆
        idle_vehicles = [v for v in self.vehicles if v not in self.active_vehicles]
        if not idle_vehicles:
            logging.info("没有空闲车辆可用，等待下一轮...")
            return
            
        vehicle = random.choice(idle_vehicles)
        
        # 添加调试钩子以捕获A*搜索过程
        def path_planner_hook(node, **context):
            """A*搜索过程钩子函数"""
            if not hasattr(self, 'debug_data'):
                self.debug_data = {}
                
            if 'current_search' not in self.debug_data:
                self.debug_data['current_search'] = {
                    'open_set': [],
                    'closed_set': [],
                    'current_path': [],
                    'current_node': None,
                    'start': start_point,
                    'goal': end_point
                }
                
            # 更新搜索状态
            search_data = self.debug_data['current_search']
            
            # 更新开放集和关闭集
            if 'open_set' in context:
                search_data['open_set'] = context['open_set']
            if 'closed_set' in context:
                search_data['closed_set'] = context['closed_set']
                
            # 更新当前节点和路径
            search_data['current_node'] = node
            if 'current_path' in context:
                search_data['current_path'] = context['current_path']
                
        # 测量规划时间
        start_time = time.time()
        
        # 规划路径
        path = self.planner.plan_path(vehicle.current_location, end_point, vehicle)
        
        # 记录规划时间
        planning_time = time.time() - start_time
        self.stats['planning_times'].append(planning_time)
        
        if path and len(path) > 1:
            # 设置车辆路径
            self.active_vehicles.append(vehicle)
            self.vehicle_paths[vehicle] = path
            self.vehicle_path_progress[vehicle] = 0
            
            # 记录路径长度
            self.stats['path_lengths'].append(len(path))
            
            logging.info(f"车辆{vehicle.vehicle_id}从{start_name}前往{end_name}，路径长度: {len(path)}，规划耗时: {planning_time:.3f}秒")
            
            # 增加测试索引
            self.current_test_idx += 1
        else:
            logging.warning(f"车辆{vehicle.vehicle_id}无法规划到{end_name}的路径")
            # 尝试下一个测试
            self.current_test_idx += 1
            self._start_new_test()
    def _show_final_stats(self):
        """显示最终的统计数据"""
        # 收集统计数据
        avg_planning_time = np.mean(self.stats['planning_times']) if self.stats['planning_times'] else 0
        avg_path_length = np.mean(self.stats['path_lengths']) if self.stats['path_lengths'] else 0
        total_conflicts = len(self.stats['conflicts'])
        success_rate = len(self.stats['path_lengths']) / max(1, self.current_test_idx)
        
        # 创建统计报告
        report = (
            f"\n{'='*60}\n"
            f"路径规划器测试完成\n"
            f"{'='*60}\n"
            f"总计测试: {self.current_test_idx}\n"
            f"规划成功率: {success_rate*100:.1f}%\n"
            f"平均规划时间: {avg_planning_time*1000:.2f}毫秒\n"
            f"平均路径长度: {avg_path_length:.1f}点\n"
            f"检测到冲突: {total_conflicts}个\n"
            f"{'='*60}\n"
        )
        
        # 显示对话框
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        import tkinter as tk
        from tkinter import messagebox
        
        # 显示结果
        messagebox.showinfo("测试完成", report)
        
        # 更新主图标题
        self.ax_main.set_title(f'路径规划测试完成 - 成功率: {success_rate*100:.1f}%')

    def run_visualization(self):
        """运行可视化"""
        # 创建动画
        ani = animation.FuncAnimation(
            self.fig, self.update_frame, 
            interval=int(100 / self.animation_speed),
            blit=True,
            cache_frame_data=False

        )
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        return ani

if __name__ == "__main__":
    # 命令行参数解析

    parser = argparse.ArgumentParser(description='路径规划器可视化测试工具')
    parser.add_argument('--vehicles', type=int, default=5, help='车辆数量 (默认: 5)')
    parser.add_argument('--points', type=int, default=10, help='测试点数量 (默认: 10)')
    parser.add_argument('--size', type=int, default=200, help='地图尺寸 (默认: 200)')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    args = parser.parse_args()
    setup_chinese_font()
    # 显示欢迎信息
    print("=" * 70)
    print("              路径规划器可视化测试工具")
    print("=" * 70)
    print("快捷键:")
    print("  空格键 - 暂停/继续")
    print("  + / -  - 调整速度")
    print("  p      - 切换路径显示")
    print("  d      - 切换调试视图")
    print("  h      - 切换热图视图")
    print("  c      - 检测路径冲突")
    print("  r      - 解决路径冲突")
    print("  s      - 切换步进模式")
    print("  n      - 在步进模式下执行下一步")
    print("=" * 70)
    
    # 创建可视化器并运行
    visualizer = PathPlannerVisualizer(
        map_size=args.size,
        num_vehicles=args.vehicles,
        num_test_points=args.points,
        debug_mode=args.debug
    )
    
    try:
        # 运行可视化
        ani = visualizer.run_visualization()
    except KeyboardInterrupt:
        print("\n可视化已中断")
    except Exception as e:
        print(f"运行时错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("可视化结束")