#!/usr/bin/env python3
"""
HybridPathPlanner 可视化测试

此脚本用于测试和可视化 HybridPathPlanner 的路径规划功能，
通过动态可视化展示路径规划过程和多车运动。
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
from matplotlib.patches import Rectangle, Circle, Arrow
from typing import List, Tuple, Dict, Optional

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
    from utils.geo_tools import GeoUtils
    from models.vehicle import MiningVehicle, VehicleState
    logging.info("成功导入所需模块")
except ImportError as e:
    logging.error(f"导入模块失败: {str(e)}")
    sys.exit(1)

class PathPlannerVisualizer:
    """路径规划器可视化类"""
    
    def __init__(self, map_size=200, num_vehicles=5, num_test_points=6):
        """初始化可视化环境"""
        # 设置地图尺寸
        self.map_size = map_size
        self.num_vehicles = num_vehicles
        self.num_test_points = num_test_points
        
        # 创建地图服务和路径规划器
        self.geo_utils = GeoUtils()
        self.map_service = MapService()
        self.planner = HybridPathPlanner(self.map_service)
        
        # 初始化测试对象
        self.vehicles = self._create_test_vehicles()
        self.test_points = self._create_test_points()
        self.obstacles = self._create_obstacles()
        
        # 将障碍物应用到规划器
        self.planner.obstacle_grids = set(self.obstacles)
        
        # 设置可视化
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.artists = {}  # 存储需要更新的图形对象
        
        # 路径规划测试参数
        self.current_test_idx = 0
        self.test_pairs = []
        self._generate_test_pairs()
        
        # 动画控制
        self.animation_speed = 1.0
        self.show_path = True
        self.pause = False
        
        # 模拟时间步长
        self.time_step = 0.2
        
        # 当前活动车辆和路径
        self.active_vehicles = []
        self.vehicle_paths = {}
        self.vehicle_path_progress = {}
        
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
                'base_location': (100, 100)
            }
            
            vehicle = MiningVehicle(
                vehicle_id=i+1,
                map_service=self.map_service,
                config=config
            )
            
            # 添加颜色属性
            vehicle.color = colors[i]
            
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
            while not valid_point:
                x = random.randint(30, self.map_size - 30)
                y = random.randint(30, self.map_size - 30)
                
                # 确保点不是障碍物
                if not self._is_obstacle_area(x, y):
                    valid_point = True
                    points[f"测试点{i}"] = (x, y)
        
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
        
        return False
    
    def _create_obstacles(self) -> List[Tuple[int, int]]:
        """创建障碍物"""
        obstacles = []
        
        # 障碍物区域
        obstacle_areas = [
            (80, 30, 120, 80),    # 中下方障碍物
            (30, 80, 80, 120),    # 左中障碍物
            (120, 80, 170, 120),  # 右中障碍物
            (80, 120, 120, 170)   # 中上方障碍物
        ]
        
        # 生成障碍点
        for area in obstacle_areas:
            x_min, y_min, x_max, y_max = area
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    obstacles.append((x, y))
        
        logging.info(f"创建了 {len(obstacles)} 个障碍点")
        return obstacles
    
    def _generate_test_pairs(self):
        """生成测试点对"""
        point_names = list(self.test_points.keys())
        
        # 确保每个点都被访问到
        for i in range(len(point_names)):
            next_idx = (i + 1) % len(point_names)
            self.test_pairs.append((point_names[i], point_names[next_idx]))
            
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
        # 设置图形
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)
        self.ax.set_aspect('equal')
        self.ax.set_title('路径规划器测试可视化')
        self.ax.set_xlabel('X坐标')
        self.ax.set_ylabel('Y坐标')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制障碍物
        obstacle_patches = []
        for obs in self.obstacles:
            rect = Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, 
                         facecolor='gray', alpha=0.5)
            self.ax.add_patch(rect)
            obstacle_patches.append(rect)
        
        # 绘制测试点
        for name, point in self.test_points.items():
            self.ax.plot(point[0], point[1], 'o', markersize=10, label=name)
            self.ax.text(point[0]+5, point[1]+5, name, fontsize=10)
        
        # 创建车辆标记
        vehicle_markers = []
        vehicle_labels = []
        vehicle_paths = []
        for vehicle in self.vehicles:
            # 车辆标记
            marker = Circle(vehicle.current_location, radius=5, 
                          color=vehicle.color, label=f'车辆{vehicle.vehicle_id}')
            self.ax.add_patch(marker)
            vehicle_markers.append(marker)
            
            # 车辆标签
            label = self.ax.text(
                vehicle.current_location[0]+5, 
                vehicle.current_location[1]+5, 
                f'{vehicle.vehicle_id}', 
                color=vehicle.color, 
                fontweight='bold'
            )
            vehicle_labels.append(label)
            
            # 车辆路径
            path_line, = self.ax.plot(
                [], [], '-', 
                color=vehicle.color, 
                alpha=0.7, 
                linewidth=2
            )
            vehicle_paths.append(path_line)
        
        # 存储需要更新的对象
        self.artists['vehicle_markers'] = vehicle_markers
        self.artists['vehicle_labels'] = vehicle_labels
        self.artists['vehicle_paths'] = vehicle_paths
        
        # 添加图例
        self.ax.legend(loc='upper right')
        
        # 添加状态文本
        self.artists['status_text'] = self.ax.text(
            10, 10, '', fontsize=10, 
            verticalalignment='bottom', 
            bbox=dict(facecolor='white', alpha=0.7)
        )
    
    def update_frame(self, frame_num):
        """动画更新函数"""
        if self.pause:
            return self.artists['vehicle_markers'] + self.artists['vehicle_labels'] + self.artists['vehicle_paths'] + [self.artists['status_text']]
        
        # 更新车辆位置和路径
        for i, vehicle in enumerate(self.vehicles):
            # 更新活动车辆
            self._update_active_vehicles()
            
            # 如果车辆在活动列表中，更新其位置
            if vehicle in self.active_vehicles:
                path = self.vehicle_paths[vehicle]
                progress = self.vehicle_path_progress[vehicle]
                
                # 获取当前位置
                if 0 <= progress < len(path):
                    current_pos = path[progress]
                    vehicle.current_location = current_pos
                    
                    # 增加进度
                    self.vehicle_path_progress[vehicle] += 1
                    
                    # 检查是否到达终点
                    if progress >= len(path) - 1:
                        self.active_vehicles.remove(vehicle)
                
            # 更新标记和标签
            marker = self.artists['vehicle_markers'][i]
            label = self.artists['vehicle_labels'][i]
            path_line = self.artists['vehicle_paths'][i]
            
            marker.center = vehicle.current_location
            label.set_position((vehicle.current_location[0]+5, vehicle.current_location[1]+5))
            
            # 更新路径线
            if vehicle in self.active_vehicles and self.show_path:
                path = self.vehicle_paths[vehicle]
                path_line.set_data([p[0] for p in path], [p[1] for p in path])
            else:
                path_line.set_data([], [])
        
        # 更新状态文本
        active_count = len(self.active_vehicles)
        completed_tests = self.current_test_idx
        total_tests = len(self.test_pairs)
        
        status = (
            f"活动车辆: {active_count}/{self.num_vehicles}\n"
            f"完成测试: {completed_tests}/{total_tests}\n"
            f"测试进度: {completed_tests/total_tests*100:.1f}%"
        )
        self.artists['status_text'].set_text(status)
        
        # 如果没有活动车辆，启动新测试
        if len(self.active_vehicles) == 0 and self.current_test_idx < len(self.test_pairs):
            self._start_new_test()
        
        return self.artists['vehicle_markers'] + self.artists['vehicle_labels'] + self.artists['vehicle_paths'] + [self.artists['status_text']]
    
    def _update_active_vehicles(self):
        """更新活动车辆列表"""
        # 移除已经到达终点的车辆
        to_remove = []
        for vehicle in self.active_vehicles:
            if vehicle not in self.vehicle_path_progress:
                to_remove.append(vehicle)
                continue
                
            progress = self.vehicle_path_progress[vehicle]
            path = self.vehicle_paths[vehicle]
            
            if progress >= len(path):
                to_remove.append(vehicle)
        
        for vehicle in to_remove:
            if vehicle in self.active_vehicles:
                self.active_vehicles.remove(vehicle)
    
    def _start_new_test(self):
        """启动新的测试"""
        if self.current_test_idx >= len(self.test_pairs):
            return
            
        # 获取测试点对
        start_name, end_name = self.test_pairs[self.current_test_idx]
        start_point = self.test_points[start_name]
        end_point = self.test_points[end_name]
        
        # 更新标题
        self.ax.set_title(f'测试 {self.current_test_idx+1}/{len(self.test_pairs)}: {start_name} → {end_name}')
        
        # 选择一辆空闲车辆
        idle_vehicles = [v for v in self.vehicles if v not in self.active_vehicles]
        if not idle_vehicles:
            return
            
        vehicle = random.choice(idle_vehicles)
        
        # 规划路径
        path = self.planner.plan_path(vehicle.current_location, start_point, vehicle)
        if path and len(path) > 1:
            # 先移动到起点
            self.active_vehicles.append(vehicle)
            self.vehicle_paths[vehicle] = path
            self.vehicle_path_progress[vehicle] = 0
            
            # 设置完成函数，当到达起点时规划到终点的路径
            def on_reach_start():
                # 规划到终点的路径
                path_to_end = self.planner.plan_path(start_point, end_point, vehicle)
                if path_to_end and len(path_to_end) > 1:
                    self.vehicle_paths[vehicle] = path_to_end
                    self.vehicle_path_progress[vehicle] = 0
                    logging.info(f"车辆{vehicle.vehicle_id}从{start_name}出发前往{end_name}，路径长度: {len(path_to_end)}")
                else:
                    logging.warning(f"车辆{vehicle.vehicle_id}无法规划从{start_name}到{end_name}的路径")
                    self.active_vehicles.remove(vehicle)
            
            # 添加到计时器
            timer = self.fig.canvas.new_timer(interval=100 * len(path))
            timer.add_callback(on_reach_start)
            timer.start()
            
            logging.info(f"车辆{vehicle.vehicle_id}开始前往起点{start_name}，路径长度: {len(path)}")
            
            # 增加测试索引
            self.current_test_idx += 1
        else:
            logging.warning(f"车辆{vehicle.vehicle_id}无法规划到起点{start_name}的路径")
            # 尝试下一个测试
            self.current_test_idx += 1
            self._start_new_test()
    
    def run_visualization(self):
        """运行可视化"""
        # 设置可视化环境
        self.setup_visualization()
        
        # 设置键盘事件处理
        def on_key(event):
            if event.key == ' ':  # 空格键
                self.pause = not self.pause
                if self.pause:
                    logging.info("动画已暂停")
                else:
                    logging.info("动画已继续")
            elif event.key == 'p':  # 显示/隐藏路径
                self.show_path = not self.show_path
            elif event.key == '+' or event.key == '=':  # 加快速度
                self.animation_speed = min(5.0, self.animation_speed * 1.5)
                logging.info(f"动画速度: {self.animation_speed:.1f}x")
            elif event.key == '-':  # 减慢速度
                self.animation_speed = max(0.1, self.animation_speed / 1.5)
                logging.info(f"动画速度: {self.animation_speed:.1f}x")
                
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # 创建动画
        ani = animation.FuncAnimation(
            self.fig, self.update_frame, 
            interval=int(100 / self.animation_speed),
            blit=True
        )
        
        # 显示图形
        plt.tight_layout()
        plt.show()
        
        return ani

if __name__ == "__main__":
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='路径规划器可视化测试')
    parser.add_argument('--vehicles', type=int, default=5, help='车辆数量')
    parser.add_argument('--points', type=int, default=10, help='测试点数量')
    parser.add_argument('--size', type=int, default=200, help='地图尺寸')
    
    args = parser.parse_args()
    
    # 创建可视化器并运行
    visualizer = PathPlannerVisualizer(
        map_size=args.size,
        num_vehicles=args.vehicles,
        num_test_points=args.points
    )
    
    # 运行可视化
    ani = visualizer.run_visualization()