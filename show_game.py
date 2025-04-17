import os
import math

from config.paths import PROJECT_ROOT
import heapq
import logging
import time
import threading
import numpy as np
import pygame
from typing import List, Tuple, Dict, Optional, Union
import networkx as nx
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dispatch import GLOBAL_CONFIG


# 导入项目模块
from utils.geo_tools import GeoUtils
from config.settings import MapConfig, PathConfig, AppConfig
from algorithm.map_service import MapService
from utils.path_tools import PathOptimizationError
from algorithm.path_planner import HybridPathPlanner
from models.vehicle import MiningVehicle, VehicleState, TransportStage

# 初始化pygame
pygame.init()

# 屏幕设置
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("矿山运输路径规划")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
GRAY = (128, 128, 128)

# 缩放因子 (将150x150的虚拟坐标映射到800x800的屏幕)
SCALE_FACTOR = SCREEN_WIDTH / 150
# 调整坐标偏移量以确保所有点都能显示在屏幕内
OFFSET_X = 20
OFFSET_Y = 20

class PathVisualizer:
    """路径可视化工具类"""
    def __init__(self, screen_width=800, screen_height=800, obstacles=None, points_config=None):
        """初始化可视化设置"""
        self.SCREEN_WIDTH = screen_width
        self.SCREEN_HEIGHT = screen_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("矿山运输路径规划")
        
        # 从外部接收或使用默认配置
        self.obstacles = obstacles if obstacles is not None else [
            [(20,60), (130,60), (130,70), (20,70)],
            [(40,30), (60,30), (60,40), (40,40)],
            [(90,100), (110,100), (110,110), (90,110)],
            [(30,20), (50,20), (50,120), (30,120)],
            [(100,20), (120,20), (120,120), (100,120)],
            [(50,40), (70,40), (70,50), (50,50)],
            [(80,80), (100,80), (100,90), (80,90)],
            [(40,40), (50,40), (50,50), (40,50)],
            [(110,90), (130,90), (130,100), (110,100)]
        ]
        
        self.points_config = points_config if points_config is not None else {
            'load1': (20, 20),
            'load2': (20, 130),
            'load3': (130, 20),
            'unload': (75, 75),
            'parking': (130, 75)
        }
        
        # 颜色定义
        self.COLORS = {
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0),
            'RED': (255, 0, 0),
            'GREEN': (0, 255, 0),
            'BLUE': (0, 0, 255),
            'YELLOW': (255, 255, 0),
            'MAGENTA': (255, 0, 255),
            'GRAY': (128, 128, 128),
            'OBSTACLE': (200, 200, 200)
        }
        
        # 缩放因子 (将150x150的虚拟坐标映射到800x800的屏幕)
        self.SCALE_FACTOR = screen_width / 150
        # 调整坐标偏移量以确保所有点都能显示在屏幕内
        self.OFFSET_X = 20
        self.OFFSET_Y = 20
        
    def draw_obstacles(self, obstacles):
        """绘制障碍物"""
        for obstacle in obstacles:
            self.draw_polygon(self.COLORS['OBSTACLE'], obstacle)
            
    def draw_map(self, points_config):
        """绘制地图关键点"""
        # 绘制装载点
        self.draw_point(self.COLORS['RED'], points_config['load1'])
        self.draw_point(self.COLORS['GREEN'], points_config['load2'])
        self.draw_point(self.COLORS['BLUE'], points_config['load3'])
        # 绘制卸载点
        self.draw_point(self.COLORS['YELLOW'], points_config['unload'])
        # 绘制停车场
        self.draw_point(self.COLORS['MAGENTA'], points_config['parking'])
    
    def scale_coord(self, coord: Tuple[float, float]) -> Tuple[int, int]:
        """将虚拟坐标转换为屏幕坐标"""
        x = max(0, min(coord[0], 150))
        y = max(0, min(coord[1], 150))
        return (int(x * self.SCALE_FACTOR) + self.OFFSET_X, 
                int(y * self.SCALE_FACTOR) + self.OFFSET_Y)
    
    def draw_polygon(self, color, polygon):
        """绘制多边形"""
        scaled_points = [self.scale_coord(p) for p in polygon]
        pygame.draw.polygon(self.screen, color, scaled_points)
    
    def draw_path(self, color, path, width=3):
        """绘制路径"""
        if len(path) < 2:
            return
        
        scaled_points = [self.scale_coord(p) for p in path]
        pygame.draw.lines(self.screen, color, False, scaled_points, width)
    
    def draw_point(self, color, point, radius=10):
        """绘制点"""
        scaled_point = self.scale_coord(point)
        pygame.draw.circle(self.screen, color, scaled_point, radius)
        
def scale_coord(coord: Tuple[float, float]) -> Tuple[int, int]:
    """将虚拟坐标转换为屏幕坐标"""
    x = max(0, min(coord[0], 150))
    y = max(0, min(coord[1], 150))
    return (int(x * SCALE_FACTOR) + OFFSET_X, 
            int(y * SCALE_FACTOR) + OFFSET_Y)

def draw_point(screen, color, point, radius=10):
    """全局绘制点函数"""
    scaled_point = scale_coord(point)
    pygame.draw.circle(screen, color, scaled_point, int(radius))
        
def display_vehicles(vehicles, dispatcher=None):
    """显示矿车位置
    :param vehicles: 矿车列表
    :param dispatcher: 调度器实例，用于获取实时位置更新
    """
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("矿车调度系统")
    visualizer = PathVisualizer()
    
    # 绘制静态地图元素
    visualizer.draw_obstacles(visualizer.obstacles)
    visualizer.draw_map(visualizer.points_config)
    
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 更新车辆位置
        if dispatcher:
            dispatcher.update_vehicle_positions()
        
        screen.fill((255, 255, 255))
        # 重新绘制静态地图元素
        visualizer.draw_obstacles(visualizer.obstacles)
        visualizer.draw_map(visualizer.points_config)
        
        # 绘制动态车辆
        for vehicle in vehicles:
            if vehicle.current_location:
                visualizer.draw_point((0, 0, 255), vehicle.current_location)
        
        pygame.display.flip()
        clock.tick(30)  # 控制帧率为30FPS
    
    pygame.quit()

class MockMapService:
    """模拟地图服务类"""
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



def plan_paths(planner, vehicle, points_config):
    """
    路径规划函数
    :param planner: 路径规划器实例
    :param vehicle: 车辆实例
    :param points_config: 关键点配置字典
    :return: 路径列表和颜色列表
    """
    paths = [
        planner.optimize_path(points_config['load1'], points_config['unload'], vehicle),
        planner.optimize_path(points_config['load2'], points_config['unload'], vehicle),
        planner.optimize_path(points_config['load3'], points_config['unload'], vehicle),
        planner.optimize_path(points_config['unload'], points_config['parking'], vehicle),
        planner.optimize_path(points_config['parking'], points_config['load1'], vehicle),
        planner.optimize_path(points_config['parking'], points_config['load2'], vehicle),
        planner.optimize_path(points_config['parking'], points_config['load3'], vehicle)
    ]
    
    path_colors = [
        'RED', 'GREEN', 'BLUE', 'YELLOW', 'MAGENTA', 
        (255, 165, 0), (0, 255, 255)
    ]
    
    return paths, path_colors

def run_animation(visualizer, paths, path_colors, points_config, fps=60, speed=0.5):
    """
    运行动画
    :param visualizer: 可视化工具实例
    :param paths: 路径列表
    :param path_colors: 路径颜色列表
    :param points_config: 关键点配置字典
    :param fps: 帧率
    :param speed: 动画速度
    """
    clock = pygame.time.Clock()
    current_positions = [0.0] * len(paths)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 清屏
        visualizer.screen.fill(visualizer.COLORS['WHITE'])
        
        # 绘制地图和障碍物
        visualizer.draw_obstacles(test_obstacles)
        visualizer.draw_map(points_config)
        
        # 更新动画位置
        for i, path in enumerate(paths):
            if path:
                # 计算总路径长度
                total_distance = sum(math.sqrt((path[k+1][0]-path[k][0])**2 + 
                                             (path[k+1][1]-path[k][1])**2) 
                                  for k in range(len(path)-1))
                
                # 更新当前进度
                current_positions[i] = min(current_positions[i] + speed, total_distance)
                
                # 计算当前点位置
                accumulated_distance = 0
                current_point = path[0]
                for k in range(len(path)-1):
                    segment_distance = math.sqrt((path[k+1][0]-path[k][0])**2 + 
                                                (path[k+1][1]-path[k][1])**2)
                    if accumulated_distance + segment_distance >= current_positions[i]:
                        # 线性插值计算当前位置
                        ratio = (current_positions[i] - accumulated_distance) / segment_distance
                        x = path[k][0] + ratio * (path[k+1][0] - path[k][0])
                        y = path[k][1] + ratio * (path[k+1][1] - path[k][1])
                        current_point = (x, y)
                        break
                    accumulated_distance += segment_distance
                    current_point = path[k+1]
                
                # 绘制路径和动点
                visualizer.draw_path(path_colors[i], path)
                visualizer.draw_point(path_colors[i], current_point)
        
        pygame.display.flip()
        clock.tick(fps)

from dispatch import GLOBAL_CONFIG

# 测试用例配置
test_obstacles = [
    [(20,60), (130,60), (130,70), (20,70)],
    [(40,30), (60,30), (60,40), (40,40)],
    [(90,100), (110,100), (110,110), (90,110)],
    [(30,20), (50,20), (50,120), (30,120)],
    [(100,20), (120,20), (120,120), (100,120)],
    [(50,40), (70,40), (70,50), (50,50)],
    [(80,80), (100,80), (100,90), (80,90)],
    [(40,40), (50,40), (50,50), (40,50)],
    [(110,90), (130,90), (130,100), (110,100)]
]

# 定义关键点
points_config = {
    'load1': (20, 20),
    'load2': (20, 130),
    'load3': (130, 20),
    'unload': (75, 75),
    'parking': (130, 75)
}

def main(obstacles=None, points_config=None):
    """主函数
    :param obstacles: 障碍物列表，可选
    :param points_config: 关键点配置字典，可选
    """
    # 使用默认配置如果未提供
    if points_config is None:
        points_config = {
            'load1': (20, 20),
            'load2': (20, 130),
            'load3': (130, 20),
            'unload': (75, 75),
            'parking': (130, 75)
        }
    
    # 初始化可视化工具
    visualizer = PathVisualizer(obstacles=obstacles, points_config=points_config)
    
    # 初始化规划器
    planner = HybridPathPlanner(MockMapService())
    
    # 标记障碍物
    planner.mark_obstacle_area(test_obstacles)
    
    # 标记障碍物
    planner.mark_obstacle_area(test_obstacles)
    
    # 标记障碍物
    planner.mark_obstacle_area(test_obstacles)
    
    # 创建测试车辆
    vehicle = MiningVehicle("test_vehicle", MockMapService(), {
        "turning_radius": 10.0,
        "max_capacity": 50,
        "max_speed": 5.0,
        "min_hardness": 2.5
    })
    
    # 规划路径
    paths, path_colors = plan_paths(planner, vehicle, points_config)
    
    # 运行动画
    run_animation(visualizer, paths, path_colors, points_config)

def display_vehicles(vehicles: List[MiningVehicle]):
    """
    显示矿车位置和任务状态
    :param vehicles: 矿车列表
    """
    # 初始化可视化工具
    visualizer = PathVisualizer()
    
    clock = pygame.time.Clock()
    FPS = 60
    
    # 矿车颜色
    vehicle_colors = [RED, GREEN, BLUE, YELLOW, MAGENTA]
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # 清屏
        visualizer.screen.fill(WHITE)
        
        # 绘制地图和障碍物
        visualizer.draw_obstacles(GLOBAL_CONFIG.obstacles)
        visualizer.draw_map(GLOBAL_CONFIG.points_config)
        
        # 绘制矿车
        for i, vehicle in enumerate(vehicles):
            if vehicle.current_location:
                color = vehicle_colors[i % len(vehicle_colors)]
                visualizer.draw_point(color, vehicle.current_location)
                
                # 显示矿车ID
                font = pygame.font.SysFont(None, 24)
                text = font.render(f"{vehicle.vehicle_id}", True, BLACK)
                visualizer.screen.blit(text, visualizer.scale_coord(vehicle.current_location))
                
                # 显示任务状态
                if vehicle.current_task:
                    task_text = f"任务: {vehicle.current_task.task_id}"
                    text = font.render(task_text, True, BLACK)
                    visualizer.screen.blit(text, (visualizer.scale_coord(vehicle.current_location)[0], 
                                visualizer.scale_coord(vehicle.current_location)[1] + 20))
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()