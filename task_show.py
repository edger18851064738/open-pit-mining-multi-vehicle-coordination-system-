import sys
import os
import math
import pygame
import random
from typing import List, Dict, Tuple
from collections import defaultdict
import time

from config.paths import PROJECT_ROOT

from models.task import TransportTask
from models.vehicle import MiningVehicle, VehicleState
from algorithm.path_planner import HybridPathPlanner
from algorithm.map_service import MapService



# 屏幕和颜色设置
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
MAGENTA = (255, 0, 255)
OBSTACLE_COLOR = (200, 200, 200)

class PathVisualizer:
    """增强版可视化工具"""
    def __init__(self, obstacles=None, points_config=None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("矿山运输任务可视化")
        
        self.obstacles = obstacles or []
        self.points_config = points_config or {}
        self.SCALE_FACTOR = SCREEN_WIDTH / 150
        self.OFFSET_X, self.OFFSET_Y = 20, 20
        
    def draw_obstacles(self):
        for obstacle in self.obstacles:
            self.draw_polygon(OBSTACLE_COLOR, obstacle)
            
    def draw_map(self):
        for point_name, color in [('load1', RED), ('load2', GREEN), 
                                ('load3', BLUE), ('unload', YELLOW), 
                                ('parking', MAGENTA)]:
            if point_name in self.points_config:
                self.draw_point(color, self.points_config[point_name])
    
    def scale_coord(self, coord):
        x = max(0, min(coord[0], 150))
        y = max(0, min(coord[1], 150))
        return (int(x * self.SCALE_FACTOR) + self.OFFSET_X, 
                int(y * self.SCALE_FACTOR) + self.OFFSET_Y)
    
    def draw_polygon(self, color, polygon):
        pygame.draw.polygon(self.screen, color, [self.scale_coord(p) for p in polygon])
    
    def draw_path(self, color, path, width=3):
        if len(path) >= 2:
            pygame.draw.lines(self.screen, color, False, [self.scale_coord(p) for p in path], width)
    
    def draw_point(self, color, point, radius=10):
        pygame.draw.circle(self.screen, color, self.scale_coord(point), radius)
    
    def draw_text(self, text, pos, color=BLACK, size=24):
        font = pygame.font.SysFont(None, size)
        text_surface = font.render(text, True, color)
        self.screen.blit(text_surface, pos)

class TaskDispatcher:
    """增强版任务分配器"""
    def __init__(self, vehicles, tasks, planner):
        self.vehicles = vehicles
        self.tasks = tasks
        self.planner = planner
        self.vehicle_colors = {
            v.vehicle_id: (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            for v in vehicles
        }
    
    def assign_tasks(self):
        """改进的任务分配算法"""
        pending_tasks = [t for t in self.tasks if t.status == 'pending']
        idle_vehicles = [v for v in self.vehicles if v.state == VehicleState.IDLE]
        
        for vehicle in idle_vehicles:
            for task in pending_tasks:
                print(f"尝试为车辆 {vehicle.vehicle_id} 分配任务 {task.task_id}...")
                
                path = self.planner.optimize_path(
                    vehicle.current_location,
                    task.end_point,
                    vehicle
                )
                
                if path:
                    print(f"规划路径结果: {path}")
                    vehicle.current_task = task
                    vehicle.current_path = path
                    vehicle.state = VehicleState.EN_ROUTE
                    task.status = 'assigned'
                    print(f"成功分配任务 {task.task_id} 给车辆 {vehicle.vehicle_id}")
                    break
                else:
                    print(f"无法为车辆 {vehicle.vehicle_id} 规划到任务 {task.task_id} 的路径")
    
    def update_vehicles(self):
        """更新所有车辆状态"""
        for vehicle in self.vehicles:
            if vehicle.state == VehicleState.EN_ROUTE and vehicle.current_path:
                vehicle.move_along_path()
                
                if vehicle.has_arrived():
                    print(f"车辆 {vehicle.vehicle_id} 完成任务 {vehicle.current_task.task_id}")
                    vehicle.complete_task()
                    self.assign_tasks()  # 尝试分配新任务

def initialize_system(vehicles, tasks, dispatcher, max_attempts=30):
    """增强的初始化流程"""
    print("\n=== 系统初始化 ===")
    
    # 初始分配尝试
    dispatcher.assign_tasks()
    
    # 等待所有可用车辆完成初始化
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        
        # 更新车辆状态
        dispatcher.update_vehicles()
        
        # 检查初始化状态
        initialized = True
        for vehicle in vehicles:
            if vehicle.state == VehicleState.IDLE and not vehicle.current_task:
                # 尝试为未分配任务的空闲车辆分配任务
                dispatcher.assign_tasks()
                initialized = False
        
        if initialized:
            print("系统初始化完成")
            return True
        
        time.sleep(0.1)
    
    print("警告: 系统初始化超时")
    return False

def main():
    # 关键点配置
    points_config = {
        'load1': (20, 20),
        'load2': (20, 130),
        'load3': (130, 20),
        'unload': (75, 75),
        'parking': (130, 75)
    }
    
    # 障碍物配置 (简化了一些障碍物)
    test_obstacles = [
        [(20,60), (130,60), (130,70), (20,70)],  # 主要障碍物
        [(40,30), (60,30), (60,40), (40,40)],
        [(90,100), (110,100), (110,110), (90,110)]
    ]
    
    # 初始化地图服务
    map_service = MapService()
    planner = HybridPathPlanner(map_service)
    planner.mark_obstacle_area(test_obstacles)
    
    # 创建车辆 (减少到2辆便于调试)
    vehicles = [
        MiningVehicle(f"vehicle_{i}", map_service, {
            'turning_radius': 10.0,
            'max_capacity': 50,
            'max_speed': 5.0,
            'min_hardness': 2.5
        }) for i in range(2)
    ]
    
    # 设置初始位置
    for vehicle in vehicles:
        vehicle.current_location = points_config['parking']
        vehicle.state = 'idle'
    
    # 生成任务 (减少到3个)
    tasks = [
        TransportTask("task_0", points_config['load1'], points_config['unload'], 'loading'),
        TransportTask("task_1", points_config['unload'], points_config['parking'], 'unloading'),
        TransportTask("task_2", points_config['load2'], points_config['unload'], 'loading')
    ]
    
    # 初始化任务分配器
    dispatcher = TaskDispatcher(vehicles, tasks, planner)
    
    # 系统初始化
    if not initialize_system(vehicles, tasks, dispatcher):
        print("初始化失败，请检查路径规划或障碍物设置")
        return
        
    # 初始化可视化
    visualizer = PathVisualizer(test_obstacles, points_config)
    
    # 主循环
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # 更新
        dispatcher.update_vehicles()
        
        # 渲染
        visualizer.screen.fill(WHITE)
        visualizer.draw_obstacles()
        visualizer.draw_map()
        
        # 绘制车辆和路径
        for vehicle in vehicles:
            if vehicle.current_path:
                visualizer.draw_path(dispatcher.vehicle_colors[vehicle.vehicle_id], vehicle.current_path)
            if vehicle.current_location:
                visualizer.draw_point(RED, vehicle.current_location)
                visualizer.draw_text(f"V{vehicle.vehicle_id}", 
                                    (visualizer.scale_coord(vehicle.current_location)[0]+15, 
                                    visualizer.scale_coord(vehicle.current_location)[1]+15))
        
        # 显示任务状态
        completed = sum(1 for t in tasks if t.status == 'completed')
        visualizer.draw_text(f"任务进度: {completed}/{len(tasks)}", (10, 10))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == "__main__":
    main()