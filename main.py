import sys
import os
import argparse
import time
import logging
import random
import pygame
import torch
from typing import List, Dict
from collections import defaultdict
from pygame.locals import *


# 项目路径配置
from config.paths import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

# 导入项目模块
from algorithm.map_service import MapService
from algorithm.dispatch_service_v1 import DispatchService
from algorithm.path_planner import HybridPathPlanner
from models.vehicle import MiningVehicle, VehicleState
from models.task import TransportTask
from train import QMixTrainer
# 显示配置
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 800
WORLD_SIZE = 500  # 模拟世界坐标范围 [-250, 250]
FPS = 60

# 颜色配置
COLORS = {
    VehicleState.IDLE: (0, 255, 0),       # 绿色
    VehicleState.EN_ROUTE: (255, 165, 0), # 橙色
    VehicleState.UNLOADING: (255, 0, 0),  # 红色
    VehicleState.PREPARING: (128, 0, 128),# 紫色
    'task_point': (0, 0, 0),              # 黑色
    'background': (255, 255, 255),        # 白色
    'loading_point': (0, 0, 255),         # 蓝色
    'unloading_point': (255, 0, 255),     # 品红
    'parking_point': (0, 255, 255),       # 青色
    'path': (200, 200, 200)               # 灰色
}

def world_to_screen(pos):
    """将世界坐标转换为屏幕坐标"""
    x, y = pos
    return (
        int(SCREEN_WIDTH/2 + x * SCREEN_WIDTH/WORLD_SIZE),
        int(SCREEN_HEIGHT/2 - y * SCREEN_HEIGHT/WORLD_SIZE)
    )

def draw_vehicle(screen, vehicle):
    """绘制车辆"""
    pos = world_to_screen(vehicle.position)
    pygame.draw.circle(screen, COLORS[vehicle.state], pos, 8)
    
    # 显示车辆ID
    font = pygame.font.SysFont(None, 20)
    text = font.render(vehicle.vehicle_id, True, (0, 0, 0))
    screen.blit(text, (pos[0] + 10, pos[1] - 10))

def draw_path(screen, path):
    """绘制路径"""
    if len(path) < 2:
        return
        
    screen_points = [world_to_screen(p) for p in path]
    pygame.draw.lines(screen, COLORS['path'], False, screen_points, 2)

def draw_task_points(screen, tasks):
    """绘制任务点"""
    for task in tasks:
        if task.task_type == 'loading':
            color = COLORS['loading_point']
        else:
            color = COLORS['unloading_point']
            
        pos = world_to_screen(task.location)
        pygame.draw.circle(screen, color, pos, 6)

def main():
    """主模拟循环"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("矿山车辆调度模拟")
    clock = pygame.time.Clock()
    
    # 初始化调度服务
    map_service = MapService()
    planner = HybridPathPlanner(map_service)
    dispatch = DispatchService(planner, map_service)
    
    # 初始化日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # 添加初始车辆和任务
    vehicles = [
        MiningVehicle(f"vehicle_{i}", map_service, {
            'speed': random.uniform(3, 7),
            'turning_radius': random.uniform(8, 15),
            'min_hardness': 2.5,
            'max_load': 50,
            'current_load': 0,
            'steering_angle': 30,
            'max_capacity': 50,
            'base_location': (0.0, 0.0),
            'current_location': (random.uniform(-200, 200), random.uniform(-200, 200))
        }) for i in range(5)
    ]
    
    tasks = [
        TransportTask(f"task_{i}",
                     (random.uniform(-200, 200), random.uniform(-200, 200)),
                     (random.uniform(-200, 200), random.uniform(-200, 200)),
                     random.choice(['loading', 'unloading']))
        for i in range(10)
    ]
    
    # 主循环
    running = True
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        
        # 更新调度
        logger.info(f"开始更新调度，当前车辆数: {len(vehicles)}，任务数: {len(tasks)}")
        dispatch.update(vehicles, tasks)
        logger.info("调度更新完成")
        
        # 更新车辆位置
        for vehicle in vehicles:
            if hasattr(vehicle, 'current_path') and vehicle.current_path:
                vehicle.move_along_path()
                logger.info(f"车辆 {vehicle.vehicle_id} 移动到位置 {vehicle.position}")
        
        # 绘制
        screen.fill(COLORS['background'])
        
        # 绘制所有车辆和路径
        logger.info("开始绘制车辆和路径")
        for vehicle in vehicles:
            draw_vehicle(screen, vehicle)
            if hasattr(vehicle, 'current_path'):
                draw_path(screen, vehicle.current_path)
        
        # 绘制任务点
        logger.info(f"开始绘制任务点，共{len(tasks)}个任务")
        draw_task_points(screen, tasks)
        logger.info("绘制完成")
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()

if __name__ == "__main__":
    main()
