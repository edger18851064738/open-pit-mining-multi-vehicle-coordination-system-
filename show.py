import os
import math
from config.paths import PROJECT_ROOT
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
from matplotlib.animation import FuncAnimation
from algorithm.path_planner import HybridPathPlanner
import numpy as np
from models.vehicle import MiningVehicle, VehicleState, TransportStage

# 配置调试日志
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
rcParams['font.sans-serif'] = ['SimHei']  # Windows系统字体
rcParams['axes.unicode_minus'] = False

class MockMapService:
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

def main():
    # 初始化规划器
    planner = HybridPathPlanner(MockMapService())
    
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
    
    # 标记障碍物
    planner.mark_obstacle_area(test_obstacles)
    
    # 定义关键点
    load_point1 = (20, 20)
    load_point2 = (20, 130)
    load_point3 = (130, 20)
    unload_point = (75, 75)
    parking_point = (130, 75)
    
    # 计算六条路径
    path1 = planner.optimize_path(load_point1, unload_point, MiningVehicle("dummy", MockMapService(), {"turning_radius": 10.0, "max_capacity": 50, "max_speed": 5.0, "min_hardness": 2.5}))
    path2 = planner.optimize_path(load_point2, unload_point, MiningVehicle("dummy", MockMapService(), {"turning_radius": 10.0, "max_capacity": 50, "max_speed": 5.0, "min_hardness": 2.5}))
    path3 = planner.optimize_path(load_point3, unload_point, MiningVehicle("dummy", MockMapService(), {"turning_radius": 10.0, "max_capacity": 50, "max_speed": 5.0, "min_hardness": 2.5}))
    path4 = planner.optimize_path(unload_point, parking_point, MiningVehicle("dummy", MockMapService(), {"turning_radius": 10.0, "max_capacity": 50, "max_speed": 5.0, "min_hardness": 2.5}))
    path5 = planner.optimize_path(parking_point, load_point1, MiningVehicle("dummy", MockMapService(), {"turning_radius": 10.0, "max_capacity": 50, "max_speed": 5.0, "min_hardness": 2.5}))
    path6 = planner.optimize_path(parking_point, load_point2, MiningVehicle("dummy", MockMapService(), {"turning_radius": 10.0, "max_capacity": 50, "max_speed": 5.0, "min_hardness": 2.5}))
    path7 = planner.optimize_path(parking_point, load_point3, MiningVehicle("dummy", MockMapService(), {"turning_radius": 10.0, "max_capacity": 50, "max_speed": 5.0, "min_hardness": 2.5}))
    
    # 可视化
    plt.figure(figsize=(15, 15))
    
    # 绘制障碍物
    for polygon in test_obstacles:
        plt.fill(*zip(*polygon), color='gray', alpha=0.5)
    
    # 绘制七条路径
    if path1:
        plt.plot(*zip(*path1), 'r-', linewidth=2, label='装载点1到卸载点')
    if path2:
        plt.plot(*zip(*path2), 'g-', linewidth=2, label='装载点2到卸载点')
    if path3:
        plt.plot(*zip(*path3), 'b-', linewidth=2, label='装载点3到卸载点')
    if path4:
        plt.plot(*zip(*path4), 'c-', linewidth=2, label='卸载点到停车场')
    if path5:
        plt.plot(*zip(*path5), 'm-', linewidth=2, label='停车场到装载点1')
    if path6:
        plt.plot(*zip(*path6), 'y-', linewidth=2, label='停车场到装载点2')
    if path7:
        plt.plot(*zip(*path7), 'k-', linewidth=2, label='停车场到装载点3')
    
    # 标记关键点
    plt.scatter(*load_point1, c='red', s=100, marker='o', label='装载点1')
    plt.scatter(*load_point2, c='green', s=100, marker='o', label='装载点2')
    plt.scatter(*load_point3, c='blue', s=100, marker='o', label='装载点3')
    plt.scatter(*unload_point, c='magenta', s=150, marker='s', label='卸载点')
    plt.scatter(*parking_point, c='yellow', s=150, marker='^', label='停车场')
    
    # 创建动画
    paths = [path1, path2, path3, path4, path5, path6, path7]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    labels = ['装载点1到卸载点', '装载点2到卸载点', '装载点3到卸载点', 
             '卸载点到停车场', '停车场到装载点1', '停车场到装载点2', '停车场到装载点3']
    
    # 创建动点
    dots = [plt.plot([], [], 'o', color=colors[i], markersize=10)[0] for i in range(len(paths))]
    
    # 计算最大路径长度
    max_length = max(len(path) for path in paths if path)
    
    def update(frame):
        for i, path in enumerate(paths):
            if path:
                # 计算总路径长度
                total_distance = sum(math.sqrt((path[k+1][0]-path[k][0])**2 + (path[k+1][1]-path[k][1])**2) 
                                   for k in range(len(path)-1))
                
                # 计算当前帧对应的路径点索引，保持匀速
                speed = 0.5  # 每帧移动0.5个单位距离
                current_distance = min(frame * speed, total_distance)
                
                # 计算当前点位置
                accumulated_distance = 0
                current_pos = path[0]
                for k in range(len(path)-1):
                    segment_distance = math.sqrt((path[k+1][0]-path[k][0])**2 + (path[k+1][1]-path[k][1])**2)
                    if accumulated_distance + segment_distance >= current_distance:
                        # 线性插值计算当前位置
                        ratio = (current_distance - accumulated_distance) / segment_distance
                        x = path[k][0] + ratio * (path[k+1][0] - path[k][0])
                        y = path[k][1] + ratio * (path[k+1][1] - path[k][1])
                        current_pos = (x, y)
                        break
                    accumulated_distance += segment_distance
                    current_pos = path[k+1]
                
                dots[i].set_data([current_pos[0]], [current_pos[1]])
        return dots
    
    ani = FuncAnimation(plt.gcf(), update, frames=max_length, 
                       interval=300, blit=True, repeat=True)
    
    plt.legend()
    plt.title('矿山运输路径规划 (150x150)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()