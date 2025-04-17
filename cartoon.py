from __future__ import annotations
import heapq
import threading
import os
import sys
from matplotlib import rcParams
import matplotlib.pyplot as plt
from config.paths import PROJECT_ROOT
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from algorithm.path_planner import HybridPathPlanner, Node, PathOptimizationError
from algorithm.map_service import MapService
from utils.geo_tools import GeoUtils
import logging
import random
import networkx as nx
import osmnx as ox

if __name__ == "__main__":
    # Configure debug logging
    logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    rcParams['font.sans-serif'] = ['SimHei']  # Windows font
    rcParams['axes.unicode_minus'] = False 
    
    # Initialize planner
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
    
    planner = HybridPathPlanner(MockMapService())

    # Key points configuration
    key_points = {
        'parking': (85, 30),      # Parking lot (adjusted position)
        'unload': (50, 50),       # Unloading point (center)
        'load1': (10, 10),        # Loading point 1 (bottom-left)
        'load2': (10, 90),        # Loading point 2 (top-left)
        'load3': (90, 10)         # Loading point 3 (bottom-right)
    }
    
    # Fixed obstacles (with passage channels)
    obstacle_polygons = [
        # Left horizontal barrier
        [(10,40), (40,40), (40,60), (10,60)],
        
        # Right horizontal barrier
        [(60,40), (90,40), (90,60), (60,60)],
        
        # Additional vertical barriers
        [(30,20), (40,20), (40,40), (30,40)],
        [(60,60), (70,60), (70,80), (60,80)],
        
        # Central obstacles
        [(40,30), (60,30), (60,40), (40,40)],
        [(40,60), (60,60), (60,70), (40,70)]
    ]

    # Mark obstacles
    planner.mark_obstacle_area(obstacle_polygons)
    
    # Validate key points are not in obstacles
    for name, point in key_points.items():
        # 只检查实际存在的障碍物
        in_obstacle = False
        for polygon in obstacle_polygons:
            if planner._point_in_polygon(point, polygon):
                in_obstacle = True
                break
    if in_obstacle:
        raise ValueError(f"关键点{name}位于障碍物内")

    # Vehicle configuration
    test_vehicle = MiningVehicle("XTR-1000", MockMapService(), {
        'min_hardness': 2.5,
        'max_load': 50,
        'speed': 5,
        'steering_angle': 30,
        'current_load': 35,
        'max_capacity': 50,
        'current_location': (0.0, 0.0)
    })

    # Define 7 paths
    path_routes = [
        (key_points['load1'], key_points['unload']),  # Load1 to Unload
        (key_points['load2'], key_points['unload']),  # Load2 to Unload
        (key_points['load3'], key_points['unload']),  # Load3 to Unload
        (key_points['unload'], key_points['parking']),  # Unload to Parking
        (key_points['parking'], key_points['load1']),  # Parking to Load1
        (key_points['parking'], key_points['load2']),  # Parking to Load2
        (key_points['parking'], key_points['load3'])   # Parking to Load3
    ]
    
    # Prepare visualization
    plt.figure(figsize=(15, 15))
    plt.rc('font', size=12)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title("Mining Vehicle Path Planning", fontsize=16, pad=20)
    
    # Plot obstacles
    plt.scatter(*zip(*planner.obstacle_grids), c='#2F4F4F', s=60, marker='s', label='Obstacles')
    
    # Color list for different paths
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    path_labels = [
        'Load1 → Unload',
        'Load2 → Unload',
        'Load3 → Unload',
        'Unload → Parking',
        'Parking → Load1',
        'Parking → Load2',
        'Parking → Load3'
    ]
    
    # Plot key points first
    for name, point in key_points.items():
        if name == 'parking':
            plt.scatter(point[0], point[1], c='yellow', s=400, edgecolors='black', 
                       marker='s', label='Parking Lot')
        elif name == 'unload':
            plt.scatter(point[0], point[1], c='magenta', s=400, 
                       edgecolors='black', marker='*', label='Unloading Point')
        else:
            plt.scatter(point[0], point[1], c='lime', s=300, 
                       edgecolors='black', marker='o', label=f'{name.capitalize()}')

    # Execute single path planning and visualization
    try:
        # 添加Fortran资源初始化
        if hasattr(planner, '_fortran_handle'):
            try:
                planner._fortran_handle.initialize()
            except Exception as e:
                logging.warning(f"Fortran资源初始化失败: {str(e)}")
                
        # 只规划第一个装载点到卸载点的路径
        start, end = path_routes[0]
        logging.info(f"正在规划路径: 装载点1到卸载点")
        test_vehicle.last_position = Node(start[0], start[1])
        path = planner.optimize_path(start, end, test_vehicle)
        logging.info(f"路径规划完成，共 {len(path)} 个点")
        print(f"Path - Valid points: {len(path)}")
        print(f"Path - Coords: {path[:2]}...{path[-2:]}")
        
        # Plot path
        plt.plot(*zip(*path), '-', linewidth=3.5, 
                label='装载点到卸载点路径', alpha=0.7, color='red')
        
        # 添加Fortran资源释放
        if hasattr(planner, '_fortran_handle'):
            try:
                planner._fortran_handle.cleanup()
            except Exception as e:
                logging.warning(f"Fortran资源释放失败: {str(e)}")
        
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.grid(True, alpha=0.3)
        
        # Adjust legend to avoid overlap
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
        
    except PathOptimizationError as e:
        logging.error(f"Path planning failed: {str(e)}")