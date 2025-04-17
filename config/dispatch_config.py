#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
露天矿无人多车协同调度系统配置模块

提供系统运行所需的各项配置参数，包括地图设置、车辆参数、调度策略等
"""

from typing import Dict, List, Tuple, Any
import os
import configparser
import logging

class DispatchConfig:
    """调度系统配置类"""
    
    def __init__(self):
        """初始化默认配置"""
        # 调度周期配置
        self.DISPATCH_INTERVAL = 30.0  # 调度周期（秒）
        self.MONITOR_INTERVAL = 10.0   # 监控周期（秒）
        self.CONFLICT_CHECK_INTERVAL = 5.0  # 冲突检测周期（秒）
        
        # 关键点坐标
        self.LOADING_POINTS = [(-100, 50), (0, 150), (100, 50)]
        self.UNLOADING_POINT = (0, -100)
        self.PARKING_AREA = (200, 200)
        self.CHARGING_STATIONS = [(200, 200)]
        
        # 调度策略参数
        self.MAX_CHARGING_VEHICLES = 2
        self.MAX_LOADING_QUEUE = 5
        self.MAX_UNLOADING_QUEUE = 5
        self.TIME_WINDOW_SIZE = 15  # 时间窗口大小（分钟）
        
        # 冲突解决策略
        self.CONFLICT_RESOLUTION_STRATEGY = "priority"  # priority, reroute, wait
        self.MAX_REROUTE_ATTEMPTS = 3
        
        # 车辆参数
        self.DEFAULT_VEHICLE_CONFIG = {
            'max_capacity': 50,
            'min_hardness': 2.5,
            'max_speed': 5.0,
            'turning_radius': 10.0,
            'steering_angle': 30
        }
        
        # 任务参数
        self.TASK_PRIORITY_LEVELS = 3
        self.TASK_TIMEOUT = 120  # 任务超时时间（分钟）
        
        # 系统参数
        self.LOG_LEVEL = logging.INFO
        self.SIMULATION_MODE = False
        
    def load_from_file(self, config_path: str) -> bool:
        """从配置文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            bool: 是否成功加载配置
        """
        if not os.path.exists(config_path):
            logging.warning(f"配置文件 {config_path} 不存在，使用默认配置")
            return False
        
        try:
            config = configparser.ConfigParser()
            config.read(config_path, encoding='utf-8')
            
            # 加载调度周期配置
            if 'DISPATCH' in config:
                self.DISPATCH_INTERVAL = config.getfloat('DISPATCH', 'dispatch_interval', fallback=self.DISPATCH_INTERVAL)
                self.MONITOR_INTERVAL = config.getfloat('DISPATCH', 'monitor_interval', fallback=self.MONITOR_INTERVAL)
                self.CONFLICT_CHECK_INTERVAL = config.getfloat('DISPATCH', 'conflict_check_interval', fallback=self.CONFLICT_CHECK_INTERVAL)
            
            # 加载关键点坐标
            if 'LOCATIONS' in config:
                # 解析装载点列表
                loading_points_str = config.get('LOCATIONS', 'loading_points', fallback=None)
                if loading_points_str:
                    try:
                        # 格式应为: "(-100,50),(0,150),(100,50)"
                        points = []
                        for point_str in loading_points_str.split('),'):
                            if not point_str.endswith(')'):
                                point_str += ')'
                            # 移除括号并分割坐标
                            coords = point_str.strip('()').split(',')
                            points.append((float(coords[0]), float(coords[1])))
                        if points:
                            self.LOADING_POINTS = points
                    except Exception as e:
                        logging.error(f"解析装载点坐标失败: {str(e)}")
                
                # 解析卸载点
                unloading_point_str = config.get('LOCATIONS', 'unloading_point', fallback=None)
                if unloading_point_str:
                    try:
                        coords = unloading_point_str.strip('()').split(',')
                        self.UNLOADING_POINT = (float(coords[0]), float(coords[1]))
                    except Exception as e:
                        logging.error(f"解析卸载点坐标失败: {str(e)}")
                
                # 解析停车场
                parking_area_str = config.get('LOCATIONS', 'parking_area', fallback=None)
                if parking_area_str:
                    try:
                        coords = parking_area_str.strip('()').split(',')
                        self.PARKING_AREA = (float(coords[0]), float(coords[1]))
                    except Exception as e:
                        logging.error(f"解析停车场坐标失败: {str(e)}")
            
            # 加载调度策略参数
            if 'STRATEGY' in config:
                self.MAX_CHARGING_VEHICLES = config.getint('STRATEGY', 'max_charging_vehicles', fallback=self.MAX_CHARGING_VEHICLES)
                self.MAX_LOADING_QUEUE = config.getint('STRATEGY', 'max_loading_queue', fallback=self.MAX_LOADING_QUEUE)
                self.MAX_UNLOADING_QUEUE = config.getint('STRATEGY', 'max_unloading_queue', fallback=self.MAX_UNLOADING_QUEUE)
                self.TIME_WINDOW_SIZE = config.getint('STRATEGY', 'time_window_size', fallback=self.TIME_WINDOW_SIZE)
                self.CONFLICT_RESOLUTION_STRATEGY = config.get('STRATEGY', 'conflict_resolution_strategy', fallback=self.CONFLICT_RESOLUTION_STRATEGY)
                self.MAX_REROUTE_ATTEMPTS = config.getint('STRATEGY', 'max_reroute_attempts', fallback=self.MAX_REROUTE_ATTEMPTS)
            
            # 加载车辆参数
            if 'VEHICLE' in config:
                self.DEFAULT_VEHICLE_CONFIG['max_capacity'] = config.getfloat('VEHICLE', 'max_capacity', fallback=self.DEFAULT_VEHICLE_CONFIG['max_capacity'])
                self.DEFAULT_VEHICLE_CONFIG['min_hardness'] = config.getfloat('VEHICLE', 'min_hardness', fallback=self.DEFAULT_VEHICLE_CONFIG['min_hardness'])
                self.DEFAULT_VEHICLE_CONFIG['max_speed'] = config.getfloat('VEHICLE', 'max_speed', fallback=self.DEFAULT_VEHICLE_CONFIG['max_speed'])
                self.DEFAULT_VEHICLE_CONFIG['turning_radius'] = config.getfloat('VEHICLE', 'turning_radius', fallback=self.DEFAULT_VEHICLE_CONFIG['turning_radius'])
                self.DEFAULT_VEHICLE_CONFIG['steering_angle'] = config.getfloat('VEHICLE', 'steering_angle', fallback=self.DEFAULT_VEHICLE_CONFIG['steering_angle'])
            
            # 加载任务参数
            if 'TASK' in config:
                self.TASK_PRIORITY_LEVELS = config.getint('TASK', 'priority_levels', fallback=self.TASK_PRIORITY_LEVELS)
                self.TASK_TIMEOUT = config.getint('TASK', 'timeout', fallback=self.TASK_TIMEOUT)
            
            # 加载系统参数
            if 'SYSTEM' in config:
                log_level_str = config.get('SYSTEM', 'log_level', fallback='INFO')
                self.LOG_LEVEL = getattr(logging, log_level_str.upper(), logging.INFO)
                self.SIMULATION_MODE = config.getboolean('SYSTEM', 'simulation_mode', fallback=self.SIMULATION_MODE)
            
            logging.info(f"成功从 {config_path} 加载配置")
            return True
            
        except Exception as e:
            logging.error(f"加载配置文件失败: {str(e)}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            'dispatch': {
                'dispatch_interval': self.DISPATCH_INTERVAL,
                'monitor_interval': self.MONITOR_INTERVAL,
                'conflict_check_interval': self.CONFLICT_CHECK_INTERVAL
            },
            'locations': {
                'loading_points': self.LOADING_POINTS,
                'unloading_point': self.UNLOADING_POINT,
                'parking_area': self.PARKING_AREA,
                'charging_stations': self.CHARGING_STATIONS
            },
            'strategy': {
                'max_charging_vehicles': self.MAX_CHARGING_VEHICLES,
                'max_loading_queue': self.MAX_LOADING_QUEUE,
                'max_unloading_queue': self.MAX_UNLOADING_QUEUE,
                'time_window_size': self.TIME_WINDOW_SIZE,
                'conflict_resolution_strategy': self.CONFLICT_RESOLUTION_STRATEGY,
                'max_reroute_attempts': self.MAX_REROUTE_ATTEMPTS
            },
            'vehicle': self.DEFAULT_VEHICLE_CONFIG,
            'task': {
                'priority_levels': self.TASK_PRIORITY_LEVELS,
                'timeout': self.TASK_TIMEOUT
            },
            'system': {
                'log_level': self.LOG_LEVEL,
                'simulation_mode': self.SIMULATION_MODE
            }
        }


# 全局配置实例
dispatch_config = DispatchConfig()


if __name__ == "__main__":
    # 测试配置加载
    import json
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建配置实例
    config = DispatchConfig()
    
    # 尝试从文件加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.ini')
    config.load_from_file(config_path)
    
    # 打印配置
    print(json.dumps(config.to_dict(), indent=2, default=str))