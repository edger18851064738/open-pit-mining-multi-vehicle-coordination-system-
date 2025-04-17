#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
露天矿无人多车协同调度系统框架

集成了路径规划、调度服务、地图服务和车辆模型等组件，
实现了多车协同调度的核心功能，包括任务分配、冲突检测与解决、路径规划优化和实时监控。
"""

import sys
import os
import time
import logging
import threading
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, deque

# 项目路径配置
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入项目模块
from algorithm.map_service import MapService
from algorithm.path_planner import HybridPathPlanner
from algorithm.dispatch_service import DispatchSystem, TransportScheduler, ConflictBasedSearch
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from utils.geo_tools import GeoUtils
from config.settings import AppConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'dispatch.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('dispatch_system')

class MineDispatchSystem:
    """露天矿无人多车协同调度系统主控制类"""
    
    def __init__(self, config_path: str = None):
        """初始化调度系统
        
        Args:
            config_path: 配置文件路径，默认为None使用默认配置
        """
        # 加载配置
        self.config = AppConfig.load(config_path or os.path.join(PROJECT_ROOT, 'config.ini'))
        
        # 初始化核心服务
        self.geo_utils = GeoUtils()
        self.map_service = MapService()
        self.path_planner = HybridPathPlanner(self.map_service)
        self.dispatch_system = DispatchSystem(self.path_planner, self.map_service)
        
        # 系统状态
        self.running = False
        self.dispatch_thread = None
        self.monitor_thread = None
        self.visualization_thread = None
        
        # 统计数据
        self.stats = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'total_distance': 0.0,
            'system_uptime': 0.0
        }
        
        # 系统锁
        self.system_lock = threading.RLock()
        
        logger.info("露天矿无人多车协同调度系统初始化完成")
    
    def register_vehicle(self, vehicle: MiningVehicle) -> None:
        """注册车辆到调度系统
        
        Args:
            vehicle: 矿用运输车辆实例
        """
        with self.system_lock:
            self.dispatch_system.vehicles[vehicle.vehicle_id] = vehicle
            logger.info(f"车辆 {vehicle.vehicle_id} 已注册到调度系统")
    
    def add_task(self, task: TransportTask) -> None:
        """添加运输任务到调度队列
        
        Args:
            task: 运输任务实例
        """
        with self.system_lock:
            self.dispatch_system.add_task(task)
            logger.info(f"任务 {task.task_id} 已添加到调度队列")
    
    def start(self) -> None:
        """启动调度系统"""
        if self.running:
            logger.warning("调度系统已在运行中")
            return
        
        with self.system_lock:
            self.running = True
            self.start_time = time.time()
            
            # 启动调度线程
            self.dispatch_thread = threading.Thread(
                target=self._dispatch_loop,
                name="DispatchThread",
                daemon=True
            )
            self.dispatch_thread.start()
            
            # 启动监控线程
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                name="MonitorThread",
                daemon=True
            )
            self.monitor_thread.start()
            
            logger.info("调度系统已启动")
    
    def stop(self) -> None:
        """停止调度系统"""
        if not self.running:
            logger.warning("调度系统未运行")
            return
        
        with self.system_lock:
            self.running = False
            
            # 等待线程结束
            if self.dispatch_thread and self.dispatch_thread.is_alive():
                self.dispatch_thread.join(timeout=5.0)
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            # 更新系统运行时间
            self.stats['system_uptime'] = time.time() - self.start_time
            
            logger.info("调度系统已停止")
    
    def _dispatch_loop(self) -> None:
        """调度循环（内部方法）"""
        dispatch_interval = 30.0  # 调度周期（秒）
        
        while self.running:
            try:
                # 执行调度周期
                self.dispatch_system.scheduling_cycle()
                
                # 更新统计数据
                with self.system_lock:
                    self.stats['tasks_completed'] = len(self.dispatch_system.completed_tasks)
                    self.stats['conflicts_detected'] += len(self.dispatch_system.cbs.find_conflicts(
                        {vid: v.current_path for vid, v in self.dispatch_system.vehicles.items() if v.current_path}
                    ))
                
                # 等待下一个调度周期
                time.sleep(dispatch_interval)
                
            except Exception as e:
                logger.error(f"调度循环异常: {str(e)}")
                time.sleep(5.0)  # 出错后短暂等待
    
    def _monitor_loop(self) -> None:
        """监控循环（内部方法）"""
        monitor_interval = 10.0  # 监控周期（秒）
        
        while self.running:
            try:
                # 更新车辆状态
                for vehicle_id, vehicle in self.dispatch_system.vehicles.items():
                    if hasattr(vehicle, 'update_position') and callable(vehicle.update_position):
                        vehicle.update_position()
                
                # 检测任务完成情况
                self._check_task_completion()
                
                # 更新统计数据
                self._update_statistics()
                
                # 等待下一个监控周期
                time.sleep(monitor_interval)
                
            except Exception as e:
                logger.error(f"监控循环异常: {str(e)}")
                time.sleep(5.0)  # 出错后短暂等待
    
    def _check_task_completion(self) -> None:
        """检查任务完成情况（内部方法）"""
        with self.system_lock:
            # 检查活跃任务是否完成
            completed_task_ids = []
            for task_id, task in self.dispatch_system.active_tasks.items():
                if hasattr(task, 'is_completed') and task.is_completed:
                    completed_task_ids.append(task_id)
                    self.dispatch_system.completed_tasks[task_id] = task
                    self.stats['tasks_completed'] += 1
                    logger.info(f"任务 {task_id} 已完成")
            
            # 从活跃任务中移除已完成任务
            for task_id in completed_task_ids:
                if task_id in self.dispatch_system.active_tasks:
                    del self.dispatch_system.active_tasks[task_id]
    
    def _update_statistics(self) -> None:
        """更新系统统计数据（内部方法）"""
        with self.system_lock:
            # 更新系统运行时间
            self.stats['system_uptime'] = time.time() - self.start_time
            
            # 更新总行驶距离
            total_distance = 0.0
            for vehicle in self.dispatch_system.vehicles.values():
                if hasattr(vehicle, 'mileage'):
                    total_distance += vehicle.mileage
            self.stats['total_distance'] = total_distance
    
    def get_system_status(self) -> Dict:
        """获取系统状态信息
        
        Returns:
            Dict: 包含系统状态的字典
        """
        with self.system_lock:
            # 基础状态信息
            status = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'running': self.running,
                'uptime': timedelta(seconds=int(self.stats['system_uptime'])),
                'vehicles': {
                    'total': len(self.dispatch_system.vehicles),
                    'active': len([v for v in self.dispatch_system.vehicles.values() 
                                  if v.state != VehicleState.IDLE]),
                    'idle': len([v for v in self.dispatch_system.vehicles.values() 
                                if v.state == VehicleState.IDLE])
                },
                'tasks': {
                    'queued': len(self.dispatch_system.task_queue),
                    'active': len(self.dispatch_system.active_tasks),
                    'completed': self.stats['tasks_completed']
                },
                'statistics': self.stats
            }
            
            return status
    
    def print_system_status(self) -> None:
        """打印系统状态信息"""
        status = self.get_system_status()
        
        print("\n===== 露天矿无人多车协同调度系统状态 =====")
        print(f"时间: {status['timestamp']}")
        print(f"运行状态: {'运行中' if status['running'] else '已停止'}")
        print(f"运行时长: {status['uptime']}")
        print("\n车辆状态:")
        print(f"  总数: {status['vehicles']['total']}")
        print(f"  活跃: {status['vehicles']['active']}")
        print(f"  空闲: {status['vehicles']['idle']}")
        print("\n任务状态:")
        print(f"  队列中: {status['tasks']['queued']}")
        print(f"  执行中: {status['tasks']['active']}")
        print(f"  已完成: {status['tasks']['completed']}")
        print("\n系统统计:")
        print(f"  总行驶距离: {status['statistics']['total_distance']:.2f} 米")
        print(f"  冲突检测次数: {status['statistics']['conflicts_detected']}")
        print(f"  冲突解决次数: {status['statistics']['conflicts_resolved']}")
        print("=========================================\n")
    
    def dispatch_vehicle_to(self, vehicle_id: str, destination: Tuple[float, float]) -> None:
        """直接调度指定车辆到目标位置
        
        Args:
            vehicle_id: 车辆ID
            destination: 目标位置坐标
        """
        with self.system_lock:
            self.dispatch_system.dispatch_vehicle_to(vehicle_id, destination)
            logger.info(f"已手动调度车辆 {vehicle_id} 前往 {destination}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='露天矿无人多车协同调度系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--vehicles', type=int, default=5, help='初始车辆数量')
    parser.add_argument('--tasks', type=int, default=10, help='初始任务数量')
    parser.add_argument('--simulation', action='store_true', help='启用模拟模式')
    args = parser.parse_args()
    
    # 初始化调度系统
    dispatch_system = MineDispatchSystem(args.config)
    
    # 添加初始车辆
    for i in range(args.vehicles):
        vehicle = MiningVehicle(
            vehicle_id=f"vehicle_{i+1}",
            map_service=dispatch_system.map_service,
            config={
                'max_capacity': 50,
                'current_location': (0, 0),
                'base_location': (0, 0),
                'min_hardness': 2.5,
                'max_speed': 5.0,
                'turning_radius': 10.0
            }
        )
        dispatch_system.register_vehicle(vehicle)
    
    # 添加初始任务
    loading_points = [(-100, 50), (0, 150), (100, 50)]
    unloading_point = (0, -100)
    
    for i in range(args.tasks):
        loading_point = loading_points[i % len(loading_points)]
        task = TransportTask(
            task_id=f"task_{i+1}",
            start_point=loading_point,
            end_point=unloading_point,
            task_type="loading",
            priority=i % 3 + 1
        )
        dispatch_system.add_task(task)
    
    # 启动系统
    dispatch_system.start()
    
    try:
        # 主循环
        while True:
            dispatch_system.print_system_status()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n接收到退出信号，正在停止系统...")
    finally:
        dispatch_system.stop()
        print("系统已安全停止")


if __name__ == "__main__":
    main()