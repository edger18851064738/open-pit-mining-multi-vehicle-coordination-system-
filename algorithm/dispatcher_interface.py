#!/usr/bin/env python3
"""
露天矿多车协同调度系统 - 调度器接口
==============================================

本模块定义了调度器的接口规范，使不同的调度算法能够以统一的方式接入系统。
任何新的调度算法实现都应该遵循此接口规范。
"""

import abc
from typing import List, Dict, Tuple, Optional, Any

class DispatcherInterface(abc.ABC):
    """
    调度器接口抽象基类
    
    定义了调度系统需要实现的基本方法，用于车辆和任务的管理与调度。
    """
    
    @abc.abstractmethod
    def add_vehicle(self, vehicle) -> None:
        """
        添加车辆到调度系统
        
        Args:
            vehicle: 车辆对象
        """
        pass
        
    @abc.abstractmethod
    def add_task(self, task) -> None:
        """
        添加任务到调度系统
        
        Args:
            task: 任务对象
        """
        pass
    
    @abc.abstractmethod
    def scheduling_cycle(self) -> bool:
        """
        执行一次调度周期
        
        执行任务分配、冲突检测和解决等工作。
        
        Returns:
            bool: 调度是否成功
        """
        pass
    
    @abc.abstractmethod
    def resolve_path_conflicts(self) -> None:
        """
        解决路径冲突
        
        检测并解决车辆之间的路径冲突，确保行驶安全。
        """
        pass
    
    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        获取调度系统状态
        
        Returns:
            Dict: 包含系统状态信息的字典
        """
        pass
    
    @abc.abstractmethod
    def start_scheduling(self, interval: float = 1.0) -> None:
        """
        启动调度循环
        
        Args:
            interval: 调度间隔时间（秒）
        """
        pass
    
    @abc.abstractmethod
    def stop_scheduling(self) -> None:
        """
        停止调度循环
        """
        pass
    
    # 可选方法
    def optimize_task_assignment(self, vehicles, tasks) -> Dict[int, str]:
        """
        优化任务分配
        
        尝试找到车辆和任务之间的最优匹配。
        
        Args:
            vehicles: 可用车辆列表
            tasks: 待分配任务列表
            
        Returns:
            Dict: 车辆ID到任务ID的映射
        """
        # 默认实现为空，子类可以根据需要重写
        return {}
    
    def predict_conflicts(self, paths) -> List[Dict]:
        """
        预测路径冲突
        
        分析路径并预测可能发生的冲突点。
        
        Args:
            paths: 车辆路径字典
            
        Returns:
            List[Dict]: 预测的冲突点列表
        """
        # 默认实现为空，子类可以根据需要重写
        return []
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        计算性能指标
        
        计算调度系统的各种性能指标，如任务完成率、车辆利用率等。
        
        Returns:
            Dict: 性能指标字典
        """
        # 默认实现为空，子类可以根据需要重写
        return {
            'tasks_completed': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'vehicle_utilization': 0.0
        }

# 用于注册自定义调度器的字典
REGISTERED_DISPATCHERS = {}

def register_dispatcher(name):
    """
    注册调度器装饰器
    
    用于将自定义调度器类注册到系统中。
    
    Args:
        name: 调度器名称
        
    Returns:
        装饰器函数
    """
    def decorator(cls):
        if not issubclass(cls, DispatcherInterface):
            raise TypeError(f"注册的调度器必须继承自DispatcherInterface，{cls.__name__}未继承")
        
        REGISTERED_DISPATCHERS[name] = cls
        return cls
        
    return decorator

def get_registered_dispatchers():
    """
    获取所有注册的调度器
    
    Returns:
        Dict: 名称到调度器类的映射
    """
    return REGISTERED_DISPATCHERS

def get_dispatcher(name):
    """
    根据名称获取调度器类
    
    Args:
        name: 调度器名称
        
    Returns:
        调度器类，如果找不到则返回None
    """
    return REGISTERED_DISPATCHERS.get(name)