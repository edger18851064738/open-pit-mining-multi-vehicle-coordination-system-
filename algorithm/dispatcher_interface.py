#!/usr/bin/env python3
"""
露天矿多车协同调度系统 - 调度器接口
==============================================

本模块定义了调度器的接口规范，使不同的调度算法能够以统一的方式接入系统。
任何新的调度算法实现都应该遵循此接口规范。

功能：
1. 提供统一的调度算法接口
2. 支持动态注册与加载不同的调度算法
3. 定义核心调度功能与可选扩展功能
4. 提供插件式架构用于自定义算法扩展
"""

import abc
import time
import logging
import importlib
import inspect
import os
import sys
from typing import List, Dict, Tuple, Optional, Any, Set, Union, Callable, Type
from datetime import datetime
from threading import RLock, Event, Thread

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 可选导入 - 用于类型注解
try:
    from models.vehicle import MiningVehicle, VehicleState, TransportStage
    from models.task import TransportTask
    from algorithm.map_service import MapService
    from algorithm.optimized_path_planner import HybridPathPlanner
except ImportError:
    logging.debug("调度器接口中的类型导入可选，仅用于类型注解")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dispatcher")

class DispatchQuality:
    """调度质量指标"""
    
    def __init__(self):
        self.task_completion_rate = 0.0  # 任务完成率
        self.vehicle_utilization = 0.0   # 车辆利用率
        self.avg_waiting_time = 0.0      # 平均等待时间
        self.conflicts_detected = 0       # 检测到的冲突数
        self.conflicts_resolved = 0       # 已解决的冲突数
        self.avg_path_length = 0.0       # 平均路径长度
        self.avg_task_duration = 0.0     # 平均任务执行时间
        self.timestamp = datetime.now()  # 时间戳

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'task_completion_rate': self.task_completion_rate,
            'vehicle_utilization': self.vehicle_utilization,
            'avg_waiting_time': self.avg_waiting_time,
            'conflicts_detected': self.conflicts_detected,
            'conflicts_resolved': self.conflicts_resolved,
            'avg_path_length': self.avg_path_length,
            'avg_task_duration': self.avg_task_duration,
            'timestamp': self.timestamp.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DispatchQuality':
        """从字典构建指标对象"""
        quality = cls()
        quality.task_completion_rate = data.get('task_completion_rate', 0.0)
        quality.vehicle_utilization = data.get('vehicle_utilization', 0.0)
        quality.avg_waiting_time = data.get('avg_waiting_time', 0.0)
        quality.conflicts_detected = data.get('conflicts_detected', 0)
        quality.conflicts_resolved = data.get('conflicts_resolved', 0)
        quality.avg_path_length = data.get('avg_path_length', 0.0)
        quality.avg_task_duration = data.get('avg_task_duration', 0.0)
        
        # 处理时间戳转换
        if 'timestamp' in data:
            try:
                quality.timestamp = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                quality.timestamp = datetime.now()
                
        return quality

class DispatcherConfig:
    """调度器配置类"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.config_dict = config_dict or {}
        
        # 调度周期设置
        self.cycle_interval = self.config_dict.get('cycle_interval', 1.0)  # 调度周期间隔(秒)
        self.conflict_check_interval = self.config_dict.get('conflict_check_interval', 0.5)  # 冲突检测间隔(秒)
        
        # 任务分配设置
        self.max_tasks_per_vehicle = self.config_dict.get('max_tasks_per_vehicle', 1)  # 每车最大任务数
        self.enable_task_priority = self.config_dict.get('enable_task_priority', True)  # 启用任务优先级
        self.enable_load_balancing = self.config_dict.get('enable_load_balancing', True)  # 启用负载均衡
        
        # 冲突解决设置
        self.max_replanning_attempts = self.config_dict.get('max_replanning_attempts', 3)  # 最大重规划次数
        self.path_safety_margin = self.config_dict.get('path_safety_margin', 5.0)  # 路径安全边距(米)
        self.conflict_detection_enabled = self.config_dict.get('conflict_detection_enabled', True)  # 启用冲突检测
        
        # 性能设置
        self.metrics_collection_enabled = self.config_dict.get('metrics_collection_enabled', True)  # 启用指标收集
        self.metrics_update_interval = self.config_dict.get('metrics_update_interval', 10)  # 指标更新间隔(秒)
        
        # 调度算法特定参数
        self.algorithm_params = self.config_dict.get('algorithm_params', {})
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'cycle_interval': self.cycle_interval,
            'conflict_check_interval': self.conflict_check_interval,
            'max_tasks_per_vehicle': self.max_tasks_per_vehicle,
            'enable_task_priority': self.enable_task_priority,
            'enable_load_balancing': self.enable_load_balancing,
            'max_replanning_attempts': self.max_replanning_attempts,
            'path_safety_margin': self.path_safety_margin,
            'conflict_detection_enabled': self.conflict_detection_enabled,
            'metrics_collection_enabled': self.metrics_collection_enabled,
            'metrics_update_interval': self.metrics_update_interval,
            'algorithm_params': self.algorithm_params
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DispatcherConfig':
        """从字典构建配置对象"""
        return cls(config_dict)

class DispatcherInterface(abc.ABC):
    """
    调度器接口抽象基类
    
    定义了调度系统需要实现的基本方法，用于车辆和任务的管理与调度。
    扩展了原有接口，添加了更多功能和配置选项。
    """
    
    def __init__(self, path_planner=None, map_service=None, config: Optional[Dict[str, Any]] = None):
        """
        初始化调度器
        
        Args:
            path_planner: 路径规划器实例
            map_service: 地图服务实例
            config: 调度器配置字典
        """
        self.path_planner = path_planner
        self.map_service = map_service
        self.config = DispatcherConfig(config)
        
        # 初始化锁和状态
        self.lock = RLock()
        self.running = False
        self.paused = False
        self.stop_event = Event()
        
        # 调度线程
        self.scheduling_thread = None
        
        # 初始化状态和指标
        self._init_metrics()
        
        # 调度器名称
        self.name = self.__class__.__name__
        
        # 回调和监听器
        self.event_listeners = {}
        
    def _init_metrics(self):
        """初始化性能指标"""
        self.metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_assigned': 0,
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'planning_count': 0,
            'scheduling_cycles': 0,
            'start_time': time.time(),
            'last_cycle_time': 0.0,
            'avg_cycle_time': 0.0,
            'cycle_times': []
        }
        
        # 质量指标
        self.quality = DispatchQuality()
        self.quality_history = []
    
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
    
    def start_scheduling(self, interval: float = None) -> None:
        """
        启动调度循环
        
        Args:
            interval: 调度间隔时间（秒），如果为None则使用配置值
        """
        if self.running:
            logging.warning(f"调度器 {self.name} 已在运行中")
            return
            
        # 使用配置值或参数值
        actual_interval = interval if interval is not None else self.config.cycle_interval
        
        # 重置停止事件
        self.stop_event.clear()
        self.running = True
        self.paused = False
        
        # 创建并启动调度线程
        self.scheduling_thread = Thread(
            target=self._scheduling_loop,
            args=(actual_interval,),
            daemon=True,
            name=f"{self.name}_scheduling_thread"
        )
        self.scheduling_thread.start()
        
        # 记录日志
        logging.info(f"调度器 {self.name} 已启动，周期间隔: {actual_interval}秒")
        
        # 通知监听器
        self._notify_listeners('scheduler_started', {'interval': actual_interval})
    
    def stop_scheduling(self) -> None:
        """
        停止调度循环
        """
        if not self.running:
            return
            
        # 设置停止事件并等待线程结束
        self.stop_event.set()
        self.running = False
        
        if self.scheduling_thread and self.scheduling_thread.is_alive():
            self.scheduling_thread.join(timeout=2.0)
            
        self.scheduling_thread = None
        
        # 记录日志
        logging.info(f"调度器 {self.name} 已停止")
        
        # 通知监听器
        self._notify_listeners('scheduler_stopped', {})
    
    def _scheduling_loop(self, interval: float) -> None:
        """
        调度循环主方法
        
        Args:
            interval: 调度间隔（秒）
        """
        next_cycle_time = time.time()
        next_conflict_check = time.time()
        next_metrics_update = time.time() + self.config.metrics_update_interval
        
        try:
            while not self.stop_event.is_set() and self.running:
                current_time = time.time()
                
                # 如果暂停，等待一会儿再检查
                if self.paused:
                    time.sleep(min(0.1, interval / 10))
                    continue
                
                # 执行调度周期
                if current_time >= next_cycle_time:
                    cycle_start = time.time()
                    try:
                        self.scheduling_cycle()
                        
                        # 更新调度周期指标
                        cycle_time = time.time() - cycle_start
                        self._update_cycle_metrics(cycle_time)
                    except Exception as e:
                        logging.error(f"调度周期执行错误: {str(e)}")
                        
                    # 计算下一个周期时间
                    next_cycle_time = max(current_time + interval, cycle_start + interval / 10)
                
                # 检查路径冲突 (如果启用)
                if self.config.conflict_detection_enabled and current_time >= next_conflict_check:
                    try:
                        self.resolve_path_conflicts()
                    except Exception as e:
                        logging.error(f"冲突解决错误: {str(e)}")
                        
                    # 计算下一次冲突检查时间
                    next_conflict_check = current_time + self.config.conflict_check_interval
                
                # 更新性能指标
                if self.config.metrics_collection_enabled and current_time >= next_metrics_update:
                    try:
                        self._update_quality_metrics()
                    except Exception as e:
                        logging.error(f"指标更新错误: {str(e)}")
                        
                    # 计算下一次指标更新时间
                    next_metrics_update = current_time + self.config.metrics_update_interval
                
                # 短暂休眠以减少CPU使用
                sleep_time = min(0.01, interval / 100)
                time.sleep(sleep_time)
                
        except Exception as e:
            logging.error(f"调度循环异常: {str(e)}")
        finally:
            self.running = False
            logging.info(f"调度循环已结束")
    
    def pause_scheduling(self) -> None:
        """暂停调度"""
        if self.running and not self.paused:
            self.paused = True
            logging.info(f"调度器 {self.name} 已暂停")
            self._notify_listeners('scheduler_paused', {})
    
    def resume_scheduling(self) -> None:
        """恢复调度"""
        if self.running and self.paused:
            self.paused = False
            logging.info(f"调度器 {self.name} 已恢复")
            self._notify_listeners('scheduler_resumed', {})
    
    def _update_cycle_metrics(self, cycle_time: float) -> None:
        """
        更新调度周期指标
        
        Args:
            cycle_time: 本次周期执行时间（秒）
        """
        with self.lock:
            self.metrics['scheduling_cycles'] += 1
            self.metrics['last_cycle_time'] = cycle_time
            
            # 添加到历史记录并保持最多100个记录
            self.metrics['cycle_times'].append(cycle_time)
            if len(self.metrics['cycle_times']) > 100:
                self.metrics['cycle_times'].pop(0)
                
            # 计算平均值
            self.metrics['avg_cycle_time'] = sum(self.metrics['cycle_times']) / len(self.metrics['cycle_times'])
    
    def _update_quality_metrics(self) -> None:
        """更新质量指标"""
        # 这个方法依赖于具体实现，因此作为可选的
        try:
            self.quality = self.calculate_metrics()
            self.quality_history.append(self.quality)
            
            # 限制历史数据长度
            if len(self.quality_history) > 100:
                self.quality_history.pop(0)
                
            # 通知监听器
            self._notify_listeners('metrics_updated', {'quality': self.quality.to_dict()})
            
        except Exception as e:
            logging.warning(f"更新质量指标失败: {str(e)}")
    
    def add_event_listener(self, event_type: str, callback: Callable) -> None:
        """
        添加事件监听器
        
        Args:
            event_type: 事件类型名称
            callback: 回调函数，接收事件数据字典作为参数
        """
        if event_type not in self.event_listeners:
            self.event_listeners[event_type] = []
            
        if callback not in self.event_listeners[event_type]:
            self.event_listeners[event_type].append(callback)
    
    def remove_event_listener(self, event_type: str, callback: Callable) -> None:
        """
        移除事件监听器
        
        Args:
            event_type: 事件类型名称
            callback: 要移除的回调函数
        """
        if event_type in self.event_listeners and callback in self.event_listeners[event_type]:
            self.event_listeners[event_type].remove(callback)
    
    def _notify_listeners(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        通知事件监听器
        
        Args:
            event_type: 事件类型名称
            data: 事件数据
        """
        if event_type not in self.event_listeners:
            return
            
        # 添加事件元数据
        event_data = data.copy()
        event_data['event_type'] = event_type
        event_data['timestamp'] = datetime.now()
        event_data['dispatcher'] = self.name
        
        # 调用回调
        for callback in self.event_listeners[event_type]:
            try:
                callback(event_data)
            except Exception as e:
                logging.error(f"事件监听器回调出错: {str(e)}")
                
    # 可选方法 - 子类可以覆盖这些方法以增强功能
    
    def optimize_task_assignment(self, vehicles: List, tasks: List) -> Dict[int, str]:
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
    
    def predict_conflicts(self, paths: Dict[str, List[Tuple]]) -> List[Dict]:
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
        
    def calculate_metrics(self) -> DispatchQuality:
        """
        计算性能指标
        
        计算调度系统的各种性能指标，如任务完成率、车辆利用率等。
        
        Returns:
            DispatchQuality: 性能指标对象
        """
        # 基本实现 - 子类应该提供更完整的实现
        quality = DispatchQuality()
        
        with self.lock:
            # 从基本指标计算值
            total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
            if total_tasks > 0:
                quality.task_completion_rate = self.metrics['tasks_completed'] / total_tasks
                
            quality.conflicts_detected = self.metrics['conflicts_detected']
            quality.conflicts_resolved = self.metrics['conflicts_resolved']
            
            # 更新时间戳
            quality.timestamp = datetime.now()
            
        return quality
    
    def get_vehicle_recommendations(self, task) -> List:
        """
        获取任务推荐车辆
        
        根据任务要求和当前状态，推荐最适合执行任务的车辆列表。
        
        Args:
            task: 任务对象
            
        Returns:
            List: 推荐车辆列表，按匹配度排序
        """
        # 默认实现返回空列表，子类可以根据需要重写
        return []
    
    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """
        更新调度器配置
        
        允许在运行时动态更新配置参数。
        
        Args:
            new_config: 新配置字典
        """
        with self.lock:
            # 更新配置字典
            if hasattr(self.config, 'config_dict'):
                self.config.config_dict.update(new_config)
                
            # 更新具体配置属性
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            
        # 通知配置已更新
        self._notify_listeners('config_updated', {'new_config': new_config})
    
    def export_state(self) -> Dict[str, Any]:
        """
        导出调度器状态
        
        导出当前调度系统的完整状态，用于持久化或迁移。
        
        Returns:
            Dict: 调度器状态
        """
        with self.lock:
            return {
                'name': self.name,
                'running': self.running,
                'paused': self.paused,
                'metrics': self.metrics.copy(),
                'quality': self.quality.to_dict(),
                'config': self.config.to_dict(),
                'timestamp': datetime.now().isoformat()
            }
    
    def import_state(self, state: Dict[str, Any]) -> bool:
        """
        导入调度器状态
        
        从之前导出的状态恢复调度系统。
        
        Args:
            state: 调度器状态字典
            
        Returns:
            bool: 是否成功导入
        """
        try:
            with self.lock:
                # 更新基本属性
                self.name = state.get('name', self.name)
                
                # 更新指标
                if 'metrics' in state:
                    self.metrics.update(state['metrics'])
                
                # 更新质量指标
                if 'quality' in state:
                    self.quality = DispatchQuality.from_dict(state['quality'])
                
                # 更新配置
                if 'config' in state:
                    self.update_configuration(state['config'])
                    
            return True
        except Exception as e:
            logging.error(f"导入状态失败: {str(e)}")
            return False

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
        logger.info(f"已注册调度算法: {name} ({cls.__name__})")
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

def find_dispatchers_in_module(module_name: str) -> List[Type]:
    """
    在指定模块中查找并注册调度器类
    
    Args:
        module_name: 模块名称
        
    Returns:
        List[Type]: 发现的调度器类列表
    """
    found_dispatchers = []
    
    try:
        # 导入模块
        module = importlib.import_module(module_name)
        
        # 查找所有继承自DispatcherInterface的类
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, DispatcherInterface) and 
                obj != DispatcherInterface):
                
                # 使用类名作为注册名称
                dispatcher_name = name.lower()
                if dispatcher_name not in REGISTERED_DISPATCHERS:
                    REGISTERED_DISPATCHERS[dispatcher_name] = obj
                    found_dispatchers.append(obj)
                    logger.info(f"自动注册调度算法: {dispatcher_name} ({name})")
    
    except (ImportError, AttributeError) as e:
        logger.warning(f"查找调度器模块 {module_name} 时出错: {str(e)}")
    
    return found_dispatchers

def load_dispatchers_from_directory(directory_path: str = None) -> Dict[str, Type]:
    """
    从目录加载所有调度器
    
    Args:
        directory_path: 目录路径，如果为None则使用algorithm目录
        
    Returns:
        Dict[str, Type]: 名称到调度器类的映射
    """
    if directory_path is None:
        directory_path = os.path.join(PROJECT_ROOT, "algorithm")
        
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        logger.warning(f"调度器目录不存在: {directory_path}")
        return {}
        
    # 查找所有Python文件
    py_files = [f for f in os.listdir(directory_path) 
                if f.endswith('.py') and not f.startswith('__')]
    
    # 导入每个文件作为模块
    for py_file in py_files:
        module_name = os.path.splitext(py_file)[0]
        full_module = f"algorithm.{module_name}"
        
        try:
            find_dispatchers_in_module(full_module)
        except Exception as e:
            logger.warning(f"加载调度器模块 {full_module} 时出错: {str(e)}")
    
    return REGISTERED_DISPATCHERS

def create_dispatcher(name: str, path_planner=None, map_service=None, config=None):
    """
    创建调度器实例
    
    Args:
        name: 调度器名称
        path_planner: 路径规划器实例
        map_service: 地图服务实例
        config: 配置字典
        
    Returns:
        调度器实例或None
    """
    dispatcher_class = get_dispatcher(name)
    if not dispatcher_class:
        logger.error(f"未找到调度算法: {name}")
        return None
        
    try:
        return dispatcher_class(path_planner, map_service, config)
    except Exception as e:
        logger.error(f"创建调度器实例失败: {str(e)}")
        return None

if __name__ != "__main__":  # 只在作为模块导入时
    # 尝试导入内置调度器
    try:
        from algorithm.cbs import ConflictBasedSearch
        from algorithm.dispatch_service import DispatchSystem
        
        # 注册内置调度器
        @register_dispatcher("cbs")
        class CBSDispatcher(DispatcherInterface):
            """CBS冲突解决调度器"""
            
            def __init__(self, path_planner=None, map_service=None, config=None):
                super().__init__(path_planner, map_service, config)
                self.cbs = ConflictBasedSearch(path_planner)
                self.vehicles = {}
                self.task_queue = []
                self.active_tasks = {}
            
            def add_vehicle(self, vehicle):
                self.vehicles[vehicle.vehicle_id] = vehicle
                
            def add_task(self, task):
                self.task_queue.append(task)
                
            def scheduling_cycle(self):
                # 为空闲车辆分配任务
                for vehicle_id, vehicle in self.vehicles.items():
                    if vehicle.state == VehicleState.IDLE and not vehicle.current_task:
                        if self.task_queue:
                            task = self.task_queue.pop(0)
                            vehicle.assign_task(task)
                            self.active_tasks[task.task_id] = task
                            self.metrics['tasks_assigned'] += 1
                            
                # 检查已完成的任务
                completed_tasks = []
                for task_id, task in self.active_tasks.items():
                    if task.is_completed:
                        completed_tasks.append(task_id)
                        self.metrics['tasks_completed'] += 1
                        
                # 从活动任务中移除已完成任务
                for task_id in completed_tasks:
                    if task_id in self.active_tasks:
                        del self.active_tasks[task_id]
                        
                return True
                
            def resolve_path_conflicts(self):
                # 收集所有车辆的路径
                vehicle_paths = {}
                for vehicle_id, vehicle in self.vehicles.items():
                    if vehicle.current_path and len(vehicle.current_path) > 1:
                        vehicle_paths[str(vehicle_id)] = vehicle.current_path
                
                if len(vehicle_paths) < 2:
                    return  # 没有足够的路径进行冲突检测
                    
                # 检测冲突
                conflicts = self.cbs.find_conflicts(vehicle_paths)
                self.metrics['conflicts_detected'] += len(conflicts)
                
                if conflicts:
                    # 解决冲突
                    resolved_paths = self.cbs.resolve_conflicts(vehicle_paths)
                    
                    # 应用解决方案
                    for vid_str, new_path in resolved_paths.items():
                        if new_path != vehicle_paths.get(vid_str, []):
                            vid = int(vid_str)
                            if vid in self.vehicles:
                                vehicle = self.vehicles[vid]
                                vehicle.assign_path(new_path)
                                self.metrics['conflicts_resolved'] += 1
                
            def get_status(self):
                return {
                    'vehicles': {
                        'total': len(self.vehicles),
                        'idle': sum(1 for v in self.vehicles.values() if v.state == VehicleState.IDLE),
                        'active': sum(1 for v in self.vehicles.values() if v.state != VehicleState.IDLE)
                    },
                    'tasks': {
                        'queued': len(self.task_queue),
                        'active': len(self.active_tasks)
                    },
                    'metrics': self.metrics
                }
                
            def calculate_metrics(self):
                quality = DispatchQuality()
                
                # 计算任务完成率
                total_tasks = self.metrics['tasks_completed'] + self.metrics['tasks_failed']
                if total_tasks > 0:
                    quality.task_completion_rate = self.metrics['tasks_completed'] / total_tasks
                
                # 计算车辆利用率
                total_vehicles = len(self.vehicles)
                if total_vehicles > 0:
                    active_vehicles = sum(1 for v in self.vehicles.values() if v.state != VehicleState.IDLE)
                    quality.vehicle_utilization = active_vehicles / total_vehicles
                
                # 计算冲突解决率
                if self.metrics['conflicts_detected'] > 0:
                    quality.conflicts_detected = self.metrics['conflicts_detected']
                    quality.conflicts_resolved = self.metrics['conflicts_resolved']
                
                return quality
                
        @register_dispatcher("basic")
        class BasicDispatcher(DispatcherInterface):
            """基本调度器（简单先来先服务策略）"""
            
            def __init__(self, path_planner=None, map_service=None, config=None):
                super().__init__(path_planner, map_service, config)
                self.vehicles = {}
                self.task_queue = []
                self.active_tasks = {}
                
            def add_vehicle(self, vehicle):
                self.vehicles[vehicle.vehicle_id] = vehicle
                
            def add_task(self, task):
                self.task_queue.append(task)
                
            def scheduling_cycle(self):
                # 简单的先来先服务任务分配
                idle_vehicles = [v for v in self.vehicles.values() if v.state == VehicleState.IDLE]
                
                for vehicle in idle_vehicles:
                    if self.task_queue:
                        task = self.task_queue.pop(0)
                        vehicle.assign_task(task)
                        self.active_tasks[task.task_id] = task
                        self.metrics['tasks_assigned'] += 1
                        
                # 检查任务完成情况
                for task_id in list(self.active_tasks.keys()):
                    task = self.active_tasks[task_id]
                    if task.is_completed:
                        del self.active_tasks[task_id]
                        self.metrics['tasks_completed'] += 1
                        
                return True
                
            def resolve_path_conflicts(self):
                # 简单版本 - 不做任何冲突检测
                pass
                
            def get_status(self):
                return {
                    'vehicles': len(self.vehicles),
                    'tasks_queued': len(self.task_queue),
                    'tasks_active': len(self.active_tasks),
                    'metrics': self.metrics
                }
                
        # 注册内置的DispatchSystem
        if hasattr(DispatchSystem, '__module__'):
            @register_dispatcher("dispatch_system")
            class DispatchSystemWrapper(DispatcherInterface):
                """DispatchSystem包装器，使其符合接口规范"""
                
                def __init__(self, path_planner=None, map_service=None, config=None):
                    super().__init__(path_planner, map_service, config)
                    self.dispatch_system = DispatchSystem(path_planner, map_service)
                    
                def add_vehicle(self, vehicle):
                    self.dispatch_system.add_vehicle(vehicle)
                    
                def add_task(self, task):
                    self.dispatch_system.add_task(task)
                    
                def scheduling_cycle(self):
                    return self.dispatch_system.scheduling_cycle()
                    
                def resolve_path_conflicts(self):
                    self.dispatch_system._resolve_path_conflicts()
                    
                def get_status(self):
                    return self.dispatch_system.get_status()
                    
                def start_scheduling(self, interval=1.0):
                    self.dispatch_system.start_scheduling(interval)
                    
                def stop_scheduling(self):
                    self.dispatch_system.stop_scheduling()
        
    except (ImportError, AttributeError) as e:
        logger.warning(f"导入内置调度器失败: {str(e)}")
        
    # 修改自动加载方式，避免重复注册
    # 只在目录中寻找其他未注册的调度器
    def load_dispatchers_from_directory(directory_path=None, exclude_classes=None):
        """
        从目录加载未注册的调度器
        Args:
            directory_path: 目录路径，默认为algorithm
            exclude_classes: 已注册的类，避免重复注册
        """
        if exclude_classes is None:
            exclude_classes = set()
        else:
            exclude_classes = set(exclude_classes)
            
        # 把已注册的调度器添加到排除列表
        for dispatcher_class in REGISTERED_DISPATCHERS.values():
            exclude_classes.add(dispatcher_class)
            
        if directory_path is None:
            directory_path = os.path.join(PROJECT_ROOT, "algorithm")
            
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            logger.warning(f"调度器目录不存在: {directory_path}")
            return {}
            
        # 查找所有Python文件
        py_files = [f for f in os.listdir(directory_path) 
                    if f.endswith('.py') and not f.startswith('__')]
        
        # 导入每个文件作为模块
        for py_file in py_files:
            module_name = os.path.splitext(py_file)[0]
            # 跳过已经处理过的内置模块
            if module_name in ['cbs', 'dispatch_service']:
                continue
                
            full_module = f"algorithm.{module_name}"
            
            try:
                # 导入模块
                module = importlib.import_module(full_module)
                
                # 查找未注册的调度器
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, DispatcherInterface) and 
                        obj != DispatcherInterface and
                        obj not in exclude_classes):
                        
                        # 使用类名作为注册名称
                        dispatcher_name = name.lower()
                        if dispatcher_name not in REGISTERED_DISPATCHERS:
                            REGISTERED_DISPATCHERS[dispatcher_name] = obj
                            logger.info(f"自动注册调度算法: {dispatcher_name} ({name})")
            except Exception as e:
                logger.warning(f"加载调度器模块 {full_module} 时出错: {str(e)}")
        
        return REGISTERED_DISPATCHERS
        
    # 使用修改后的加载方法
    load_dispatchers_from_directory()