from __future__ import annotations
import os
import sys
import math
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import logging
from enum import Enum, auto
from threading import RLock
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Set, TYPE_CHECKING, Callable
import time
from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError


# 状态枚举增强
class TransportStage(Enum):
    APPROACHING = auto()
    TRANSPORTING = auto()
    RETURNING = auto()

class VehicleState(Enum):
    IDLE = auto()
    PREPARING = auto()
    EN_ROUTE = auto()
    UNLOADING = auto()
    EMERGENCY_STOP = auto()

if TYPE_CHECKING:
    from .task import TransportTask
else:
    # 使用字符串前向引用
    TransportTask = 'TransportTask'

from algorithm.map_service import MapService
from utils.path_tools import PathOptimizationError 

class ConcurrentTaskError(Exception):
    """并发任务异常"""
    def __init__(self, vehicle_id: int):
        super().__init__(f"Vehicle {vehicle_id} already has active task")
        self.code = "CONCURRENT_TASK"

"""
增强型矿用运输车辆实体模型
"""
class MiningVehicle:
    """统一状态管理的矿用运输车辆模型"""
    
    def __init__(self, vehicle_id: int, map_service: MapService, config: Dict):
        self._state_lock = RLock()
        self.vehicle_id = vehicle_id
        self.map_service = map_service
        
        # 容量参数
        self.max_capacity = config.get('max_capacity', 50)
        self.current_load = config.get('current_load', 0)
        
        # 性能参数
        self.max_speed = config.get('max_speed', 5.0)
        self.turning_radius = config.get('turning_radius', 10.0)
        self.min_hardness = config.get('min_hardness', 2.5)
        
        # 位置和路径
        self.current_location = config.get('current_location', (0.0, 0.0))
        self.last_position = self.current_location
        self.base_location = config.get('base_location', self.current_location)
        self.current_path = []
        self.path_index = 0
        
        # 状态系统（统一使用state枚举，移除status）
        state_value = config.get('status', VehicleState.IDLE)
        # 兼容处理：如果传入的是字符串，转换为枚举
        if isinstance(state_value, str):
            try:
                self.state = VehicleState[state_value.upper()]
            except KeyError:
                self.state = VehicleState.IDLE
        else:
            self.state = state_value
            
        # 任务和阶段
        self.current_task = None
        self.transport_stage = None
        
        # 监控指标
        self.mileage = 0.0
        self.operation_hours = 0.0
        self.last_update = datetime.now()
        self.fault_codes = set()
        self.maintenance_records = []
        
        # 回调管理
        self.status_callbacks = []
        self.observers = []
        
        # 性能指标
        self.metrics = {
            'tasks_completed': 0,
            'total_distance': 0.0,
            'waiting_time': 0.0,
            'conflicts': 0
        }
        
    def add_observer(self, observer):
        """添加状态观察者"""
        with self._state_lock:
            if observer not in self.observers:
                self.observers.append(observer)
                
    def remove_observer(self, observer):
        """移除状态观察者"""
        with self._state_lock:
            if observer in self.observers:
                self.observers.remove(observer)
                
    def _notify_observers(self):
        """通知所有观察者状态变更"""
        for observer in self.observers:
            try:
                observer.update_vehicle_status(self)
            except Exception as e:
                logging.warning(f"观察者通知失败: {str(e)}")
                
    def assign_task(self, task: TransportTask):
        """分配任务给车辆"""
        with self._state_lock:
            if self.current_task is not None:
                raise ConcurrentTaskError(self.vehicle_id)
                
            if self.state not in (VehicleState.IDLE, VehicleState.PREPARING):
                raise ValueError(f"车辆 {self.vehicle_id} 当前状态({self.state})无法接受新任务")
                
            self.current_task = task
            self.state = VehicleState.EN_ROUTE
            
            # 根据任务类型设置运输阶段
            if task.task_type == "loading":
                self.transport_stage = TransportStage.APPROACHING
            elif task.task_type == "unloading":
                self.transport_stage = TransportStage.TRANSPORTING
            else:
                self.transport_stage = TransportStage.RETURNING
                
            # 记录任务分配时间
            task.assigned_to = self.vehicle_id
            task.assigned_time = datetime.now()
            
            # 通知观察者
            for callback in self.status_callbacks:
                try:
                    callback(self)
                except Exception as e:
                    logging.warning(f"状态回调执行失败: {str(e)}")
                    
            self._notify_observers()
            logging.info(f"车辆 {self.vehicle_id} 已接受任务 {task.task_id}")
            
    def assign_path(self, path: List[Tuple[float, float]]):
        """分配路径给车辆"""
        with self._state_lock:
            if not path:
                logging.warning(f"车辆 {self.vehicle_id} 分配了空路径")
                return
                
            # 保存之前的路径用于指标计算
            old_path_len = len(self.current_path) if self.current_path else 0
            
            # 标准化路径点为二维
            normalized_path = []
            for point in path:
                if isinstance(point, tuple):
                    if len(point) >= 2:
                        normalized_path.append((float(point[0]), float(point[1])))
                    else:
                        logging.warning(f"车辆 {self.vehicle_id} 路径点维度异常: {point}")
                        return
                else:
                    logging.warning(f"车辆 {self.vehicle_id} 路径点格式异常: {point}")
                    return
            
            self.current_path = normalized_path
            self.path_index = 0
            
            # 确保当前位置是二维点
            current_loc_2d = (self.current_location[0], self.current_location[1]) if len(self.current_location) > 2 else self.current_location
            path_start = self.current_path[0]
            
            # 安全计算距离
            try:
                distance = math.sqrt((current_loc_2d[0] - path_start[0])**2 + (current_loc_2d[1] - path_start[1])**2)
                if distance > 0.1:
                    logging.debug(f"车辆 {self.vehicle_id} 当前位置({current_loc_2d})与路径起点({path_start})不一致，已自动调整")
                    self.current_location = path_start
            except Exception as e:
                logging.error(f"车辆 {self.vehicle_id} 路径起点检查出错: {str(e)}")
            
    def update_position(self):
        """更新车辆位置（沿着路径移动）"""
        with self._state_lock:
            if self.state != VehicleState.EN_ROUTE or not self.current_path:
                return
                
            if self.path_index >= len(self.current_path) - 1:
                # 已到达路径终点
                self._handle_path_completion()
                return
                
            # 记录上一个位置
            self.last_position = self.current_location
            
            # 更新到下一个路径点
            self.path_index += 1
            self.current_location = self.current_path[self.path_index]
            
            # 计算移动距离并更新里程
            distance = math.dist(self.last_position, self.current_location)
            self.mileage += distance
            self.metrics['total_distance'] += distance
            
            # 记录操作时间
            current_time = datetime.now()
            elapsed = (current_time - self.last_update).total_seconds()
            self.operation_hours += elapsed / 3600  # 转换为小时
            self.last_update = current_time
            
            # 检查是否到达任务终点
            if self.path_index == len(self.current_path) - 1:
                logging.debug(f"车辆 {self.vehicle_id} 到达路径终点")
                
    def _handle_path_completion(self):
        """处理路径完成事件"""
        if not self.current_task:
            return
            
        # 根据运输阶段处理
        if self.transport_stage == TransportStage.APPROACHING:
            # 从接近装载点到运输阶段
            self.state = VehicleState.PREPARING
            # 模拟装载过程（实际应该有延迟）
            self.current_load = self.max_capacity
            self.transport_stage = TransportStage.TRANSPORTING
            logging.info(f"车辆 {self.vehicle_id} 完成装载，准备运输")
            
        elif self.transport_stage == TransportStage.TRANSPORTING:
            # 从运输到卸载阶段
            self.state = VehicleState.UNLOADING
            # 模拟卸载过程
            self.current_load = 0
            self.transport_stage = TransportStage.RETURNING
            logging.info(f"车辆 {self.vehicle_id} 完成卸载，准备返回")
            
        elif self.transport_stage == TransportStage.RETURNING:
            # 完成整个任务
            self._complete_task()
            
    def _complete_task(self):
        """完成当前任务"""
        if not self.current_task:
            return
            
        logging.info(f"车辆 {self.vehicle_id} 完成任务 {self.current_task.task_id}")
        
        # 更新任务状态
        self.current_task.is_completed = True
        
        # 更新车辆状态
        self.state = VehicleState.IDLE
        self.transport_stage = None
        self.current_path = []
        self.path_index = 0
        
        # 更新指标
        self.metrics['tasks_completed'] += 1
        
        # 保存任务引用后清除
        completed_task = self.current_task
        self.current_task = None
        
        # 通知观察者
        for callback in self.status_callbacks:
            try:
                callback(self, completed_task)
            except Exception as e:
                logging.warning(f"完成任务回调失败: {str(e)}")
                
        self._notify_observers()
        
    def perform_emergency_stop(self):
        """执行紧急停车"""
        with self._state_lock:
            prev_state = self.state
            self.state = VehicleState.EMERGENCY_STOP
            
            if prev_state != VehicleState.EMERGENCY_STOP:
                logging.warning(f"车辆 {self.vehicle_id} 执行紧急停车")
                self._notify_observers()
                
    def resume_operation(self):
        """恢复正常运行"""
        with self._state_lock:
            if self.state == VehicleState.EMERGENCY_STOP:
                if self.current_task:
                    self.state = VehicleState.EN_ROUTE
                else:
                    self.state = VehicleState.IDLE
                    
                logging.info(f"车辆 {self.vehicle_id} 恢复运行")
                self._notify_observers()
                
    def perform_maintenance(self, maintenance_type: str):
        """执行维护操作"""
        with self._state_lock:
            prev_state = self.state
            
            # 记录维护信息
            self.maintenance_records.append({
                'timestamp': datetime.now(),
                'type': maintenance_type,
                'pre_status': prev_state.name,
                'mileage': self.mileage
            })
            
            # 根据维护类型重置指标
            if maintenance_type == 'full_service':
                self.mileage = 0.0
                self.operation_hours = 0.0
                self.fault_codes.clear()
                
            elif maintenance_type == 'quick_check':
                # 快速检查不重置指标，只清除故障码
                self.fault_codes.clear()
                
            logging.info(f"车辆 {self.vehicle_id} 完成{maintenance_type}维护")
            
    def get_health_report(self) -> Dict:
        """获取健康状态报告"""
        with self._state_lock:
            return {
                'vehicle_id': self.vehicle_id,
                'state': self.state.name,
                'transport_stage': self.transport_stage.name if self.transport_stage else "NONE",
                'current_load': f"{self.current_load}/{self.max_capacity}",
                'mileage': f"{self.mileage:.1f}km",
                'operation_hours': f"{self.operation_hours:.1f}h",
                'faults': list(self.fault_codes),
                'last_maintenance': self.maintenance_records[-1] if self.maintenance_records else None,
                'position': self.current_location,
                'metrics': {
                    'completed_tasks': self.metrics['tasks_completed'],
                    'total_distance': f"{self.metrics['total_distance']:.1f}",
                    'conflicts': self.metrics['conflicts']
                }
            }
            
    def register_status_callback(self, callback):
        """注册状态变更回调"""
        if callback not in self.status_callbacks:
            self.status_callbacks.append(callback)
            
    def unregister_status_callback(self, callback):
        """取消状态变更回调"""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)


if __name__ == "__main__":
    import math
    from datetime import timedelta
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    class MockMapService:
        """模拟地图服务"""
        def get_virtual_origin(self):
            return (0.0, 0.0)
            
        def plan_route(self, start, end, vehicle_type):
            try:
                # 直接返回简单路径，不再依赖self.map_service
                return {
                    'path': [start, (start[0]+1, start[1]+1), end],
                    'distance': math.dist(start, end)
                }
            except Exception as e:
                logging.error(f"路径规划失败: {str(e)}")
                # 降级方案：简单直线路径
                return {
                    'path': [start, (start[0]+1, start[1]+1), end],
                    'distance': math.dist(start, end)
                }

    # 测试任务对象
    class TestTask:
        task_id = "T20230701"
        start_point = (10.0, 10.0)
        end_point = (50.0, 50.0)
        priority = 1  # 添加优先级属性

    # 初始化车辆
    config = {'max_capacity': 5000}
    vehicle = MiningVehicle(1, MockMapService(), config)
    
    try:
        # 测试任务分配
        vehicle.register_task_assignment(TestTask())
        
        # 模拟运输过程
        for _ in range(3):
            vehicle.update_position()
            print(f"当前位置: {vehicle.current_location}")
            time.sleep(0.5)
            
        # 测试维护操作
        vehicle.perform_maintenance("full_service")
        print(f"维护后里程: {vehicle.mileage}")
        
    except ConcurrentTaskError as e:
        print(f"并发任务错误: {str(e)}")
    except PathOptimizationError as e:
        print(f"路径规划失败: {str(e)}")