from __future__ import annotations
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Union, TYPE_CHECKING
import logging
import os
import sys
from threading import RLock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if TYPE_CHECKING:
    from models.vehicle import MiningVehicle, VehicleState, TransportStage


from utils.geo_tools import GeoUtils
from algorithm.path_planner import Node  # 导入已存
from models.vehicle import VehicleState, TransportStage
class ConcurrentTaskError(Exception):
    """并发任务异常"""
    def __init__(self, vehicle_id: int):
        super().__init__(f"Vehicle {vehicle_id} already has active task")
        self.code = "CONCURRENT_TASK"
class TaskValidationError(Exception):
    """任务验证异常基类"""
    code = "TASK_VALIDATION_ERROR"

class TaskAssignError(TaskValidationError):
    """任务分配异常"""
    def __init__(self, message: str, vehicle_id: Optional[int] = None):
        super().__init__(f"[{vehicle_id}] {message}" if vehicle_id else message)
        self.code = "TASK_ASSIGN_ERROR"

class RouteValidationError(TaskValidationError):
    """路径验证异常"""
    code = "ROUTE_VALIDATION"

"""
增强型运输任务模型
"""
class TransportTask:
    """增强型运输任务模型"""
    
    STATUS_FLOW = {
        'pending': ['assigned', 'canceled'],
        'assigned': ['in_progress', 'canceled'],
        'in_progress': ['completed', 'failed'],
        'completed': [],
        'canceled': ['pending'],
        'failed': ['pending']
    }
    
    def __init__(self, 
                 task_id: str,
                 start_point: Union[Tuple[float, float], Node], 
                 end_point: Union[Tuple[float, float], Node],
                 task_type: str,
                 total_weight: float = 50000,
                 waypoints: List[Union[Tuple[float, float], Node]] = None,  
                 priority: int = 1,
                 max_retries: int = 3):
        
        # 核心属性
        self.task_id = task_id
        self.task_type = task_type
        self.start_point = self._validate_coordinate(start_point)
        self.end_point = self._validate_coordinate(end_point)
        self.location = self.start_point  # 当前位置，初始为起点
        self.total_weight = total_weight
        self.remaining_weight = total_weight
        self.priority = priority
        self.max_retries = max_retries
        self.retry_count = 0
        
        # 状态管理
        self.status = 'pending'
        self.is_completed = False
        self.created_time = datetime.now()
        self.deadline = self.created_time + timedelta(hours=2)
        self._status_lock = RLock()
        
        # 分配管理
        self.assigned_to = None
        self.assigned_time = None
        self.assigned_vehicle = None
        
        # 路径管理
        self.waypoints = []
        if waypoints:
            try:
                self.waypoints = [self._validate_coordinate(p) for p in waypoints]
            except ValueError as e:
                logging.error(f"任务{task_id}路径点验证失败: {str(e)}")
                
        self.current_waypoint = 0
        self.planned_route = []
        
        # 性能监控
        self.position_history = []
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        
        # 资源计算
        self._approach_distance = 0.0
        self._task_distance = 0.0
        self._total_distance = 0.0
        
    def _validate_coordinate(self, point) -> Tuple[float, float]:
        """统一坐标验证与转换"""
        if isinstance(point, tuple) and len(point) == 2:
            return (float(point[0]), float(point[1]))
        elif isinstance(point, Node):
            return (float(point.x), float(point.y))
        elif isinstance(point, str):
            try:
                # 处理字符串格式 "(x,y)"
                x, y = map(float, point.strip('()').split(','))
                return (x, y)
            except Exception:
                raise ValueError(f"无效坐标字符串格式: {point}")
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            # 处理通用点对象
            return (float(point.x), float(point.y))
            
        raise ValueError(f"不支持的坐标类型: {type(point)}")
        
    def validate_for_vehicle(self, vehicle) -> bool:
        """验证车辆是否适合执行此任务"""
        # 检查车辆状态
        if vehicle.state != VehicleState.IDLE and vehicle.state != VehicleState.PREPARING:
            return False
            
        # 检查载重能力
        if self.task_type == "loading" and vehicle.current_load + self.remaining_weight > vehicle.max_capacity:
            return False
            
        # 检查是否有故障
        if hasattr(vehicle, 'fault_codes') and vehicle.fault_codes:
            return False
            
        return True
        
    def assign_to_vehicle(self, vehicle) -> None:
        """分配任务给车辆"""
        with self._status_lock:
            if not self._validate_status_transition('assigned'):
                raise TaskAssignError(
                    f"任务状态不允许分配: {self.status} → assigned",
                    vehicle.vehicle_id
                )
                
            if not self.validate_for_vehicle(vehicle):
                raise TaskAssignError(
                    f"车辆不满足任务要求",
                    vehicle.vehicle_id
                )
                
            self.assigned_vehicle = vehicle
            self.assigned_to = vehicle.vehicle_id
            self.assigned_time = datetime.now()
            self.status = 'assigned'
            
            logging.info(f"任务 {self.task_id} 已分配给车辆 {vehicle.vehicle_id}")
            
    def start_execution(self) -> None:
        """开始执行任务"""
        with self._status_lock:
            if not self._validate_status_transition('in_progress'):
                logging.warning(f"任务 {self.task_id} 状态({self.status})不允许开始执行")
                return
                
            self.status = 'in_progress'
            self.start_time = datetime.now()
            
            # 尝试更新分配的车辆状态
            if self.assigned_vehicle:
                self.assigned_vehicle.state = VehicleState.EN_ROUTE
                
            logging.info(f"任务 {self.task_id} 开始执行")
            
    def complete_task(self) -> None:
        """完成任务"""
        with self._status_lock:
            if not self._validate_status_transition('completed'):
                logging.warning(f"任务 {self.task_id} 状态({self.status})不允许标记为完成")
                return
                
            self.status = 'completed'
            self.is_completed = True
            self.end_time = datetime.now()
            
            if self.start_time:
                self.execution_time = (self.end_time - self.start_time).total_seconds()
                
            # 清理资源
            self._cleanup_resources()
            
            logging.info(f"任务 {self.task_id} 已完成，耗时: {self.execution_time:.1f}秒")
            
    def fail_task(self, reason: str = "") -> None:
        """标记任务失败"""
        with self._status_lock:
            if not self._validate_status_transition('failed'):
                logging.warning(f"任务 {self.task_id} 状态({self.status})不允许标记为失败")
                return
                
            self.status = 'failed'
            self.end_time = datetime.now()
            
            # 尝试重试
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                self._requeue_task()
                logging.warning(f"任务 {self.task_id} 失败，正在尝试第{self.retry_count}次重试: {reason}")
            else:
                logging.error(f"任务 {self.task_id} 失败且达到最大重试次数: {reason}")
                self._cleanup_resources()
                
    def cancel_task(self) -> None:
        """取消任务"""
        with self._status_lock:
            if not self._validate_status_transition('canceled'):
                logging.warning(f"任务 {self.task_id} 状态({self.status})不允许取消")
                return
                
            self.status = 'canceled'
            self.end_time = datetime.now()
            
            # 清理资源
            self._cleanup_resources()
            
            logging.info(f"任务 {self.task_id} 已取消")
            
    def _validate_status_transition(self, new_status: str) -> bool:
        """验证状态转换是否允许"""
        with self._status_lock:
            allowed_transitions = self.STATUS_FLOW.get(self.status, [])
            return new_status in allowed_transitions
            
    def _cleanup_resources(self) -> None:
        """清理任务资源"""
        # 恢复车辆负载
        if self.assigned_vehicle and hasattr(self.assigned_vehicle, 'current_load'):
            # 如果是装载任务，从车辆上移除负载
            if self.task_type == 'loading' and self.status == 'completed':
                pass  # 装载任务完成后不移除负载
            elif self.task_type == 'unloading' and self.status == 'completed':
                self.assigned_vehicle.current_load = 0  # 卸载任务完成后清空负载
            
        self.assigned_vehicle = None
        
    def _requeue_task(self) -> None:
        """重新入队任务"""
        self.status = 'pending'
        self.assigned_vehicle = None
        self.assigned_to = None
        self.assigned_time = None
        self.start_time = None
        self.end_time = None
        self.execution_time = None
        self.priority += 1  # 提高优先级
        self.deadline = datetime.now() + timedelta(hours=1)  # 设置新截止时间
        
    @property
    def estimated_duration(self) -> float:
        """估计任务执行时间（秒）"""
        if not self._total_distance:
            return 0.0
            
        # 如果有分配的车辆，使用车辆速度计算
        if self.assigned_vehicle and hasattr(self.assigned_vehicle, 'max_speed'):
            avg_speed = self.assigned_vehicle.max_speed
        else:
            avg_speed = 5.0  # 默认速度 5 m/s
            
        return self._total_distance / avg_speed
        
    @property
    def urgency(self) -> float:
        """计算任务紧急度（0-1）"""
        # 基于截止时间和优先级的紧急度
        if self.status in ('completed', 'canceled', 'failed'):
            return 0.0
            
        # 计算截止时间紧急度
        now = datetime.now()
        total_time = (self.deadline - self.created_time).total_seconds()
        remaining_time = (self.deadline - now).total_seconds()
        
        if remaining_time <= 0:
            time_urgency = 1.0  # 超时
        else:
            time_ratio = remaining_time / total_time if total_time > 0 else 0.0
            time_urgency = 1.0 - time_ratio
            
        # 优先级影响（1-5）
        priority_factor = min(1.0, (self.priority / 5.0))
        
        # 综合紧急度（优先级占40%，时间占60%）
        return (time_urgency * 0.6) + (priority_factor * 0.4)
        
    def set_route(self, route: List[Tuple[float, float]]) -> None:
        """设置计划路线"""
        if not route:
            return
            
        self.planned_route = route
        
        # 计算总距离
        self._task_distance = sum(
            math.dist(route[i], route[i+1])
            for i in range(len(route)-1)
        )
        
        self._total_distance = self._approach_distance + self._task_distance
        
    def update_approach_distance(self, distance: float) -> None:
        """更新接近距离"""
        self._approach_distance = distance
        self._total_distance = self._approach_distance + self._task_distance
        
    def add_position_history(self, position: Tuple[float, float]) -> None:
        """记录位置历史"""
        self.position_history.append((datetime.now(), position))
        self.location = position  # 更新当前位置
        
    def __repr__(self) -> str:
        """字符串表示"""
        progress = (1 - self.remaining_weight/self.total_weight) * 100 if self.total_weight > 0 else 0
        
        return (f"<TransportTask {self.task_id} | {self.task_type} | {self.status} | "
                f"进度: {progress:.1f}% | 优先级: {self.priority} | "
                f"分配: {self.assigned_to or '无'} | 紧急度: {self.urgency:.2f}>")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    class MockVehicle:
        def __init__(self):
            self.vehicle_id = 1001
            self.status = VehicleState.IDLE.value  # 确保使用枚举值
            self.max_capacity = 100000
            self.current_load = 0
            self.max_speed = 5.0
            self.current_path = [(10.0, 10.0), (20.0, 20.0)]
            self.map_service = type('MockMap', (), {'validate_path': lambda x,y: True})()
            self.fault_codes = set()
            self.task_history = []
        
        def complete_task(self):
            self.status = 'idle'
            self.current_load = 0
    
    try:
        print("=== 开始运输任务系统测试 ===")
        
        test_task = TransportTask(
            task_id="T1001",
            start_point=(25.0, 25.0),
            end_point=(75.0, 75.0),
            task_type="loading",  # 新增task_type参数
            waypoints=[(30.0, 30.0), (50.0, 50.0)],
            total_weight=50000,
            priority=3
        )
        
        print("\n*** 测试用例1：有效任务分配 ***")
        test_vehicle = MockVehicle()
        test_route = [(25.0, 25.0), (30.0, 30.0), (50.0, 50.0), (75.0, 75.0)]
        
        test_task.assign_to_vehicle(test_vehicle, test_route)
        print(f"任务状态：{test_task.status} | 车辆状态：{test_vehicle.status}")
        print(f"预估耗时：{test_task.estimated_duration}")
        
        print("\n*** 测试用例2：非法状态转换 ***")
        try:
            test_task.assign_to_vehicle(test_vehicle, test_route)
        except TaskAssignError as e:
            print(f"捕获状态异常：{str(e)}")
        
        print("\n*** 测试用例3：任务生命周期测试 ***")
        try:
            test_task.start_execution()
            print(f"执行中状态：{test_task.status}")
            test_task.update_progress(50000)
            print(f"完成状态：{test_task.status}")
            
        except Exception as e:
            print(f"测试异常：{str(e)}")
            
    except Exception as e:
        logging.error(f"测试失败：{str(e)}", exc_info=True)