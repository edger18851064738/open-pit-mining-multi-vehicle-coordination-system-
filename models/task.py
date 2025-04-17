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

class TransportTask:
    """增强型运输任务实体模型"""
    
    STATUS_FLOW = {
        'pending': ['assigned'],
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
                 task_type: str,  # 新增task_type参数
                 total_weight: float = 50000,
                 waypoints: List[Union[Tuple[float, float], Node]] = None,  
                 priority: int = 1,
                 max_retries: int = 3):
        # 位置属性
        self.location = start_point
        # 核心属性
        self.task_id = task_id
        self.task_type = task_type  # 新增任务类型字段
        self.start_point = self._validate_coordinate(start_point)
        self.end_point = self._validate_coordinate(end_point)
        self.total_weight = total_weight
        self.remaining_weight = total_weight
        self.priority = priority
        self.max_retries = max_retries
        self.retry_count = 0
        
        # 状态管理
        self.status = 'pending'
        self.created_time = datetime.now()
        self.deadline = self.created_time + timedelta(hours=2)
        self._status_lock = RLock()
        
        # 运输路径
        self.waypoints = []
        if waypoints:
            try:
                self.waypoints = [self._validate_coordinate(p) for p in waypoints]
            except ValueError as e:
                logging.error(f"任务{task_id}路径点验证失败: {str(e)}")
                self.waypoints = []
        self.current_waypoint = 0
        self.planned_route: List[Tuple[float, float]] = []
        self.assigned_vehicle: Optional[MiningVehicle] = None
        self.assigned_to = None  # 新增任务分

        # 资源需求
        self._approach_distance: float = 0.0
        self._task_distance: float = 0.0
        self._total_distance: float = 0.0

        # 运输监控
        self.position_history: List[Tuple[float, float]] = []
        self.speed_monitor: Dict[datetime, float] = {}
        self.transport_stage: Optional[TransportStage] = None
        self.created_time = datetime.now()
        self.assigned_time: Optional[datetime] = None


    def _validate_coordinate(self, point) -> Tuple[float, float]:
        """统一坐标验证方法"""
        if isinstance(point, tuple) and len(point) == 2:
            return (float(point[0]), float(point[1]))
        if isinstance(point, str):
            try:
                x, y = map(float, point.strip('()').split(','))
                return (x, y)
            except:
                raise ValueError(f"无效坐标字符串格式：{point}")
        raise ValueError(f"不支持的坐标类型：{type(point)}")
    def _calculate_total_distance(self, path: List[Tuple[float, float]]) -> float:
        """统一距离计算逻辑（适配虚拟坐标系）"""
        geo_utils = GeoUtils()
        return sum(
            geo_utils.haversine(path[i], path[i+1])
            for i in range(len(path)-1)
        )

    def _estimate_duration(self) -> timedelta:
        """动态耗时预估（基于车辆性能）"""
        if not self.assigned_vehicle:
            return timedelta(seconds=0)
            
        avg_speed = min(
            self.assigned_vehicle.max_speed,
            15.0
        )
        return timedelta(seconds=self._total_distance / avg_speed)

    def validate_for_vehicle(self, vehicle: MiningVehicle) -> bool:
            return all([
                vehicle.status == VehicleState.IDLE.value,  # 修改为枚举值比较
                vehicle.current_load + self.remaining_weight <= vehicle.max_capacity,
                not vehicle.fault_codes
            ])

    def assign_to_vehicle(self, 
                        vehicle: MiningVehicle,
                        route: List[Tuple[float, float]]) -> None:
        """增强型任务分配（集成路径元数据）"""
        with self._status_lock:
            if not self._validate_status_transition('assigned'):
                raise TaskAssignError(
                    f"非法状态转换：{self.status} → assigned",
                    vehicle.vehicle_id
                )

            if not self.validate_for_vehicle(vehicle):
                raise TaskAssignError(
                    "车辆条件不满足任务需求",
                    vehicle.vehicle_id
                )

            if not vehicle.map_service.validate_path(route):
                raise TaskValidationError("无效运输路径")

            # 记录路径元数据
            self._approach_distance = self._calculate_total_distance(vehicle.current_path)
            self._task_distance = self._calculate_total_distance(route)
            self._total_distance = self._approach_distance + self._task_distance

            self.assigned_vehicle = vehicle
            self.planned_route = route
            self.status = 'assigned'
            self.assigned_time = datetime.now()
            self.estimated_duration = self._estimate_duration()

    def update_progress(self, delivered_weight: float) -> None:
        """增强进度更新（带实时校验）"""
        self.remaining_weight = max(0, self.remaining_weight - delivered_weight)
        
        if not self.assigned_vehicle or self.assigned_vehicle.status != 'moving':
            raise TaskValidationError("关联车辆状态异常")

        if self.remaining_weight <= 0:
            self.complete_task()
        elif datetime.now() > self.deadline:
            self.handle_timeout()

    @property
    def urgency(self) -> float:
        """动态紧急度计算（适配实际剩余时间）"""
        total_time = (self.deadline - self.created_time).total_seconds()
        elapsed_time = (datetime.now() - self.created_time).total_seconds()
        time_ratio = elapsed_time / total_time if total_time > 0 else 1.0
        
        return min(1.0, max(0.0, 
            time_ratio * 0.6 + 
            (self.priority / 5) * 0.4
        ))

    def start_execution(self) -> None:
        if self._validate_status_transition('in_progress'):
            self.status = 'in_progress'
            self.assigned_vehicle.status = 'moving'
            self.assigned_vehicle.current_load += self.remaining_weight

    def complete_task(self) -> None:
        if self._validate_status_transition('completed'):
            self.status = 'completed'
            self.assigned_vehicle.complete_task()
            self._cleanup_resources()

    def handle_timeout(self) -> None:
        if self.retry_count < self.max_retries:
            self.retry_count += 1
            self.status = 'failed'
            self._requeue_task()
        else:
            self.status = 'failed'
            self._cleanup_resources()

    def _validate_status_transition(self, new_status: str) -> bool:
        with self._status_lock:
            allowed = self.STATUS_FLOW.get(self.status, [])
            return new_status in allowed

    def _cleanup_resources(self) -> None:
        if self.assigned_vehicle:
            self.assigned_vehicle.current_load -= (
                self.total_weight - self.remaining_weight
            )
            self.assigned_vehicle = None

    def _requeue_task(self) -> None:
        self.priority += 1
        self.status = 'pending'
        self.assigned_vehicle = None
        self.planned_route = []
        self.deadline = datetime.now() + timedelta(hours=1)

    def __repr__(self) -> str:
        return (f"<TransportTask {self.task_id} | {self.status} | "
                f"进度：{1 - self.remaining_weight/self.total_weight:.1%} | "
                f"紧急度：{self.urgency:.2f}>")

    def _convert_to_tuple(self, point) -> Tuple[float, float]:
        """增强型坐标转换"""
        if isinstance(point, Node):
            return (point.x, point.y)
        elif isinstance(point, (tuple, list)) and len(point) >= 2:
            return (float(point[0]), float(point[1]))
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            return (point.x, point.y)
        raise ValueError(f"无效坐标格式：{type(point)}")

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