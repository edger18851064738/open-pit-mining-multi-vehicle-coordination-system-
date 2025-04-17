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
from typing import Optional, List, Tuple, Dict, Set, TYPE_CHECKING
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

class MiningVehicle:
    """智能矿用运输车辆实体模型（强化状态机版本）"""

    def __init__(self, vehicle_id: int, map_service: MapService, config: Dict):
        self.observers = []  # 观察者列表
        self._state_lock = RLock()

    def add_observer(self, observer):
        """添加任务观察者"""
        with self._state_lock:
            if observer not in self.observers:
                self.observers.append(observer)

    def remove_observer(self, observer):
        """移除任务观察者"""
        with self._state_lock:
            if observer in self.observers:
                self.observers.remove(observer)

    def _notify_observers(self):
        """通知所有观察者状态变更"""
        with self._state_lock:
            for observer in self.observers:
                observer.update_vehicle_status(self)

    def get_state_vector(self) -> List[float]:
        """
        获取车辆状态向量，用于强化学习
        
        返回:
            List[float]: 包含以下信息的向量:
                - 当前位置x坐标
                - 当前位置y坐标
                - 当前负载百分比
                - 车辆状态编码
                - 当前任务优先级(如有)
                - 到目标点的距离(如有)
        """
        state = [
            self.current_location[0],
            self.current_location[1],
            self.current_load / self.max_capacity if self.max_capacity > 0 else 0,
            self.state.value
        ]
        
        if self.current_task:
            state.append(self.current_task.priority)
            state.append(math.dist(self.current_location, self.current_task.end_point))
        else:
            state.extend([0, 0])
            
        return state
        
    def get_next_state_vector(self) -> List[float]:
        """
        获取车辆下一个状态的向量表示，用于强化学习
        
        返回:
            List[float]: 包含以下信息的向量:
                - 预测下一个位置x坐标
                - 预测下一个位置y坐标
                - 预测下一个负载百分比
                - 预测下一个状态编码
                - 当前任务优先级(如有)
                - 预测到目标点的距离(如有)
        """
        if self.state == VehicleState.EN_ROUTE and self.current_path:
            next_pos = self.current_path[0] if len(self.current_path) > 0 else self.current_location
            next_dist = math.dist(next_pos, self.current_task.end_point) if self.current_task else 0
        else:
            next_pos = self.current_location
            next_dist = math.dist(self.current_location, self.current_task.end_point) if self.current_task else 0
            
        state = [
            next_pos[0],
            next_pos[1],
            self.current_load / self.max_capacity if self.max_capacity > 0 else 0,
            self.state.value
        ]
        
        if self.current_task:
            state.append(self.current_task.priority)
            state.append(next_dist)
        else:
            state.extend([0, 0])
            
        return state
        
    def __init__(self, vehicle_id: int, map_service: MapService, config: Dict):
        self._state_lock = RLock()
        self.vehicle_id = vehicle_id
        self.max_capacity = config['max_capacity']
        self.map_service = map_service
        self.current_action = 0  # Initialize current_action with default value
        self.current_reward = 0.0  # Initialize current_reward with default value
        self.task_completed = False  # 标记任务是否完成
        self.position = config.get('current_location', (0.0, 0.0))  # 添加position属性，与current_location同步
        self.turning_radius = config.get('turning_radius', 10.0)
        
        # 坐标系统
        self._init_coordinate_system(config)
        
        # 运输状态
        self.current_task: Optional[TransportTask] = None  # 修改类型为TransportTask
        self.current_path: List[Tuple[float, float]] = []
        self.transport_stage: Optional[TransportStage] = None  # 初始状态应为空
        self.path_index: int = 0
        
        # 设备状态
        self.current_load: int = 0
        self.mileage: float = 0.0
        self.operation_hours: float = 0.0
        self.last_brake_check: datetime = datetime.now()
        
        # 安全系统
        self.fault_codes: Set[str] = set()
        self.maintenance_records: List[Dict] = []
        
        # 状态回调
# 导入 Callable 以解决未定义问题
        from typing import Callable
        self.status_callbacks: List[Callable] = []
        self.state: VehicleState = VehicleState.IDLE
        # 移除冗余的status属性，统一使用state枚举管理状态
        self.current_location = config.get('current_location', (0.0, 0.0))
        self.last_position = self.current_location  # 解决_update_position中的last_position未定义问题
        self.last_update: Optional[datetime] = None  # 明确可选类型        
    def _init_coordinate_system(self, config: Dict):
        """修正坐标初始化方法"""
        # 从配置直接获取原点坐标
        origin = (
            config.get('base_location', (0.0, 0.0)),  # 优先使用base_location
            config.get('virtual_origin', (0.0, 0.0))  # 备用参数
        )
        
        # 直接使用原始坐标（不再进行坐标转换）
        self.current_location = config.get('current_location', (0.0, 0.0))
        self.base_location = config.get('base_location', self.current_location)
        self.min_hardness = config.get('min_hardness', 2.5)  # 新增默认值
        self.max_speed = config.get('max_speed', 5.0)  # 新增最大速度属性
    def register_task_assignment(self, task: TransportTask) -> None:
        """强化版任务注册方法"""
        with self._state_lock:
            # 严格检查current_task属性
            if not hasattr(self, 'current_task'):
                self.current_task = None
                
            # 双重检查当前任务和状态
            if self.current_task is not None:
                raise ConcurrentTaskError(self.vehicle_id)
            
            # 增强状态验证
            if self.state not in (VehicleState.IDLE, VehicleState.PREPARING):
                raise ConcurrentTaskError(self.vehicle_id)
                
            # 验证任务准备状态
            self._validate_task_readiness()
            
            try:
                # 分阶段路径规划
                approach_path, transport_path = self._plan_routes(task)
                if not hasattr(self, 'transport_path'):
                    self.transport_path = []
                self._init_transport_parameters(approach_path, transport_path)
                
                # 启动运输流程
                self.current_task = task
                self.state = VehicleState.EN_ROUTE
                self._notify_status_change()
                
                # 添加详细日志记录
                logging.info(f"Vehicle {self.vehicle_id} accepted task {task.task_id}")
                logging.debug(f"Vehicle {self.vehicle_id} task details: {task.__dict__}")

            except Exception as e:
                logging.error(f"Route planning failed: {str(e)}")
                # 重置状态以防不一致
                self.current_task = None
                self.state = VehicleState.IDLE
                self._notify_status_change()
                if isinstance(e, PathOptimizationError):
                    raise
                elif isinstance(e, ConcurrentTaskError):
                    raise
                else:
                    raise PathOptimizationError(f"路径规划失败: {str(e)}") from e

    def _plan_routes(self, task: TransportTask) -> tuple:
        """分阶段路径规划"""
        current_coord = (self.current_location[0], self.current_location[1])
        task_start = (float(task.start_point[0]), float(task.start_point[1]))
        
        approach_result = self.map_service.plan_route(
            start=current_coord,
            end=task_start,
            vehicle_type='empty'
        )
        transport_result = self.map_service.plan_route(
            start=task_start,
            end=(float(task.end_point[0]), float(task.end_point[1])),
            vehicle_type='loaded'
        )
        
        return (
            self._parse_path(approach_result.get('path', [])),
            self._parse_path(transport_result.get('path', []))
        )

    def _parse_path(self, raw_path: List) -> List[Tuple[float, float]]:
        """路径数据格式标准化"""
        try:
            if isinstance(raw_path, dict) and 'path' in raw_path:
                raw_path = raw_path['path']
            return [(float(p[0]), float(p[1])) for p in raw_path]
        except (ValueError, TypeError, IndexError) as e:
            logging.error(f"路径解析错误: {e}, 原始路径数据: {raw_path}")
            return []

    def _plan_transport_path(self) -> List[Tuple[float, float]]:
        """规划运输路径"""
        if not self.current_task:
            raise PathOptimizationError("当前没有运输任务")
            
        try:
            transport_result = self.map_service.plan_route(
                start=self.current_task.start_point,
                end=self.current_task.end_point,
                vehicle_type='loaded'
            )
            return self._parse_path(transport_result.get('path', []))
        except Exception as e:
            raise PathOptimizationError(f"运输路径规划失败: {str(e)}")
            
    def _init_transport_parameters(self, approach_path: List, transport_path: List):
        """初始化运输参数"""
        if len(approach_path) < 2 or len(transport_path) < 2:
            raise PathOptimizationError("无效路径规划结果")
            
        self.approach_path = approach_path
        self.transport_path = transport_path
        self.current_path = approach_path
        self.transport_stage = TransportStage.APPROACHING
        self.current_load = 0
        self.vehicle_type = 'loaded' if self.current_load else 'empty'
        
        # Initialize path tracking variables
        self.path_index = 0
        
        # Ensure transport_path is initialized
        self.transport_path = transport_path if transport_path else []
    def perform_emergency_stop(self):
        """增强版紧急制动"""
        with self._state_lock:
            if self.state == VehicleState.EMERGENCY_STOP:
                return
                
            self.state = VehicleState.EMERGENCY_STOP
            self.current_path = []
            self.path_index = 0
            self._notify_status_change()
            logging.warning(f"Vehicle {self.vehicle_id} 紧急制动 | 最后位置: {self.current_location}")

    def perform_maintenance(self, maintenance_type: str) -> None:
        """维护操作强化"""
        with self._state_lock:
            if maintenance_type == "full_service":
                self.mileage = 0.0
                self.operation_hours = 0.0
                self.last_brake_check = datetime.now()
                self.fault_codes.clear()
            
            self.maintenance_records.append({
                'timestamp': datetime.now(),
                'type': maintenance_type,
                'pre_status': self.state.name
            })
            logging.info(f"Vehicle {self.vehicle_id} 完成维护: {maintenance_type}")

    def get_health_report(self) -> Dict:
        """设备健康报告"""
        with self._state_lock:
            return {
                'vehicle_id': self.vehicle_id,
                'mileage': f"{self.mileage:.1f}m",
                'operation_hours': f"{self.operation_hours/3600:.1f}h",
                'load_status': f"{self.current_load}/{self.max_capacity}kg",
                'last_brake_check': self.last_brake_check.isoformat(),
                'active_faults': list(self.fault_codes)
            }

    def _notify_status_change(self):
        """状态变更通知"""
        for callback in self.status_callbacks:
            try:
                callback(self)
            except Exception as e:
                logging.error(f"状态回调执行失败: {str(e)}")

    def update_position(self) -> None:
        """增强版位置更新"""
        with self._state_lock:
            if self.state != VehicleState.EN_ROUTE or not self.current_path:
                return

            self._advance_position()
            self._handle_stage_transition()

    def _advance_position(self):
        """推进路径点"""
        next_index = min(self.path_index + 1, len(self.current_path)-1)
        new_pos = self.current_path[next_index]
        self._update_position(new_pos)
        self.path_index = next_index

    def _update_position(self, new_pos: Tuple[float, float]):
        """原子位置更新"""
        self.current_location = new_pos
        self.last_update = datetime.now()
        self.mileage += self._calculate_distance(self.last_position, new_pos)
        self.last_position = new_pos

    def _calculate_distance(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """坐标距离计算"""
        return math.hypot(end[0]-start[0], end[1]-start[1])

    def _handle_stage_transition(self):
        """处理运输阶段转换"""
        if self.path_index < len(self.current_path)-1:
            return

        if self.transport_stage == TransportStage.APPROACHING:
            self._transition_to_transport()
        else:
            self._complete_task()

    def _transition_to_transport(self):
        """切换到运输阶段"""
        if not hasattr(self, 'transport_path') or not self.transport_path:
            # 尝试重新规划路径
            if hasattr(self, 'current_task') and self.current_task:
                try:
                    self.transport_path = self._plan_transport_path()
                except Exception as e:
                    raise PathOptimizationError(f"运输路径未初始化且无法重新规划: {str(e)}")
            else:
                raise PathOptimizationError("运输路径未初始化且无当前任务")
            
        self.current_path = self.transport_path
        self.path_index = 0
        self.transport_stage = TransportStage.TRANSPORTING
        self.current_load = self.max_capacity
        logging.info(f"Vehicle {self.vehicle_id} 开始运输阶段")
    def _validate_task_readiness(self):
        """新增任务准备校验"""
        if self.state != VehicleState.IDLE:
            raise ConcurrentTaskError(self.vehicle_id)

    def _complete_task(self):
        """增强版任务完成处理（移除充电依赖）"""
        with self._state_lock:
            # 运输阶段状态转换 ▼▼▼
            if self.transport_stage == TransportStage.TRANSPORTING:
                # 切换到返回基地阶段
                try:
                    return_path = self.map_service.plan_return_path(
                        current_pos=self.current_location,
                        base_pos=self.base_location,
                        vehicle_type='loaded' if self.current_load else 'empty'
                    )
                    if return_path:
                        self.current_path = self._parse_path(return_path)
                        self.path_index = 0
                        self.transport_stage = TransportStage.RETURNING
                        logging.info(f"车辆 {self.vehicle_id} 开始返回基地")
                        return
                except Exception as e:
                    logging.error(f"返回路径规划失败: {str(e)}")
            
            # 无论如何，确保任务完成后状态重置为IDLE
            self.transport_stage = None
            self.state = VehicleState.IDLE
            self.operation_hours = 0  # 移除充电计时
            self.current_task = None
            self.current_path = []
            logging.info(f"车辆 {self.vehicle_id} 任务完成，进入空闲状态")

    def assign_task(self, task: TransportTask):
        """任务分配方法（修复路径初始化问题）"""
        with self._state_lock:
            if hasattr(self, 'current_task') and self.current_task is not None:
                raise ConcurrentTaskError(f"Vehicle {self.vehicle_id} already has active task {self.current_task.task_id}")
                
            self.current_task = task
            self.transport_stage = TransportStage.APPROACHING
            
            # 生成模拟路径（临时方案）
            if not self.current_path:
                self.current_path = [
                    (self.current_location[0] + i*0.1, 
                    self.current_location[1] + i*0.1)
                    for i in range(20)
                ]
            self.path_index = 0
            self.state = VehicleState.EN_ROUTE
            
            logging.info(f"Vehicle {self.vehicle_id} 已接受任务 {task.task_id}")
            self._notify_status_change()

    def assign_path(self, path: List[Tuple[float, float]]):
        """路径分配方法（新增）"""
        with self._state_lock:
            if not path:
                raise ValueError("无效路径")
                
            self.current_path = path
            self.path_index = 0
            if self.current_location != path[0]:
                logging.warning(f"车辆 {self.vehicle_id} 当前位置与路径起点不一致，已重置位置")
                self.current_location = path[0]
    def move_along_path(self):
        """沿路径移动车辆"""
        if self.status == VehicleState.EN_ROUTE and self.current_path:
            # 模拟移动逻辑
            if self.path_index < len(self.current_path):
                self.current_location = self.current_path[self.path_index]
                self.path_index += 1
                self.mileage += 0.5  # 模拟里程增加
                
            # 到达终点后重置状态
            if self.path_index >= len(self.current_path):
                self.state = VehicleState.IDLE
                self.current_path = []
                # 触发回调通知调度系统生成新任务
                for callback in self.status_callbacks:
                    callback(self)
                    
    def update_state(self):
        """完整的状态更新方法"""
        self.move_along_path()
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