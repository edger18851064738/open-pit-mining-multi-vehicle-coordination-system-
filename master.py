import os
import sys
import time
import logging
import random
from typing import List, Tuple, Dict

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入项目模块
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from utils.geo_tools import GeoUtils
from algorithm.map_service import MapService
from algorithm.optimized_path_planner import HybridPathPlanner
from algorithm.cbs import ConflictBasedSearch  # 假设已移动到独立文件

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_simulation(duration=120, update_interval=1.0, num_vehicles=5, num_tasks=10):
    """
    运行多车调度系统模拟
    
    Args:
        duration: 模拟持续时间(秒)
        update_interval: 状态更新间隔(秒)
        num_vehicles: 车辆数量
        num_tasks: 任务数量
    """
    logging.info("初始化系统组件...")
    
    # 创建核心组件
    map_service = MapService()
    path_planner = HybridPathPlanner(map_service)
    
    # 创建测试场景的障碍物
    create_test_obstacles(path_planner)
    # 创建车辆
    vehicles = create_vehicles(map_service, num_vehicles)    
    # 创建CBS冲突解决器
    cbs = ConflictBasedSearch(path_planner)
    class MockDispatch:
        def __init__(self):
            self.vehicles = {}

    if not path_planner.dispatch:
        mock_dispatch = MockDispatch()
        for vehicle in vehicles:
            mock_dispatch.vehicles[vehicle.vehicle_id] = vehicle
        path_planner.dispatch = mock_dispatch    

    
    # 创建任务
    tasks = create_tasks(num_tasks)
    
    # 分配初始任务给车辆
    assign_initial_tasks(vehicles, tasks, path_planner, cbs)
    
    # 运行模拟
    logging.info(f"开始模拟，持续时间: {duration}秒")
    start_time = time.time()
    last_status_time = start_time
    
    try:
        while time.time() - start_time < duration:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # 更新车辆位置
            update_vehicles(vehicles)
            
            # 检查任务完成情况
            check_completed_tasks(vehicles, tasks)
            
            # 分配新任务给空闲车辆
            assign_new_tasks(vehicles, tasks, path_planner, cbs)
            
            # 冲突检测与解决
            resolve_conflicts(vehicles, cbs)
            
            # 定期打印状态
            if current_time - last_status_time >= 5.0:
                print_status(vehicles, tasks, elapsed)
                last_status_time = current_time
                
            # 等待下一个更新周期
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        logging.info("模拟被用户中断")
    
    # 打印最终状态
    print_final_results(vehicles, tasks, time.time() - start_time)

def create_test_obstacles(path_planner):
    """创建测试场景的障碍物"""
    # 预定义障碍物多边形
    obstacles = [
        [(20,60), (80,60), (80,70), (20,70)],         # 水平障碍墙1
        [(120,60), (180,60), (180,70), (120,70)],     # 水平障碍墙2
        [(40,30), (60,30), (60,40), (40,40)],         # 小障碍物1
        [(140,30), (160,30), (160,40), (140,40)],     # 小障碍物2
        [(90,100), (110,100), (110,120), (90,120)],   # 中央障碍物
    ]
    
    # 标记障碍物区域
    path_planner.mark_obstacle_area(obstacles)
    logging.info(f"已创建{len(path_planner.obstacle_grids)}个障碍物点")

def create_vehicles(map_service, num_vehicles):
    """创建测试车辆"""
    vehicles = []
    
    # 车辆起始位置，分散在地图四周
    start_positions = [
        (180, 180),  # 右上角
        (20, 180),   # 左上角
        (20, 20),    # 左下角
        (180, 20),   # 右下角
        (100, 20),   # 下方中央
    ]
    
    # 确保有足够的起始位置
    while len(start_positions) < num_vehicles:
        start_positions.append(
            (random.randint(20, 180), random.randint(20, 180))
        )
    
    # 创建车辆
    for i in range(num_vehicles):
        config = {
            'vehicle_id': i+1,
            'current_location': start_positions[i],
            'max_capacity': 50,
            'max_speed': random.uniform(5.0, 8.0),
            'base_location': (100, 190),  # 基地位置
            'status': VehicleState.IDLE
        }
        
        vehicle = MiningVehicle(
            vehicle_id=i+1,
            map_service=map_service,
            config=config
        )
        
        # 确保必要属性存在
        if not hasattr(vehicle, 'current_path'):
            vehicle.current_path = []
        if not hasattr(vehicle, 'path_index'):
            vehicle.path_index = 0
            
        vehicles.append(vehicle)
        
    logging.info(f"已创建{len(vehicles)}辆车辆")
    return vehicles

def create_tasks(num_tasks):
    """创建测试任务"""
    tasks = []
    
    # 任务起点和终点
    locations = [
        (30, 30),   # 左下区域
        (30, 170),  # 左上区域
        (170, 30),  # 右下区域
        (170, 170), # 右上区域
        (100, 30),  # 下方中央
        (100, 170), # 上方中央
        (30, 100),  # 左侧中央
        (170, 100), # 右侧中央
    ]
    
    # 创建任务
    for i in range(num_tasks):
        # 随机选择起点和终点（确保不同）
        start_idx = random.randint(0, len(locations)-1)
        end_idx = start_idx
        while end_idx == start_idx:
            end_idx = random.randint(0, len(locations)-1)
            
        start_point = locations[start_idx]
        end_point = locations[end_idx]
        
        # 随机任务类型
        task_type = random.choice(["transport", "loading", "unloading"])
        
        # 创建任务
        task = TransportTask(
            task_id=f"TASK-{i+1}",
            start_point=start_point,
            end_point=end_point,
            task_type=task_type,
            priority=random.randint(1, 3)
        )
        
        tasks.append(task)
    
    logging.info(f"已创建{len(tasks)}个任务")
    return tasks

def assign_initial_tasks(vehicles, tasks, path_planner, cbs):
    """分配初始任务给车辆"""
    # 确保每辆车都有任务
    for i, vehicle in enumerate(vehicles):
        if i < len(tasks):
            task = tasks[i]
            
            # 规划路径
            path = path_planner.plan_path(vehicle.current_location, task.end_point, vehicle)
            
            # 检查冲突
            vehicle_paths = {str(v.vehicle_id): v.current_path for v in vehicles if v.current_path}
            vehicle_paths[str(vehicle.vehicle_id)] = path
            
            # 解决冲突
            resolved_paths = cbs.resolve_conflicts(vehicle_paths)
            
            # 更新路径
            if str(vehicle.vehicle_id) in resolved_paths:
                path = resolved_paths[str(vehicle.vehicle_id)]
            
            # 分配任务和路径
            vehicle.assign_task(task)
            vehicle.assign_path(path)
            task.assigned_to = vehicle.vehicle_id
            
            logging.info(f"已将任务{task.task_id}分配给车辆{vehicle.vehicle_id}，路径长度: {len(path)}")

def update_vehicles(vehicles):
    """更新车辆位置"""
    for vehicle in vehicles:
        # 检查车辆是否有路径
        if vehicle.current_path and vehicle.path_index < len(vehicle.current_path) - 1:
            # 移动到下一个路径点
            vehicle.path_index += 1
            vehicle.current_location = vehicle.current_path[vehicle.path_index]

def check_completed_tasks(vehicles, tasks):
    """检查任务完成情况"""
    for vehicle in vehicles:
        if vehicle.current_task and vehicle.path_index >= len(vehicle.current_path) - 1:
            # 车辆已到达终点，标记任务完成
            task = vehicle.current_task
            task.is_completed = True
            
            # 更新车辆状态
            vehicle.current_task = None
            vehicle.state = VehicleState.IDLE
            vehicle.current_path = []
            vehicle.path_index = 0
            
            logging.info(f"车辆{vehicle.vehicle_id}完成任务{task.task_id}")

def assign_new_tasks(vehicles, tasks, path_planner, cbs):
    """分配新任务给空闲车辆"""
    # 获取空闲车辆
    idle_vehicles = [v for v in vehicles if not v.current_task]
    
    available_tasks = []
    for task in tasks:
        # 确保任务未完成且未被分配或已经完成
        if hasattr(task, 'is_completed') and not task.is_completed:
            # 检查任务是否已被分配给正在执行的车辆
            already_assigned = False
            for v in vehicles:
                if v.current_task and v.current_task.task_id == task.task_id:
                    already_assigned = True
                    break
            
            if not already_assigned:
                available_tasks.append(task)
        elif not hasattr(task, 'is_completed') or (not task.is_completed and not hasattr(task, 'assigned_to')):
            available_tasks.append(task)
    
    
    # 分配任务
    for vehicle in idle_vehicles:
        if not available_tasks:
            break
            
        # 选择任务 (基于优先级和距离)
        best_task = None
        best_score = float('inf')
        
        for task in available_tasks:
            # 计算距离分数
            distance = math.dist(vehicle.current_location, task.end_point)
            
            # 计算优先级分数 (优先级越高，分数越低)
            priority_score = task.priority
            
            # 综合分数
            score = distance * priority_score
            
            if score < best_score:
                best_score = score
                best_task = task
        
        if best_task:
            # 规划路径
            path = path_planner.plan_path(vehicle.current_location, best_task.end_point, vehicle)
            
            # 检查冲突
            vehicle_paths = {str(v.vehicle_id): v.current_path for v in vehicles if v.current_path}
            vehicle_paths[str(vehicle.vehicle_id)] = path
            
            # 解决冲突
            resolved_paths = cbs.resolve_conflicts(vehicle_paths)
            
            # 更新路径
            if str(vehicle.vehicle_id) in resolved_paths:
                path = resolved_paths[str(vehicle.vehicle_id)]
            
            # 分配任务和路径
            vehicle.assign_task(best_task)
            vehicle.assign_path(path)
            best_task.assigned_to = vehicle.vehicle_id
            
            # 从可用任务中移除
            available_tasks.remove(best_task)
            
            logging.info(f"已将任务{best_task.task_id}分配给空闲车辆{vehicle.vehicle_id}，路径长度: {len(path)}")

def resolve_conflicts(vehicles, cbs):
    """检测并解决车辆之间的冲突"""
    # 收集所有车辆的路径
    vehicle_paths = {}
    for vehicle in vehicles:
        if vehicle.current_path and len(vehicle.current_path) > vehicle.path_index:
            # 只考虑当前位置之后的路径
            remaining_path = vehicle.current_path[vehicle.path_index:]
            if len(remaining_path) > 1:
                vehicle_paths[str(vehicle.vehicle_id)] = remaining_path
    
    # 检测和解决冲突
    if vehicle_paths:
        # 解决冲突前的路径数
        before_count = len(vehicle_paths)
        
        # 执行冲突解决
        resolved_paths = cbs.resolve_conflicts(vehicle_paths)
        
        # 计算修改的路径数
        changed_count = sum(1 for vid in vehicle_paths if vid in resolved_paths and 
                           vehicle_paths[vid] != resolved_paths[vid])
        
        if changed_count > 0:
            logging.info(f"解决了{changed_count}条路径的冲突")
            
            # 更新车辆路径
            for vid_str, path in resolved_paths.items():
                if path and path != vehicle_paths.get(vid_str, []):
                    vid = int(vid_str)
                    for vehicle in vehicles:
                        if vehicle.vehicle_id == vid:
                            # 保留当前位置
                            current_pos = vehicle.current_location
                            # 确保新路径从当前位置开始
                            if len(path) > 0 and path[0] != current_pos:
                                path = [current_pos] + path
                            vehicle.assign_path(path)
                            vehicle.path_index = 0  # 重置路径索引
                            break

def print_status(vehicles, tasks, elapsed_time):
    """打印系统状态"""
    # 计算状态统计
    total_vehicles = len(vehicles)
    idle_count = sum(1 for v in vehicles if not v.current_task)
    moving_count = total_vehicles - idle_count
    
    total_tasks = len(tasks)
    completed_count = sum(1 for t in tasks if hasattr(t, 'is_completed') and t.is_completed)
    in_progress_count = sum(1 for t in tasks if hasattr(t, 'assigned_to') and t.assigned_to and 
                           (not hasattr(t, 'is_completed') or not t.is_completed))
    pending_count = total_tasks - completed_count - in_progress_count
    
    # 打印状态
    print("\n" + "="*40)
    print(f"模拟时间: {int(elapsed_time)}秒")
    print(f"车辆: {moving_count}活动/{total_vehicles}总数 ({idle_count}空闲)")
    print(f"任务: {completed_count}完成/{total_tasks}总数 ({in_progress_count}进行中, {pending_count}等待)")
    
    # 打印车辆详情
    print("\n车辆状态:")
    for vehicle in vehicles:
        status = "空闲" if not vehicle.current_task else "执行任务"
        task_id = vehicle.current_task.task_id if vehicle.current_task else "无"
        position = f"({vehicle.current_location[0]:.1f}, {vehicle.current_location[1]:.1f})"
        progress = ""
        
        if vehicle.current_path and vehicle.path_index > 0:
            progress = f"{vehicle.path_index}/{len(vehicle.current_path)}点"
            
        print(f"  车辆{vehicle.vehicle_id}: {status} | 任务: {task_id} | 位置: {position} | 进度: {progress}")
    
    print("="*40)

def print_final_results(vehicles, tasks, elapsed_time):
    """打印最终结果"""
    # 计算完成任务
    completed_tasks = [t for t in tasks if hasattr(t, 'is_completed') and t.is_completed]
    
    # 计算每辆车完成的任务数
    vehicle_completions = {}
    for task in completed_tasks:
        if hasattr(task, 'assigned_to') and task.assigned_to:
            vid = task.assigned_to
            vehicle_completions[vid] = vehicle_completions.get(vid, 0) + 1
    
    print("\n" + "="*60)
    print("模拟结束 - 最终结果")
    print("="*60)
    print(f"总时间: {elapsed_time:.1f}秒")
    print(f"总车辆: {len(vehicles)}")
    print(f"总任务: {len(tasks)}")
    print(f"完成任务: {len(completed_tasks)}/{len(tasks)} ({len(completed_tasks)/len(tasks)*100:.1f}%)")
    
    # 打印每辆车的任务完成情况
    print("\n车辆任务完成情况:")
    for vehicle in vehicles:
        completions = vehicle_completions.get(vehicle.vehicle_id, 0)
        print(f"  车辆{vehicle.vehicle_id}: {completions}个任务")
    
    print("\n任务详情:")
    for task in tasks:
        status = "已完成" if (hasattr(task, 'is_completed') and task.is_completed) else "未完成"
        assigned = f"车辆{task.assigned_to}" if hasattr(task, 'assigned_to') and task.assigned_to else "未分配"
        print(f"  任务{task.task_id}: {status} | {assigned} | 优先级: {task.priority}")
    
    print("="*60)

if __name__ == "__main__":
    import math  # 导入math模块(用于计算点距离)
    
    # 运行模拟
    run_simulation(
        duration=120,      # 模拟2分钟
        update_interval=0.5,  # 每0.5秒更新一次
        num_vehicles=5,    # 5辆车辆
        num_tasks=10      # 10个任务
    )