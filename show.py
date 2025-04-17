"""
露天矿多车协同调度系统可视化工具 (Tkinter版)
=====================================

提供以下功能:
1. 使用tkinter创建实时交互式可视化界面
2. 在100×100地图上显示障碍物、两个装载点、一个卸载点和一个停车场
3. 展示四台车辆执行任务的实时动态
4. 任务循环：装载点→卸载点→停车场→装载点
"""

import os
import sys
import time
import threading
import random
import logging
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.font_manager as fm

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Set
import math
from datetime import datetime
from queue import Queue
from enum import Enum, auto

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("visualization")

# 颜色配置
COLORS = {
    'background': '#f5f5f5',
    'grid': '#e0e0e0',
    'obstacle': '#555555',
    'loading': '#4CAF50',    # 绿色
    'unloading': '#F44336',  # 红色
    'parking': '#2196F3',    # 蓝色
    'vehicle': {
        'idle': '#2196F3',       # 蓝色
        'en_route': '#FF9800',   # 橙色
        'loading': '#4CAF50',    # 绿色
        'unloading': '#F44336',  # 红色
        'returning': '#9C27B0',  # 紫色
    },
    'path': '#78909C',       # 灰蓝色
}

# 任务类型枚举
class TaskType(Enum):
    LOADING = auto()    # 前往装载点
    UNLOADING = auto()  # 前往卸载点
    RETURNING = auto()  # 返回停车场
    IDLE = auto()       # 空闲

# 简化的车辆状态
class VehicleState(Enum):
    IDLE = auto()
    EN_ROUTE = auto()
    LOADING = auto()
    UNLOADING = auto()
    RETURNING = auto()

# 简化的车辆类
class Vehicle:
    def __init__(self, vehicle_id, initial_pos=(0, 0)):
        self.vehicle_id = vehicle_id
        self.current_location = initial_pos
        self.target_location = None
        self.current_path = []
        self.path_index = 0
        self.state = VehicleState.IDLE
        self.task_type = None
        self.speed = 1.0
        self.history = [initial_pos]  # 历史轨迹
        
    def assign_task(self, task_type, target):
        """分配任务"""
        self.task_type = task_type
        self.target_location = target
        self.current_path = self._generate_path(self.current_location, target)
        self.path_index = 0
        
        # 设置车辆状态
        if task_type == TaskType.LOADING:
            self.state = VehicleState.EN_ROUTE
        elif task_type == TaskType.UNLOADING:
            self.state = VehicleState.EN_ROUTE
        elif task_type == TaskType.RETURNING:
            self.state = VehicleState.RETURNING
        
        return True
        
    def _generate_path(self, start, end):
        """生成从起点到终点的简单路径"""
        # 这里使用简单直线，实际系统中会使用A*等算法
        # 生成多个中间点以使路径更自然
        path = [start]
        
        steps = int(max(
            abs(end[0] - start[0]), 
            abs(end[1] - start[1])
        ) // 2)
        
        if steps > 0:
            dx = (end[0] - start[0]) / steps
            dy = (end[1] - start[1]) / steps
            
            for i in range(1, steps):
                path.append((
                    start[0] + dx * i,
                    start[1] + dy * i
                ))
        
        path.append(end)
        return path
        
    def update(self):
        """更新车辆位置"""
        if self.state == VehicleState.IDLE:
            return False
            
        if self.path_index < len(self.current_path) - 1:
            # 移动到下一个路径点
            self.path_index += 1
            self.current_location = self.current_path[self.path_index]
            self.history.append(self.current_location)
            return True
        elif self.current_location == self.target_location:
            # 已到达目的地
            if self.task_type == TaskType.LOADING:
                self.state = VehicleState.LOADING
            elif self.task_type == TaskType.UNLOADING:
                self.state = VehicleState.UNLOADING
            elif self.task_type == TaskType.RETURNING:
                self.state = VehicleState.IDLE
                self.task_type = None
            return False
        return False
        
    def is_task_completed(self):
        """检查当前任务是否已完成"""
        if self.task_type == TaskType.LOADING and self.state == VehicleState.LOADING:
            return True
        elif self.task_type == TaskType.UNLOADING and self.state == VehicleState.UNLOADING:
            return True
        elif self.task_type == TaskType.RETURNING and self.state == VehicleState.IDLE:
            return True
        return False
        
    def get_color(self):
        """获取车辆状态对应的颜色"""
        state_name = self.state.name.lower()
        return COLORS['vehicle'].get(state_name, COLORS['vehicle']['idle'])

# 调度系统
class DispatchSystem:
    def __init__(self, map_size=100):
        self.map_size = map_size
        self.vehicles = {}
        self.loading_points = []
        self.unloading_points = []
        self.parking_point = None
        self.obstacles = set()
        self.task_queue = Queue()
        self.active_tasks = {}
        self.completed_tasks = []
        self.system_time = 0
        
        # 统计信息
        self.stats = {
            'tasks_completed': 0,
            'tasks_generated': 0,
            'vehicle_idle_time': 0,
            'total_distance': 0,
        }
        
    def setup_environment(self):
        """设置地图环境"""
        # 设置装载点
        self.loading_points = [
            (20, 80),   # 左上
            (80, 80),   # 右上
        ]
        
        # 设置卸载点
        self.unloading_points = [
            (50, 20),   # 底部中间
        ]
        
        # 设置停车场
        self.parking_point = (50, 50)  # 中心
        
        # 生成障碍物
        self._generate_obstacles()
        
    def _generate_obstacles(self):
        """生成地图障碍物"""
        # 清空现有障碍物
        self.obstacles = set()
        
        # 障碍区1：左侧垂直障碍带
        for x in range(30, 40):
            for y in range(10, 70):
                self.obstacles.add((x, y))
        
        # 障碍区2：右侧垂直障碍带
        for x in range(60, 70):
            for y in range(30, 90):
                self.obstacles.add((x, y))
                
        # 障碍区3：底部水平障碍带
        for x in range(10, 90):
            for y in range(30, 40):
                # 在中间留一个通道
                if not (45 <= x <= 55):
                    self.obstacles.add((x, y))
                    
        # 随机障碍物
        for _ in range(50):
            x = random.randint(5, self.map_size-5)
            y = random.randint(5, self.map_size-5)
            
            # 确保不会挡住关键点
            min_distance = 10
            if all(math.dist((x, y), point) > min_distance for point in self.loading_points + self.unloading_points + [self.parking_point]):
                # 添加小型障碍群
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        self.obstacles.add((x+dx, y+dy))
        
    def add_vehicle(self, vehicle):
        """添加车辆"""
        self.vehicles[vehicle.vehicle_id] = vehicle
        
    def generate_task(self):
        """生成随机任务"""
        for vid, vehicle in self.vehicles.items():
            if vehicle.state == VehicleState.IDLE:
                # 如果车辆空闲，分配新任务
                loading_point = random.choice(self.loading_points)
                task = {
                    'vehicle_id': vid,
                    'task_type': TaskType.LOADING,
                    'target': loading_point,
                    'created_time': self.system_time
                }
                self.task_queue.put(task)
                self.stats['tasks_generated'] += 1
            elif vehicle.state == VehicleState.LOADING:
                # 如果车辆正在装载，接下来去卸载
                unloading_point = random.choice(self.unloading_points)
                task = {
                    'vehicle_id': vid,
                    'task_type': TaskType.UNLOADING,
                    'target': unloading_point,
                    'created_time': self.system_time
                }
                self.task_queue.put(task)
                self.stats['tasks_generated'] += 1
            elif vehicle.state == VehicleState.UNLOADING:
                # 如果车辆正在卸载，接下来返回停车场
                task = {
                    'vehicle_id': vid,
                    'task_type': TaskType.RETURNING,
                    'target': self.parking_point,
                    'created_time': self.system_time
                }
                self.task_queue.put(task)
                self.stats['tasks_generated'] += 1
                
    def dispatch_tasks(self):
        """分配任务给车辆"""
        # 从队列中取出任务并分配
        while not self.task_queue.empty():
            task = self.task_queue.get()
            vehicle_id = task['vehicle_id']
            
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles[vehicle_id]
                
                # 检查车辆是否已准备好接受新任务
                if ((task['task_type'] == TaskType.LOADING and vehicle.state == VehicleState.IDLE) or
                    (task['task_type'] == TaskType.UNLOADING and vehicle.state == VehicleState.LOADING) or
                    (task['task_type'] == TaskType.RETURNING and vehicle.state == VehicleState.UNLOADING)):
                    
                    # 分配任务
                    success = vehicle.assign_task(task['task_type'], task['target'])
                    
                    if success:
                        # 添加到活动任务列表
                        task_id = f"TASK-{self.stats['tasks_generated']}-{vehicle_id}"
                        self.active_tasks[task_id] = {
                            **task,
                            'task_id': task_id,
                            'status': 'active',
                            'start_time': self.system_time
                        }
                        logger.info(f"任务 {task_id} 已分配给车辆 {vehicle_id}")
                    else:
                        # 如果分配失败，放回队列
                        self.task_queue.put(task)
                else:
                    # 车辆未准备好，放回队列
                    self.task_queue.put(task)
    
    def update_system(self):
        """更新系统状态"""
        self.system_time += 1
        
        # 更新车辆位置
        for vid, vehicle in self.vehicles.items():
            vehicle.update()
            
            # 检查任务是否完成
            if vehicle.is_task_completed():
                for task_id, task in list(self.active_tasks.items()):
                    if task['vehicle_id'] == vid and task['task_type'] == vehicle.task_type:
                        # 更新任务状态为已完成
                        task['status'] = 'completed'
                        task['end_time'] = self.system_time
                        self.completed_tasks.append(task)
                        del self.active_tasks[task_id]
                        self.stats['tasks_completed'] += 1
                        logger.info(f"任务 {task_id} 已完成")
                        break
        
        # 生成新任务
        self.generate_task()
        
        # 分配任务
        self.dispatch_tasks()
        
    def get_system_status(self):
        """获取系统状态信息"""
        vehicle_states = {}
        for state in VehicleState:
            vehicle_states[state.name] = 0
            
        for vehicle in self.vehicles.values():
            vehicle_states[vehicle.state.name] += 1
            
        return {
            'system_time': self.system_time,
            'active_tasks': len(self.active_tasks),
            'completed_tasks': self.stats['tasks_completed'],
            'vehicle_states': vehicle_states,
            'stats': self.stats
        }

# 可视化应用类
class DispatchVisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("露天矿多车协同调度系统可视化")
        self.root.geometry("1200x800")
        
        # 创建调度系统
        self.dispatch_system = DispatchSystem(map_size=100)
        self.dispatch_system.setup_environment()
        
        # 添加车辆
        self.setup_vehicles()
        
        # 设置UI
        self.setup_ui()
        
        # 动画控制
        self.animation_speed = 100  # 毫秒
        self.is_running = False
        self.animation_job = None
        
    def setup_vehicles(self):
        """初始化车辆"""
        # 在停车场创建四辆车
        parking = self.dispatch_system.parking_point
        
        for i in range(1, 5):
            # 稍微错开初始位置
            pos = (parking[0] + (i-2.5)*3, parking[1] + (i-2.5)*3)
            vehicle = Vehicle(i, initial_pos=pos)
            self.dispatch_system.add_vehicle(vehicle)
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建左侧地图和控制区域
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 创建地图画布
        self.fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加工具栏
        toolbar_frame = ttk.Frame(left_frame)
        toolbar_frame.pack(fill=tk.X, padx=5)
        
        ttk.Button(toolbar_frame, text="开始", command=self.start_animation).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(toolbar_frame, text="暂停", command=self.pause_animation).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(toolbar_frame, text="重置", command=self.reset_simulation).pack(side=tk.LEFT, padx=5, pady=5)
        
        # 速度控制
        ttk.Label(toolbar_frame, text="动画速度:").pack(side=tk.LEFT, padx=(20, 5), pady=5)
        speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(toolbar_frame, from_=0.1, to=3.0, variable=speed_var, orient=tk.HORIZONTAL, length=100)
        speed_scale.pack(side=tk.LEFT, padx=5, pady=5)
        speed_scale.bind("<Motion>", lambda e: self.update_animation_speed(speed_var.get()))
        
        # 创建右侧状态面板
        right_frame = ttk.Frame(main_frame, width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        # 系统状态
        system_frame = ttk.LabelFrame(right_frame, text="系统状态", padding="10")
        system_frame.pack(fill=tk.X, pady=5)
        
        self.status_text = tk.Text(system_frame, height=10, width=30, wrap=tk.WORD)
        self.status_text.pack(fill=tk.X)
        
        # 车辆状态
        vehicle_frame = ttk.LabelFrame(right_frame, text="车辆状态", padding="10")
        vehicle_frame.pack(fill=tk.X, pady=5)
        
        self.vehicle_text = tk.Text(vehicle_frame, height=10, width=30, wrap=tk.WORD)
        self.vehicle_text.pack(fill=tk.X)
        
        # 任务状态
        task_frame = ttk.LabelFrame(right_frame, text="任务状态", padding="10")
        task_frame.pack(fill=tk.X, pady=5)
        
        self.task_text = tk.Text(task_frame, height=10, width=30, wrap=tk.WORD)
        self.task_text.pack(fill=tk.X)
        
        # 绘制初始地图
        self.draw_map()
        
    def draw_map(self):
        """绘制地图"""
        self.ax.clear()
        
        # 设置坐标范围
        self.ax.set_xlim(0, self.dispatch_system.map_size)
        self.ax.set_ylim(0, self.dispatch_system.map_size)
        
        # 设置标题
        self.ax.set_title("露天矿调度系统地图", fontsize=14)
        
        # 绘制网格
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制障碍物
        obstacles_x = [point[0] for point in self.dispatch_system.obstacles]
        obstacles_y = [point[1] for point in self.dispatch_system.obstacles]
        if obstacles_x:
            self.ax.scatter(obstacles_x, obstacles_y, color=COLORS['obstacle'], 
                           marker='s', s=40, alpha=0.8, label='障碍物')
        
        # 绘制装载点
        for i, point in enumerate(self.dispatch_system.loading_points):
            self.ax.scatter(point[0], point[1], color=COLORS['loading'], 
                           marker='s', s=150, edgecolor='black', linewidth=1.5,
                           label='装载点' if i == 0 else "")
            self.ax.text(point[0], point[1], f'L{i+1}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        # 绘制卸载点
        for i, point in enumerate(self.dispatch_system.unloading_points):
            self.ax.scatter(point[0], point[1], color=COLORS['unloading'], 
                           marker='s', s=150, edgecolor='black', linewidth=1.5,
                           label='卸载点')
            self.ax.text(point[0], point[1], f'U{i+1}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        # 绘制停车场
        parking = self.dispatch_system.parking_point
        self.ax.scatter(parking[0], parking[1], color=COLORS['parking'], 
                       marker='s', s=150, edgecolor='black', linewidth=1.5,
                       label='停车场')
        self.ax.text(parking[0], parking[1], f'P', 
                    ha='center', va='center', color='white', fontweight='bold')
        
        # 绘制车辆
        for vid, vehicle in self.dispatch_system.vehicles.items():
            pos = vehicle.current_location
            
            # 绘制车辆轨迹
            if len(vehicle.history) > 1:
                history_x = [point[0] for point in vehicle.history]
                history_y = [point[1] for point in vehicle.history]
                self.ax.plot(history_x, history_y, '-', color=vehicle.get_color(), 
                            linewidth=1.5, alpha=0.5)
            
            # 绘制当前路径
            if vehicle.state != VehicleState.IDLE and vehicle.current_path:
                path_x = [point[0] for point in vehicle.current_path[vehicle.path_index:]]
                path_y = [point[1] for point in vehicle.current_path[vehicle.path_index:]]
                self.ax.plot(path_x, path_y, '--', color=COLORS['path'], 
                            linewidth=1.0, alpha=0.7)
            
            # 绘制车辆位置
            self.ax.scatter(pos[0], pos[1], color=vehicle.get_color(), 
                           marker='o', s=150, edgecolor='black', linewidth=1.5,
                           label=f'车辆' if vid == 1 else "")
            self.ax.text(pos[0], pos[1], f'{vid}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        # 添加图例
        self.ax.legend(loc='upper right')
        
        # 更新画布
        self.canvas.draw()
        
    def update_status_panels(self):
        """更新状态面板信息"""
        status = self.dispatch_system.get_system_status()
        
        # 系统状态信息
        system_info = [
            f"系统时间: {status['system_time']}",
            f"活动任务: {status['active_tasks']}",
            f"完成任务: {status['completed_tasks']}",
            f"\n车辆状态统计:",
        ]
        
        for state, count in status['vehicle_states'].items():
            if count > 0:
                system_info.append(f"- {state}: {count}")
        
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, "\n".join(system_info))
        
        # 车辆状态信息
        vehicle_info = []
        for vid, vehicle in self.dispatch_system.vehicles.items():
            task_status = "无任务" if vehicle.state == VehicleState.IDLE else f"{vehicle.task_type.name}"
            vehicle_info.append(f"车辆 {vid}: {vehicle.state.name} | {task_status}")
            
        self.vehicle_text.delete(1.0, tk.END)
        self.vehicle_text.insert(tk.END, "\n".join(vehicle_info))
        
        # 任务状态信息
        task_info = ["活动任务:"]
        for task_id, task in self.dispatch_system.active_tasks.items():
            task_info.append(f"- {task_id}: 车辆{task['vehicle_id']} → {task['task_type'].name}")
            
        if self.dispatch_system.completed_tasks:
            task_info.append("\n最近完成的任务:")
            for task in self.dispatch_system.completed_tasks[-5:]:
                task_info.append(f"- {task['task_id']}: 车辆{task['vehicle_id']} → {task['task_type'].name}")
        
        self.task_text.delete(1.0, tk.END)
        self.task_text.insert(tk.END, "\n".join(task_info))
    
    def update_animation_speed(self, value):
        """更新动画速度"""
        # 反转值使得大值对应快速（低延迟）
        self.animation_speed = int(1000 / value)
        
        # 如果动画正在运行，重新调度
        if self.is_running and self.animation_job:
            self.root.after_cancel(self.animation_job)
            self.animation_job = self.root.after(self.animation_speed, self.animation_step)
    
    def animation_step(self):
        """单步更新动画"""
        if not self.is_running:
            return
            
        try:
            # 更新系统
            self.dispatch_system.update_system()
            
            # 更新绘图
            self.draw_map()
            
            # 更新状态面板
            self.update_status_panels()
            
            # 继续下一步
            self.animation_job = self.root.after(self.animation_speed, self.animation_step)
            
        except Exception as e:
            logger.error(f"动画步骤发生错误: {str(e)}")
            messagebox.showerror("错误", f"动画步骤发生错误: {str(e)}")
            self.pause_animation()
    
    def start_animation(self):
        """开始动画"""
        if not self.is_running:
            self.is_running = True
            self.animation_job = self.root.after(self.animation_speed, self.animation_step)
            logger.info("动画已开始")
    
    def pause_animation(self):
        """暂停动画"""
        if self.is_running:
            self.is_running = False
            if self.animation_job:
                self.root.after_cancel(self.animation_job)
                self.animation_job = None
            logger.info("动画已暂停")
    
    def reset_simulation(self):
        """重置模拟"""
        # 暂停当前动画
        self.pause_animation()
        
        # 重新初始化调度系统
        self.dispatch_system = DispatchSystem(map_size=100)
        self.dispatch_system.setup_environment()
        self.setup_vehicles()
        
        # 重新绘制地图
        self.draw_map()
        
        # 更新状态面板
        self.update_status_panels()
        
        logger.info("模拟已重置")

def main():
    """主入口函数"""
    root = tk.Tk()
    app = DispatchVisualizationApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.pause_animation(), root.destroy()))
    
    # 自动开始动画
    root.after(1000, app.start_animation)
    
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"程序发生错误: {str(e)}", exc_info=True)
        print(f"程序发生错误: {str(e)}")