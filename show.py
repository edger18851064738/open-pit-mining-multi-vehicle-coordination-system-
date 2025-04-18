"""
增强型露天矿多车协同调度系统可视化模拟
"""
import os
import sys
import math
import random
import configparser
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import threading
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入项目模块
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from utils.geo_tools import GeoUtils
from algorithm.map_service import MapService
from algorithm.optimized_path_planner import HybridPathPlanner
from algorithm.dispatch_service import DispatchSystem, ConflictBasedSearch, TransportScheduler

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class MineSimulator:
    """露天矿运输系统模拟器"""
    
    def __init__(self):
        self.geo_utils = GeoUtils()
        self.map_service = MapService()
        self.planner = HybridPathPlanner(self.map_service)
        self.dispatch = DispatchSystem(self.planner, self.map_service)
        
        # 模拟参数
        self.config = self._load_config()
        self.sim_speed = self.config.get('simulation_speed', 1.0)
        self.running = False
        self.sim_time = 0
        
        # 初始化障碍物
        self.obstacles = self._generate_obstacles()
        self.planner.obstacle_grids = set(self._flatten_obstacles())
        
        # 初始化车辆
        self.vehicles = []
        self._init_vehicles()
        
        # 模拟线程
        self.sim_thread = None
        self.vis_thread = None
        
        # 可视化相关
        self.fig = None
        self.ax = None
        self.vehicle_plots = {}
        self.path_plots = {}
        self.anim = None
        
        logging.info("模拟器初始化完成")
        
    def _load_config(self) -> dict:
        """加载模拟器配置"""
        config = {
            'grid_size': 200,
            'num_vehicles': 5,
            'simulation_speed': 2.0,
            'task_generation_rate': 0.2,
            'visualization_interval': 0.5,
            'scheduling_interval': 2.0
        }
        
        # 尝试从配置文件加载
        config_path = os.path.join(PROJECT_ROOT, 'config.ini')
        if os.path.exists(config_path):
            try:
                parser = configparser.ConfigParser()
                parser.read(config_path)
                if 'SIMULATION' in parser:
                    for key in config:
                        if key in parser['SIMULATION']:
                            # 尝试转换为适当的类型
                            try:
                                config[key] = eval(parser['SIMULATION'][key])
                            except:
                                config[key] = parser['SIMULATION'][key]
            except Exception as e:
                logging.warning(f"读取配置文件异常: {str(e)}")
                
        return config
        
    def _generate_obstacles(self) -> List[List[Tuple[float, float]]]:
        """生成障碍物"""
        # 固定样式的障碍物布局
        obstacles = [
            [(20,60), (80,60), (80,70), (20,70)],         # 水平障碍墙1
            [(120,60), (180,60), (180,70), (120,70)],     # 水平障碍墙2
            [(40,30), (60,30), (60,40), (40,40)],         # 小障碍物1
            [(140,30), (160,30), (160,40), (140,40)],     # 小障碍物2
            [(90,100), (110,100), (110,120), (90,120)],   # 中央障碍物
            [(30,20), (50,20), (50,180), (30,180)],       # 垂直障碍墙1
            [(150,20), (170,20), (170,180), (150,180)],   # 垂直障碍墙2
            [(50,90), (80,90), (80,110), (50,110)],       # 左侧障碍区
            [(120,90), (150,90), (150,110), (120,110)]    # 右侧障碍区
        ]
        return obstacles
        
    def _flatten_obstacles(self) -> List[Tuple[float, float]]:
        """将障碍物多边形展平为点集"""
        flat_obstacles = []
        for polygon in self.obstacles:
            # 计算多边形内部所有点
            min_x = min(p[0] for p in polygon)
            max_x = max(p[0] for p in polygon)
            min_y = min(p[1] for p in polygon)
            max_y = max(p[1] for p in polygon)
            
            for x in range(int(min_x), int(max_x)+1):
                for y in range(int(min_y), int(max_y)+1):
                    if self._point_in_polygon((x, y), polygon):
                        flat_obstacles.append((x, y))
                        
        return flat_obstacles
        
    def _point_in_polygon(self, point, polygon) -> bool:
        """判断点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        
        for i in range(n):
            p1 = polygon[i]
            p2 = polygon[(i+1)%n]
            
            if (y > min(p1[1], p2[1]) and y <= max(p1[1], p2[1])) and (x <= max(p1[0], p2[0])):
                if p1[1] != p2[1]:
                    x_intersect = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                    if p1[0] == p2[0] or x <= x_intersect:
                        inside = not inside
                        
        return inside
        
    def _init_vehicles(self):
        """初始化车辆"""
        dispatch_config = self.dispatch._load_config()
        parking_area = dispatch_config['parking_area']
        
        num_vehicles = self.config.get('num_vehicles', 3)
        for i in range(1, num_vehicles + 1):
            # 在停车场周围随机位置生成车辆
            offset_x = random.uniform(-10, 10)
            offset_y = random.uniform(-10, 10)
            start_pos = (parking_area[0] + offset_x, parking_area[1] + offset_y)
            
            vehicle = MiningVehicle(
                vehicle_id=i,
                map_service=self.map_service,
                config={
                    'current_location': start_pos,
                    'max_capacity': 50,
                    'max_speed': random.uniform(5.0, 8.0),  # 随机速度
                    'base_location': parking_area,
                    'status': VehicleState.IDLE
                }
            )
            
            self.vehicles.append(vehicle)
            self.dispatch.vehicles[i] = vehicle
            
        logging.info(f"已初始化{num_vehicles}辆车辆")
        
    def _generate_initial_tasks(self):
        """生成初始任务"""
        num_tasks = len(self.vehicles)  # 与车辆数量相同的初始任务
        dispatch_config = self.dispatch._load_config()
        
        for i in range(1, num_tasks + 1):
            # 随机选择装载点
            loading_point = random.choice(dispatch_config['loading_points'])
            
            task = TransportTask(
                task_id=f"INITIAL-{i}",
                start_point=loading_point,
                end_point=dispatch_config['unloading_point'],
                task_type="loading",
                priority=random.randint(1, 3)
            )
            
            self.dispatch.add_task(task)
            
        logging.info(f"已生成{num_tasks}个初始任务")
        
    def _generate_random_task(self) -> bool:
        """按概率生成随机任务"""
        if random.random() < self.config.get('task_generation_rate', 0.2):
            dispatch_config = self.dispatch._load_config()
            task_type = random.choice(["loading", "unloading"])
            
            if task_type == "loading":
                start_point = random.choice(dispatch_config['loading_points'])
                end_point = dispatch_config['unloading_point']
            else:
                start_point = dispatch_config['unloading_point']
                end_point = dispatch_config['parking_area']
                
            task = TransportTask(
                task_id=f"{task_type.upper()}-{int(time.time() * 1000) % 10000}",
                start_point=start_point,
                end_point=end_point,
                task_type=task_type,
                priority=random.randint(1, 3)
            )
            
            self.dispatch.add_task(task)
            logging.debug(f"已生成随机任务: {task.task_id}")
            return True
            
        return False
        
    def start_simulation(self):
        """启动模拟"""
        if self.running:
            logging.warning("模拟已在运行中")
            return
            
        self.running = True
        
        # 创建初始任务
        self._generate_initial_tasks()
        
        # 启动模拟线程
        self.sim_thread = threading.Thread(target=self._simulation_loop)
        self.sim_thread.daemon = True
        self.sim_thread.start()
        
        # 启动调度线程
        scheduling_thread = threading.Thread(
            target=self.dispatch.start_scheduling,
            args=(self.config.get('scheduling_interval', 2.0),)
        )
        scheduling_thread.daemon = True
        scheduling_thread.start()
        
        logging.info("模拟系统已启动")
        
        # 启动可视化
        self.start_visualization()
        
    def stop_simulation(self):
        """停止模拟"""
        self.running = False
        self.dispatch.running = False
        logging.info("模拟系统已停止")
        
    def _simulation_loop(self):
        """模拟主循环"""
        last_task_time = time.time()
        last_status_time = time.time()
        
        while self.running:
            # 更新模拟时间
            self.sim_time += self.config.get('simulation_speed', 1.0)
            
            # 定期生成随机任务
            current_time = time.time()
            if current_time - last_task_time > 5:  # 每5秒尝试生成新任务
                if self._generate_random_task():
                    last_task_time = current_time
            
            # 定期打印状态
            if current_time - last_status_time > 10:  # 每10秒打印一次状态
                self.dispatch.print_system_status()
                last_status_time = current_time
                
            # 等待下一帧
            time.sleep(0.1)
            
    def start_visualization(self):
        """启动可视化"""
        # 在主线程中初始化图形
        plt.ion()  # 开启交互模式
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(-50, 250)
        self.ax.set_ylim(-50, 250)
        self.ax.set_title('露天矿多车协同调度系统')
        self.ax.set_xlabel('X坐标')
        self.ax.set_ylabel('Y坐标')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # 绘制障碍物
        for polygon in self.obstacles:
            poly = plt.Polygon(polygon, facecolor='gray', alpha=0.5)
            self.ax.add_patch(poly)
            
        # 绘制关键点
        dispatch_config = self.dispatch._load_config()
        
        # 绘制装载点
        for i, lp in enumerate(dispatch_config['loading_points']):
            self.ax.plot(lp[0], lp[1], 'go', markersize=12, label=f'装载点{i+1}' if i==0 else "")
            self.ax.text(lp[0]+5, lp[1]+5, f'装载点{i+1}', fontsize=10)
            
        # 绘制卸载点
        unload = dispatch_config['unloading_point']
        self.ax.plot(unload[0], unload[1], 'rs', markersize=12, label='卸载点')
        self.ax.text(unload[0]+5, unload[1]+5, '卸载点', fontsize=10)
        
        # 绘制停车场
        parking = dispatch_config['parking_area']
        self.ax.plot(parking[0], parking[1], 'b^', markersize=12, label='停车场')
        self.ax.text(parking[0]+5, parking[1]+5, '停车场', fontsize=10)
        
        # 车辆图例
        self.vehicle_plots = {}
        for i, vehicle in enumerate(self.vehicles):
            color = plt.cm.tab10(i % 10)
            vehicle_plot, = self.ax.plot(
                vehicle.current_location[0], 
                vehicle.current_location[1], 
                'o', 
                color=color,
                markersize=10,
                label=f'车辆{vehicle.vehicle_id}'
            )
            self.vehicle_plots[vehicle.vehicle_id] = {
                'plot': vehicle_plot,
                'color': color,
                'path_plot': None,
                'label': None
            }
            
        # 添加图例
        self.ax.legend(loc='upper right')
            
        # 启动更新线程（只负责准备数据，不直接操作图形）
        self.vis_thread = threading.Thread(target=self._prepare_visualization_data)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def _prepare_visualization_data(self):
        """准备可视化数据（在线程中运行）"""
        try:
            last_update = time.time()
            while self.running:
                current_time = time.time()
                if current_time - last_update >= self.config.get('visualization_interval', 0.5):
                    # 只收集数据，不直接更新图形
                    self._collect_visualization_data()
                    last_update = current_time
                    
                time.sleep(0.1)
        except Exception as e:
            logging.error(f"可视化数据准备异常: {str(e)}")

    def _collect_visualization_data(self):
        """收集可视化数据"""
        # 设置一个标志，表示有新数据需要在主线程中更新
        self.visualization_data_ready = True

    def _update_visualization(self):
        """更新可视化（在主线程中调用）"""
        if not hasattr(self, 'visualization_data_ready') or not self.visualization_data_ready:
            return
            
        try:
            # 更新车辆位置
            for vid, vehicle in self.dispatch.vehicles.items():
                if vid in self.vehicle_plots:
                    # 更新车辆位置
                    self.vehicle_plots[vid]['plot'].set_data(
                        vehicle.current_location[0], 
                        vehicle.current_location[1]
                    )
                    
                    # 更新状态标签
                    if self.vehicle_plots[vid]['label'] is None:
                        self.vehicle_plots[vid]['label'] = self.ax.text(
                            vehicle.current_location[0], 
                            vehicle.current_location[1] + 5,
                            f"{vid}: {vehicle.state.name}",
                            fontsize=8,
                            color=self.vehicle_plots[vid]['color']
                        )
                    else:
                        self.vehicle_plots[vid]['label'].set_position(
                            (vehicle.current_location[0], vehicle.current_location[1] + 5)
                        )
                        self.vehicle_plots[vid]['label'].set_text(
                            f"{vid}: {vehicle.state.name}"
                        )
                    
                    # 更新路径
                    if hasattr(vehicle, 'current_path') and vehicle.current_path:
                        path_x = [p[0] for p in vehicle.current_path]
                        path_y = [p[1] for p in vehicle.current_path]
                        
                        if self.vehicle_plots[vid]['path_plot'] is None:
                            path_plot, = self.ax.plot(
                                path_x, path_y, '--', 
                                color=self.vehicle_plots[vid]['color'],
                                alpha=0.5,
                                linewidth=1
                            )
                            self.vehicle_plots[vid]['path_plot'] = path_plot
                        else:
                            self.vehicle_plots[vid]['path_plot'].set_data(path_x, path_y)
                    elif self.vehicle_plots[vid]['path_plot'] is not None:
                        # 清除路径
                        self.vehicle_plots[vid]['path_plot'].set_data([], [])
            
            # 更新仿真时间
            self.ax.set_title(f'露天矿多车协同调度系统 - 仿真时间: {self.sim_time:.1f}s')
            
            # 刷新图形
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
            # 重置标志
            self.visualization_data_ready = False
            
        except Exception as e:
            logging.error(f"更新可视化异常: {str(e)}")

def main():
    """主函数"""
    print("=" * 60)
    print("   露天矿多车协同调度系统模拟 v1.0")
    print("=" * 60)
    print("\n初始化中...")
    
    simulator = MineSimulator()
    
    try:
        simulator.start_simulation()
        
        # 主线程等待，接收用户输入的命令
        while simulator.running:
            try:
                # 更新可视化（在主线程中）
                if hasattr(simulator, 'fig') and simulator.fig:
                    simulator._update_visualization()
                    plt.pause(0.01)  # 短暂暂停以允许GUI事件处理
                
                # 非阻塞方式检查是否有输入
                import msvcrt
                if msvcrt.kbhit():
                    cmd = input("\n输入命令(help查看帮助)> ").strip().lower()
                    
                    if cmd == 'quit' or cmd == 'exit':
                        simulator.stop_simulation()
                        break
                    elif cmd == 'help':
                        print("\n可用命令:")
                        print("  status  - 显示系统状态")
                        print("  map     - 显示ASCII地图")
                        print("  add     - 添加随机任务")
                        print("  move id x y - 移动指定车辆到坐标")
                        print("  quit    - 退出模拟")
                    elif cmd == 'status':
                        simulator.dispatch.print_system_status()
                    elif cmd == 'map':
                        simulator.dispatch.print_ascii_map()
                    elif cmd == 'add':
                        simulator._generate_random_task()
                        print("已添加随机任务")
                    elif cmd.startswith('move '):
                        parts = cmd.split()
                        if len(parts) == 4:
                            try:
                                vid = int(parts[1])
                                x = float(parts[2])
                                y = float(parts[3])
                                simulator.dispatch.dispatch_vehicle_to(vid, (x, y))
                                print(f"已调度车辆{vid}前往({x}, {y})")
                            except ValueError:
                                print("参数错误：请确保输入正确的车辆ID和坐标")
                            except Exception as e:
                                print(f"调度失败：{str(e)}")
                        else:
                            print("格式错误：使用'move 车辆ID x坐标 y坐标'")
                    else:
                        print("未知命令，输入'help'查看帮助")
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n接收到中断信号，退出模拟...")
                simulator.stop_simulation()
                break
                
    except Exception as e:
        logging.error(f"模拟运行异常: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        simulator.stop_simulation()
        if hasattr(simulator, 'fig') and simulator.fig:
            plt.close(simulator.fig)
        print("\n模拟已停止")

if __name__ == "__main__":
    main()