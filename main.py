"""
露天矿多车协同调度系统集成方案
"""
import os
import sys
import argparse
import configparser
import logging
import threading
import time
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入项目模块
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from utils.geo_tools import GeoUtils
from algorithm.map_service import MapService
from algorithm.path_planner import HybridPathPlanner
from algorithm.dispatch_service import DispatchSystem, ConflictBasedSearch, TransportScheduler

class MineDispatchSystem:
    """露天矿调度系统集成类"""
    
    def __init__(self, config_path=None):
        # 初始化配置
        self.config = self._load_config(config_path)
        self._configure_logging()
        
        logging.info("初始化露天矿调度系统")
        
        # 初始化坐标工具
        self.geo_utils = GeoUtils()
        
        # 初始化核心组件
        self.map_service = MapService()  # 地图服务
        self.path_planner = HybridPathPlanner(self.map_service)  # 路径规划器
        self.dispatch = DispatchSystem(self.path_planner, self.map_service)  # 调度系统
        
        # 系统状态
        self.running = False
        self.simulation_mode = self.config.getboolean('SIMULATION', 'simulation_mode', fallback=True)
        
        # 控制线程
        self.dispatch_thread = None
        self.simulation_thread = None
        self.visualization_thread = None
        
        # 可视化相关
        self.fig = None
        self.ax = None
        self.vehicle_plots = {}
        
        logging.info("调度系统初始化完成")
        
    def _load_config(self, config_path=None) -> configparser.ConfigParser:
        """加载配置文件"""
        config = configparser.ConfigParser()
        
        # 默认配置文件路径
        if not config_path:
            config_path = os.path.join(PROJECT_ROOT, 'config.ini')
            
        # 如果配置文件不存在，创建默认配置
        if not os.path.exists(config_path):
            self._create_default_config(config_path)
            
        # 读取配置文件
        config.read(config_path)
        
        return config
        
    def _create_default_config(self, config_path):
        """创建默认配置文件"""
        config = configparser.ConfigParser()
        
        # MAP部分
        config['MAP'] = {
            'grid_size': '200',
            'grid_nodes': '50',
            'safe_radius': '30',
            'obstacle_density': '0.15',
            'data_type': 'virtual',
            'virtual_origin_x': '0',
            'virtual_origin_y': '0',
            'max_grade': '15.0',
            'min_turn_radius': '15.0'
        }
        
        # DISPATCH部分
        config['DISPATCH'] = {
            'loading_points': '[(-100,50), (0,150), (100,50)]',
            'unloading_point': '(0,-100)',
            'parking_area': '(200,200)',
            'max_charging_vehicles': '2',
            'scheduling_interval': '2.0',
            'conflict_resolution_method': 'priority',
            'task_assignment_method': 'nearest'
        }
        
        # VEHICLE部分
        config['VEHICLE'] = {
            'default_speed': '5.0',
            'default_capacity': '50',
            'default_hardness': '2.5',
            'default_turning_radius': '10.0',
            'battery_capacity': '100.0',
            'power_consumption': '2.0',
            'maintenance_interval': '500'
        }
        
        # TASK部分
        config['TASK'] = {
            'default_priority': '1',
            'default_weight': '50000',
            'deadline_hours': '2',
            'max_retries': '3'
        }
        
        # SIMULATION部分
        config['SIMULATION'] = {
            'simulation_mode': 'True',
            'num_vehicles': '5',
            'simulation_speed': '2.0',
            'task_generation_rate': '0.2',
            'visualization_interval': '0.5',
            'scheduling_interval': '2.0',
            'random_seed': '42'
        }
        
        # LOGGING部分
        config['LOGGING'] = {
            'level': 'INFO',
            'console_output': 'True',
            'file_output': 'True',
            'log_file': 'dispatch.log',
            'rotate_logs': 'True',
            'max_file_size': '10485760',
            'backup_count': '5'
        }
        
        # 写入配置文件
        with open(config_path, 'w') as f:
            config.write(f)
            
        print(f"已创建默认配置文件: {config_path}")
        
    def _configure_logging(self):
        """配置日志系统"""
        log_level_str = self.config.get('LOGGING', 'level', fallback='INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # 配置根日志记录器
        logging.basicConfig(level=log_level, format=log_format)
        
        # 如果需要文件输出
        if self.config.getboolean('LOGGING', 'file_output', fallback=True):
            log_file = self.config.get('LOGGING', 'log_file', fallback='dispatch.log')
            file_path = os.path.join(PROJECT_ROOT, log_file)
            
            # 如果需要循环日志
            if self.config.getboolean('LOGGING', 'rotate_logs', fallback=True):
                from logging.handlers import RotatingFileHandler
                
                max_size = self.config.getint('LOGGING', 'max_file_size', fallback=10485760)
                backup_count = self.config.getint('LOGGING', 'backup_count', fallback=5)
                
                file_handler = RotatingFileHandler(
                    file_path, maxBytes=max_size, backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(file_path)
                
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(log_format))
            
            # 添加到根日志记录器
            logging.getLogger('').addHandler(file_handler)
            
    def start(self):
        """启动调度系统"""
        if self.running:
            logging.warning("系统已在运行中")
            return
            
        self.running = True
        
        # 设置随机种子
        random_seed = self.config.getint('SIMULATION', 'random_seed', fallback=42)
        random.seed(random_seed)
        
        if self.simulation_mode:
            # 初始化模拟环境
            self._init_simulation()
            
            # 启动模拟线程
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
        # 启动调度线程
        scheduling_interval = self.config.getfloat('DISPATCH', 'scheduling_interval', fallback=2.0)
        self.dispatch_thread = threading.Thread(
            target=self.dispatch.start_scheduling,
            args=(scheduling_interval,)
        )
        self.dispatch_thread.daemon = True
        self.dispatch_thread.start()
        
        # 启动可视化（如果在模拟模式下）
        if self.simulation_mode:
            self.start_visualization()
            
        logging.info("调度系统已启动")
        
    def stop(self):
        """停止调度系统"""
        if not self.running:
            logging.warning("系统未在运行")
            return
            
        self.running = False
        self.dispatch.running = False
        
        logging.info("调度系统已停止")
        
    def _init_simulation(self):
        """初始化模拟环境"""
        # 生成障碍物
        self._generate_obstacles()
        
        # 初始化车辆
        self._init_vehicles()
        
        # 生成初始任务
        self._generate_initial_tasks()
        
        logging.info("模拟环境初始化完成")
        
    def _generate_obstacles(self):
        """生成障碍物"""
        # 预定义一些障碍物
        obstacles = [
            [(20,60), (80,60), (80,70), (20,70)],  # 水平障碍墙1
            [(120,60), (180,60), (180,70), (120,70)],  # 水平障碍墙2
            [(40,30), (60,30), (60,40), (40,40)],  # 小障碍物1
            [(140,30), (160,30), (160,40), (140,40)],  # 小障碍物2
            [(90,100), (110,100), (110,120), (90,120)],  # 中央障碍物
            [(30,20), (50,20), (50,180), (30,180)],  # 垂直障碍墙1
            [(150,20), (170,20), (170,180), (150,180)],  # 垂直障碍墙2
            [(50,90), (80,90), (80,110), (50,110)],  # 左侧障碍区
            [(120,90), (150,90), (150,110), (120,110)]  # 右侧障碍区
        ]
        
        # 处理每个障碍物多边形
        flat_obstacles = []
        for polygon in obstacles:
            # 计算多边形内部所有点
            min_x = min(p[0] for p in polygon)
            max_x = max(p[0] for p in polygon)
            min_y = min(p[1] for p in polygon)
            max_y = max(p[1] for p in polygon)
            
            for x in range(int(min_x), int(max_x)+1):
                for y in range(int(min_y), int(max_y)+1):
                    if self._point_in_polygon((x, y), polygon):
                        flat_obstacles.append((x, y))
                        
        # 将障碍物设置到路径规划器中
        self.path_planner.obstacle_grids = set(flat_obstacles)
        
        logging.debug(f"已生成{len(flat_obstacles)}个障碍物点")
        
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
        
        # 从配置读取车辆数量和属性
        num_vehicles = self.config.getint('SIMULATION', 'num_vehicles', fallback=5)
        default_speed = self.config.getfloat('VEHICLE', 'default_speed', fallback=5.0)
        default_capacity = self.config.getfloat('VEHICLE', 'default_capacity', fallback=50)
        default_hardness = self.config.getfloat('VEHICLE', 'default_hardness', fallback=2.5)
        default_turning_radius = self.config.getfloat('VEHICLE', 'default_turning_radius', fallback=10.0)
        
        for i in range(1, num_vehicles + 1):
            # 在停车场周围随机位置生成车辆
            offset_x = random.uniform(-10, 10)
            offset_y = random.uniform(-10, 10)
            start_pos = (parking_area[0] + offset_x, parking_area[1] + offset_y)
            
            # 创建车辆对象
            vehicle = MiningVehicle(
                vehicle_id=i,
                map_service=self.map_service,
                config={
                    'current_location': start_pos,
                    'max_capacity': default_capacity,
                    'max_speed': random.uniform(default_speed*0.8, default_speed*1.2),  # 添加随机性
                    'min_hardness': default_hardness,
                    'turning_radius': default_turning_radius,
                    'base_location': parking_area,
                    'status': VehicleState.IDLE
                }
            )
            
            # 注册到调度系统
            self.dispatch.vehicles[i] = vehicle
            
        logging.info(f"已初始化{num_vehicles}辆车辆")
        
    def _generate_initial_tasks(self):
        """生成初始任务"""
        dispatch_config = self.dispatch._load_config()
        num_vehicles = self.config.getint('SIMULATION', 'num_vehicles', fallback=5)
        
        # 为每辆车生成一个初始任务
        for i in range(1, num_vehicles + 1):
            # 随机选择装载点
            loading_point = random.choice(dispatch_config['loading_points'])
            
            # 创建装载任务
            task = TransportTask(
                task_id=f"INITIAL-{i}",
                start_point=loading_point,
                end_point=dispatch_config['unloading_point'],
                task_type="loading",
                priority=random.randint(1, 3)
            )
            
            # 添加到调度系统
            self.dispatch.add_task(task)
            
        logging.info(f"已生成{num_vehicles}个初始任务")
        
    def _simulation_loop(self):
        """模拟主循环"""
        sim_speed = self.config.getfloat('SIMULATION', 'simulation_speed', fallback=2.0)
        task_rate = self.config.getfloat('SIMULATION', 'task_generation_rate', fallback=0.2)
        
        last_task_time = time.time()
        last_status_time = time.time()
        
        while self.running:
            # 随机生成任务
            current_time = time.time()
            if current_time - last_task_time > 5:  # 每5秒检查一次
                if random.random() < task_rate:
                    self._generate_random_task()
                    last_task_time = current_time
                    
            # 定期打印状态
            if current_time - last_status_time > 10:  # 每10秒打印一次状态
                self.dispatch.print_system_status()
                last_status_time = current_time
                
            # 等待下一个循环
            time.sleep(0.1 / sim_speed)
            
    def _generate_random_task(self) -> bool:
        """生成随机任务"""
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
        logging.debug(f"已生成随机任务: {task.task_id} ({task_type})")
        return True
        
    def start_visualization(self):
        """启动可视化"""
        if not self.simulation_mode:
            logging.warning("可视化仅在模拟模式下可用")
            return
            
        self.visualization_thread = threading.Thread(target=self._visualization_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        
        logging.info("可视化线程已启动")
        
    def _visualization_loop(self):
        """可视化线程主循环"""
        try:
            # 设置交互模式
            plt.ion()
            
            # 创建图形和坐标轴
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
            self.ax.set_xlim(-50, 250)
            self.ax.set_ylim(-50, 250)
            self.ax.set_title('露天矿多车协同调度系统')
            self.ax.set_xlabel('X坐标')
            self.ax.set_ylabel('Y坐标')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # 绘制障碍物
            for point in self.path_planner.obstacle_grids:
                self.ax.plot(point[0], point[1], 'k.', markersize=1, alpha=0.5)
                
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
            
            # 初始化车辆图形
            self.vehicle_plots = {}
            for i, (vid, vehicle) in enumerate(self.dispatch.vehicles.items()):
                color = plt.cm.tab10(i % 10)
                vehicle_plot, = self.ax.plot(
                    vehicle.current_location[0], 
                    vehicle.current_location[1], 
                    'o', 
                    color=color,
                    markersize=10,
                    label=f'车辆{vid}'
                )
                self.vehicle_plots[vid] = {
                    'plot': vehicle_plot,
                    'color': color,
                    'path_plot': None,
                    'label': None
                }
                
            # 添加图例
            self.ax.legend(loc='upper right')
            
            # 更新循环
            vis_interval = self.config.getfloat('SIMULATION', 'visualization_interval', fallback=0.5)
            last_update = time.time()
            
            while self.running:
                current_time = time.time()
                if current_time - last_update >= vis_interval:
                    self._update_visualization()
                    last_update = current_time
                    
                plt.pause(0.1)
                
        except Exception as e:
            logging.error(f"可视化异常: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            plt.ioff()
            
    def _update_visualization(self):
        """更新可视化"""
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
            
            # 更新标题（显示任务数量）
            active_tasks = len(self.dispatch.active_tasks)
            queued_tasks = len(self.dispatch.task_queue)
            completed_tasks = len(self.dispatch.completed_tasks)
            self.ax.set_title(f'露天矿多车协同调度系统 - 活动任务: {active_tasks} | 队列: {queued_tasks} | 已完成: {completed_tasks}')
            
            # 刷新图形
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            logging.error(f"更新可视化异常: {str(e)}")
            
    def export_statistics(self, output_file=None):
        """导出系统运行统计信息"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(PROJECT_ROOT, f"stats_{timestamp}.txt")
            
        # 收集统计信息
        stats = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'runtime': "运行中" if self.running else "已停止",
            'vehicles': len(self.dispatch.vehicles),
            'active_tasks': len(self.dispatch.active_tasks),
            'completed_tasks': len(self.dispatch.completed_tasks),
            'conflicts': self.dispatch.performance_metrics.get('conflict_count', 0),
            'resolved_conflicts': self.dispatch.performance_metrics.get('resolved_conflicts', 0),
        }
        
        # 车辆统计
        vehicle_stats = []
        for vid, vehicle in self.dispatch.vehicles.items():
            v_stat = {
                'id': vid,
                'state': vehicle.state.name,
                'mileage': vehicle.mileage,
                'tasks_completed': vehicle.metrics.get('tasks_completed', 0),
                'conflicts': vehicle.metrics.get('conflicts', 0)
            }
            vehicle_stats.append(v_stat)
            
        # 写入文件
        with open(output_file, 'w') as f:
            f.write("===== 露天矿调度系统统计信息 =====\n\n")
            
            f.write("系统概况:\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
                
            f.write("\n车辆统计:\n")
            for v in vehicle_stats:
                f.write(f"车辆 {v['id']} - 状态:{v['state']} | 里程:{v['mileage']:.1f}km | "
                       f"完成任务:{v['tasks_completed']} | 冲突:{v['conflicts']}\n")
                
        logging.info(f"统计信息已导出至: {output_file}")
        return output_file
        
    def run_interactive(self):
        """运行交互式命令行界面"""
        print("=" * 60)
        print("   露天矿多车协同调度系统 v1.0")
        print("=" * 60)
        
        try:
            self.start()
            print("\n系统已启动，输入 'help' 查看可用命令")
            
            while self.running:
                try:
                    cmd = input("\n> ").strip().lower()
                    
                    if cmd == 'quit' or cmd == 'exit':
                        self.stop()
                        break
                    elif cmd == 'help':
                        print("\n可用命令:")
                        print("  status  - 显示系统状态")
                        print("  map     - 显示ASCII地图")
                        print("  add     - 添加随机任务")
                        print("  stats   - 导出统计信息")
                        print("  move id x y - 移动指定车辆到坐标")
                        print("  quit    - 退出系统")
                    elif cmd == 'status':
                        self.dispatch.print_system_status()
                    elif cmd == 'map':
                        self.dispatch.print_ascii_map()
                    elif cmd == 'add':
                        self._generate_random_task()
                        print("已添加随机任务")
                    elif cmd == 'stats':
                        output_file = self.export_statistics()
                        print(f"统计信息已导出至: {output_file}")
                    elif cmd.startswith('move '):
                        parts = cmd.split()
                        if len(parts) == 4:
                            try:
                                vid = int(parts[1])
                                x = float(parts[2])
                                y = float(parts[3])
                                self.dispatch.dispatch_vehicle_to(vid, (x, y))
                                print(f"已调度车辆{vid}前往({x}, {y})")
                            except ValueError:
                                print("参数错误：请确保输入正确的车辆ID和坐标")
                            except Exception as e:
                                print(f"调度失败：{str(e)}")
                        else:
                            print("格式错误：使用'move 车辆ID x坐标 y坐标'")
                    else:
                        print("未知命令，输入'help'查看帮助")
                        
                except KeyboardInterrupt:
                    print("\n接收到中断信号，正在停止系统...")
                    self.stop()
                    break
                    
        except Exception as e:
            logging.error(f"系统运行异常: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
            print("\n系统已停止")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='露天矿多车协同调度系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--simulation', action='store_true', help='启用模拟模式')
    parser.add_argument('--vehicles', type=int, help='模拟车辆数量')
    parser.add_argument('--speed', type=float, help='模拟速度')
    parser.add_argument('--stats-only', action='store_true', help='仅导出统计信息后退出')
    
    args = parser.parse_args()
    
    # 创建系统实例
    system = MineDispatchSystem(config_path=args.config)
    
    # 应用命令行参数
    if args.simulation:
        system.simulation_mode = True
    if args.vehicles:
        system.config.set('SIMULATION', 'num_vehicles', str(args.vehicles))
    if args.speed:
        system.config.set('SIMULATION', 'simulation_speed', str(args.speed))
        
    # 运行系统
    if args.stats_only:
        # 仅导出统计信息
        system.start()
        time.sleep(5)  # 等待系统初始化
        system.export_statistics()
        system.stop()
    else:
        # 运行交互式界面
        system.run_interactive()

if __name__ == "__main__":
    main()