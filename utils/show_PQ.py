#!/usr/bin/env python3
"""
HybridPathPlanner 增强可视化测试工具 - PyQtGraph版

使用PyQtGraph进行高性能可视化，解决Matplotlib性能不足问题
"""

import os
import sys
import time
import math
import random
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
import threading
import argparse

# PyQt和PyQtGraph导入
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QGridLayout
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
import pyqtgraph as pg

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 导入项目模块
try:
    from algorithm.optimized_path_planner import HybridPathPlanner
    from algorithm.map_service import MapService
    from algorithm.cbs import ConflictBasedSearch
    from utils.geo_tools import GeoUtils
    from models.vehicle import MiningVehicle, VehicleState
    logging.info("成功导入所需模块")
except ImportError as e:
    logging.error(f"导入模块失败: {str(e)}")
    sys.exit(1)

# 设置PyQtGraph全局配置
pg.setConfigOptions(antialias=True, background='w', foreground='k')

class SimulationThread(QObject):
    """模拟线程，处理路径规划和车辆移动"""
    update_signal = pyqtSignal(dict)  # 发送更新数据的信号
    
    def __init__(self, visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.running = False
    
    def start(self):
        """启动模拟线程"""
        self.running = True
        self._run_simulation()
    
    def stop(self):
        """停止模拟线程"""
        self.running = False
    
    def _run_simulation(self):
        """运行模拟"""
        while self.running:
            # 更新车辆位置
            self.visualizer._update_vehicles()
            
            # 如果没有活动车辆，启动新测试
            if (len(self.visualizer.active_vehicles) == 0 and 
                self.visualizer.current_test_idx < len(self.visualizer.test_pairs)):
                self.visualizer._start_new_test()
            
            # 准备更新数据
            update_data = {
                'vehicles': [(v.vehicle_id, v.current_location) for v in self.visualizer.vehicles],
                'paths': {v.vehicle_id: self.visualizer.vehicle_paths.get(v, []) 
                         for v in self.visualizer.active_vehicles},
                'stats': self.visualizer.stats,
                'test_info': {
                    'current': self.visualizer.current_test_idx,
                    'total': len(self.visualizer.test_pairs)
                }
            }
            
            # 发送更新信号
            self.update_signal.emit(update_data)
            
            # 控制更新速度
            time.sleep(0.05 / self.visualizer.animation_speed)


class PathPlannerVisualizer(QMainWindow):
    """路径规划器高性能可视化工具 - PyQtGraph版"""
    
    def __init__(self, map_size=200, num_vehicles=5, num_test_points=6, debug_mode=False):
        """初始化可视化环境"""
        super().__init__()
        
        # 设置地图尺寸
        self.map_size = map_size
        self.num_vehicles = num_vehicles
        self.num_test_points = num_test_points
        self.debug_mode = debug_mode
        
        # 创建地图服务和路径规划器
        self.geo_utils = GeoUtils()
        self.map_service = MapService()
        self.planner = HybridPathPlanner(self.map_service)
        
        # 初始化冲突检测器
        self.cbs = ConflictBasedSearch(self.planner)
        
        # 初始化测试对象
        self.vehicles = self._create_test_vehicles()
        self.test_points = self._create_test_points()
        self.obstacles = self._create_obstacles()
        
        # 将障碍物应用到规划器
        self.planner.obstacle_grids = set(self.obstacles)
        
        # Mock dispatch 对象 (路径规划器需要)
        class MockDispatch:
            def __init__(self):
                self.vehicles = {}
        
        mock_dispatch = MockDispatch()
        for vehicle in self.vehicles:
            mock_dispatch.vehicles[vehicle.vehicle_id] = vehicle
        self.planner.dispatch = mock_dispatch
        
        # 路径规划测试参数
        self.current_test_idx = 0
        self.test_pairs = []
        self._generate_test_pairs()
        
        # 动画控制
        self.animation_speed = 1.0
        self.show_path = True
        self.pause = False
        self.step_mode = False
        
        # 当前活动车辆和路径
        self.active_vehicles = []
        self.vehicle_paths = {}
        self.vehicle_path_progress = {}
        
        # 性能数据收集
        self.stats = {
            'planning_times': [],
            'path_lengths': [],
            'conflicts': [],
            'visited_nodes_count': []
        }
        
        # 视图控制
        self.view_mode = "normal"  # normal, debug, heatmap
        
        # 创建模拟线程
        self.sim_thread = SimulationThread(self)
        self.sim_thread.update_signal.connect(self.update_display)
        
        # 设置UI
        self.setup_ui()
        
        logging.info("可视化环境初始化完成")
    
    def setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("路径规划可视化")
        self.setGeometry(100, 100, 1200, 800)
        
        # 主布局
        main_layout = QGridLayout()
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 创建主视图
        self.main_view = pg.PlotWidget(title="路径规划器测试可视化")
        self.main_view.setAspectLocked(True)
        self.main_view.setRange(xRange=(0, self.map_size), yRange=(0, self.map_size))
        self.main_view.showGrid(x=True, y=True, alpha=0.5)
        
        # 创建调试视图
        self.debug_view = pg.PlotWidget(title="路径规划调试")
        self.debug_view.setAspectLocked(True)
        self.debug_view.setRange(xRange=(0, self.map_size), yRange=(0, self.map_size))
        self.debug_view.showGrid(x=True, y=True, alpha=0.5)
        
        # 创建状态面板
        self.status_panel = QWidget()
        status_layout = QVBoxLayout()
        self.status_panel.setLayout(status_layout)
        self.status_label = QLabel("系统状态")
        status_layout.addWidget(self.status_label)
        
        # 创建统计面板
        self.stats_panel = QWidget()
        stats_layout = QVBoxLayout()
        self.stats_panel.setLayout(stats_layout)
        self.stats_label = QLabel("性能统计")
        stats_layout.addWidget(self.stats_label)
        
        # 创建控制按钮
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)
        
        self.start_btn = QPushButton("开始")
        self.start_btn.clicked.connect(self.toggle_simulation)
        control_layout.addWidget(self.start_btn)
        
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        control_layout.addWidget(self.pause_btn)
        
        self.speed_up_btn = QPushButton("加速")
        self.speed_up_btn.clicked.connect(self.speed_up)
        control_layout.addWidget(self.speed_up_btn)
        
        self.slow_down_btn = QPushButton("减速")
        self.slow_down_btn.clicked.connect(self.slow_down)
        control_layout.addWidget(self.slow_down_btn)
        
        self.toggle_path_btn = QPushButton("显示/隐藏路径")
        self.toggle_path_btn.clicked.connect(self.toggle_path)
        control_layout.addWidget(self.toggle_path_btn)
        
        self.check_conflicts_btn = QPushButton("检测冲突")
        self.check_conflicts_btn.clicked.connect(self._check_path_conflicts)
        control_layout.addWidget(self.check_conflicts_btn)
        
        # 添加到主布局
        main_layout.addWidget(self.main_view, 0, 0)
        main_layout.addWidget(self.status_panel, 0, 1)
        main_layout.addWidget(self.debug_view, 1, 0)
        main_layout.addWidget(self.stats_panel, 1, 1)
        main_layout.addWidget(control_panel, 2, 0, 1, 2)
        
        # 绘制地图和障碍物
        self._draw_map_background()
        
        # 初始化车辆标记
        self.vehicle_markers = {}
        self.path_items = {}
        self._initialize_vehicle_markers()
        
        # 初始化调试视图元素
        self._initialize_debug_view()
    
    def _draw_map_background(self):
        """绘制地图背景和障碍物"""
        # 绘制测试点
        for name, point in self.test_points.items():
            point_item = pg.ScatterPlotItem([point[0]], [point[1]], size=10, pen=pg.mkPen('b'), brush=pg.mkBrush('b'))
            self.main_view.addItem(point_item)
            text_item = pg.TextItem(name, anchor=(0, 0))
            text_item.setPos(point[0]+5, point[1]+5)
            self.main_view.addItem(text_item)
            
            # 也添加到调试视图
            debug_point_item = pg.ScatterPlotItem([point[0]], [point[1]], size=10, pen=pg.mkPen('b'), brush=pg.mkBrush('b'))
            self.debug_view.addItem(debug_point_item)
            debug_text_item = pg.TextItem(name, anchor=(0, 0))
            debug_text_item.setPos(point[0]+5, point[1]+5)
            self.debug_view.addItem(debug_text_item)
        
        # 绘制矩形障碍物区域
        obstacle_areas = [
            (80, 30, 120, 80),    # 中下方障碍物
            (30, 80, 80, 120),    # 左中障碍物
            (120, 80, 170, 120),  # 右中障碍物
            (80, 120, 120, 170)   # 中上方障碍物
        ]
        
        for area in obstacle_areas:
            x_min, y_min, x_max, y_max = area
            width, height = x_max - x_min, y_max - y_min
            
            # 主视图障碍物
            obstacle_rect = pg.QtGui.QGraphicsRectItem(x_min, y_min, width, height)
            obstacle_rect.setPen(pg.mkPen('k'))
            obstacle_rect.setBrush(pg.mkBrush(100, 100, 100, 100))
            self.main_view.addItem(obstacle_rect)
            
            # 调试视图障碍物
            debug_obstacle_rect = pg.QtGui.QGraphicsRectItem(x_min, y_min, width, height)
            debug_obstacle_rect.setPen(pg.mkPen('k'))
            debug_obstacle_rect.setBrush(pg.mkBrush(100, 100, 100, 100))
            self.debug_view.addItem(debug_obstacle_rect)
        
        # 绘制多边形障碍物
        for polygon_vertices in self._create_polygon_obstacles():
            x_values = [p[0] for p in polygon_vertices]
            y_values = [p[1] for p in polygon_vertices]
            
            # 主视图多边形
            poly_item = pg.PlotDataItem(x_values + [x_values[0]], y_values + [y_values[0]], 
                                    pen=pg.mkPen('k'), fillLevel=0, fillBrush=pg.mkBrush(100, 100, 100, 100))
            self.main_view.addItem(poly_item)
            
            # 调试视图多边形
            debug_poly_item = pg.PlotDataItem(x_values + [x_values[0]], y_values + [y_values[0]], 
                                         pen=pg.mkPen('k'), fillLevel=0, fillBrush=pg.mkBrush(100, 100, 100, 100))
            self.debug_view.addItem(debug_poly_item)
    
    def _initialize_vehicle_markers(self):
        """初始化车辆标记"""
        # 为每个车辆创建一个标记
        for vehicle in self.vehicles:
            # 随机颜色
            color = pg.intColor(vehicle.vehicle_id % 10)
            
            # 车辆标记
            marker = pg.ScatterPlotItem([vehicle.current_location[0]], [vehicle.current_location[1]], 
                                    size=10, pen=pg.mkPen(color), brush=pg.mkBrush(color))
            self.main_view.addItem(marker)
            
            # 车辆ID标签
            label = pg.TextItem(str(vehicle.vehicle_id), anchor=(0, 0))
            label.setPos(vehicle.current_location[0]+5, vehicle.current_location[1]+5)
            self.main_view.addItem(label)
            
            # 路径线
            path_item = pg.PlotDataItem([], [], pen=pg.mkPen(color, width=2))
            self.main_view.addItem(path_item)
            
            # 存储这些元素
            self.vehicle_markers[vehicle.vehicle_id] = {
                'marker': marker,
                'label': label,
                'path': path_item,
                'color': color
            }
    
    def _initialize_debug_view(self):
        """初始化调试视图元素"""
        # A*搜索可视化
        self.debug_elements = {
            'open_set': pg.ScatterPlotItem([], [], pen=None, brush=pg.mkBrush(0, 255, 0, 100), size=6),
            'closed_set': pg.ScatterPlotItem([], [], pen=None, brush=pg.mkBrush(255, 0, 0, 100), size=6),
            'current_path': pg.PlotDataItem([], [], pen=pg.mkPen('b', width=2)),
            'current_node': pg.ScatterPlotItem([], [], pen=None, brush=pg.mkBrush(255, 255, 0, 200), size=12),
            'start_point': pg.ScatterPlotItem([], [], pen=None, brush=pg.mkBrush(0, 255, 0, 200), size=12),
            'end_point': pg.ScatterPlotItem([], [], pen=None, brush=pg.mkBrush(255, 0, 0, 200), size=12)
        }
        
        # 添加到调试视图
        for item in self.debug_elements.values():
            self.debug_view.addItem(item)
        
        # 添加热图层 (初始为空)
        self.heatmap_img = pg.ImageItem()
        self.debug_view.addItem(self.heatmap_img)
        self.heatmap_img.setZValue(-1)  # 确保在底层
        self.heatmap_img.setOpacity(0)  # 初始不可见
        
        # 创建颜色映射
        colormap = pg.ColorMap([0, 0.5, 1], [(0, 0, 0, 0), (255, 165, 0, 100), (255, 0, 0, 200)])
        self.heatmap_img.setLookupTable(colormap.getLookupTable())
    
    def toggle_simulation(self):
        """开始/停止模拟"""
        if self.sim_thread.running:
            self.sim_thread.stop()
            self.start_btn.setText("开始")
        else:
            self.sim_thread.start()
            self.start_btn.setText("停止")
    
    def toggle_pause(self):
        """暂停/继续模拟"""
        self.pause = not self.pause
        self.pause_btn.setText("继续" if self.pause else "暂停")
    
    def speed_up(self):
        """加快模拟速度"""
        self.animation_speed = min(5.0, self.animation_speed * 1.5)
        logging.info(f"动画速度: {self.animation_speed:.1f}x")
    
    def slow_down(self):
        """减慢模拟速度"""
        self.animation_speed = max(0.1, self.animation_speed / 1.5)
        logging.info(f"动画速度: {self.animation_speed:.1f}x")
    
    def toggle_path(self):
        """显示/隐藏路径"""
        self.show_path = not self.show_path
        for vehicle_id in self.vehicle_markers:
            path_item = self.vehicle_markers[vehicle_id]['path']
            if not self.show_path:
                path_item.setData([], [])
    
    def update_display(self, data):
        """更新显示 (由模拟线程调用)"""
        if self.pause:
            return
            
        # 更新车辆位置和路径
        vehicles_data = data['vehicles']
        paths_data = data['paths']
        
        for vehicle_id, position in vehicles_data:
            if vehicle_id in self.vehicle_markers:
                marker_data = self.vehicle_markers[vehicle_id]
                
                # 更新标记位置
                marker_data['marker'].setData([position[0]], [position[1]])
                
                # 更新标签位置
                marker_data['label'].setPos(position[0]+5, position[1]+5)
                
                # 更新路径
                if self.show_path and vehicle_id in paths_data:
                    path = paths_data[vehicle_id]
                    marker_data['path'].setData(
                        [p[0] for p in path], 
                        [p[1] for p in path]
                    )
                else:
                    marker_data['path'].setData([], [])
        
        # 更新状态文本
        test_info = data['test_info']
        active_count = len(self.active_vehicles)
        completed_tests = test_info['current']
        total_tests = test_info['total']
        
        # 状态文本
        status_html = f"""
        <h3>系统状态</h3>
        <p>活动车辆: {active_count}/{self.num_vehicles}</p>
        <p>完成测试: {completed_tests}/{total_tests} ({completed_tests/total_tests*100:.1f}%)</p>
        <p>动画速度: {self.animation_speed:.1f}x</p>
        <p>视图模式: {self.view_mode}</p>
        <p>路径显示: {'开启' if self.show_path else '关闭'}</p>
        <br>
        <p><b>当前活动:</b></p>
        """
        
        # 添加活动车辆信息
        for vehicle in self.active_vehicles:
            task_info = f"{vehicle.current_task.task_id}" if hasattr(vehicle, 'current_task') and vehicle.current_task else "无任务"
            progress = self.vehicle_path_progress.get(vehicle, 0)
            path_len = len(self.vehicle_paths.get(vehicle, []))
            if path_len > 0:
                progress_pct = min(100, progress / path_len * 100)
            else:
                progress_pct = 0
                
            status_html += f"<p>车辆{vehicle.vehicle_id}: {task_info} - 进度: {progress_pct:.1f}%</p>"
            
        self.status_label.setText(status_html)
        
        # 更新统计文本
        stats_data = data['stats']
        avg_planning_time = np.mean(stats_data['planning_times']) if stats_data['planning_times'] else 0
        avg_path_length = np.mean(stats_data['path_lengths']) if stats_data['path_lengths'] else 0
        conflict_count = len(stats_data['conflicts'])
        
        stats_html = f"""
        <h3>性能统计</h3>
        <p>平均规划时间: {avg_planning_time*1000:.2f}毫秒</p>
        <p>平均路径长度: {avg_path_length:.1f}点</p>
        <p>检测到的冲突: {conflict_count}个</p>
        <p>规划成功率: {100 * len(stats_data['path_lengths']) / max(1, test_info['current']):.1f}%</p>
        <br>
        <p><b>路径规划指标:</b></p>
        """
        
        if stats_data['path_lengths']:
            stats_html += f"<p>最短路径: {min(stats_data['path_lengths'])}点</p>"
            stats_html += f"<p>最长路径: {max(stats_data['path_lengths'])}点</p>"
            
        if stats_data['planning_times']:
            stats_html += f"<p>最快规划: {min(stats_data['planning_times'])*1000:.2f}毫秒</p>"
            stats_html += f"<p>最慢规划: {max(stats_data['planning_times'])*1000:.2f}毫秒</p>"
            
        self.stats_label.setText(stats_html)
        
        # 更新视图标题
        self.main_view.setTitle(f"测试 {completed_tests+1}/{total_tests}: 当前{self.view_mode}模式")
        
        # 根据视图模式更新调试视图
        if self.view_mode == "debug":
            self._update_debug_view()
        elif self.view_mode == "heatmap":
            self._update_heatmap_view()