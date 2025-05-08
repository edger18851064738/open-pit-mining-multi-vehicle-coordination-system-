"""
露天矿多车协同调度系统可视化界面

集成了:
1. 混合A*路径规划器
2. CBS冲突解决算法
3. 中央调度系统
4. 交互式可视化界面
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
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# PyQt和PyQtGraph导入
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QPushButton, QGridLayout, QComboBox, QSlider, QCheckBox, 
    QFileDialog, QTextEdit, QGroupBox, QSplitter, QTabWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint
from PyQt5.QtGui import QColor, QPen, QBrush, QFont
import pyqtgraph as pg

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 导入项目模块
try:
    from algorithm.hybrid_path_planner import HybridPathPlanner
    from algorithm.map_service import MapService
    from algorithm.cbs import ConflictBasedSearch
    from algorithm.dispatch_service import DispatchSystem
    from utils.geo_tools import GeoUtils
    from models.vehicle import MiningVehicle, VehicleState, TransportStage
    from models.task import TransportTask
    logging.info("成功导入所需模块")
except ImportError as e:
    logging.error(f"导入模块失败: {str(e)}")
    sys.exit(1)

# 设置PyQtGraph全局配置
pg.setConfigOptions(antialias=True, background='w', foreground='k')

class SimulationThread(QThread):
    """模拟线程，处理路径规划和车辆移动"""
    update_signal = pyqtSignal(dict)  # 发送更新数据的信号
    finished_signal = pyqtSignal(dict)  # 测试完成信号
    
    def __init__(self, main_app):
        super().__init__()
        self.main_app = main_app
        self.running = False
        self.paused = False
    
    def run(self):
        """线程运行函数"""
        self.running = True
        last_update_time = time.time()
        update_interval = 0.05  # 秒
        
        while self.running:
            if not self.paused:
                current_time = time.time()
                
                # 调度系统更新
                self.main_app.dispatch.scheduling_cycle()
                
                # 更新车辆位置
                self.main_app._update_vehicles()
                
                # 如果所有测试完成，发送测试完成信号
                if self.main_app.all_tests_completed:
                    final_stats = self.main_app.get_test_stats()
                    self.finished_signal.emit(final_stats)
                    self.running = False
                    break
                
                # 如果没有活动车辆，启动新测试
                if len(self.main_app.active_vehicles) == 0 and not self.main_app.all_tests_completed:
                    self.main_app._start_new_test()
                
                # 定时更新UI
                if current_time - last_update_time >= update_interval:
                    # 准备更新数据
                    update_data = {
                        'vehicles': {v.vehicle_id: v.current_location for v in self.main_app.vehicles},
                        'vehicle_states': {v.vehicle_id: v.state.name if hasattr(v.state, 'name') else str(v.state) 
                                          for v in self.main_app.vehicles},
                        'paths': {v.vehicle_id: self.main_app.vehicle_paths.get(v, []) 
                                 for v in self.main_app.active_vehicles},
                        'stats': self.main_app.test_stats,
                        'dispatch_status': self.main_app.dispatch.get_status(),
                        'test_info': self.main_app.current_test_info,
                        'conflicts': self.main_app.current_conflicts
                    }
                    
                    # 发送更新信号
                    self.update_signal.emit(update_data)
                    last_update_time = current_time
            
            # 控制更新速度
            sim_delay = 0.01 / self.main_app.simulation_speed
            time.sleep(sim_delay)
    
    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()  # 等待线程完成
    
    def pause(self):
        """暂停模拟"""
        self.paused = True
    
    def resume(self):
        """继续模拟"""
        self.paused = False


class MiningDispatchApp(QMainWindow):
    """露天矿调度系统主应用程序"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("露天矿多车协同调度系统")
        self.resize(1280, 800)
        
        # 初始化测试参数
        self.map_size = 200
        self.num_vehicles = 5
        self.num_test_pairs = 10
        self.simulation_speed = 1.0
        self.show_paths = True
        self.show_vehicle_states = True
        self.show_conflicts = True
        self.show_heatmap = False
        
        # 测试状态
        self.test_running = False
        self.test_paused = False
        self.all_tests_completed = False
        
        # 初始化系统组件
        self._init_system_components()
        
        # 当前测试信息
        self.current_test_info = {
            'current_idx': 0,
            'total_tests': 0,
            'current_test': None
        }
        
        # 测试统计数据
        self.test_stats = {
            'planning_times': [],
            'path_lengths': [],
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'total_tests': 0,
            'start_time': None,
        }
        
        # 当前冲突
        self.current_conflicts = []
        
        # 测试数据
        self.vehicles = []
        self.test_points = {}
        self.test_pairs = []
        self.vehicle_paths = {}
        self.active_vehicles = []
        
        # 创建模拟线程
        self.sim_thread = SimulationThread(self)
        self.sim_thread.update_signal.connect(self.update_display)
        self.sim_thread.finished_signal.connect(self.on_test_completed)
        
        # 创建UI
        self.setup_ui()
        
        # 热图数据
        self.heatmap_data = np.zeros((self.map_size, self.map_size))
        
        logging.info("调度系统应用程序初始化完成")
    
    def _init_system_components(self):
        """初始化系统核心组件"""
        try:
            # 初始化基础工具
            self.geo_utils = GeoUtils()
            
            # 初始化地图服务
            self.map_service = MapService()
            
            # 初始化路径规划器
            self.planner = HybridPathPlanner(self.map_service)
            
            # 初始化调度系统
            self.dispatch = DispatchSystem(self.planner, self.map_service)
            
            # 初始化CBS冲突解决器
            self.cbs = self.dispatch.cbs
            
            logging.info("系统组件初始化成功")
        except Exception as e:
            logging.critical(f"系统组件初始化失败: {str(e)}")
            if hasattr(self, 'update_status_text'):
                self.update_status_text(f"系统初始化失败: {str(e)}")
            else:
                print(f"系统初始化失败: {str(e)}")
            raise
    
    def setup_ui(self):
        """设置用户界面"""
        # 创建主部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建上下分割的布局
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # 上部分：地图显示区域和控制面板
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        splitter.addWidget(top_widget)
        
        # 下部分：状态和统计信息
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        splitter.addWidget(bottom_widget)
        
        # 设置分割比例
        splitter.setSizes([600, 200])
        
        # 左侧：地图显示区域
        map_widget = QWidget()
        map_layout = QVBoxLayout(map_widget)
        
        # 创建主视图
        self.main_view = pg.PlotWidget(title="调度系统地图")
        self.main_view.setAspectLocked(True)
        self.main_view.setRange(xRange=(0, self.map_size), yRange=(0, self.map_size))
        self.main_view.showGrid(x=True, y=True, alpha=0.5)
        self.main_view.addLegend()
        map_layout.addWidget(self.main_view)
        
        top_layout.addWidget(map_widget, 3)
        
        # 右侧：控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        top_layout.addWidget(control_panel, 1)
        
        # 测试参数设置
        params_group = QGroupBox("调度参数")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)
        
        # 车辆数量
        params_layout.addWidget(QLabel("车辆数量:"), 0, 0)
        self.vehicles_slider = QSlider(Qt.Horizontal)
        self.vehicles_slider.setMinimum(1)
        self.vehicles_slider.setMaximum(20)
        self.vehicles_slider.setValue(5)
        self.vehicles_slider.setTickPosition(QSlider.TicksBelow)
        self.vehicles_slider.setTickInterval(1)
        self.vehicles_slider.valueChanged.connect(self.on_vehicles_changed)
        params_layout.addWidget(self.vehicles_slider, 0, 1)
        self.vehicles_label = QLabel("5")
        params_layout.addWidget(self.vehicles_label, 0, 2)
        
        # 测试数量
        params_layout.addWidget(QLabel("测试数量:"), 1, 0)
        self.tests_slider = QSlider(Qt.Horizontal)
        self.tests_slider.setMinimum(5)
        self.tests_slider.setMaximum(50)
        self.tests_slider.setValue(10)
        self.tests_slider.setTickPosition(QSlider.TicksBelow)
        self.tests_slider.setTickInterval(5)
        self.tests_slider.valueChanged.connect(self.on_tests_changed)
        params_layout.addWidget(self.tests_slider, 1, 1)
        self.tests_label = QLabel("10")
        params_layout.addWidget(self.tests_label, 1, 2)
        
        # 是否使用复杂地图
        params_layout.addWidget(QLabel("复杂地图:"), 2, 0)
        self.complex_map_checkbox = QCheckBox()
        self.complex_map_checkbox.setChecked(True)
        params_layout.addWidget(self.complex_map_checkbox, 2, 1)
        
        # 显示冲突
        params_layout.addWidget(QLabel("显示冲突:"), 3, 0)
        self.show_conflicts_checkbox = QCheckBox()
        self.show_conflicts_checkbox.setChecked(True)
        self.show_conflicts_checkbox.stateChanged.connect(self.on_show_conflicts_changed)
        params_layout.addWidget(self.show_conflicts_checkbox, 3, 1)
        
        control_layout.addWidget(params_group)
        
        # 控制按钮
        buttons_group = QGroupBox("运行控制")
        buttons_layout = QGridLayout()
        buttons_group.setLayout(buttons_layout)
        
        # 启动测试按钮
        self.start_btn = QPushButton("启动系统")
        self.start_btn.clicked.connect(self.start_test)
        buttons_layout.addWidget(self.start_btn, 0, 0)
        
        # 暂停测试按钮
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        buttons_layout.addWidget(self.pause_btn, 0, 1)
        
        # 速度控制
        buttons_layout.addWidget(QLabel("模拟速度:"), 1, 0)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        buttons_layout.addWidget(self.speed_slider, 1, 1)
        self.speed_label = QLabel("1.0x")
        buttons_layout.addWidget(self.speed_label, 1, 2)
        
        # 显示设置
        buttons_layout.addWidget(QLabel("显示路径:"), 2, 0)
        self.show_paths_checkbox = QCheckBox()
        self.show_paths_checkbox.setChecked(True)
        self.show_paths_checkbox.stateChanged.connect(self.on_show_paths_changed)
        buttons_layout.addWidget(self.show_paths_checkbox, 2, 1)
        
        buttons_layout.addWidget(QLabel("显示车辆状态:"), 3, 0)
        self.show_states_checkbox = QCheckBox()
        self.show_states_checkbox.setChecked(True)
        self.show_states_checkbox.stateChanged.connect(self.on_show_states_changed)
        buttons_layout.addWidget(self.show_states_checkbox, 3, 1)
        
        buttons_layout.addWidget(QLabel("显示热图:"), 4, 0)
        self.show_heatmap_checkbox = QCheckBox()
        self.show_heatmap_checkbox.setChecked(False)
        self.show_heatmap_checkbox.stateChanged.connect(self.on_show_heatmap_changed)
        buttons_layout.addWidget(self.show_heatmap_checkbox, 4, 1)
        
        control_layout.addWidget(buttons_group)
        
        # 功能按钮
        functions_group = QGroupBox("冲突管理")
        functions_layout = QGridLayout()
        functions_group.setLayout(functions_layout)
        
        # 检测冲突按钮
        self.check_conflicts_btn = QPushButton("检测冲突")
        self.check_conflicts_btn.clicked.connect(self.check_path_conflicts)
        self.check_conflicts_btn.setEnabled(False)
        functions_layout.addWidget(self.check_conflicts_btn, 0, 0)
        
        # 解决冲突按钮
        self.resolve_conflicts_btn = QPushButton("解决冲突")
        self.resolve_conflicts_btn.clicked.connect(self.resolve_path_conflicts)
        self.resolve_conflicts_btn.setEnabled(False)
        functions_layout.addWidget(self.resolve_conflicts_btn, 0, 1)
        
        # 保存结果按钮
        self.save_results_btn = QPushButton("保存结果")
        self.save_results_btn.clicked.connect(self.save_test_results)
        self.save_results_btn.setEnabled(False)
        functions_layout.addWidget(self.save_results_btn, 1, 0, 1, 2)
        
        control_layout.addWidget(functions_group)
        
        # 弹性空间
        control_layout.addStretch()
        
        # 下部分：左侧状态信息
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(150)
        status_layout.addWidget(self.status_text)
        
        bottom_layout.addWidget(status_group)
        
        # 右侧：统计信息
        stats_group = QGroupBox("调度统计")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        
        bottom_layout.addWidget(stats_group)
        
        # 初始化状态
        self.update_status_text("系统初始化完成，请点击'启动系统'开始调度")
        self.update_stats_text("尚未开始调度")
    
    def on_vehicles_changed(self, value):
        """车辆数量变更处理"""
        self.num_vehicles = value
        self.vehicles_label.setText(str(value))
        self.update_status_text(f"车辆数量已更改为 {value}")
    
    def on_tests_changed(self, value):
        """测试数量变更处理"""
        self.num_test_pairs = value
        self.tests_label.setText(str(value))
        self.update_status_text(f"测试数量已更改为 {value}")
    
    def on_speed_changed(self, value):
        """模拟速度变更处理"""
        self.simulation_speed = value / 10.0
        self.speed_label.setText(f"{self.simulation_speed:.1f}x")
    
    def on_show_paths_changed(self, state):
        """路径显示变更处理"""
        self.show_paths = state == Qt.Checked
        self.update_display({})  # 触发重绘
    
    def on_show_states_changed(self, state):
        """车辆状态显示变更处理"""
        self.show_vehicle_states = state == Qt.Checked
        self.update_display({})  # 触发重绘
    
    def on_show_conflicts_changed(self, state):
        """冲突显示变更处理"""
        self.show_conflicts = state == Qt.Checked
        self.update_display({})  # 触发重绘
    
    def on_show_heatmap_changed(self, state):
        """热图显示变更处理"""
        self.show_heatmap = state == Qt.Checked
        
        if self.show_heatmap:
            # 如果还没有热图，创建一个新的
            if not hasattr(self, 'heatmap_img'):
                self.heatmap_img = pg.ImageItem()
                self.main_view.addItem(self.heatmap_img)
                
                # 创建颜色映射
                pos = np.array([0.0, 0.33, 0.66, 1.0])
                color = np.array([
                    [0, 0, 0, 0],
                    [0, 0, 255, 50],
                    [255, 255, 0, 100],
                    [255, 0, 0, 150]
                ])
                cmap = pg.ColorMap(pos, color)
                self.heatmap_img.setLookupTable(cmap.getLookupTable())
                
                # 设置位置和缩放
                self.heatmap_img.setRect(pg.QtCore.QRectF(0, 0, self.map_size, self.map_size))
                
            # 显示热图
            self.heatmap_img.setOpacity(0.7)
        else:
            # 隐藏热图
            if hasattr(self, 'heatmap_img'):
                self.heatmap_img.setOpacity(0)
    
    def update_status_text(self, message):
        """更新状态文本"""
        self.status_text.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # 滚动到底部
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )
    
    def update_stats_text(self, stats_html):
        """更新统计信息文本"""
        self.stats_text.setHtml(stats_html)
    
    def update_display(self, update_data):
        """更新显示内容"""
        try:
            # 清除主视图
            self.main_view.clear()
            
            # 重新绘制地图和障碍物
            if hasattr(self, 'obstacles'):
                self._draw_map_and_obstacles()
            
            # 更新热图
            if self.show_heatmap and hasattr(self, 'heatmap_img'):
                self._update_heatmap_view()
            
            # 更新车辆位置
            if 'vehicles' in update_data:
                for i, vehicle in enumerate(self.vehicles):
                    if vehicle.vehicle_id in update_data['vehicles']:
                        vehicle.current_location = update_data['vehicles'][vehicle.vehicle_id]
                
                # 绘制车辆
                self._draw_vehicles(update_data.get('vehicle_states', {}))
            
            # 更新路径显示
            if 'paths' in update_data and self.show_paths:
                self._draw_paths(update_data['paths'])
            
            # 更新冲突显示
            if 'conflicts' in update_data and self.show_conflicts:
                self._draw_conflicts(update_data['conflicts'])
            
            # 更新统计信息
            if 'stats' in update_data:
                self._update_stats_display(update_data['stats'], update_data.get('dispatch_status', {}))
            
            # 更新测试信息
            if 'test_info' in update_data:
                self.current_test_info = update_data['test_info']
                self._update_status_display()
            
            # 刷新应用
            QApplication.processEvents()
            
        except Exception as e:
            logging.error(f"更新显示时出错: {str(e)}")
    
    def _draw_vehicles(self, vehicle_states):
        """绘制车辆"""
        # 状态颜色映射
        state_colors = {
            'IDLE': (100, 100, 255),       # 蓝色
            'PREPARING': (255, 200, 0),    # 黄色
            'EN_ROUTE': (0, 200, 0),       # 绿色
            'UNLOADING': (255, 100, 100),  # 红色
            'EMERGENCY_STOP': (255, 0, 0)  # 鲜红色
        }
        
        for vehicle in self.vehicles:
            # 获取车辆状态
            state_name = vehicle_states.get(vehicle.vehicle_id, 'IDLE')
            state_color = state_colors.get(state_name, (150, 150, 150))
            
            # 创建车辆点
            vehicle_point = pg.ScatterPlotItem(
                [vehicle.current_location[0]], 
                [vehicle.current_location[1]],
                size=10, 
                brush=pg.mkBrush(*state_color),
                pen=pg.mkPen(width=1.5, color='k')
            )
            self.main_view.addItem(vehicle_point)
            
            # 添加车辆ID标签
            vehicle_label = pg.TextItem(
                text=str(vehicle.vehicle_id),
                color='k',
                anchor=(0.5, 0.5)
            )
            vehicle_label.setPos(vehicle.current_location[0], vehicle.current_location[1])
            self.main_view.addItem(vehicle_label)
            
            # 如果显示车辆状态
            if self.show_vehicle_states:
                state_label = pg.TextItem(
                    text=state_name,
                    color=(50, 50, 50),
                    anchor=(0.5, 0)
                )
                state_label.setPos(vehicle.current_location[0], vehicle.current_location[1] + 8)
                self.main_view.addItem(state_label)
    
    def _draw_paths(self, paths):
        """绘制路径"""
        # 路径样式
        path_pen = pg.mkPen(color=(100, 100, 200), width=2, style=Qt.DashLine)
        
        for vehicle_id, path in paths.items():
            if not path or len(path) < 2:
                continue
                
            # 创建路径线
            x_data = [p[0] for p in path]
            y_data = [p[1] for p in path]
            
            path_line = pg.PlotDataItem(
                x_data, y_data,
                pen=path_pen
            )
            self.main_view.addItem(path_line)
            
            # 绘制路径起点和终点
            start_point = pg.ScatterPlotItem(
                [x_data[0]], [y_data[0]],
                size=8, symbol='o', brush=pg.mkBrush(0, 200, 0)
            )
            end_point = pg.ScatterPlotItem(
                [x_data[-1]], [y_data[-1]],
                size=8, symbol='s', brush=pg.mkBrush(200, 0, 0)
            )
            
            self.main_view.addItem(start_point)
            self.main_view.addItem(end_point)
    
    def _draw_conflicts(self, conflicts):
        """绘制冲突点"""
        if not conflicts:
            return
            
        for conflict in conflicts:
            location = conflict.get("location")
            if not location:
                continue
                
            # 绘制冲突点
            conflict_point = pg.ScatterPlotItem(
                [location[0]], [location[1]],
                size=15, symbol='x', 
                brush=pg.mkBrush(255, 0, 0, 150),
                pen=pg.mkPen(width=2, color='r')
            )
            self.main_view.addItem(conflict_point)
            
            # 添加冲突类型标签
            conflict_type = conflict.get("type", "未知")
            type_label = pg.TextItem(
                text=f"{conflict_type}冲突",
                color='r',
                anchor=(0.5, 0)
            )
            type_label.setPos(location[0], location[1] + 10)
            self.main_view.addItem(type_label)
    
    def _update_stats_display(self, stats, dispatch_status):
        """更新统计信息显示"""
        # 计算平均值
        avg_planning_time = np.mean(stats['planning_times']) if stats['planning_times'] else 0
        avg_path_length = np.mean(stats['path_lengths']) if stats['path_lengths'] else 0
        
        # 计算成功率
        total_plans = stats['successful_plans'] + stats['failed_plans']
        success_rate = stats['successful_plans'] / max(1, total_plans) * 100
        
        # 计算测试进度
        progress = (stats['successful_plans'] + stats['failed_plans']) / max(1, stats['total_tests']) * 100
        
        # 计算运行时间
        if stats['start_time']:
            run_time = time.time() - stats['start_time']
        else:
            run_time = 0
        
        # 获取调度系统状态
        vehicles_status = dispatch_status.get('vehicles', {})
        tasks_status = dispatch_status.get('tasks', {})
        metrics = dispatch_status.get('metrics', {})
        
        # 创建HTML格式的统计信息
        stats_html = f"""
        <h3>调度系统统计</h3>
        <table>
            <tr><td>测试进度:</td><td>{progress:.1f}% ({stats['successful_plans'] + stats['failed_plans']}/{stats['total_tests']})</td></tr>
            <tr><td>规划成功率:</td><td>{success_rate:.1f}%</td></tr>
            <tr><td>平均规划时间:</td><td>{avg_planning_time*1000:.2f} 毫秒</td></tr>
            <tr><td>平均路径长度:</td><td>{avg_path_length:.1f} 点</td></tr>
            <tr><td>车辆状态:</td><td>总数: {vehicles_status.get('total', 0)} | 空闲: {vehicles_status.get('idle', 0)} | 活动: {vehicles_status.get('active', 0)}</td></tr>
            <tr><td>任务状态:</td><td>队列: {tasks_status.get('queued', 0)} | 活动: {tasks_status.get('active', 0)} | 完成: {tasks_status.get('completed', 0)}</td></tr>
            <tr><td>检测到冲突:</td><td>{metrics.get('conflicts_detected', 0)} 个</td></tr>
            <tr><td>解决的冲突:</td><td>{metrics.get('conflicts_resolved', 0)} 个</td></tr>
            <tr><td>运行时间:</td><td>{run_time:.1f} 秒</td></tr>
        </table>
        """
        
        self.update_stats_text(stats_html)
    
    def _update_status_display(self):
        """更新状态信息显示"""
        if not hasattr(self, 'current_test_info'):
            return
            
        test_info = self.current_test_info
        
        # 如果有当前测试信息，显示它
        if test_info.get('current_test'):
            test = test_info['current_test']
            start_name = test.get('start_name', '未知')
            end_name = test.get('end_name', '未知')
            
            status_msg = f"当前测试 {test_info['current_idx']}/{test_info['total_tests']}: "
            status_msg += f"从 {start_name} 到 {end_name}"
            
            self.update_status_text(status_msg)
    
    def _update_heatmap_view(self):
        """更新热图视图"""
        # 应用平滑处理
        from scipy.ndimage import gaussian_filter
        smoothed_data = gaussian_filter(self.heatmap_data, sigma=2)
        
        # 更新图像数据
        self.heatmap_img.setImage(smoothed_data.T)
    
    def start_test(self):
        """启动测试"""
        self.reset_system()
        if self.test_running:
            # 如果测试已在运行，停止它
            self.stop_test()
            return
        
        try:
            # 重置测试状态
            self.all_tests_completed = False
            self.test_running = True
            self.test_paused = False
            self.reset_system()
            # 重置统计数据
            self.test_stats = {
                'planning_times': [],
                'path_lengths': [],
                'conflicts_detected': 0,
                'conflicts_resolved': 0,
                'successful_plans': 0,
                'failed_plans': 0,
                'total_tests': 0,
                'start_time': time.time()
            }
            
            # 创建测试组件
            self._create_test_components()
            
            # 更新UI
            self.start_btn.setText("停止系统")
            self.pause_btn.setEnabled(True)
            self.check_conflicts_btn.setEnabled(True)
            self.resolve_conflicts_btn.setEnabled(True)
            self.save_results_btn.setEnabled(False)
            
            # 禁用测试参数设置
            self._set_params_enabled(False)
            
            # 清除视图
            self.main_view.clear()
            
            # 绘制地图和障碍物
            self._draw_map_and_obstacles()
            
            # 初始化热图数据
            self.heatmap_data = np.zeros((self.map_size, self.map_size))
            
            # 启动模拟线程
            self.sim_thread.start()
            
            self.update_status_text("调度系统已启动")
            
        except Exception as e:
            logging.error(f"启动系统时出错: {str(e)}")
            self.update_status_text(f"启动系统失败: {str(e)}")
            self.test_running = False
    
    def stop_test(self):
        """停止测试"""
        if not self.test_running:
            return
            
        # 停止模拟线程
        self.sim_thread.stop()
        
        # 更新状态
        self.test_running = False
        self.test_paused = False
        
        # 更新UI
        self.start_btn.setText("启动系统")
        self.pause_btn.setEnabled(False)
        self.check_conflicts_btn.setEnabled(False)
        self.resolve_conflicts_btn.setEnabled(False)
        self.save_results_btn.setEnabled(True)
        
        # 启用测试参数设置
        self._set_params_enabled(True)
        
        self.update_status_text("调度系统已停止")
    
    def toggle_pause(self):
        """暂停/继续测试"""
        if not self.test_running:
            return
            
        self.test_paused = not self.test_paused
        
        if self.test_paused:
            # 暂停模拟线程
            self.sim_thread.pause()
            self.pause_btn.setText("继续")
            self.update_status_text("系统已暂停")
        else:
            # 继续模拟线程
            self.sim_thread.resume()
            self.pause_btn.setText("暂停")
            self.update_status_text("系统已继续")
    
    def _set_params_enabled(self, enabled):
        """设置参数控件是否可用"""
        self.vehicles_slider.setEnabled(enabled)
        self.tests_slider.setEnabled(enabled)
        self.complex_map_checkbox.setEnabled(enabled)
    
    def _create_test_components(self):
        """创建测试组件"""
        # 重新初始化系统组件
        self._init_system_components()
        
        # 创建测试点和任务
        self.test_points = self._create_test_points()
        self.test_pairs = self._generate_test_pairs()
        
        # 创建车辆
        self.vehicles = self._create_test_vehicles()
        
        # 为每个车辆创建调度系统引用
        for vehicle in self.vehicles:
            self.dispatch.add_vehicle(vehicle)
        
        # 创建障碍物
        self.obstacles = self._create_obstacles()
        
        # 将障碍物应用到规划器
        self.planner.obstacle_grids = set(self.obstacles)
        
        # 重置测试索引和活动车辆
        self.current_test_info = {
            'current_idx': 0,
            'total_tests': len(self.test_pairs),
            'current_test': None
        }
        
        self.vehicle_paths = {}
        self.active_vehicles = []
        self.current_conflicts = []
    
    def _create_test_points(self) -> Dict[str, Tuple[float, float]]:
        """创建测试点"""
        points = {}
        
        # 网格尺寸
        grid_step = self.map_size / 10
        
        # 关键点位置
        key_points = {
            "装载点1": (grid_step, self.map_size - grid_step),
            "装载点2": (grid_step, grid_step),
            "装载点3": (self.map_size - grid_step, grid_step),
            "卸载点": (self.map_size - grid_step, self.map_size - grid_step),
            "停车场": (self.map_size // 2, self.map_size - grid_step),
            "检查点1": (self.map_size // 2, grid_step),
            "检查点2": (grid_step, self.map_size // 2),
            "检查点3": (self.map_size - grid_step, self.map_size // 2)
        }
        
        points.update(key_points)
        
        # 添加额外随机点
        num_random_points = max(0, 5)
        
        for i in range(num_random_points):
            # 尝试找到一个不在障碍物上的随机点
            for attempt in range(20):  # 最多尝试20次
                x = random.uniform(grid_step, self.map_size - grid_step)
                y = random.uniform(grid_step, self.map_size - grid_step)
                point = (x, y)
                
                # 检查点是否远离障碍物
                if not hasattr(self, 'obstacles') or not self._is_near_obstacle(point, 15):
                    points[f"辅助点{i+1}"] = point
                    break
        
        logging.info(f"创建了 {len(points)} 个任务点")
        return points
    
    def _is_near_obstacle(self, point, threshold=10.0):
        """检查点是否靠近障碍物"""
        if not hasattr(self, 'obstacles') or not self.obstacles:
            return False
            
        # 快速检查：如果点与任何障碍物的距离小于阈值，则认为靠近障碍物
        for obs_point in self.obstacles:
            dx = abs(point[0] - obs_point[0])
            dy = abs(point[1] - obs_point[1])
            
            # 曼哈顿距离快速检查
            if dx + dy < threshold:
                # 欧几里得距离精确检查
                if math.sqrt(dx*dx + dy*dy) < threshold:
                    return True
                    
        return False
    
    def _generate_test_pairs(self):
        """生成测试路径对"""
        test_pairs = []
        
        # 如果测试点少于两个，无法生成测试对
        if len(self.test_points) < 2:
            return test_pairs
            
        # 获取所有点名称
        point_names = list(self.test_points.keys())
        
        # 主要测试集：装载点到卸载点的路径
        loading_points = [p for p in point_names if "装载" in p]
        unloading_points = [p for p in point_names if "卸载" in p]
        
        # 创建通常的矿山运输任务
        for i in range(min(self.num_test_pairs, len(loading_points) * len(unloading_points))):
            loading_idx = i % len(loading_points)
            unloading_idx = i % len(unloading_points)
            
            start_name = loading_points[loading_idx]
            end_name = unloading_points[unloading_idx]
            
            start_point = self.test_points[start_name]
            end_point = self.test_points[end_name]
            
            test_pairs.append({
                'start': start_point,
                'end': end_point,
                'start_name': start_name,
                'end_name': end_name,
                'task_type': 'transport'
            })
        
        # 如果需要更多测试对，添加返回停车场的路径
        parking_points = [p for p in point_names if "停车" in p]
        while len(test_pairs) < self.num_test_pairs and parking_points:
            random_point_name = random.choice([p for p in point_names if "停车" not in p])
            parking_name = parking_points[0]
            
            start_point = self.test_points[random_point_name]
            end_point = self.test_points[parking_name]
            
            test_pairs.append({
                'start': start_point,
                'end': end_point,
                'start_name': random_point_name,
                'end_name': parking_name,
                'task_type': 'return'
            })
        
        logging.info(f"生成了 {len(test_pairs)} 个测试路径对")
        return test_pairs
    
    def _create_test_vehicles(self):
        """创建测试车辆"""
        vehicles = []
        
        # 计算车辆初始位置
        positions = []
        
        # 停车位置
        parking_point = None
        for name, point in self.test_points.items():
            if "停车" in name:
                parking_point = point
                break
        
        if not parking_point:
            # 如果没有找到停车场，使用默认位置
            parking_point = (self.map_size // 2, self.map_size // 2)
        
        # 在停车场周围分布车辆
        for i in range(self.num_vehicles):
            angle = i * (2 * math.pi / self.num_vehicles)
            radius = 10
            x = parking_point[0] + radius * math.cos(angle)
            y = parking_point[1] + radius * math.sin(angle)
            positions.append((x, y))
        
        # 创建车辆
        for i in range(self.num_vehicles):
            position = positions[i] if i < len(positions) else (0, 0)
            
            config = {
                'current_location': position,
                'max_capacity': 50000,
                'max_speed': random.uniform(5.0, 8.0),
                'min_hardness': 2.5,
                'turning_radius': 10.0,
                'base_location': parking_point
            }
            
            vehicle = MiningVehicle(
                vehicle_id=i+1,
                map_service=self.map_service,
                config=config
            )
            
            vehicles.append(vehicle)
        
        logging.info(f"创建了 {len(vehicles)} 辆测试车辆")
        return vehicles
    
    def _create_obstacles(self):
        """创建障碍物点集"""
        obstacles = []
        
        if not self.complex_map_checkbox.isChecked():
            # 简单地图，只创建几个矩形障碍物
            obstacles.extend(self._create_simple_obstacles())
        else:
            # 复杂地图
            # 1. 添加矩形障碍物
            obstacles.extend(self._create_rectangular_obstacles())
            
            # 2. 添加随机障碍物点
            obstacles.extend(self._create_random_obstacles())
        
        logging.info(f"创建了 {len(obstacles)} 个障碍物点")
        return obstacles
    
    def _create_simple_obstacles(self):
        """创建简单障碍物"""
        obstacles = []
        
        # 创建十字形障碍物
        middle_x = self.map_size // 2
        middle_y = self.map_size // 2
        width = self.map_size // 4
        height = self.map_size // 10
        
        # 水平障碍物
        for x in range(middle_x - width//2, middle_x + width//2):
            for y in range(middle_y - height//2, middle_y + height//2):
                obstacles.append((x, y))
        
        # 垂直障碍物
        for x in range(middle_x - height//2, middle_x + height//2):
            for y in range(middle_y - width//2, middle_y + width//2):
                obstacles.append((x, y))
        
        return obstacles
    
    def _create_rectangular_obstacles(self):
        """创建矩形障碍物"""
        obstacles = []
        
        # 定义几个矩形障碍物区域: (x1, y1, x2, y2)
        rectangles = [
            (80, 30, 120, 50),    # 中下方障碍物
            (30, 80, 50, 120),    # 左中障碍物
            (150, 80, 170, 120),  # 右中障碍物
            (80, 150, 120, 170)   # 中上方障碍物
        ]
        
        # 添加L形障碍物
        l_shapes = [
            # L形: ((x1,y1,x2,y2), (x3,y3,x4,y4))
            ((40, 40, 60, 60), (40, 40, 80, 50)),
            ((140, 140, 160, 160), (140, 140, 180, 150))
        ]
        
        # 光栅化矩形
        for rect in rectangles:
            x1, y1, x2, y2 = rect
            for x in range(int(x1), int(x2)+1):
                for y in range(int(y1), int(y2)+1):
                    obstacles.append((x, y))
        
        # 光栅化L形
        for l_shape in l_shapes:
            rect1, rect2 = l_shape
            x1, y1, x2, y2 = rect1
            x3, y3, x4, y4 = rect2
            
            # 第一个矩形
            for x in range(int(x1), int(x2)+1):
                for y in range(int(y1), int(y2)+1):
                    obstacles.append((x, y))
            
            # 第二个矩形
            for x in range(int(x3), int(x4)+1):
                for y in range(int(y3), int(y4)+1):
                    obstacles.append((x, y))
        
        return obstacles
    
    def _create_random_obstacles(self):
        """创建随机障碍物点"""
        obstacles = []
        
        # 障碍物密度
        obstacle_density = 0.01
        
        # 障碍物数量与地图大小成比例
        num_obstacles = int(obstacle_density * self.map_size * self.map_size)
        
        # 限制最大数量
        num_obstacles = min(num_obstacles, 300)
        
        for _ in range(num_obstacles):
            x = random.randint(0, self.map_size)
            y = random.randint(0, self.map_size)
            
            # 避免关键点附近有障碍物
            if not any(math.dist((x, y), point) < 15 for point in self.test_points.values()):
                obstacles.append((x, y))
        
        return obstacles
    
    def _draw_map_and_obstacles(self):
        """绘制地图和障碍物"""
        # 绘制测试点
        for name, point in self.test_points.items():
            # 根据点类型选择颜色
            color = 'b'  # 默认蓝色
            if "装载" in name:
                color = 'g'  # 装载点绿色
            elif "卸载" in name:
                color = 'r'  # 卸载点红色
            elif "停车" in name:
                color = 'y'  # 停车场黄色
            
            # 绘制点
            point_item = pg.ScatterPlotItem(
                [point[0]], [point[1]], 
                size=10, 
                pen=pg.mkPen(color),
                brush=pg.mkBrush(color)
            )
            self.main_view.addItem(point_item)
            
            # 点标签
            text_item = pg.TextItem(name, anchor=(0, 0), color=color)
            text_item.setPos(point[0] + 5, point[1] + 5)
            self.main_view.addItem(text_item)
        
        # 绘制障碍物
        if hasattr(self, 'obstacles') and self.obstacles:
            x_coords = [p[0] for p in self.obstacles]
            y_coords = [p[1] for p in self.obstacles]
            
            obstacle_item = pg.ScatterPlotItem(
                x_coords, y_coords,
                size=2, pen=None, brush=pg.mkBrush(100, 100, 100, 150)
            )
            self.main_view.addItem(obstacle_item)
    
    def _start_new_test(self):
        """启动新的测试"""
        current_idx = self.current_test_info['current_idx']
        
        # 检查是否所有测试都已完成
        if current_idx >= len(self.test_pairs):
            self.all_tests_completed = True
            self.update_status_text("所有测试完成!")
            return
        
        # 获取当前测试
        test_pair = self.test_pairs[current_idx]
        start_point = test_pair['start']
        end_point = test_pair['end']
        start_name = test_pair['start_name']
        end_name = test_pair['end_name']
        task_type = test_pair.get('task_type', 'transport')
        
        # 更新测试信息
        self.current_test_info = {
            'current_idx': current_idx,
            'total_tests': len(self.test_pairs),
            'current_test': test_pair
        }
        
        # 测试开始时间
        test_start_time = time.time()
        
        # 更新状态文本
        self.update_status_text(f"开始测试 {current_idx+1}/{len(self.test_pairs)}: {start_name} -> {end_name}")
        
        # 为每个车辆分配测试
        active_count = 0
        for vehicle in self.vehicles:
            # 设置车辆起点
            vehicle.current_location = start_point
            
            # 创建任务
            task = TransportTask(
                task_id=f"T{current_idx}_{vehicle.vehicle_id}",
                start_point=start_point,
                end_point=end_point,
                task_type=task_type,
                priority=1
            )
            
            try:
                # 将任务添加到调度系统
                self.dispatch.add_task(task)
                
                # 执行一次调度周期
                self.dispatch.scheduling_cycle()
                
                # 检查任务是否被分配
                if vehicle.current_task:
                    # 获取规划路径
                    if str(vehicle.vehicle_id) in self.dispatch.vehicle_paths:
                        path = self.dispatch.vehicle_paths[str(vehicle.vehicle_id)]
                        
                        # 确保路径点是二维的 - 添加此转换
                        path = [(p[0], p[1]) if len(p) > 2 else p for p in path]
                        
                        # 保存路径
                        self.vehicle_paths[vehicle] = path
                    
                    # 将车辆添加到活动列表
                    self.active_vehicles.append(vehicle)
                    active_count += 1
                    
                    # 更新统计信息
                    self.test_stats['path_lengths'].append(len(path))
                    self.test_stats['successful_plans'] += 1
                    
                    # 每次最多使用3辆车，以避免冲突过多
                    if active_count >= min(3, self.num_vehicles):
                        break
            except Exception as e:
                logging.error(f"车辆 {vehicle.vehicle_id} 任务分配错误: {str(e)}")        
        # 如果没有活动车辆，进入下一个测试
        if not self.active_vehicles:
            self.current_test_info['current_idx'] += 1
            self._start_new_test()
        
        # 检测冲突
        self.check_path_conflicts()
        
        # 更新总测试计数
        self.test_stats['total_tests'] += 1
    def reset_system(self):
        """重置系统状态，清除所有遗留数据"""
        # 重置调度系统
        self._init_system_components()
        
        # 清除测试数据
        self.vehicles = []
        self.test_points = {}
        self.test_pairs = []
        self.vehicle_paths = {}
        self.active_vehicles = []
        
        # 重置统计数据
        self.test_stats = {
            'planning_times': [],
            'path_lengths': [],
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'successful_plans': 0,
            'failed_plans': 0,
            'total_tests': 0,
            'start_time': None
        }
        
        # 清除热图数据
        self.heatmap_data = np.zeros((self.map_size, self.map_size))
        
        logging.info("系统状态已重置")    
    def _update_vehicles(self):
        """更新车辆位置"""
        # 检查是否有活动车辆
        if not self.active_vehicles:
            return
            
        # 记录已完成的车辆
        completed = []
        
        for vehicle in self.active_vehicles:
            # 检查车辆是否已完成任务
            if vehicle.state == VehicleState.IDLE:
                completed.append(vehicle)
                continue
        
        # 移除已完成的车辆
        for vehicle in completed:
            self.active_vehicles.remove(vehicle)
        
        # 检查是否所有车辆都完成测试
        if not self.active_vehicles:
            # 移动到下一个测试
            self.current_test_info['current_idx'] += 1
    
    def check_path_conflicts(self):
        """检查路径冲突"""
        if not self.active_vehicles or len(self.active_vehicles) < 2:
            self.update_status_text("没有足够的活动车辆进行冲突检测")
            return
            
        # 收集路径
        vehicle_paths = {}
        for vehicle in self.active_vehicles:
            path = self.vehicle_paths.get(vehicle, [])
            if path and len(path) > 1:
                vehicle_paths[str(vehicle.vehicle_id)] = path
        
        if not vehicle_paths or len(vehicle_paths) < 2:
            self.update_status_text("没有足够的有效路径进行冲突检测")
            return
            
        # 检测冲突
        conflicts = self.cbs.find_conflicts(vehicle_paths)
        
        # 更新统计信息
        self.test_stats['conflicts_detected'] += len(conflicts)
        
        # 保存当前冲突
        self.current_conflicts = conflicts
        
        # 更新状态
        self.update_status_text(f"检测到 {len(conflicts)} 个冲突")
        
        return conflicts
    
    def resolve_path_conflicts(self):
        """解决路径冲突"""
        # 首先检测冲突
        conflicts = self.check_path_conflicts()
        
        if not conflicts:
            self.update_status_text("没有需要解决的冲突")
            return
            
        # 收集路径
        vehicle_paths = {}
        for vehicle in self.active_vehicles:
            path = self.vehicle_paths.get(vehicle, [])
            if path and len(path) > 1:
                vehicle_paths[str(vehicle.vehicle_id)] = path
        
        # 使用CBS解决冲突
        resolved_paths = self.cbs.resolve_conflicts(vehicle_paths)
        
        # 计算解决的冲突数
        resolved_count = 0
        for vid_str, new_path in resolved_paths.items():
            if vehicle_paths.get(vid_str) != new_path:
                resolved_count += 1
                
                # 更新车辆路径
                vid = int(vid_str)
                for vehicle in self.active_vehicles:
                    if vehicle.vehicle_id == vid:
                        self.vehicle_paths[vehicle] = new_path
                        
                        # 更新调度系统中的路径
                        self.dispatch.vehicle_paths[vid_str] = new_path
                        
                        # 如果有车辆对象，同步更新
                        if vid in self.dispatch.vehicles:
                            v = self.dispatch.vehicles[vid]
                            v.assign_path(new_path)
                        
                        break
        
        # 更新统计信息
        self.test_stats['conflicts_resolved'] += resolved_count
        
        # 清除冲突列表
        self.current_conflicts = []
        
        # 更新状态
        self.update_status_text(f"已解决 {resolved_count} 个冲突")
    
    def on_test_completed(self, stats):
        """测试完成回调"""
        self.update_status_text("所有测试完成!")
        
        # 启用保存按钮
        self.save_results_btn.setEnabled(True)
        
        # 显示测试结果统计
        self._show_test_results()
    
    def _show_test_results(self):
        """显示测试结果"""
        # 计算平均值
        avg_planning_time = np.mean(self.test_stats['planning_times']) if self.test_stats['planning_times'] else 0
        avg_path_length = np.mean(self.test_stats['path_lengths']) if self.test_stats['path_lengths'] else 0
        
        # 创建结果消息
        result_msg = (
            f"测试完成！\n"
            f"总测试数: {self.test_stats['total_tests']}\n"
            f"规划成功: {self.test_stats['successful_plans']}\n"
            f"规划失败: {self.test_stats['failed_plans']}\n"
            f"成功率: {self.test_stats['successful_plans']/max(1, self.test_stats['total_tests'])*100:.1f}%\n"
            f"平均规划时间: {avg_planning_time*1000:.2f}毫秒\n"
            f"平均路径长度: {avg_path_length:.1f}点\n"
            f"检测冲突: {self.test_stats['conflicts_detected']}\n"
            f"解决冲突: {self.test_stats['conflicts_resolved']}\n"
        )
        
        # 显示消息框
        QMessageBox.information(self, "测试完成", result_msg)
    
    def save_test_results(self):
        """保存测试结果"""
        # 打开文件对话框
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存测试结果", "", "CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if not filename:
            return
            
        # 确保文件有.csv扩展名
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        try:
            # 打开文件
            with open(filename, 'w') as f:
                # 写入标题行
                f.write("测试,起点,终点,任务类型,规划时间(毫秒),路径长度,冲突数,成功\n")
                
                # 写入每次测试的结果
                for i in range(min(len(self.test_pairs), len(self.test_stats['planning_times']))):
                    test_pair = self.test_pairs[i]
                    planning_time = self.test_stats['planning_times'][i] * 1000  # 转换为毫秒
                    path_length = self.test_stats['path_lengths'][i] if i < len(self.test_stats['path_lengths']) else 0
                    
                    # 写入一行
                    f.write(f"{i+1},{test_pair['start_name']},{test_pair['end_name']},{test_pair.get('task_type', 'transport')},{planning_time:.2f},{path_length},0,是\n")
                
                # 写入汇总信息
                f.write("\n总结,,,,,\n")
                
                # 计算平均值
                avg_planning_time = np.mean(self.test_stats['planning_times']) * 1000 if self.test_stats['planning_times'] else 0
                avg_path_length = np.mean(self.test_stats['path_lengths']) if self.test_stats['path_lengths'] else 0
                
                # 写入汇总行
                f.write(f"总测试数,{self.test_stats['total_tests']},,,,,\n")
                f.write(f"规划成功,{self.test_stats['successful_plans']},,,,,\n")
                f.write(f"规划失败,{self.test_stats['failed_plans']},,,,,\n")
                f.write(f"成功率,{self.test_stats['successful_plans']/max(1, self.test_stats['total_tests'])*100:.1f}%,,,,,\n")
                f.write(f"平均规划时间,{avg_planning_time:.2f}毫秒,,,,,\n")
                f.write(f"平均路径长度,{avg_path_length:.1f},,,,,\n")
                f.write(f"检测冲突,{self.test_stats['conflicts_detected']},,,,,\n")
                f.write(f"解决冲突,{self.test_stats['conflicts_resolved']},,,,,\n")
            
            self.update_status_text(f"测试结果已保存到 {filename}")
            
        except Exception as e:
            logging.error(f"保存测试结果时出错: {str(e)}")
            QMessageBox.warning(self, "保存失败", f"保存测试结果时出错: {str(e)}")
    
    def get_test_stats(self):
        """获取测试统计数据"""
        # 计算派生统计数据
        if self.test_stats['total_tests'] > 0:
            success_rate = self.test_stats['successful_plans'] / self.test_stats['total_tests']
        else:
            success_rate = 0
            
        if self.test_stats['conflicts_detected'] > 0:
            conflict_resolution_rate = self.test_stats['conflicts_resolved'] / self.test_stats['conflicts_detected']
        else:
            conflict_resolution_rate = 1.0  # 没有冲突时为100%
        
        # 返回统计数据字典
        return {
            'total_tests': self.test_stats['total_tests'],
            'successful_plans': self.test_stats['successful_plans'],
            'failed_plans': self.test_stats['failed_plans'],
            'success_rate': success_rate,
            'avg_planning_time': np.mean(self.test_stats['planning_times']) if self.test_stats['planning_times'] else 0,
            'avg_path_length': np.mean(self.test_stats['path_lengths']) if self.test_stats['path_lengths'] else 0,
            'conflicts_detected': self.test_stats['conflicts_detected'],
            'conflicts_resolved': self.test_stats['conflicts_resolved'],
            'conflict_resolution_rate': conflict_resolution_rate
        }


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="露天矿多车协同调度系统")
    parser.add_argument("--vehicles", type=int, default=5, help="车辆数量")
    parser.add_argument("--tests", type=int, default=10, help="测试对数量")
    args = parser.parse_args()
    
    # 创建Qt应用程序
    app = QApplication(sys.argv)
    
    # 创建主应用程序
    main_app = MiningDispatchApp()
    
    # 应用命令行参数
    if args.vehicles:
        main_app.vehicles_slider.setValue(args.vehicles)
    
    if args.tests:
        main_app.tests_slider.setValue(args.tests)
    
    # 显示窗口
    main_app.show()
    
    # 运行应用程序
    sys.exit(app.exec_())