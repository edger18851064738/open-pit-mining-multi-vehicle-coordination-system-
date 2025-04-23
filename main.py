#!/usr/bin/env python3
"""
露天矿多车协同调度系统 - 主程序
==============================================

此程序集成了调度器接口和调度服务，并提供了可视化界面，用于:
1. 测试并可视化不同调度算法在复杂地图上的表现
2. 切换不同的调度策略（CBS、QMIX等）
3. 监控性能指标和冲突解决效果
4. 实时显示车辆路径和调度状态
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
import heapq
from collections import deque 
# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# PyQt和PyQtGraph导入
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, 
    QLabel, QPushButton, QGridLayout, QComboBox, QSlider, QCheckBox, 
    QFileDialog, QTextEdit, QGroupBox, QSplitter, QTabWidget, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QPoint
from PyQt5.QtGui import QColor, QPen, QBrush, QFont
import pyqtgraph as pg

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
    from algorithm.dispatch_service import DispatchSystem
    from algorithm.dispatcher_interface import (
        DispatcherInterface, load_dispatchers_from_directory,
        get_registered_dispatchers, create_dispatcher, DispatcherConfig
    )
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
    
    def __init__(self, manager):
        super().__init__()
        self.manager = manager
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
                
                # 执行调度循环
                if self.manager.dispatcher:
                    self.manager.dispatcher.scheduling_cycle()
                
                # 更新车辆位置（如果不是使用调度器自动更新）
                if not self.manager.use_dispatcher_movement:
                    self.manager._update_vehicles()
                
                # 如果所有测试完成，发送测试完成信号
                if self.manager.all_tests_completed:
                    final_stats = self.manager.get_simulation_stats()
                    self.finished_signal.emit(final_stats)
                    self.running = False
                    break
                
                # 如果没有活动车辆且测试模式，启动新测试
                if self.manager.test_mode and len(self.manager.active_vehicles) == 0 and not self.manager.all_tests_completed:
                    self.manager._start_new_test()
                
                # 定时更新UI
                if current_time - last_update_time >= update_interval:
                    # 准备更新数据
                    update_data = {
                        'vehicles': {v.vehicle_id: v.current_location for v in self.manager.vehicles},
                        'vehicle_states': {v.vehicle_id: v.state for v in self.manager.vehicles},
                        'paths': {v.vehicle_id: v.current_path 
                                 for v in self.manager.vehicles if hasattr(v, 'current_path') and v.current_path},
                        'stats': self.manager.dispatcher.get_status() if self.manager.dispatcher else {},
                        'test_info': self.manager.current_test_info if hasattr(self.manager, 'current_test_info') else {},
                        'tasks': {t.task_id: {
                            'status': t.status,
                            'start': t.start_point,
                            'end': t.end_point,
                            'assigned_to': t.assigned_to
                        } for t in self.manager.tasks} if hasattr(self.manager, 'tasks') else {}
                    }
                    
                    # 发送更新信号
                    self.update_signal.emit(update_data)
                    last_update_time = current_time
            
            # 控制更新速度
            sim_delay = 0.01 / self.manager.simulation_speed
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


class MiningDispatchManager(QMainWindow):
    """露天矿调度系统管理器"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("露天矿多车协同调度系统")
        self.resize(1280, 800)
        
        # 初始化参数
        self.map_size = 200
        self.num_vehicles = 5
        self.num_tasks = 10
        self.simulation_speed = 1.0
        self.show_paths = True
        self.show_tasks = True
        
        # 测试与模拟设置
        self.test_mode = False
        self.use_dispatcher_movement = True
        self.auto_assign_tasks = True
        self.test_running = False
        self.test_paused = False
        self.all_tests_completed = False
        
        # 加载可用的调度器
        self.available_dispatchers = self._load_dispatchers()
        
        # 初始化地图和规划器
        self._init_map_and_planner()
        
        # 初始化调度器（默认为None，由用户选择）
        self.dispatcher = None
        self.selected_dispatcher_name = None
        
        # 模拟组件
        self.vehicles = []
        self.tasks = []
        self.active_vehicles = []
        self.obstacle_areas = []
        
        # 创建模拟线程
        self.sim_thread = SimulationThread(self)
        self.sim_thread.update_signal.connect(self.update_display)
        self.sim_thread.finished_signal.connect(self.on_simulation_completed)
        
        # 创建UI
        self.setup_ui()
        
        # 热图数据
        self.heatmap_data = np.zeros((self.map_size, self.map_size))
        
        logging.info("露天矿调度系统初始化完成")
    
    def _load_dispatchers(self) -> Dict[str, type]:
        """加载所有可用的调度器"""
        # 先加载目录中的调度器
        load_dispatchers_from_directory()
        
        # 获取所有注册的调度器
        dispatchers = get_registered_dispatchers()
        
        if not dispatchers:
            logging.warning("未找到可用的调度器")
            return {"none": None}
            
        logging.info(f"找到 {len(dispatchers)} 个可用调度器: {', '.join(dispatchers.keys())}")
        return dispatchers
    
    def _init_map_and_planner(self):
        """初始化地图和路径规划器"""
        # 创建地图服务和路径规划器
        try:
            self.geo_utils = GeoUtils()
            self.map_service = MapService()
            self.planner = HybridPathPlanner(self.map_service)
            self.cbs = ConflictBasedSearch(self.planner)
            logging.info("地图和规划器初始化成功")
        except Exception as e:
            logging.error(f"初始化地图和规划器时出错: {str(e)}")
            QMessageBox.critical(self, "初始化错误", f"初始化地图和规划器失败: {str(e)}")
    
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
        self.main_view = pg.PlotWidget(title="露天矿车辆调度地图")
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
        
        # 模拟参数设置
        params_group = QGroupBox("模拟参数")
        params_layout = QGridLayout()
        params_group.setLayout(params_layout)
        
        # 地图尺寸
        params_layout.addWidget(QLabel("地图尺寸:"), 0, 0)
        self.map_size_combo = QComboBox()
        self.map_size_combo.addItems(["100x100", "200x200", "300x300", "400x400"])
        self.map_size_combo.setCurrentIndex(1)  # 默认200x200
        self.map_size_combo.currentIndexChanged.connect(self.on_map_size_changed)
        params_layout.addWidget(self.map_size_combo, 0, 1)
        
        # 车辆数量
        params_layout.addWidget(QLabel("车辆数量:"), 1, 0)
        self.vehicles_slider = QSlider(Qt.Horizontal)
        self.vehicles_slider.setMinimum(1)
        self.vehicles_slider.setMaximum(20)
        self.vehicles_slider.setValue(5)
        self.vehicles_slider.setTickPosition(QSlider.TicksBelow)
        self.vehicles_slider.setTickInterval(1)
        self.vehicles_slider.valueChanged.connect(self.on_vehicles_changed)
        params_layout.addWidget(self.vehicles_slider, 1, 1)
        self.vehicles_label = QLabel("5")
        params_layout.addWidget(self.vehicles_label, 1, 2)
        
        # 任务数量
        params_layout.addWidget(QLabel("任务数量:"), 2, 0)
        self.tasks_slider = QSlider(Qt.Horizontal)
        self.tasks_slider.setMinimum(5)
        self.tasks_slider.setMaximum(50)
        self.tasks_slider.setValue(10)
        self.tasks_slider.setTickPosition(QSlider.TicksBelow)
        self.tasks_slider.setTickInterval(5)
        self.tasks_slider.valueChanged.connect(self.on_tasks_changed)
        params_layout.addWidget(self.tasks_slider, 2, 1)
        self.tasks_label = QLabel("10")
        params_layout.addWidget(self.tasks_label, 2, 2)
        
        # 是否使用复杂地图
        params_layout.addWidget(QLabel("复杂地图:"), 3, 0)
        self.complex_map_checkbox = QCheckBox()
        self.complex_map_checkbox.setChecked(True)
        params_layout.addWidget(self.complex_map_checkbox, 3, 1)
        
        # 测试模式
        params_layout.addWidget(QLabel("测试模式:"), 4, 0)
        self.test_mode_checkbox = QCheckBox()
        self.test_mode_checkbox.setChecked(False)
        self.test_mode_checkbox.stateChanged.connect(self.on_test_mode_changed)
        params_layout.addWidget(self.test_mode_checkbox, 4, 1)
        
        control_layout.addWidget(params_group)
        
        # 调度器设置
        dispatcher_group = QGroupBox("调度器设置")
        dispatcher_layout = QGridLayout()
        dispatcher_group.setLayout(dispatcher_layout)
        
        # 调度器选择
        dispatcher_layout.addWidget(QLabel("调度算法:"), 0, 0)
        self.dispatcher_combo = QComboBox()
        self.dispatcher_combo.addItems(list(self.available_dispatchers.keys()))
        self.dispatcher_combo.currentIndexChanged.connect(self.on_dispatcher_changed)
        dispatcher_layout.addWidget(self.dispatcher_combo, 0, 1)
        
        # 自动分配任务
        dispatcher_layout.addWidget(QLabel("自动分配任务:"), 1, 0)
        self.auto_assign_checkbox = QCheckBox()
        self.auto_assign_checkbox.setChecked(True)
        self.auto_assign_checkbox.stateChanged.connect(self.on_auto_assign_changed)
        dispatcher_layout.addWidget(self.auto_assign_checkbox, 1, 1)
        
        # 使用调度器移动
        dispatcher_layout.addWidget(QLabel("使用调度器移动:"), 2, 0)
        self.dispatcher_movement_checkbox = QCheckBox()
        self.dispatcher_movement_checkbox.setChecked(True)
        self.dispatcher_movement_checkbox.stateChanged.connect(self.on_dispatcher_movement_changed)
        dispatcher_layout.addWidget(self.dispatcher_movement_checkbox, 2, 1)
        
        control_layout.addWidget(dispatcher_group)
        
        # 控制按钮
        buttons_group = QGroupBox("模拟控制")
        buttons_layout = QGridLayout()
        buttons_group.setLayout(buttons_layout)
        
        # 启动模拟按钮
        self.start_btn = QPushButton("启动模拟")
        self.start_btn.clicked.connect(self.start_simulation)
        buttons_layout.addWidget(self.start_btn, 0, 0)
        
        # 暂停模拟按钮
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
        
        buttons_layout.addWidget(QLabel("显示任务:"), 3, 0)
        self.show_tasks_checkbox = QCheckBox()
        self.show_tasks_checkbox.setChecked(True)
        self.show_tasks_checkbox.stateChanged.connect(self.on_show_tasks_changed)
        buttons_layout.addWidget(self.show_tasks_checkbox, 3, 1)
        
        control_layout.addWidget(buttons_group)
        
        # 功能按钮
        functions_group = QGroupBox("功能")
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
        
        # 添加随机任务按钮
        self.add_task_btn = QPushButton("添加随机任务")
        self.add_task_btn.clicked.connect(self.add_random_task)
        self.add_task_btn.setEnabled(False)
        functions_layout.addWidget(self.add_task_btn, 1, 0)
        
        # 保存结果按钮
        self.save_results_btn = QPushButton("保存结果")
        self.save_results_btn.clicked.connect(self.save_simulation_results)
        self.save_results_btn.setEnabled(False)
        functions_layout.addWidget(self.save_results_btn, 1, 1)
        
        control_layout.addWidget(functions_group)
        
        # 弹性空间
        control_layout.addStretch()
        
        # 下部分：左侧状态信息
        status_group = QGroupBox("状态信息")
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
        self.update_status_text("准备就绪，请设置参数后点击'启动模拟'")
        self.update_stats_text("尚未开始模拟")
    
    def on_map_size_changed(self, index):
        """地图尺寸变更处理"""
        size_text = self.map_size_combo.currentText()
        self.map_size = int(size_text.split('x')[0])
        self.main_view.setRange(xRange=(0, self.map_size), yRange=(0, self.map_size))
        
        self.update_status_text(f"地图尺寸已更改为 {self.map_size}x{self.map_size}")
    
    def on_vehicles_changed(self, value):
        """车辆数量变更处理"""
        self.num_vehicles = value
        self.vehicles_label.setText(str(value))
        self.update_status_text(f"车辆数量已更改为 {value}")
    
    def on_tasks_changed(self, value):
        """任务数量变更处理"""
        self.num_tasks = value
        self.tasks_label.setText(str(value))
        self.update_status_text(f"任务数量已更改为 {value}")
    
    def on_speed_changed(self, value):
        """模拟速度变更处理"""
        self.simulation_speed = value / 10.0
        self.speed_label.setText(f"{self.simulation_speed:.1f}x")
    
    def on_test_mode_changed(self, state):
        """测试模式变更处理"""
        self.test_mode = state == Qt.Checked
        
        # 更新UI元素可用性
        self.tasks_slider.setEnabled(not self.test_mode)
        self.auto_assign_checkbox.setEnabled(not self.test_mode)
        
        self.update_status_text(f"测试模式: {'开启' if self.test_mode else '关闭'}")
    
    def on_dispatcher_changed(self, index):
        """调度器变更处理"""
        self.selected_dispatcher_name = self.dispatcher_combo.currentText()
        self.update_status_text(f"已选择调度器: {self.selected_dispatcher_name}")
    
    def on_auto_assign_changed(self, state):
        """自动分配任务变更处理"""
        self.auto_assign_tasks = state == Qt.Checked
        self.update_status_text(f"自动分配任务: {'开启' if self.auto_assign_tasks else '关闭'}")
    
    def on_dispatcher_movement_changed(self, state):
        """调度器移动控制变更处理"""
        self.use_dispatcher_movement = state == Qt.Checked
        self.update_status_text(f"使用调度器移动: {'开启' if self.use_dispatcher_movement else '关闭'}")
    
    def on_show_paths_changed(self, state):
        """路径显示变更处理"""
        self.show_paths = state == Qt.Checked
        self.update_display({})  # 触发重绘
    
    def on_show_tasks_changed(self, state):
        """任务显示变更处理"""
        self.show_tasks = state == Qt.Checked
        self.update_display({})  # 触发重绘
    
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
            # 清除视图
            if not hasattr(self, 'initialized_display') or not self.initialized_display:
                self.main_view.clear()
                self._draw_map_and_obstacles()
                self.initialized_display = True
            
            # 绘制车辆
            self._draw_vehicles(update_data.get('vehicles', {}), update_data.get('vehicle_states', {}))
            
            # 绘制路径
            if self.show_paths:
                self._draw_paths(update_data.get('paths', {}))
            
            # 绘制任务
            if self.show_tasks:
                self._draw_tasks(update_data.get('tasks', {}))
            
            # 更新统计信息
            if 'stats' in update_data:
                self._update_stats_display(update_data['stats'])
            
            # 刷新应用
            QApplication.processEvents()
            
        except Exception as e:
            logging.error(f"更新显示时出错: {str(e)}")
    
    def _draw_vehicles(self, vehicles_positions, vehicle_states):
        """绘制车辆"""
        # 清除现有车辆点
        if hasattr(self, 'vehicle_markers'):
            for marker in self.vehicle_markers:
                self.main_view.removeItem(marker)
        
        self.vehicle_markers = []
        
        # 绘制车辆标记
        for vid, pos in vehicles_positions.items():
            # 获取车辆状态
            state = vehicle_states.get(vid, None)
            
            # 根据状态设置颜色
            color = QColor('blue')  # 默认颜色
            if state is not None:
                if hasattr(state, 'name'):  # 如果是枚举
                    state_name = state.name
                else:
                    state_name = str(state)
                    
                if 'IDLE' in state_name:
                    color = QColor('green')
                elif 'PREPARING' in state_name:
                    color = QColor('yellow')
                elif 'EN_ROUTE' in state_name:
                    color = QColor('blue')
                elif 'UNLOADING' in state_name:
                    color = QColor('orange')
                elif 'EMERGENCY' in state_name:
                    color = QColor('red')
            
            # 创建车辆标记
            vehicle_marker = pg.ScatterPlotItem(
                [pos[0]], [pos[1]],
                size=15, pen=pg.mkPen(color, width=2), brush=pg.mkBrush(color, alpha=100)
            )
            
            # 添加车辆ID标签
            text_item = pg.TextItem(str(vid), anchor=(0.5, 0.5), color='white')
            text_item.setPos(pos[0], pos[1])
            
            self.main_view.addItem(vehicle_marker)
            self.main_view.addItem(text_item)
            
            self.vehicle_markers.append(vehicle_marker)
            self.vehicle_markers.append(text_item)
    
    def _draw_paths(self, vehicle_paths):
        """绘制车辆路径"""
        # 清除现有路径
        if hasattr(self, 'path_items'):
            for path_item in self.path_items:
                self.main_view.removeItem(path_item)
        
        self.path_items = []
        
        # 绘制每个车辆的路径
        for vid, path in vehicle_paths.items():
            if not path or len(path) < 2:
                continue
                
            # 提取路径点坐标
            x_data = [p[0] for p in path]
            y_data = [p[1] for p in path]
            
            # 创建路径线
            path_line = pg.PlotDataItem(
                x_data, y_data,
                pen=pg.mkPen(color=(100, 100, 255), width=2, style=Qt.DashLine),
                name=f"车辆{vid}路径"
            )
            
            self.main_view.addItem(path_line)
            self.path_items.append(path_line)
    
    def _draw_tasks(self, tasks):
        """绘制任务"""
        # 清除现有任务标记
        if hasattr(self, 'task_markers'):
            for marker in self.task_markers:
                self.main_view.removeItem(marker)
        
        self.task_markers = []
        
        # 绘制每个任务
        for task_id, task_info in tasks.items():
            start_point = task_info.get('start')
            end_point = task_info.get('end')
            status = task_info.get('status')
            assigned_to = task_info.get('assigned_to')
            
            # 根据任务状态设置颜色
            color = QColor('gray')  # 默认颜色
            if status == 'pending':
                color = QColor('cyan')
            elif status == 'assigned':
                color = QColor('magenta')
            elif status == 'in_progress':
                color = QColor('yellow')
            elif status == 'completed':
                color = QColor('green')
            elif status == 'failed':
                color = QColor('red')
            
            # 绘制起点标记
            if start_point:
                start_marker = pg.ScatterPlotItem(
                    [start_point[0]], [start_point[1]],
                    size=12, pen=pg.mkPen(color, width=2), brush=pg.mkBrush('white'),
                    symbol='o'
                )
                
                self.main_view.addItem(start_marker)
                self.task_markers.append(start_marker)
                
                # 添加任务ID标签
                text_item = pg.TextItem(f"任务{task_id}", anchor=(0, 0), color=color)
                text_item.setPos(start_point[0] + 5, start_point[1] + 5)
                
                self.main_view.addItem(text_item)
                self.task_markers.append(text_item)
            
            # 绘制终点标记
            if end_point:
                end_marker = pg.ScatterPlotItem(
                    [end_point[0]], [end_point[1]],
                    size=12, pen=pg.mkPen(color, width=2), brush=pg.mkBrush(color, alpha=150),
                    symbol='t'
                )
                
                self.main_view.addItem(end_marker)
                self.task_markers.append(end_marker)
            
            # 如果已分配，绘制连接线
            if start_point and end_point and assigned_to:
                task_line = pg.PlotDataItem(
                    [start_point[0], end_point[0]], [start_point[1], end_point[1]],
                    pen=pg.mkPen(color, width=1, style=Qt.DotLine)
                )
                
                self.main_view.addItem(task_line)
                self.task_markers.append(task_line)
    
    def _update_stats_display(self, stats):
        """更新统计信息显示"""
        try:
            # 提取车辆统计数据
            vehicles = stats.get('vehicles', {})
            tasks = stats.get('tasks', {})
            metrics = stats.get('metrics', {})
            
            # 格式化HTML显示
            stats_html = "<h3>调度系统状态</h3>"
            stats_html += "<table>"
            
            # 车辆状态
            if isinstance(vehicles, dict) and 'total' in vehicles:
                stats_html += f"<tr><td>车辆总数:</td><td>{vehicles['total']}</td></tr>"
                stats_html += f"<tr><td>空闲车辆:</td><td>{vehicles['idle']}</td></tr>"
                stats_html += f"<tr><td>活动车辆:</td><td>{vehicles['active']}</td></tr>"
            else:
                stats_html += f"<tr><td>车辆总数:</td><td>{len(self.vehicles)}</td></tr>"
            
            # 任务状态
            if isinstance(tasks, dict):
                stats_html += f"<tr><td>队列任务:</td><td>{tasks.get('queued', 0)}</td></tr>"
                stats_html += f"<tr><td>活动任务:</td><td>{tasks.get('active', 0)}</td></tr>"
                stats_html += f"<tr><td>完成任务:</td><td>{tasks.get('completed', 0)}</td></tr>"
            else:
                stats_html += f"<tr><td>任务总数:</td><td>{len(self.tasks)}</td></tr>"
            
            # 性能指标
            stats_html += f"<tr><td>检测冲突:</td><td>{metrics.get('conflicts_detected', 0)}</td></tr>"
            stats_html += f"<tr><td>解决冲突:</td><td>{metrics.get('conflicts_resolved', 0)}</td></tr>"
            stats_html += f"<tr><td>路径规划次数:</td><td>{metrics.get('planning_count', 0)}</td></tr>"
            
            if 'last_cycle_time' in metrics:
                stats_html += f"<tr><td>最近周期时间:</td><td>{metrics['last_cycle_time']*1000:.2f}毫秒</td></tr>"
            
            stats_html += "</table>"
            
            self.update_stats_text(stats_html)
            
        except Exception as e:
            logging.error(f"更新统计信息出错: {str(e)}")
    
    def _draw_map_and_obstacles(self):
        """绘制地图和障碍物"""
        # 创建障碍物
        self.obstacles = self._create_obstacles()
        
        # 绘制障碍物
        if self.obstacles:
            x_coords = [p[0] for p in self.obstacles]
            y_coords = [p[1] for p in self.obstacles]
            
            obstacle_item = pg.ScatterPlotItem(
                x_coords, y_coords,
                size=2, pen=None, brush=pg.mkBrush(100, 100, 100, 150)
            )
            self.main_view.addItem(obstacle_item)
        
        # 绘制关键点
        key_points = self._create_key_points()
        for name, point in key_points.items():
            # 点标记
            point_item = pg.ScatterPlotItem(
                [point[0]], [point[1]], 
                size=15, pen=pg.mkPen('b'), brush=pg.mkBrush(100, 100, 255, 150),
                symbol='s'
            )
            self.main_view.addItem(point_item)
            
            # 点标签
            text_item = pg.TextItem(name, anchor=(0, 0), color='b')
            text_item.setPos(point[0] + 5, point[1] + 5)
            self.main_view.addItem(text_item)
    
    def _create_key_points(self) -> Dict[str, Tuple[float, float]]:
        """创建关键点"""
        points = {}
        
        # 网格尺寸
        grid_step = self.map_size / 10
        
        # 关键点位置
        key_points = {
            "基地": (self.map_size // 2, self.map_size - grid_step),
            "装载点1": (grid_step, self.map_size - grid_step),
            "装载点2": (self.map_size - grid_step, self.map_size - grid_step),
            "卸载点1": (grid_step, grid_step),
            "卸载点2": (self.map_size - grid_step, grid_step),
            "停车场": (self.map_size // 2, grid_step),
            "维修站": (self.map_size // 2, self.map_size // 2)
        }
        
        points.update(key_points)
        return points
    
    def _create_obstacles(self) -> List[Tuple[int, int]]:
        """创建迷宫式障碍物点集"""
        obstacles = []
        
        # 创建迷宫式障碍物布局
        obstacles.extend(self._create_maze_obstacles())
        
        # 验证关键点之间的连通性
        key_points = self._create_key_points()
        key_point_positions = list(key_points.values())
        
        # 确保关键点周围的区域是可通行的
        for point in key_point_positions:
            self._ensure_path_around_point(point, obstacles, clearance=15)
        
        # 验证并确保关键点之间的连通性
        self._ensure_connectivity(key_point_positions, obstacles)
        
        logging.info(f"创建了 {len(obstacles)} 个障碍物点")
        return obstacles

    def _create_maze_obstacles(self) -> List[Tuple[int, int]]:
        """创建简化迷宫式障碍物"""
        obstacles = []
        
        # 减小墙壁宽度
        wall_width = 4  # 从8减少到4
        
        # 简化垂直墙，减少数量
        vertical_walls = [
            (0.2, 0.2, 0.2, 0.6),     # 左侧竖墙
            (0.6, 0.3, 0.6, 0.7),     # 中间竖墙
        ]
        
        # 简化水平墙，减少数量
        horizontal_walls = [
            (0.1, 0.3, 0.4, 0.3),     # 上部横墙
            (0.4, 0.7, 0.8, 0.7),     # 下部横墙
        ]
        
        # 减少大型障碍物区域
        obstacle_areas = [
            (0.75, 0.1, 0.85, 0.2),   # 右上角小障碍
        ]
        
        # 转换相对坐标为实际地图坐标并生成墙壁
        for x1, y1, x2, y2 in vertical_walls:
            x1, y1 = int(x1 * self.map_size), int(y1 * self.map_size)
            x2, y2 = int(x2 * self.map_size), int(y2 * self.map_size)
            
            # 创建竖直墙壁
            for y in range(y1, y2 + 1):
                for x in range(x1 - wall_width//2, x1 + wall_width//2 + 1):
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        obstacles.append((x, y))
        
        for x1, y1, x2, y2 in horizontal_walls:
            x1, y1 = int(x1 * self.map_size), int(y1 * self.map_size)
            x2, y2 = int(x2 * self.map_size), int(y2 * self.map_size)
            
            # 创建水平墙壁
            for x in range(x1, x2 + 1):
                for y in range(y1 - wall_width//2, y1 + wall_width//2 + 1):
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        obstacles.append((x, y))
        
        # 创建障碍区域
        for x1, y1, x2, y2 in obstacle_areas:
            x1, y1 = int(x1 * self.map_size), int(y1 * self.map_size)
            x2, y2 = int(x2 * self.map_size), int(y2 * self.map_size)
            
            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    if 0 <= x < self.map_size and 0 <= y < self.map_size:
                        obstacles.append((x, y))
        
        # 减少随机障碍物数量
        num_random_obstacles = int(self.map_size * self.map_size * 0.001)  # 从0.5%减少到0.1%
        
        for _ in range(num_random_obstacles):
            # 随机位置
            x = random.randint(0, self.map_size - 1)
            y = random.randint(0, self.map_size - 1)
            
            # 创建更小的障碍区域
            size = random.randint(2, 4)  # 从3-7减小到2-4
            for dx in range(-size//2, size//2 + 1):
                for dy in range(-size//2, size//2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        obstacles.append((nx, ny))
        
        return obstacles

    def _ensure_path_around_point(self, point, obstacles, clearance=15):
        """确保关键点周围区域可通行"""
        x, y = point
        
        # 清除关键点周围的障碍物
        obstacles_to_remove = []
        for obstacle in obstacles:
            ox, oy = obstacle
            if math.sqrt((ox - x)**2 + (oy - y)**2) < clearance:
                obstacles_to_remove.append(obstacle)
        
        for obstacle in obstacles_to_remove:
            if obstacle in obstacles:
                obstacles.remove(obstacle)

    def _ensure_connectivity(self, key_points, obstacles):
        """确保关键点之间有可行的路径"""
        # 使用A*算法检查关键点之间是否有路径
        # 如果没有，尝试移除部分障碍物创建路径
        
        def is_path_possible(start, end, obstacles_set):
            """使用简化版A*检查两点间是否有可行路径"""
            open_set = [(0, start)]
            closed_set = set()
            came_from = {}
            g_score = {start: 0}
            
            while open_set:
                _, current = heapq.heappop(open_set)
                
                if current[0] == end[0] and current[1] == end[1]:
                    return True
                    
                if current in closed_set:
                    continue
                    
                closed_set.add(current)
                
                # 8个方向的移动
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                            (1, 1), (1, -1), (-1, 1), (-1, -1)]
                
                for dx, dy in directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # 检查边界
                    if not (0 <= neighbor[0] < self.map_size and 0 <= neighbor[1] < self.map_size):
                        continue
                        
                    # 检查障碍物
                    if neighbor in obstacles_set:
                        continue
                        
                    # 计算新的g值
                    tentative_g = g_score[current] + math.sqrt(dx**2 + dy**2)
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_value = tentative_g + math.sqrt((neighbor[0] - end[0])**2 + (neighbor[1] - end[1])**2)
                        heapq.heappush(open_set, (f_value, neighbor))
            
            return False
        
        obstacles_set = set(obstacles)
        
        # 检查每对关键点的连通性
        for i in range(len(key_points)):
            for j in range(i+1, len(key_points)):
                start = key_points[i]
                end = key_points[j]
                
                # 检查是否可达
                if not is_path_possible(start, end, obstacles_set):
                    # 如果不可达，创建一条路径
                    logging.info(f"创建从 {start} 到 {end} 的路径")
                    self._create_path_between(start, end, obstacles)
                    
                    # 更新障碍物集合
                    obstacles_set = set(obstacles)

    def _create_path_between(self, start, end, obstacles):
        """在两点之间创建一条路径，移除必要的障碍物"""
        # 计算直线路径上的点
        path_points = []
        
        # 使用Bresenham算法获取路径线
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        sx = 1 if start[0] < end[0] else -1
        sy = 1 if start[1] < end[1] else -1
        err = dx - dy
        
        x, y = start
        while (x, y) != end:
            path_points.append((x, y))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        path_points.append(end)
        
        # 沿路径移除障碍物，并创建宽度更大的通道
        path_width = 15  # 从10增加到15
        obstacles_to_remove = []
        
        for x, y in path_points:
            for dx in range(-path_width//2, path_width//2 + 1):
                for dy in range(-path_width//2, path_width//2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                        obstacle = (nx, ny)
                        if obstacle in obstacles:
                            obstacles_to_remove.append(obstacle)
        
        # 实际移除障碍物
        for obstacle in obstacles_to_remove:
            if obstacle in obstacles:
                obstacles.remove(obstacle)
        
        # 对于避免使路径过于复杂，我们可以简化路径弯曲部分，或提高操作成功的阈值
        if len(path_points) > 10:  # 从5增加到10，减少弯曲
            midpoint_idx = len(path_points) // 2
            midpoint = path_points[midpoint_idx]
            
            # 添加偏移，但减小偏移距离
            offset_distance = random.randint(10, 20)  # 从20-40减小到10-20
            perpendicular_x = -(end[1] - start[1])
            perpendicular_y = end[0] - start[0]
            
            # 标准化并缩放
            length = math.sqrt(perpendicular_x**2 + perpendicular_y**2)
            if length > 0:
                perpendicular_x = perpendicular_x / length * offset_distance
                perpendicular_y = perpendicular_y / length * offset_distance
            
            # 计算偏移点
            offset_point = (int(midpoint[0] + perpendicular_x), int(midpoint[1] + perpendicular_y))
            
            # 确保偏移点在地图范围内
            offset_point = (
                max(0, min(self.map_size - 1, offset_point[0])),
                max(0, min(self.map_size - 1, offset_point[1]))
            )
            
            # 递归地创建从起点到偏移点，以及从偏移点到终点的路径
            self._create_path_between(start, offset_point, obstacles)
            self._create_path_between(offset_point, end, obstacles)
    
    def _create_simple_obstacles(self) -> List[Tuple[int, int]]:
        """创建简单障碍物"""
        obstacles = []
        
        # 创建矩形障碍物
        middle_x = self.map_size // 2
        middle_y = self.map_size // 2
        width = self.map_size // 4
        height = self.map_size // 10
        
        # 水平障碍物
        for x in range(middle_x - width//2, middle_x + width//2):
            for y in range(middle_y - height//2, middle_y + height//2):
                obstacles.append((x, y))
        
        return obstacles
    
    def _create_rectangular_obstacles(self) -> List[Tuple[int, int]]:
        """创建矩形障碍物"""
        obstacles = []
        
        # 定义几个矩形障碍物区域: (x1, y1, x2, y2)
        rectangles = [
            (80, 30, 120, 80),    # 中下方障碍物
            (30, 80, 80, 120),    # 左中障碍物
            (120, 80, 170, 120),  # 右中障碍物
            (80, 120, 120, 170)   # 中上方障碍物
        ]
        
        # 光栅化矩形
        for rect in rectangles:
            x1, y1, x2, y2 = rect
            for x in range(int(x1), int(x2)+1):
                for y in range(int(y1), int(y2)+1):
                    obstacles.append((x, y))
        
        return obstacles
    
    def _create_polygon_obstacles(self) -> List[List[Tuple[int, int]]]:
        """创建多边形障碍物"""
        polygons = []
        
        # 定义几个多边形（顶点列表）
        poly1 = [(60, 30), (90, 50), (80, 70), (50, 60)]  # 不规则四边形
        poly2 = [(120, 140), (140, 150), (150, 130), (130, 120)]  # 不规则四边形
        poly3 = [(170, 40), (180, 60), (190, 60), (180, 40)]  # 不规则四边形
        
        # 如果是复杂地图，添加更多形状
        if self.complex_map_checkbox.isChecked():
            # 添加五边形
            poly4 = [(30, 140), (40, 160), (30, 180), (20, 170), (20, 150)]
            polygons.append(poly4)
            
            # 添加更大的多边形
            poly5 = [(130, 30), (150, 40), (160, 30), (150, 20), (130, 20)]
            polygons.append(poly5)
        
        polygons.extend([poly1, poly2, poly3])
        return polygons
    
    def _rasterize_polygon(self, polygon: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """将多边形光栅化为点集"""
        if not polygon or len(polygon) < 3:
            return []
            
        points = []
        
        # 找出多边形的边界框
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # 对边界框内的每个点检查是否在多边形内
        for x in range(int(min_x), int(max_x) + 1):
            for y in range(int(min_y), int(max_y) + 1):
                if self._point_in_polygon((x, y), polygon):
                    points.append((x, y))
                    
        return points
    
    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """判断点是否在多边形内 (射线法)"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    def _create_random_obstacles(self) -> List[Tuple[int, int]]:
        """创建随机障碍物点"""
        obstacles = []
        
        # 障碍物数量与地图大小成比例
        num_obstacles = int(0.01 * self.map_size * self.map_size)
        
        # 限制最大数量，避免创建太多障碍物
        num_obstacles = min(num_obstacles, 500)
        
        # 获取关键点以避开它们
        key_points = list(self._create_key_points().values())
        
        for _ in range(num_obstacles):
            x = random.randint(0, self.map_size)
            y = random.randint(0, self.map_size)
            
            # 避免关键点附近有障碍物
            if not any(math.dist((x, y), point) < 15 for point in key_points):
                obstacles.append((x, y))
        
        return obstacles
    
    def start_simulation(self):
        """启动模拟"""
        if self.test_running:
            # 如果模拟已在运行，停止它
            self.stop_simulation()
            return
        
        try:
            # 检查是否选择了调度器
            if not self.selected_dispatcher_name or self.selected_dispatcher_name == "none":
                QMessageBox.warning(self, "未选择调度器", "请先选择一个调度算法")
                return
            
            # 重置模拟状态
            self.all_tests_completed = False
            self.test_running = True
            self.test_paused = False
            
            # 创建模拟组件
            self._create_simulation_components()
            
            # 更新UI
            self.start_btn.setText("停止模拟")
            self.pause_btn.setEnabled(True)
            self.check_conflicts_btn.setEnabled(True)
            self.resolve_conflicts_btn.setEnabled(True)
            self.add_task_btn.setEnabled(True)
            self.save_results_btn.setEnabled(False)
            
            # 禁用模拟参数设置
            self._set_params_enabled(False)
            
            # 清除视图并重新绘制
            self.main_view.clear()
            self.initialized_display = False  # 强制重新初始化显示
            
            # 启动模拟线程
            self.sim_thread.start()
            
            self.update_status_text("模拟已启动")
            
        except Exception as e:
            logging.error(f"启动模拟时出错: {str(e)}")
            self.update_status_text(f"启动模拟失败: {str(e)}")
            self.test_running = False
    
    def stop_simulation(self):
        """停止模拟"""
        if not self.test_running:
            return
            
        # 停止模拟线程
        self.sim_thread.stop()
        
        # 如果有调度器，停止它
        if self.dispatcher:
            try:
                self.dispatcher.stop_scheduling()
            except:
                pass
        
        # 更新状态
        self.test_running = False
        self.test_paused = False
        
        # 更新UI
        self.start_btn.setText("启动模拟")
        self.pause_btn.setEnabled(False)
        self.check_conflicts_btn.setEnabled(False)
        self.resolve_conflicts_btn.setEnabled(False)
        self.add_task_btn.setEnabled(False)
        self.save_results_btn.setEnabled(True)
        
        # 启用模拟参数设置
        self._set_params_enabled(True)
        
        self.update_status_text("模拟已停止")
    
    def toggle_pause(self):
        """暂停/继续模拟"""
        if not self.test_running:
            return
            
        self.test_paused = not self.test_paused
        
        if self.test_paused:
            # 暂停模拟线程
            self.sim_thread.pause()
            
            # 如果有调度器，暂停它
            if self.dispatcher:
                try:
                    self.dispatcher.pause_scheduling()
                except:
                    pass
                
            self.pause_btn.setText("继续")
            self.update_status_text("模拟已暂停")
        else:
            # 继续模拟线程
            self.sim_thread.resume()
            
            # 如果有调度器，继续它
            if self.dispatcher:
                try:
                    self.dispatcher.resume_scheduling()
                except:
                    pass
                
            self.pause_btn.setText("暂停")
            self.update_status_text("模拟已继续")
    
    def _set_params_enabled(self, enabled):
        """设置参数控件是否可用"""
        self.map_size_combo.setEnabled(enabled)
        self.vehicles_slider.setEnabled(enabled)
        self.tasks_slider.setEnabled(enabled)
        self.complex_map_checkbox.setEnabled(enabled)
        self.test_mode_checkbox.setEnabled(enabled)
        self.dispatcher_combo.setEnabled(enabled)
    
    def _create_simulation_components(self):
        """创建模拟组件"""
        # 重新创建一个干净的规划器和地图服务
        self._init_map_and_planner()
        
        # 创建障碍物
        self.obstacles = self._create_obstacles()
        
        # 将障碍物应用到规划器
        self.planner.obstacle_grids = set(self.obstacles)
        
        # 创建关键点
        self.key_points = self._create_key_points()
        
        # 创建车辆
        self.vehicles = self._create_vehicles()
        
        # 创建任务
        self.tasks = self._create_tasks()
        
        # 创建调度器
        self._create_dispatcher()
        
        # 初始化活动车辆列表
        self.active_vehicles = []
    
    def _create_vehicles(self) -> List[MiningVehicle]:
        """创建车辆"""
        vehicles = []
        
        # 计算车辆初始位置
        positions = []
        
        # 沿地图边缘均匀分布车辆
        margin = self.map_size // 10
        
        # 上边缘
        top_count = self.num_vehicles // 4
        for i in range(top_count):
            x = margin + i * (self.map_size - 2*margin) / max(1, top_count-1)
            positions.append((x, self.map_size - margin))
            
        # 右边缘
        right_count = self.num_vehicles // 4
        for i in range(right_count):
            y = margin + i * (self.map_size - 2*margin) / max(1, right_count-1)
            positions.append((self.map_size - margin, y))
            
        # 下边缘
        bottom_count = self.num_vehicles // 4
        for i in range(bottom_count):
            x = margin + i * (self.map_size - 2*margin) / max(1, bottom_count-1)
            positions.append((x, margin))
            
        # 左边缘
        left_count = self.num_vehicles - (top_count + right_count + bottom_count)
        for i in range(left_count):
            y = margin + i * (self.map_size - 2*margin) / max(1, left_count-1)
            positions.append((margin, y))
        
        # 确保位置不在障碍物上
        valid_positions = []
        for pos in positions:
            if hasattr(self, 'obstacles'):
                # 确保位置不在障碍物上
                if any(math.dist(pos, obstacle) < 5 for obstacle in self.obstacles):
                    # 寻找附近的安全位置
                    for r in range(1, 20):
                        for dx in [-r, 0, r]:
                            for dy in [-r, 0, r]:
                                if dx == 0 and dy == 0:
                                    continue
                                new_pos = (pos[0] + dx, pos[1] + dy)
                                if not any(math.dist(new_pos, obstacle) < 5 for obstacle in self.obstacles):
                                    pos = new_pos
                                    break
            valid_positions.append(pos)
        
        # 创建车辆
        for i in range(min(self.num_vehicles, len(valid_positions))):
            position = valid_positions[i]
            
            config = {
                'current_location': position,
                'max_capacity': 50,
                'max_speed': random.uniform(5.0, 8.0),
                'turning_radius': 10.0,
                'min_hardness': 2.5,
                'base_location': self.key_points.get('基地', (self.map_size // 2, self.map_size - margin))
            }
            
            vehicle = MiningVehicle(
                vehicle_id=i+1,
                map_service=self.map_service,
                config=config
            )
            
            vehicles.append(vehicle)
            
            # 如果使用调度器，将车辆添加到调度器
            if self.dispatcher:
                self.dispatcher.add_vehicle(vehicle)
        
        logging.info(f"创建了 {len(vehicles)} 辆车辆")
        return vehicles
    
    def _create_tasks(self) -> List[TransportTask]:
        """创建任务"""
        tasks = []
        
        if self.test_mode:
            # 测试模式下不预创建任务
            return tasks
        
        # 创建常规任务
        key_points = self.key_points
        loading_points = [key_points[k] for k in key_points if '装载点' in k]
        unloading_points = [key_points[k] for k in key_points if '卸载点' in k]
        
        # 确保至少有一个装载点和一个卸载点
        if not loading_points:
            loading_points = [(self.map_size // 4, self.map_size // 2)]
        if not unloading_points:
            unloading_points = [(3 * self.map_size // 4, self.map_size // 2)]
        
        # 创建装载/卸载任务对
        for i in range(self.num_tasks):
            # 随机选择装载点和卸载点
            loading_point = random.choice(loading_points)
            unloading_point = random.choice(unloading_points)
            
            # 创建装载任务
            loading_task = TransportTask(
                task_id=f"L{i+1}",
                start_point=loading_point,
                end_point=unloading_point,
                task_type="loading",
                priority=random.randint(1, 3)
            )
            
            tasks.append(loading_task)
            
            # 添加到调度器
            if self.dispatcher:
                self.dispatcher.add_task(loading_task)
        
        logging.info(f"创建了 {len(tasks)} 个任务")
        return tasks
    
    def _create_dispatcher(self):
        """创建调度器"""
        # 获取选择的调度器类
        dispatcher_class = self.available_dispatchers.get(self.selected_dispatcher_name)
        
        if not dispatcher_class:
            # 如果没有找到调度器，使用默认的DispatchSystem
            logging.warning(f"未找到调度器 '{self.selected_dispatcher_name}'，使用默认调度系统")
            try:
                self.dispatcher = DispatchSystem(self.planner, self.map_service)
            except Exception as e:
                logging.error(f"创建默认调度系统失败: {str(e)}")
                self.dispatcher = None
            return
        
        # 创建调度器配置
        config = {
            'cycle_interval': 0.1,  # 调度周期间隔(秒)
            'conflict_check_interval': 0.2,  # 冲突检测间隔(秒)
            'path_safety_margin': 5.0,  # 路径安全边距
            'conflict_detection_enabled': True  # 启用冲突检测
        }
        
        try:
            # 尝试创建调度器
            self.dispatcher = create_dispatcher(
                self.selected_dispatcher_name,
                self.planner,
                self.map_service,
                config
            )
            
            if not self.dispatcher:
                raise ValueError(f"创建调度器 '{self.selected_dispatcher_name}' 失败")
                
            # 初始化车辆和任务
            for vehicle in self.vehicles:
                self.dispatcher.add_vehicle(vehicle)
                
            for task in self.tasks:
                self.dispatcher.add_task(task)
                
            logging.info(f"成功创建调度器: {self.selected_dispatcher_name}")
            
        except Exception as e:
            logging.error(f"创建调度器失败: {str(e)}")
            QMessageBox.warning(self, "调度器错误", f"创建调度器失败: {str(e)}")
            self.dispatcher = None
    
    def _update_vehicles(self):
        """更新车辆位置（自定义移动）"""
        if not self.vehicles or self.use_dispatcher_movement:
            return
            
        # 手动更新车辆位置
        for vehicle in self.vehicles:
            # 如果车辆有分配的路径
            if hasattr(vehicle, 'current_path') and vehicle.current_path and hasattr(vehicle, 'path_index'):
                # 检查是否到达终点
                if vehicle.path_index >= len(vehicle.current_path) - 1:
                    # 已到达路径终点
                    vehicle.state = VehicleState.IDLE
                    
                    # 如果有任务，标记完成
                    if hasattr(vehicle, 'current_task') and vehicle.current_task:
                        vehicle.current_task.complete_task()
                        vehicle.current_task = None
                        
                    if vehicle in self.active_vehicles:
                        self.active_vehicles.remove(vehicle)
                        
                    continue
                
                # 更新到下一个路径点
                vehicle.path_index += 1
                vehicle.current_location = vehicle.current_path[vehicle.path_index]
                
                # 如果不在活动车辆列表中，添加它
                if vehicle not in self.active_vehicles:
                    self.active_vehicles.append(vehicle)
    
    def _start_new_test(self):
        """测试模式：开始新的测试"""
        if not self.test_mode:
            return
            
        # 定义测试点（可以是关键点位置）
        points = list(self.key_points.values())
        
        # 随机选择起点和终点
        start_idx = random.randint(0, len(points)-1)
        end_idx = random.randint(0, len(points)-1)
        
        # 确保起点和终点不同
        while end_idx == start_idx:
            end_idx = random.randint(0, len(points)-1)
            
        start_point = points[start_idx]
        end_point = points[end_idx]
        
        # 创建测试任务
        task = TransportTask(
            task_id=f"TEST-{len(self.tasks)+1}",
            start_point=start_point,
            end_point=end_point,
            task_type="transport",
            priority=1
        )
        
        self.tasks.append(task)
        
        # 添加到调度器
        if self.dispatcher:
            self.dispatcher.add_task(task)
            
        self.update_status_text(f"创建测试任务: {start_point} → {end_point}")
        
        # 如果调度器会自动分配任务，不需要手动分配
        if not self.auto_assign_tasks:
            # 随机选择一个空闲车辆
            idle_vehicles = [v for v in self.vehicles if v.state == VehicleState.IDLE]
            if idle_vehicles:
                vehicle = random.choice(idle_vehicles)
                
                # 规划路径
                try:
                    path = self.planner.plan_path(vehicle.current_location, start_point, vehicle)
                    
                    if path and len(path) > 1:
                        # 分配路径和任务
                        vehicle.assign_path(path)
                        vehicle.assign_task(task)
                        
                        # 添加到活动车辆
                        if vehicle not in self.active_vehicles:
                            self.active_vehicles.append(vehicle)
                            
                        self.update_status_text(f"将任务分配给车辆 {vehicle.vehicle_id}")
                except Exception as e:
                    logging.error(f"规划路径出错: {str(e)}")
    
    def check_path_conflicts(self):
        """检查路径冲突"""
        if not self.dispatcher:
            QMessageBox.warning(self, "未初始化", "调度器未初始化")
            return
            
        try:
            # 调用调度器的冲突检测
            self.dispatcher.resolve_path_conflicts()
            
            # 获取状态
            status = self.dispatcher.get_status()
            conflicts = status.get('metrics', {}).get('conflicts_detected', 0)
            
            self.update_status_text(f"检测到 {conflicts} 个冲突")
            
        except Exception as e:
            logging.error(f"检查冲突时出错: {str(e)}")
            self.update_status_text(f"检查冲突失败: {str(e)}")
    
    def resolve_path_conflicts(self):
        """解决路径冲突"""
        if not self.dispatcher:
            QMessageBox.warning(self, "未初始化", "调度器未初始化")
            return
            
        try:
            # 调用调度器的冲突解决（与检测相同方法，因为检测后会自动解决）
            self.dispatcher.resolve_path_conflicts()
            
            # 获取状态
            status = self.dispatcher.get_status()
            resolved = status.get('metrics', {}).get('conflicts_resolved', 0)
            
            self.update_status_text(f"已解决 {resolved} 个冲突")
            
        except Exception as e:
            logging.error(f"解决冲突时出错: {str(e)}")
            self.update_status_text(f"解决冲突失败: {str(e)}")
    
    def add_random_task(self):
        """添加随机任务"""
        if not self.key_points:
            QMessageBox.warning(self, "未初始化", "关键点未初始化")
            return
            
        try:
            # 选择随机起点和终点
            key_point_names = list(self.key_points.keys())
            
            # 如果有装载点和卸载点，优先使用这些
            loading_points = [k for k in key_point_names if '装载点' in k]
            unloading_points = [k for k in key_point_names if '卸载点' in k]
            
            if loading_points and unloading_points:
                # 选择一个装载点作为起点
                start_name = random.choice(loading_points)
                # 选择一个卸载点作为终点
                end_name = random.choice(unloading_points)
                
                start_point = self.key_points[start_name]
                end_point = self.key_points[end_name]
                task_type = "loading"
            else:
                # 随机选择任意两个不同的关键点
                start_name = random.choice(key_point_names)
                end_name = start_name
                
                while end_name == start_name:
                    end_name = random.choice(key_point_names)
                    
                start_point = self.key_points[start_name]
                end_point = self.key_points[end_name]
                task_type = random.choice(["transport", "loading", "unloading"])
            
            # 创建新任务
            task = TransportTask(
                task_id=f"RAND-{len(self.tasks)+1}",
                start_point=start_point,
                end_point=end_point,
                task_type=task_type,
                priority=random.randint(1, 3)
            )
            
            # 添加到任务列表
            self.tasks.append(task)
            
            # 添加到调度器
            if self.dispatcher:
                self.dispatcher.add_task(task)
                
            self.update_status_text(f"添加随机任务: {start_name} → {end_name} (类型: {task_type})")
            
        except Exception as e:
            logging.error(f"添加随机任务时出错: {str(e)}")
            self.update_status_text(f"添加随机任务失败: {str(e)}")
    
    def on_simulation_completed(self, stats):
        """模拟完成回调"""
        self.update_status_text("模拟完成!")
        
        # 启用保存按钮
        self.save_results_btn.setEnabled(True)
        
        # 显示模拟结果统计
        self._show_simulation_results(stats)
    
    def _show_simulation_results(self, stats):
        """显示模拟结果"""
        try:
            # 提取统计数据
            vehicles = stats.get('vehicles', {})
            tasks = stats.get('tasks', {})
            metrics = stats.get('metrics', {})
            
            # 格式化结果HTML
            result_html = """
            <h3>模拟结果统计</h3>
            <table>
            """
            
            # 车辆统计
            if isinstance(vehicles, dict) and 'total' in vehicles:
                result_html += f"<tr><td>车辆总数:</td><td>{vehicles['total']}</td></tr>"
                result_html += f"<tr><td>活动车辆:</td><td>{vehicles['active']}</td></tr>"
            
            # 任务统计
            if isinstance(tasks, dict):
                completed = tasks.get('completed', 0)
                total = completed + tasks.get('queued', 0) + tasks.get('active', 0)
                
                if total > 0:
                    completion_rate = completed / total * 100
                else:
                    completion_rate = 0
                    
                result_html += f"<tr><td>任务完成数:</td><td>{completed}</td></tr>"
                result_html += f"<tr><td>任务总数:</td><td>{total}</td></tr>"
                result_html += f"<tr><td>完成率:</td><td>{completion_rate:.1f}%</td></tr>"
            
            # 冲突统计
            detected = metrics.get('conflicts_detected', 0)
            resolved = metrics.get('conflicts_resolved', 0)
            
            if detected > 0:
                resolution_rate = resolved / detected * 100
            else:
                resolution_rate = 100
                
            result_html += f"<tr><td>检测到冲突:</td><td>{detected}</td></tr>"
            result_html += f"<tr><td>解决的冲突:</td><td>{resolved}</td></tr>"
            result_html += f"<tr><td>冲突解决率:</td><td>{resolution_rate:.1f}%</td></tr>"
            
            # 规划统计
            planning_count = metrics.get('planning_count', 0)
            result_html += f"<tr><td>路径规划次数:</td><td>{planning_count}</td></tr>"
            
            result_html += "</table>"
            
            # 更新统计面板
            self.update_stats_text(result_html)
            
            # 显示消息框
            QMessageBox.information(self, "模拟完成", "露天矿调度模拟已完成!")
            
        except Exception as e:
            logging.error(f"显示模拟结果时出错: {str(e)}")
    
    def save_simulation_results(self):
        """保存模拟结果"""
        # 打开文件对话框
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存模拟结果", "", "CSV文件 (*.csv);;所有文件 (*)"
        )
        
        if not filename:
            return
            
        # 确保文件有.csv扩展名
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        try:
            # 获取模拟统计信息
            stats = self.get_simulation_stats()
            
            # 打开文件写入
            with open(filename, 'w', encoding='utf-8') as f:
                # 写入标题
                f.write("指标,值\n")
                
                # 写入车辆数据
                f.write(f"车辆总数,{len(self.vehicles)}\n")
                
                # 计算不同状态的车辆数
                if self.vehicles:
                    idle_count = sum(1 for v in self.vehicles if v.state == VehicleState.IDLE)
                    active_count = len(self.vehicles) - idle_count
                    f.write(f"空闲车辆,{idle_count}\n")
                    f.write(f"活动车辆,{active_count}\n")
                
                # 写入任务数据
                if hasattr(self, 'tasks'):
                    completed = sum(1 for t in self.tasks if hasattr(t, 'is_completed') and t.is_completed)
                    pending = sum(1 for t in self.tasks if hasattr(t, 'status') and t.status == 'pending')
                    active = len(self.tasks) - completed - pending
                    
                    f.write(f"任务总数,{len(self.tasks)}\n")
                    f.write(f"完成任务,{completed}\n")
                    f.write(f"活动任务,{active}\n")
                    f.write(f"等待任务,{pending}\n")
                
                # 写入调度器指标
                if self.dispatcher:
                    status = self.dispatcher.get_status()
                    metrics = status.get('metrics', {})
                    
                    # 写入冲突数据
                    conflicts_detected = metrics.get('conflicts_detected', 0)
                    conflicts_resolved = metrics.get('conflicts_resolved', 0)
                    
                    f.write(f"检测到冲突,{conflicts_detected}\n")
                    f.write(f"解决的冲突,{conflicts_resolved}\n")
                    
                    # 写入规划统计
                    planning_count = metrics.get('planning_count', 0)
                    f.write(f"路径规划次数,{planning_count}\n")
                
                # 写入调度系统信息
                f.write(f"\n调度算法,{self.selected_dispatcher_name}\n")
                f.write(f"地图尺寸,{self.map_size}x{self.map_size}\n")
                f.write(f"复杂地图,{'是' if self.complex_map_checkbox.isChecked() else '否'}\n")
                f.write(f"测试模式,{'是' if self.test_mode else '否'}\n")
            
            self.update_status_text(f"模拟结果已保存到 {filename}")
            
        except Exception as e:
            logging.error(f"保存模拟结果时出错: {str(e)}")
            QMessageBox.warning(self, "保存失败", f"保存模拟结果时出错: {str(e)}")
    
    def get_simulation_stats(self):
        """获取模拟统计数据"""
        stats = {}
        
        # 如果有调度器，获取其状态
        if self.dispatcher:
            stats = self.dispatcher.get_status()
        else:
            # 否则构建基本统计信息
            stats = {
                'vehicles': {
                    'total': len(self.vehicles),
                    'idle': sum(1 for v in self.vehicles if v.state == VehicleState.IDLE),
                    'active': sum(1 for v in self.vehicles if v.state != VehicleState.IDLE)
                },
                'tasks': {
                    'queued': sum(1 for t in self.tasks if hasattr(t, 'status') and t.status == 'pending'),
                    'active': sum(1 for t in self.tasks if hasattr(t, 'status') and t.status in ['assigned', 'in_progress']),
                    'completed': sum(1 for t in self.tasks if hasattr(t, 'is_completed') and t.is_completed)
                },
                'metrics': {
                    'conflicts_detected': 0,
                    'conflicts_resolved': 0,
                    'planning_count': 0
                }
            }
        
        return stats


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="露天矿调度系统")
    parser.add_argument("--vehicles", type=int, default=5, help="车辆数量")
    parser.add_argument("--tasks", type=int, default=10, help="任务数量")
    parser.add_argument("--map-size", type=int, default=200, help="地图尺寸")
    parser.add_argument("--complex-map", action="store_true", help="使用复杂地图")
    parser.add_argument("--dispatcher", type=str, default="", help="指定调度算法")
    parser.add_argument("--test-mode", action="store_true", help="启用测试模式")
    args = parser.parse_args()
    
    # 创建Qt应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    manager = MiningDispatchManager()
    
    # 应用命令行参数
    if args.vehicles:
        manager.vehicles_slider.setValue(args.vehicles)
    
    if args.tasks:
        manager.tasks_slider.setValue(args.tasks)
    
    if args.map_size:
        # 找到最接近的地图尺寸选项
        sizes = [100, 200, 300, 400]
        closest_size = min(sizes, key=lambda x: abs(x - args.map_size))
        index = sizes.index(closest_size)
        manager.map_size_combo.setCurrentIndex(index)
    
    if args.complex_map:
        manager.complex_map_checkbox.setChecked(True)
    
    if args.test_mode:
        manager.test_mode_checkbox.setChecked(True)
    
    if args.dispatcher:
        # 尝试找到匹配的调度器
        index = manager.dispatcher_combo.findText(args.dispatcher)
        if index >= 0:
            manager.dispatcher_combo.setCurrentIndex(index)
    
    # 显示窗口
    manager.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()