#!/usr/bin/env python3
"""
optimized_path_planner 复杂地图测试脚本

此脚本用于测试路径规划器在复杂地图上的性能和可视化效果
使用PyQtGraph进行高性能可视化，支持：
1. 复杂地图和障碍物测试
2. 路径规划效果分析
3. 性能指标监测
4. A*搜索过程可视化
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    from utils.geo_tools import GeoUtils
    from models.vehicle import MiningVehicle, VehicleState
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
    
    def __init__(self, tester):
        super().__init__()
        self.tester = tester
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
                
                # 更新车辆位置
                self.tester._update_vehicles()
                
                # 如果所有测试完成，发送测试完成信号
                if self.tester.all_tests_completed:
                    final_stats = self.tester.get_test_stats()
                    self.finished_signal.emit(final_stats)
                    self.running = False
                    break
                
                # 如果没有活动车辆，启动新测试
                if len(self.tester.active_vehicles) == 0 and not self.tester.all_tests_completed:
                    self.tester._start_new_test()
                
                # 定时更新UI
                if current_time - last_update_time >= update_interval:
                    # 准备更新数据
                    update_data = {
                        'vehicles': {v.vehicle_id: v.current_location for v in self.tester.vehicles},
                        'paths': {v.vehicle_id: self.tester.vehicle_paths.get(v, []) 
                                 for v in self.tester.active_vehicles},
                        'stats': self.tester.test_stats,
                        'test_info': self.tester.current_test_info,
                        'debug_data': self.tester.debug_data if hasattr(self.tester, 'debug_data') else {}
                    }
                    
                    # 发送更新信号
                    self.update_signal.emit(update_data)
                    last_update_time = current_time
            
            # 控制更新速度
            sim_delay = 0.01 / self.tester.simulation_speed
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


class PathPlannerTester(QMainWindow):
    """路径规划器测试工具"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("路径规划器复杂地图测试")
        self.resize(1280, 800)
        
        # 初始化测试参数
        self.map_size = 200
        self.num_vehicles = 5
        self.num_test_pairs = 10
        self.simulation_speed = 1.0
        self.show_paths = True
        self.show_debug = False
        self.show_heatmap = False
        self.debug_mode = False
        
        # 测试状态
        self.test_running = False
        self.test_paused = False
        self.all_tests_completed = False
        
        # 初始化地图和规划器
        self._init_map_and_planner()
        
        # 当前测试信息
        self.current_test_info = {
            'current_idx': 0,
            'total_tests': 0,
            'current_test': None
        }
        
        # 测试统计数据
        self.test_stats = {
            'planning_times': [],    # 规划时间列表 (秒)
            'path_lengths': [],      # 路径长度列表 (点数)
            'conflicts_detected': 0, # 检测到的冲突数
            'conflicts_resolved': 0, # 解决的冲突数
            'successful_plans': 0,   # 成功规划数
            'failed_plans': 0,       # 失败规划数
            'total_tests': 0,        # 总测试数
            'start_time': None,      # A*搜索次数
            'astar_node_expansions': [], # A*节点展开数
            'test_durations': []     # 测试持续时间列表 (秒)
        }
        
        # 调试数据
        self.debug_data = {
            'open_set': [],
            'closed_set': [],
            'current_path': [],
            'current_node': None,
            'start_point': None,
            'end_point': None
        }
        
        # 测试数据
        self.vehicles = []
        self.test_points = {}
        self.test_pairs = []
        self.vehicle_paths = {}
        self.active_vehicles = []
        self.vehicle_path_progress = {}
        
        # 创建模拟线程
        self.sim_thread = SimulationThread(self)
        self.sim_thread.update_signal.connect(self.update_display)
        self.sim_thread.finished_signal.connect(self.on_test_completed)
        
        # 创建UI
        self.setup_ui()
        
        # 热图数据
        self.heatmap_data = np.zeros((self.map_size, self.map_size))
        
        logging.info("路径规划器测试工具初始化完成")
    
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
        self.main_view = pg.PlotWidget(title="路径规划测试地图")
        self.main_view.setAspectLocked(True)
        self.main_view.setRange(xRange=(0, self.map_size), yRange=(0, self.map_size))
        self.main_view.showGrid(x=True, y=True, alpha=0.5)
        self.main_view.addLegend()
        map_layout.addWidget(self.main_view)
        
        # 调试视图(初始隐藏)
        self.debug_view = pg.PlotWidget(title="A*搜索过程")
        self.debug_view.setAspectLocked(True)
        self.debug_view.setRange(xRange=(0, self.map_size), yRange=(0, self.map_size))
        self.debug_view.showGrid(x=True, y=True, alpha=0.5)
        self.debug_view.addLegend()
        self.debug_view.hide()
        map_layout.addWidget(self.debug_view)
        
        top_layout.addWidget(map_widget, 3)
        
        # 右侧：控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        top_layout.addWidget(control_panel, 1)
        
        # 测试参数设置
        params_group = QGroupBox("测试参数")
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
        
        # 测试数量
        params_layout.addWidget(QLabel("测试数量:"), 2, 0)
        self.tests_slider = QSlider(Qt.Horizontal)
        self.tests_slider.setMinimum(5)
        self.tests_slider.setMaximum(50)
        self.tests_slider.setValue(10)
        self.tests_slider.setTickPosition(QSlider.TicksBelow)
        self.tests_slider.setTickInterval(5)
        self.tests_slider.valueChanged.connect(self.on_tests_changed)
        params_layout.addWidget(self.tests_slider, 2, 1)
        self.tests_label = QLabel("10")
        params_layout.addWidget(self.tests_label, 2, 2)
        
        # 是否使用复杂地图
        params_layout.addWidget(QLabel("复杂地图:"), 3, 0)
        self.complex_map_checkbox = QCheckBox()
        self.complex_map_checkbox.setChecked(True)
        params_layout.addWidget(self.complex_map_checkbox, 3, 1)
        
        # 是否启用调试模式
        params_layout.addWidget(QLabel("调试模式:"), 4, 0)
        self.debug_checkbox = QCheckBox()
        self.debug_checkbox.setChecked(False)
        self.debug_checkbox.stateChanged.connect(self.on_debug_mode_changed)
        params_layout.addWidget(self.debug_checkbox, 4, 1)
        
        control_layout.addWidget(params_group)
        
        # 控制按钮
        buttons_group = QGroupBox("测试控制")
        buttons_layout = QGridLayout()
        buttons_group.setLayout(buttons_layout)
        
        # 启动测试按钮
        self.start_btn = QPushButton("启动测试")
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
        
        buttons_layout.addWidget(QLabel("显示调试视图:"), 3, 0)
        self.show_debug_checkbox = QCheckBox()
        self.show_debug_checkbox.setChecked(False)
        self.show_debug_checkbox.stateChanged.connect(self.on_show_debug_changed)
        buttons_layout.addWidget(self.show_debug_checkbox, 3, 1)
        
        buttons_layout.addWidget(QLabel("显示热图:"), 4, 0)
        self.show_heatmap_checkbox = QCheckBox()
        self.show_heatmap_checkbox.setChecked(False)
        self.show_heatmap_checkbox.stateChanged.connect(self.on_show_heatmap_changed)
        buttons_layout.addWidget(self.show_heatmap_checkbox, 4, 1)
        
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
        
        # 保存结果按钮
        self.save_results_btn = QPushButton("保存结果")
        self.save_results_btn.clicked.connect(self.save_test_results)
        self.save_results_btn.setEnabled(False)
        functions_layout.addWidget(self.save_results_btn, 1, 0, 1, 2)
        
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
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout()
        stats_group.setLayout(stats_layout)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(150)
        stats_layout.addWidget(self.stats_text)
        
        bottom_layout.addWidget(stats_group)
        
        # 初始化状态
        self.update_status_text("准备就绪，请设置参数后点击'启动测试'")
        self.update_stats_text("尚未开始测试")
    
    def on_map_size_changed(self, index):
        """地图尺寸变更处理"""
        size_text = self.map_size_combo.currentText()
        self.map_size = int(size_text.split('x')[0])
        self.main_view.setRange(xRange=(0, self.map_size), yRange=(0, self.map_size))
        self.debug_view.setRange(xRange=(0, self.map_size), yRange=(0, self.map_size))
        
        self.update_status_text(f"地图尺寸已更改为 {self.map_size}x{self.map_size}")
    
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
    
    def on_debug_mode_changed(self, state):
        """调试模式变更处理"""
        self.debug_mode = state == Qt.Checked
        
        # 如果开启调试模式，自动显示调试视图
        if self.debug_mode:
            self.show_debug_checkbox.setChecked(True)
        
        self.update_status_text(f"调试模式: {'开启' if self.debug_mode else '关闭'}")
    
    def on_show_paths_changed(self, state):
        """路径显示变更处理"""
        self.show_paths = state == Qt.Checked
        self.update_display({})  # 触发重绘
    
    def on_show_debug_changed(self, state):
        """调试视图显示变更处理"""
        self.show_debug = state == Qt.Checked
        
        if self.show_debug:
            self.debug_view.show()
        else:
            self.debug_view.hide()
    
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
    
    def start_test(self):
        """启动测试"""
        if self.test_running:
            # 如果测试已在运行，停止它
            self.stop_test()
            return
        
        try:
            # 重置测试状态
            self.all_tests_completed = False
            self.test_running = True
            self.test_paused = False
            
            # 重置统计数据
            self.test_stats = {
                'planning_times': [],
                'path_lengths': [],
                'conflicts_detected': 0,
                'conflicts_resolved': 0,
                'successful_plans': 0,
                'failed_plans': 0,
                'total_tests': 0,
                'start_time': time.time(),
                'astar_node_expansions': [],
                'test_durations': []
            }
            
            # 创建测试组件
            self._create_test_components()
            
            # 更新UI
            self.start_btn.setText("停止测试")
            self.pause_btn.setEnabled(True)
            self.check_conflicts_btn.setEnabled(True)
            self.resolve_conflicts_btn.setEnabled(True)
            self.save_results_btn.setEnabled(False)
            
            # 禁用测试参数设置
            self._set_params_enabled(False)
            
            # 清除视图
            self._clear_views()
            
            # 绘制地图和障碍物
            self._draw_map_and_obstacles()
            
            # 初始化热图数据
            self.heatmap_data = np.zeros((self.map_size, self.map_size))
            
            # 启动模拟线程
            self.sim_thread.start()
            
            self.update_status_text("测试已启动")
            
        except Exception as e:
            logging.error(f"启动测试时出错: {str(e)}")
            self.update_status_text(f"启动测试失败: {str(e)}")
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
        self.start_btn.setText("启动测试")
        self.pause_btn.setEnabled(False)
        self.check_conflicts_btn.setEnabled(False)
        self.resolve_conflicts_btn.setEnabled(False)
        self.save_results_btn.setEnabled(True)
        
        # 启用测试参数设置
        self._set_params_enabled(True)
        
        self.update_status_text("测试已停止")
    
    def toggle_pause(self):
        """暂停/继续测试"""
        if not self.test_running:
            return
            
        self.test_paused = not self.test_paused
        
        if self.test_paused:
            # 暂停模拟线程
            self.sim_thread.pause()
            self.pause_btn.setText("继续")
            self.update_status_text("测试已暂停")
        else:
            # 继续模拟线程
            self.sim_thread.resume()
            self.pause_btn.setText("暂停")
            self.update_status_text("测试已继续")
    
    def _set_params_enabled(self, enabled):
        """设置参数控件是否可用"""
        self.map_size_combo.setEnabled(enabled)
        self.vehicles_slider.setEnabled(enabled)
        self.tests_slider.setEnabled(enabled)
        self.complex_map_checkbox.setEnabled(enabled)
        self.debug_checkbox.setEnabled(enabled)
    
    def _clear_views(self):
        """清除视图内容"""
        self.main_view.clear()
        self.debug_view.clear()
    
    def _create_test_components(self):
        """创建测试组件"""
        # 重新创建一个干净的规划器
        self._init_map_and_planner()
        
        # 创建测试点和任务
        self.test_points = self._create_test_points()
        self.test_pairs = self._generate_test_pairs()
        
        # 创建车辆
        self.vehicles = self._create_test_vehicles()
        
        # 为规划器设置mock dispatch对象
        class MockDispatch:
            def __init__(self):
                self.vehicles = {}
                
        mock_dispatch = MockDispatch()
        for vehicle in self.vehicles:
            mock_dispatch.vehicles[vehicle.vehicle_id] = vehicle
        
        self.planner.dispatch = mock_dispatch
        
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
        self.vehicle_path_progress = {}
        
        # 初始化性能监控钩子
        self._install_performance_hooks()
    
    def _create_test_points(self) -> Dict[str, Tuple[float, float]]:
        """创建测试点"""
        points = {}
        
        # 网格尺寸
        grid_step = self.map_size / 10
        
        # 关键点位置
        key_points = {
            "中心点": (self.map_size // 2, self.map_size // 2),
            "左上角": (grid_step, self.map_size - grid_step),
            "右上角": (self.map_size - grid_step, self.map_size - grid_step),
            "左下角": (grid_step, grid_step),
            "右下角": (self.map_size - grid_step, grid_step),
            "上中": (self.map_size // 2, self.map_size - grid_step),
            "下中": (self.map_size // 2, grid_step),
            "左中": (grid_step, self.map_size // 2),
            "右中": (self.map_size - grid_step, self.map_size // 2)
        }
        
        points.update(key_points)
        
        # 添加额外随机点
        num_random_points = max(0, self.num_test_pairs - len(key_points))
        
        for i in range(num_random_points):
            # 尝试找到一个不在障碍物上的随机点
            for attempt in range(20):  # 最多尝试20次
                x = random.uniform(grid_step, self.map_size - grid_step)
                y = random.uniform(grid_step, self.map_size - grid_step)
                point = (x, y)
                
                # 检查点是否远离障碍物
                if not self._is_near_obstacle(point, 15):  # 15是安全距离
                    points[f"随机点{i+1}"] = point
                    break
        
        logging.info(f"创建了 {len(points)} 个测试点")
        return points
    def _create_test_vehicles(self):
        """创建测试车辆"""
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
            for attempt in range(10):
                # 如果位置在障碍物上，略微移动位置
                if hasattr(self, 'obstacles') and self._is_near_obstacle(pos, 10):
                    offset_x = random.uniform(-10, 10)
                    offset_y = random.uniform(-10, 10)
                    new_pos = (pos[0] + offset_x, pos[1] + offset_y)
                    pos = new_pos
                else:
                    break
            valid_positions.append(pos)
        
        # 创建车辆
        for i in range(min(self.num_vehicles, len(valid_positions))):
            position = valid_positions[i]
            
            config = {
                'current_location': position,
                'max_capacity': 50,
                'max_speed': random.uniform(5.0, 8.0),
                'min_hardness': 2.5,
                'turning_radius': 10.0,
                'base_location': (self.map_size // 2, self.map_size // 2)
            }
            
            vehicle = MiningVehicle(
                vehicle_id=i+1,
                map_service=self.map_service,
                config=config
            )
            
            # 确保必要属性存在
            if not hasattr(vehicle, 'current_path'):
                vehicle.current_path = []
            if not hasattr(vehicle, 'path_index'):
                vehicle.path_index = 0
            
            # 随机颜色
            vehicle.color = pg.intColor(i % 10)
            
            vehicles.append(vehicle)
        
        logging.info(f"创建了 {len(vehicles)} 辆测试车辆")
        return vehicles

    def _install_performance_hooks(self):
        """安装性能监控钩子"""
        # 保存原始方法
        if not hasattr(self.planner, '_original_astar'):
            self.planner._original_astar = getattr(self.planner, '_astar', None)
        
        # 添加A*搜索性能监控
        if self.planner._original_astar:
            def monitored_astar(start, end, vehicle=None):
                # 保存起点和终点
                self.debug_data['start_point'] = start
                self.debug_data['end_point'] = end
                
                # 统计开始时间
                start_time = time.time()
                
                # 清除上一次的调试数据
                self.debug_data['open_set'] = []
                self.debug_data['closed_set'] = []
                self.debug_data['current_path'] = []
                self.debug_data['current_node'] = None
                
                # 保存当前搜索次数
                current_expansion = [0]
                
                # 调试钩子
                def astar_hook(current_node, open_set=None, closed_set=None, current_path=None):
                    # 更新调试数据
                    if open_set is not None:
                        self.debug_data['open_set'] = list(open_set)
                    if closed_set is not None:
                        self.debug_data['closed_set'] = list(closed_set)
                    if current_path is not None:
                        self.debug_data['current_path'] = current_path
                    
                    self.debug_data['current_node'] = current_node
                    
                    # 统计节点展开
                    current_expansion[0] += 1
                
                # 如果处于调试模式，附加钩子
                if self.debug_mode:
                    try:
                        # 尝试添加hook参数
                        path = self.planner._original_astar(start, end, vehicle, hook=astar_hook)
                    except TypeError:
                        # 如果原方法不支持hook参数，使用原始方法
                        path = self.planner._original_astar(start, end, vehicle)
                else:
                    path = self.planner._original_astar(start, end, vehicle)
                
                # 统计结束时间
                planning_time = time.time() - start_time
                
                # 记录性能数据
                self.test_stats['planning_times'].append(planning_time)
                self.test_stats['astar_node_expansions'].append(current_expansion[0])
                
                return path
            
            # 替换A*方法
            if hasattr(self.planner, '_astar'):
                self.planner._astar = monitored_astar

    def update_display(self, data):
        """更新显示"""
        if self.test_paused or not self.test_running:
            return
        
        try:
            # 清除主视图中的动态元素
            self.vehicle_markers = self._clear_vehicle_markers()
            
            # 绘制车辆和路径
            self._draw_vehicles_and_paths(data.get('vehicles', {}), data.get('paths', {}))
            
            # 更新热图
            if self.show_heatmap:
                self._update_heatmap_data(data.get('paths', {}))
            
            # 更新调试视图
            if self.show_debug:
                self._update_debug_view(data.get('debug_data', {}))
            
            # 更新状态和统计信息
            if 'stats' in data:
                self._update_status_and_stats(data['stats'], data.get('test_info', {}))
        
        except Exception as e:
            logging.error(f"更新显示时出错: {str(e)}")

    def _clear_vehicle_markers(self):
        """清除车辆标记并返回新的字典"""
        # 清除所有现有车辆标记
        for marker in getattr(self, 'vehicle_markers', {}).values():
            for item in marker.values():
                if isinstance(item, pg.PlotDataItem) or isinstance(item, pg.ScatterPlotItem):
                    self.main_view.removeItem(item)
        
        return {}

    def _draw_vehicles_and_paths(self, vehicles_data, paths_data):
        """绘制车辆和路径"""
        self.vehicle_markers = {}
        
        for vid, position in vehicles_data.items():
            # 生成颜色
            color = pg.intColor(vid % 10)
            
            # 车辆标记
            marker = pg.ScatterPlotItem(
                [position[0]], [position[1]], 
                size=10, pen=pg.mkPen(color), brush=pg.mkBrush(color)
            )
            self.main_view.addItem(marker)
            
            # 车辆标签
            label = pg.TextItem(text=str(vid), anchor=(0.5, 0.5), color=color)
            label.setPos(position[0], position[1])
            self.main_view.addItem(label)
            
            # 路径线条
            path_item = None
            if self.show_paths and vid in paths_data and paths_data[vid]:
                path = paths_data[vid]
                path_item = pg.PlotDataItem(
                    [p[0] for p in path], 
                    [p[1] for p in path],
                    pen=pg.mkPen(color, width=2, style=Qt.DashLine)
                )
                self.main_view.addItem(path_item)
            
            # 存储到字典
            self.vehicle_markers[vid] = {
                'marker': marker,
                'label': label,
                'path': path_item
            }

    def _update_heatmap_data(self, paths_data):
        """更新热图数据"""
        # 递减现有热图数据
        self.heatmap_data *= 0.95
        
        # 添加新的路径点
        for path in paths_data.values():
            if path:
                for x, y in path:
                    if 0 <= int(x) < self.map_size and 0 <= int(y) < self.map_size:
                        self.heatmap_data[int(y), int(x)] += 0.5
        
        # 为冲突点添加更高的热度
        if hasattr(self, 'conflict_points'):
            for x, y in self.conflict_points:
                if 0 <= int(x) < self.map_size and 0 <= int(y) < self.map_size:
                    self.heatmap_data[int(y), int(x)] += 2.0
        
        # 更新热图显示
        if hasattr(self, 'heatmap_img'):
            self.heatmap_img.setImage(self.heatmap_data.T)

    def _update_debug_view(self, debug_data):
        """更新调试视图"""
        # 清除现有调试元素
        self.debug_view.clear()
        
        # 绘制地图和障碍物到调试视图
        self._draw_map_on_debug_view()
        
        # 从debug_data中提取调试信息
        open_set = debug_data.get('open_set', [])
        closed_set = debug_data.get('closed_set', [])
        current_path = debug_data.get('current_path', [])
        current_node = debug_data.get('current_node')
        start_point = debug_data.get('start_point')
        end_point = debug_data.get('end_point')
        
        # 绘制开放集
        if open_set:
            open_set_item = pg.ScatterPlotItem(
                [p[0] for p in open_set], 
                [p[1] for p in open_set],
                size=5, pen=None, brush=pg.mkBrush(0, 255, 0, 100)
            )
            self.debug_view.addItem(open_set_item)
        
        # 绘制关闭集
        if closed_set:
            closed_set_item = pg.ScatterPlotItem(
                [p[0] for p in closed_set], 
                [p[1] for p in closed_set],
                size=5, pen=None, brush=pg.mkBrush(255, 0, 0, 100)
            )
            self.debug_view.addItem(closed_set_item)
        
        # 绘制当前路径
        if current_path:
            path_item = pg.PlotDataItem(
                [p[0] for p in current_path], 
                [p[1] for p in current_path],
                pen=pg.mkPen('b', width=2)
            )
            self.debug_view.addItem(path_item)
        
        # 绘制当前节点
        if current_node:
            current_node_item = pg.ScatterPlotItem(
                [current_node[0]], [current_node[1]],
                size=10, pen=None, brush=pg.mkBrush(255, 255, 0, 200)
            )
            self.debug_view.addItem(current_node_item)
        
        # 绘制起点和终点
        if start_point:
            start_item = pg.ScatterPlotItem(
                [start_point[0]], [start_point[1]],
                size=12, pen=None, brush=pg.mkBrush(0, 255, 0, 200)
            )
            self.debug_view.addItem(start_item)
            
        if end_point:
            end_item = pg.ScatterPlotItem(
                [end_point[0]], [end_point[1]],
                size=12, pen=None, brush=pg.mkBrush(255, 0, 0, 200)
            )
            self.debug_view.addItem(end_item)

    def _draw_map_on_debug_view(self):
        """在调试视图上绘制地图和障碍物"""
        # 绘制障碍物
        if hasattr(self, 'obstacles'):
            obstacle_item = pg.ScatterPlotItem(
                [p[0] for p in self.obstacles], 
                [p[1] for p in self.obstacles],
                size=2, pen=None, brush=pg.mkBrush(100, 100, 100, 150)
            )
            self.debug_view.addItem(obstacle_item)
        
        # 绘制测试点
        for name, point in self.test_points.items():
            point_item = pg.ScatterPlotItem(
                [point[0]], [point[1]], 
                size=8, pen=pg.mkPen('b'), brush=pg.mkBrush('b')
            )
            self.debug_view.addItem(point_item)
            
            text_item = pg.TextItem(name, anchor=(0, 0), color='b')
            text_item.setPos(point[0]+5, point[1]+5)
            self.debug_view.addItem(text_item)