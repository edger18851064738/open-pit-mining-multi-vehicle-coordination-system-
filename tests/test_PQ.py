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
    from algorithm.hybrid_path_planner import HybridPathPlanner
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
    # 在PathPlannerTester类中添加update_display方法
    def update_display(self, update_data):
        """更新显示内容"""
        try:
            # 更新车辆位置
            for i, vehicle in enumerate(self.vehicles):
                if vehicle.vehicle_id in update_data['vehicles']:
                    vehicle.current_location = update_data['vehicles'][vehicle.vehicle_id]
            
            # 更新路径显示
            self._update_path_display(update_data['paths'])
            
            # 更新统计信息
            if 'stats' in update_data:
                self._update_stats_display(update_data['stats'])
            
            # 更新测试信息
            if 'test_info' in update_data:
                self.current_test_info = update_data['test_info']
                self._update_status_display()
            
            # 更新调试数据
            if 'debug_data' in update_data and self.show_debug:
                self._update_debug_display(update_data['debug_data'])
            
            # 更新热图
            if self.show_heatmap and 'paths' in update_data:
                self._update_heatmap(update_data['paths'])
            
            # 刷新应用
            QApplication.processEvents()
            
        except Exception as e:
            logging.error(f"更新显示时出错: {str(e)}")

    def _update_path_display(self, paths):
        """更新路径显示，保留之前的路径并使用渐变透明度"""
        # 初始化路径存储
        if not hasattr(self, 'displayed_paths'):
            self.displayed_paths = {}
            self.path_items = []  # 存储所有路径项以便控制透明度
        
        # 如果不显示路径，隐藏所有路径
        if not self.show_paths:
            for item in self.path_items:
                item.setOpacity(0)
            return
        else:
            # 恢复显示所有路径
            for item in self.path_items:
                item.setOpacity(1)
        
        # 降低之前路径的透明度
        for i, item in enumerate(self.path_items):
            # 使旧路径逐渐变淡，但保持可见
            opacity = max(0.2, 1.0 - (len(self.path_items) - i) * 0.1)
            try:
                pen = item.opts['pen']
                pen.setWidth(max(1, pen.width() - 0.2))  # 逐渐减小线宽
                item.setPen(pen)
                item.setOpacity(opacity)
            except:
                pass
        
        # 绘制新的路径
        for vehicle_id, path in paths.items():
            if not path or len(path) < 2:
                continue
                
            # 找到对应的车辆
            vehicle = next((v for v in self.vehicles if v.vehicle_id == vehicle_id), None)
            if not vehicle:
                continue
            
            # 检查此路径是否与上一个相同
            path_key = str(path)
            if vehicle_id in self.displayed_paths and self.displayed_paths[vehicle_id] == path_key:
                continue  # 如果路径相同，跳过
            
            # 创建路径线
            x_data = [p[0] for p in path]
            y_data = [p[1] for p in path]
            
            # 使用更明显的样式表示新路径
            path_line = pg.PlotDataItem(
                x_data, y_data,
                pen=pg.mkPen(color=vehicle.color, width=3, style=Qt.SolidLine),
                name=f"车辆{vehicle_id}-路径{len(self.path_items)}"
            )
            
            self.main_view.addItem(path_line)
            self.path_items.append(path_line)  # 添加到路径项列表
            self.displayed_paths[vehicle_id] = path_key  # 记录显示的路径
            
            # 如果路径项过多，限制数量以避免性能问题
            max_paths = 50
            if len(self.path_items) > max_paths:
                # 移除最旧的路径
                old_item = self.path_items.pop(0)
                self.main_view.removeItem(old_item)

    def _update_stats_display(self, stats):
        """更新统计信息显示"""
        # 计算平均值
        avg_planning_time = np.mean(stats['planning_times']) if stats['planning_times'] else 0
        avg_path_length = np.mean(stats['path_lengths']) if stats['path_lengths'] else 0
        avg_node_expansions = np.mean(stats['astar_node_expansions']) if stats['astar_node_expansions'] else 0
        
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
        
        # 创建HTML格式的统计信息
        stats_html = f"""
        <h3>测试统计</h3>
        <table>
            <tr><td>测试进度:</td><td>{progress:.1f}% ({stats['successful_plans'] + stats['failed_plans']}/{stats['total_tests']})</td></tr>
            <tr><td>成功率:</td><td>{success_rate:.1f}%</td></tr>
            <tr><td>平均规划时间:</td><td>{avg_planning_time*1000:.2f} 毫秒</td></tr>
            <tr><td>平均路径长度:</td><td>{avg_path_length:.1f} 点</td></tr>
            <tr><td>平均节点展开:</td><td>{avg_node_expansions:.1f} 个</td></tr>
            <tr><td>检测到冲突:</td><td>{stats['conflicts_detected']} 个</td></tr>
            <tr><td>解决的冲突:</td><td>{stats['conflicts_resolved']} 个</td></tr>
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

    def _update_debug_display(self, debug_data):
        """更新调试视图显示"""
        # 清除旧的调试视图内容
        self.debug_view.clear()
        
        # 绘制障碍物
        if hasattr(self, 'obstacles'):
            obstacle_points = np.array(self.obstacles)
            if len(obstacle_points) > 0:
                self.debug_view.plot(
                    obstacle_points[:, 0], obstacle_points[:, 1],
                    pen=None, symbol='s', symbolSize=3, symbolBrush='gray',
                    name="障碍物"
                )
        
        # 绘制开放集
        if 'open_set' in debug_data and debug_data['open_set']:
            open_set = np.array(debug_data['open_set'])
            if len(open_set) > 0:
                self.debug_view.plot(
                    open_set[:, 0], open_set[:, 1],
                    pen=None, symbol='o', symbolSize=5, symbolBrush='green',
                    name="开放集"
                )
        
        # 绘制关闭集
        if 'closed_set' in debug_data and debug_data['closed_set']:
            closed_set = np.array(debug_data['closed_set'])
            if len(closed_set) > 0:
                self.debug_view.plot(
                    closed_set[:, 0], closed_set[:, 1],
                    pen=None, symbol='x', symbolSize=5, symbolBrush='red',
                    name="关闭集"
                )
        
        # 绘制当前路径
        if 'current_path' in debug_data and debug_data['current_path']:
            path = debug_data['current_path']
            x_data = [p[0] for p in path]
            y_data = [p[1] for p in path]
            self.debug_view.plot(
                x_data, y_data,
                pen=pg.mkPen('blue', width=2),
                name="当前路径"
            )
        
        # 绘制当前节点
        if 'current_node' in debug_data and debug_data['current_node']:
            node = debug_data['current_node']
            self.debug_view.plot(
                [node[0]], [node[1]],
                pen=None, symbol='*', symbolSize=15, symbolBrush='yellow',
                name="当前节点"
            )
        
        # 绘制起点和终点
        if 'start_point' in debug_data and debug_data['start_point']:
            start = debug_data['start_point']
            self.debug_view.plot(
                [start[0]], [start[1]],
                pen=None, symbol='s', symbolSize=10, symbolBrush='green',
                name="起点"
            )
        
        if 'end_point' in debug_data and debug_data['end_point']:
            end = debug_data['end_point']
            self.debug_view.plot(
                [end[0]], [end[1]],
                pen=None, symbol='s', symbolSize=10, symbolBrush='red',
                name="终点"
            )

    def _update_heatmap(self, paths):
        """更新热图数据"""
        # 对所有活动路径进行热图更新
        for vehicle_id, path in paths.items():
            if not path:
                continue
                
            # 将路径点添加到热图数据
            for x, y in path:
                # 确保坐标在地图范围内
                if 0 <= int(x) < self.map_size and 0 <= int(y) < self.map_size:
                    self.heatmap_data[int(y), int(x)] += 1
        
        # 更新热图显示
        if hasattr(self, 'heatmap_img'):
            # 应用平滑处理
            from scipy.ndimage import gaussian_filter
            smoothed_data = gaussian_filter(self.heatmap_data, sigma=2)
            
            # 更新图像数据
            self.heatmap_img.setImage(smoothed_data)

    def get_test_stats(self):
        """获取测试统计数据"""
        return self.test_stats    
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
        
        # 跳过第一个测试案例 (中心点到外面的路径)
        # 生成测试对，从索引1开始而不是0
        for i in range(1, min(self.num_test_pairs + 1, len(point_names))):
            start_idx = i % len(point_names)
            end_idx = (i + len(point_names) // 2) % len(point_names)  # 确保起点和终点不同
            
            if start_idx == end_idx:
                end_idx = (end_idx + 1) % len(point_names)
                
            start_name = point_names[start_idx]
            end_name = point_names[end_idx]
            
            # 跳过使用"中心点"的测试对
            if "中心点" in (start_name, end_name):
                continue
                
            start_point = self.test_points[start_name]
            end_point = self.test_points[end_name]
            
            test_pairs.append({
                'start': start_point,
                'end': end_point,
                'start_name': start_name,
                'end_name': end_name
            })
        
        # 如果需要更多测试对，添加随机组合，但避开中心点
        while len(test_pairs) < self.num_test_pairs:
            # 随机选择起点和终点
            start_idx = random.randint(0, len(point_names) - 1)
            end_idx = random.randint(0, len(point_names) - 1)
            
            # 确保起点和终点不同
            while end_idx == start_idx:
                end_idx = random.randint(0, len(point_names) - 1)
                
            start_name = point_names[start_idx]
            end_name = point_names[end_idx]
            
            # 跳过使用"中心点"的测试对
            if "中心点" in (start_name, end_name):
                continue
            
            start_point = self.test_points[start_name]
            end_point = self.test_points[end_name]
            
            test_pairs.append({
                'start': start_point,
                'end': end_point,
                'start_name': start_name,
                'end_name': end_name
            })
        
        logging.info(f"生成了 {len(test_pairs)} 个测试路径对")
        return test_pairs
        
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
        
    def _create_obstacles(self):
        """创建障碍物点集"""
        obstacles = []
        
        if not self.complex_map_checkbox.isChecked():
            # 简单地图，只创建几个简单障碍物
            obstacles.extend(self._create_simple_obstacles())
        else:
            # 复杂地图，创建多个不同类型的障碍物
            # 1. 添加矩形障碍物
            obstacles.extend(self._create_rectangular_obstacles())
            
            # 2. 添加多边形障碍物
            for polygon in self._create_polygon_obstacles():
                obstacles.extend(self._rasterize_polygon(polygon))
                
            # 3. 添加随机障碍物点
            obstacles.extend(self._create_random_obstacles())
        
        logging.info(f"创建了 {len(obstacles)} 个障碍物点")
        return obstacles
        
    def _create_simple_obstacles(self):
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
        
    def _create_rectangular_obstacles(self):
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
        
    def _create_polygon_obstacles(self):
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
        
    def _rasterize_polygon(self, polygon):
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
        
    def _point_in_polygon(self, point, polygon):
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
        
    def _create_random_obstacles(self):
        """创建随机障碍物点"""
        obstacles = []
        
        # 障碍物数量与地图大小成比例
        num_obstacles = int(0.01 * self.map_size * self.map_size)
        
        # 限制最大数量，避免创建太多障碍物
        num_obstacles = min(num_obstacles, 500)
        
        for _ in range(num_obstacles):
            x = random.randint(0, self.map_size)
            y = random.randint(0, self.map_size)
            
            # 避免关键点附近有障碍物
            if not any(math.dist((x, y), point) < 15 for point in self.test_points.values()):
                obstacles.append((x, y))
        
        return obstacles

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

    def _draw_map_and_obstacles(self):
        """绘制地图和障碍物"""
        # 绘制测试点
        for name, point in self.test_points.items():
            # 主视图测试点
            point_item = pg.ScatterPlotItem(
                [point[0]], [point[1]], 
                size=10, pen=pg.mkPen('b'), brush=pg.mkBrush('b')
            )
            self.main_view.addItem(point_item)
            
            # 点标签
            text_item = pg.TextItem(name, anchor=(0, 0), color='b')
            text_item.setPos(point[0] + 5, point[1] + 5)
            self.main_view.addItem(text_item)
        
        # 绘制障碍物
        x_coords = [p[0] for p in self.obstacles]
        y_coords = [p[1] for p in self.obstacles]
        
        obstacle_item = pg.ScatterPlotItem(
            x_coords, y_coords,
            size=2, pen=None, brush=pg.mkBrush(100, 100, 100, 150)
        )
        self.main_view.addItem(obstacle_item)
        
        # 同样在调试视图中绘制地图和障碍物
        if self.show_debug:
            self._draw_map_on_debug_view()

    def _draw_map_on_debug_view(self):
        """在调试视图上绘制地图和障碍物"""
        # 绘制障碍物
        x_coords = [p[0] for p in self.obstacles]
        y_coords = [p[1] for p in self.obstacles]
        
        obstacle_item = pg.ScatterPlotItem(
            x_coords, y_coords,
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
        
        # 更新测试信息
        self.current_test_info = {
            'current_idx': current_idx,
            'total_tests': len(self.test_pairs),
            'current_test': test_pair
        }
        
        test_start_time = time.time()
        
        # 更新状态文本
        self.update_status_text(f"开始测试 {current_idx+1}/{len(self.test_pairs)}: {start_name} -> {end_name}")
        
        # 为每个车辆分配测试
        for vehicle in self.vehicles:
            # 设置车辆起点
            vehicle.current_location = start_point
            vehicle.path_index = 0
            
            # 规划路径
            try:
                path = self.planner.plan_path(start_point, end_point, vehicle)
                
                # 检查路径是否有效
                if path and len(path) > 1:
                    # 保存路径
                    self.vehicle_paths[vehicle] = path
                    
                    # 初始化路径进度
                    self.vehicle_path_progress[vehicle] = 0
                    
                    # 添加到活动车辆列表
                    self.active_vehicles.append(vehicle)
                    
                    # 更新统计信息
                    self.test_stats['path_lengths'].append(len(path))
                    self.test_stats['successful_plans'] += 1
                else:
                    logging.warning(f"车辆 {vehicle.vehicle_id} 路径规划失败: 路径为空或太短")
                    self.test_stats['failed_plans'] += 1
            except Exception as e:
                logging.error(f"车辆 {vehicle.vehicle_id} 路径规划出错: {str(e)}")
                self.test_stats['failed_plans'] += 1
        
        # 如果启用了调试模式，添加更新热图
        if self.show_heatmap:
            self._update_heatmap_from_obstacles()
        
        # 如果没有活动车辆，立即进入下一个测试
        if not self.active_vehicles:
            self.current_test_info['current_idx'] += 1
            test_duration = time.time() - test_start_time
            self.test_stats['test_durations'].append(test_duration)
            self._start_new_test()
        
        # 更新总测试计数
        self.test_stats['total_tests'] += 1

    def _update_vehicles(self):
        """更新车辆位置"""
        if not self.active_vehicles:
            return
            
        for vehicle in list(self.active_vehicles):
            path = self.vehicle_paths.get(vehicle, [])
            
            if not path or len(path) < 2:
                self.active_vehicles.remove(vehicle)
                continue
                
            # 获取当前路径进度
            progress = self.vehicle_path_progress.get(vehicle, 0)
            
            # 检查是否到达终点
            if progress >= len(path) - 1:
                # 车辆到达终点
                self.active_vehicles.remove(vehicle)
                continue
                
            # 更新路径进度
            progress += self.simulation_speed * 0.1  # 根据模拟速度调整步长
            progress = min(progress, len(path) - 1)  # 确保不超过路径长度
            self.vehicle_path_progress[vehicle] = progress
            
            # 计算当前位置
            progress_int = int(progress)
            progress_frac = progress - progress_int
            
            if progress_int < len(path) - 1:
                current = path[progress_int]
                next_point = path[progress_int + 1]
                
                # 插值计算当前位置
                vehicle.current_location = (
                    current[0] + (next_point[0] - current[0]) * progress_frac,
                    current[1] + (next_point[1] - current[1]) * progress_frac
                )
            else:
                # 到达终点
                vehicle.current_location = path[-1]
        
        # 检查是否所有车辆都完成测试
        if not self.active_vehicles:
            # 记录测试持续时间
            test_duration = time.time() - self.test_stats.get('start_time', time.time())
            self.test_stats['test_durations'].append(test_duration)
            
            # 移动到下一个测试
            self.current_test_info['current_idx'] += 1
            
            # 更新热图
            if self.show_heatmap:
                self._update_heatmap_from_paths()

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
        
        # 存储冲突点（用于热图）
        self.conflict_points = []
        for conflict in conflicts:
            self.conflict_points.append(conflict["location"])
            
        # 在热图上更新冲突点
        if self.show_heatmap:
            self._update_heatmap_from_conflicts()
            
        # 更新状态
        plural = "个" if len(conflicts) != 1 else "个"
        self.update_status_text(f"检测到 {len(conflicts)} {plural}冲突")
        
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
                        # 重置进度
                        self.vehicle_path_progress[vehicle] = 0
                        break
        
        # 更新统计信息
        self.test_stats['conflicts_resolved'] += resolved_count
        
        # 清除冲突点
        self.conflict_points = []
        
        # 更新状态
        self.update_status_text(f"已解决 {resolved_count} 个冲突")

    def _update_debug_view(self):
        """更新调试视图"""
        # 清除现有调试元素
        self.debug_view.clear()
        
        # 绘制地图和障碍物到调试视图
        self._draw_map_on_debug_view()
        
        # 从debug_data中提取调试信息
        open_set = self.debug_data.get('open_set', [])
        closed_set = self.debug_data.get('closed_set', [])
        current_path = self.debug_data.get('current_path', [])
        current_node = self.debug_data.get('current_node')
        start_point = self.debug_data.get('start_point')
        end_point = self.debug_data.get('end_point')
        
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

    def _update_heatmap_view(self):
        """更新热图视图"""
        # 确保热图对象已创建
        if not hasattr(self, 'heatmap_img'):
            # 创建热图对象
            self.heatmap_img = pg.ImageItem()
            self.main_view.addItem(self.heatmap_img)
            
            # 设置颜色映射
            pos = np.array([0.0, 0.33, 0.66, 1.0])
            color = np.array([
                [0, 0, 0, 0],
                [0, 0, 255, 50],
                [255, 255, 0, 100],
                [255, 0, 0, 150]
            ])
            cmap = pg.ColorMap(pos, color)
            self.heatmap_img.setLookupTable(cmap.getLookupTable())
            
            # 设置位置和尺寸
            self.heatmap_img.setRect(pg.QtCore.QRectF(0, 0, self.map_size, self.map_size))
        
        # 更新热图数据
        self.heatmap_img.setImage(self.heatmap_data.T)
        self.heatmap_img.setOpacity(0.7)

    def _update_heatmap_from_paths(self):
        """从路径更新热图数据"""
        # 衰减现有热图数据
        self.heatmap_data *= 0.95
        
        # 为每个车辆的路径添加热度
        for vehicle in self.vehicles:
            path = self.vehicle_paths.get(vehicle, [])
            if not path:
                continue
                
            for x, y in path:
                # 确保坐标在地图范围内
                if 0 <= int(x) < self.map_size and 0 <= int(y) < self.map_size:
                    self.heatmap_data[int(x), int(y)] += 0.5
        
        # 更新热图
        if self.show_heatmap:
            self._update_heatmap_view()

    def _update_heatmap_from_conflicts(self):
        """从冲突点更新热图数据"""
        # 为冲突点添加高热度
        if hasattr(self, 'conflict_points'):
            for x, y in self.conflict_points:
                # 确保坐标在地图范围内
                if 0 <= int(x) < self.map_size and 0 <= int(y) < self.map_size:
                    self.heatmap_data[int(x), int(y)] += 3.0  # 冲突点热度高
                    
                    # 为冲突点周围添加热度
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            nx, ny = int(x) + dx, int(y) + dy
                            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                                distance = math.sqrt(dx*dx + dy*dy)
                                if distance <= 3:
                                    self.heatmap_data[nx, ny] += 2.0 * (1 - distance/3)
        
        # 更新热图
        if self.show_heatmap:
            self._update_heatmap_view()

    def _update_heatmap_from_obstacles(self):
        """从障碍物更新热图数据"""
        # 障碍物周围区域有一定热度
        for x, y in self.obstacles:
            if 0 <= int(x) < self.map_size and 0 <= int(y) < self.map_size:
                self.heatmap_data[int(x), int(y)] += 0.2
        
        # 更新热图
        if self.show_heatmap:
            self._update_heatmap_view()

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
        avg_nodes = np.mean(self.test_stats['astar_node_expansions']) if self.test_stats['astar_node_expansions'] else 0
        
        # 格式化结果HTML
        result_html = f"""
        <h3>测试结果统计</h3>
        <p>总测试数: {self.test_stats['total_tests']}</p>
        <p>规划成功: {self.test_stats['successful_plans']}</p>
        <p>规划失败: {self.test_stats['failed_plans']}</p>
        <p>成功率: {self.test_stats['successful_plans']/max(1, self.test_stats['total_tests'])*100:.1f}%</p>
        <p>平均规划时间: {avg_planning_time*1000:.2f}毫秒</p>
        <p>平均路径长度: {avg_path_length:.1f}点</p>
        <p>平均展开节点: {avg_nodes:.1f}个</p>
        <p>检测冲突: {self.test_stats['conflicts_detected']}</p>
        <p>解决冲突: {self.test_stats['conflicts_resolved']}</p>
        """
        
        # 更新统计面板
        self.update_stats_text(result_html)
        
        # 显示消息框
        QMessageBox.information(self, "测试完成", "所有路径规划测试已完成!")

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
                f.write("测试,起点,终点,规划时间(毫秒),路径长度,展开节点,成功\n")
                
                # 写入每次测试的结果
                for i in range(min(len(self.test_pairs), len(self.test_stats['planning_times']))):
                    test_pair = self.test_pairs[i]
                    planning_time = self.test_stats['planning_times'][i] * 1000  # 转换为毫秒
                    path_length = self.test_stats['path_lengths'][i] if i < len(self.test_stats['path_lengths']) else 0
                    nodes_count = self.test_stats['astar_node_expansions'][i] if i < len(self.test_stats['astar_node_expansions']) else 0
                    
                    # 写入一行
                    f.write(f"{i+1},{test_pair['start_name']},{test_pair['end_name']},{planning_time:.2f},{path_length},{nodes_count},是\n")
                
                # 写入失败的规划
                for i in range(len(self.test_stats['planning_times']), len(self.test_pairs)):
                    test_pair = self.test_pairs[i]
                    f.write(f"{i+1},{test_pair['start_name']},{test_pair['end_name']},0,0,0,否\n")
                
                # 写入汇总信息
                f.write("\n总结,,,,,\n")
                
                # 计算平均值
                avg_planning_time = np.mean(self.test_stats['planning_times']) * 1000 if self.test_stats['planning_times'] else 0
                avg_path_length = np.mean(self.test_stats['path_lengths']) if self.test_stats['path_lengths'] else 0
                avg_nodes = np.mean(self.test_stats['astar_node_expansions']) if self.test_stats['astar_node_expansions'] else 0
                
                # 写入汇总行
                f.write(f"总测试数,{self.test_stats['total_tests']},,,,\n")
                f.write(f"规划成功,{self.test_stats['successful_plans']},,,,\n")
                f.write(f"规划失败,{self.test_stats['failed_plans']},,,,\n")
                f.write(f"成功率,{self.test_stats['successful_plans']/max(1, self.test_stats['total_tests'])*100:.1f}%,,,,\n")
                f.write(f"平均规划时间,{avg_planning_time:.2f}毫秒,,,,\n")
                f.write(f"平均路径长度,{avg_path_length:.1f},,,,\n")
                f.write(f"平均展开节点,{avg_nodes:.1f},,,,\n")
                f.write(f"检测冲突,{self.test_stats['conflicts_detected']},,,,\n")
                f.write(f"解决冲突,{self.test_stats['conflicts_resolved']},,,,\n")
            
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
            'avg_nodes': np.mean(self.test_stats['astar_node_expansions']) if self.test_stats['astar_node_expansions'] else 0,
            'conflicts_detected': self.test_stats['conflicts_detected'],
            'conflicts_resolved': self.test_stats['conflicts_resolved'],
            'conflict_resolution_rate': conflict_resolution_rate
        }


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="路径规划器测试工具")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    parser.add_argument("--vehicles", type=int, default=5, help="测试车辆数量")
    parser.add_argument("--tests", type=int, default=10, help="测试对数量")
    args = parser.parse_args()
    
    # 创建Qt应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    tester = PathPlannerTester()
    
    # 应用命令行参数
    if args.debug:
        tester.debug_checkbox.setChecked(True)
    
    if args.vehicles:
        tester.vehicles_slider.setValue(args.vehicles)
    
    if args.tests:
        tester.tests_slider.setValue(args.tests)
    
    # 显示窗口
    tester.show()
    
    # 运行应用程序
    sys.exit(app.exec_())