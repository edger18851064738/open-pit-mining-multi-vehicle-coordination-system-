#!/usr/bin/env python3
"""
HybridPathPlanner 测试脚本

此脚本用于测试 HybridPathPlanner 中的路径规划算法。
它会创建一个含有障碍物的测试环境，并测试不同的路径规划方法，
包括 _fast_astar, _optimized_astar, simple_astar 和 plan_path 等。

测试结果会以可视化图表展示，便于比较不同算法的性能和路径质量。
"""

import os
import sys
import time
import math
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

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
    from algorithm.path_planner import HybridPathPlanner, Node, global_node_pool
    from algorithm.map_service import MapService
    from utils.geo_tools import GeoUtils
    from models.vehicle import MiningVehicle
    logging.info("成功导入所需模块")
except ImportError as e:
    logging.error(f"导入模块失败: {str(e)}")
    sys.exit(1)

class PathPlannerTester:
    """路径规划器测试类"""
    
    def __init__(self, map_size=200):
        """初始化测试环境"""
        self.map_size = map_size
        self.geo_utils = GeoUtils()
        self.map_service = MapService()
        self.planner = HybridPathPlanner(self.map_service)
        
        # 创建测试车辆
        self.test_vehicle = self._create_test_vehicle()
        
        # 测试点
        self.test_points = self._create_test_points()
        
        # 创建障碍物
        self.obstacles = self._create_test_obstacles()
        self.planner.obstacle_grids = set(self.obstacles)
        
        # 测试结果
        self.results = {}
        
        logging.info("测试环境初始化完成")
    
    def _create_test_vehicle(self) -> MiningVehicle:
        """创建测试用车辆对象"""
        vehicle = MiningVehicle(
            vehicle_id="test", 
            map_service=self.map_service,
            config={
                'current_location': (100, 100),
                'max_capacity': 50,
                'max_speed': 8.0,
                'min_hardness': 2.5,
                'turning_radius': 10.0,
                'base_location': (0, 0)
            }
        )
        return vehicle
    
    def _create_test_points(self) -> Dict[str, Tuple[float, float]]:
        """创建测试点"""
        return {
            "左上": (20, 180),
            "右上": (180, 180),
            "左下": (20, 20),
            "右下": (180, 20),
            "中心": (100, 100),
            "装载点": (50, 150),
            "卸载点": (150, 50),
            "充电站": (100, 20)
        }
    
    def _create_test_obstacles(self) -> List[Tuple[float, float]]:
        """创建测试障碍物"""
        obstacles = []
        
        # 预定义障碍物区域 (x_min, y_min, x_max, y_max)
        obstacle_areas = [
            (80, 80, 120, 120),   # 中心障碍物
            (40, 40, 60, 60),     # 左下障碍物
            (140, 140, 160, 160), # 右上障碍物
            (40, 140, 60, 160),   # 左上障碍物
            (140, 40, 160, 60),   # 右下障碍物
            (90, 30, 110, 170)    # 垂直长障碍物
        ]
        
        # 生成障碍点
        for area in obstacle_areas:
            x_min, y_min, x_max, y_max = area
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    obstacles.append((x, y))
        
        logging.info(f"创建了 {len(obstacles)} 个障碍点")
        return obstacles
    
    def run_tests(self):
        """运行所有测试"""
        logging.info("开始路径规划测试")
        
        # 定义测试用例 (起点名称, 终点名称)
        test_cases = [
            ("左上", "右下"),
            ("左下", "右上"),
            ("中心", "装载点"),
            ("中心", "卸载点"),
            ("装载点", "卸载点"),
            ("充电站", "装载点"),
        ]
        
        # 运行每个测试用例
        for case in test_cases:
            start_name, end_name = case
            start_point = self.test_points[start_name]
            end_point = self.test_points[end_name]
            
            logging.info(f"测试用例: {start_name} -> {end_name}")
            case_key = f"{start_name}->{end_name}"
            self.results[case_key] = {}
            
            # 测试不同的规划方法
            self._test_planner_methods(start_point, end_point, case_key)
        
        # 显示测试结果
        self._show_results()
    
    def _test_planner_methods(self, start: Tuple[float, float], end: Tuple[float, float], case_key: str):
        """测试不同的路径规划方法"""
        # 1. 测试 plan_path 方法 (默认方法)
        try:
            start_time = time.time()
            path1 = self.planner.plan_path(start, end, self.test_vehicle)
            elapsed = time.time() - start_time
            
            self.results[case_key]["plan_path"] = {
                "path": path1,
                "time": elapsed,
                "length": self._calculate_path_length(path1),
                "success": bool(path1 and len(path1) > 1)
            }
            logging.info(f"plan_path: 成功={bool(path1 and len(path1) > 1)}, 耗时={elapsed:.3f}秒, 路径长度={len(path1) if path1 else 0}")
        except Exception as e:
            logging.error(f"plan_path 方法失败: {str(e)}")
            self.results[case_key]["plan_path"] = {
                "path": None,
                "time": 0,
                "length": 0,
                "success": False
            }
        
        # 2. 测试 simple_astar 方法
        try:
            start_time = time.time()
            path2 = self.planner.simple_astar(start, end, self.test_vehicle)
            elapsed = time.time() - start_time
            
            self.results[case_key]["simple_astar"] = {
                "path": path2,
                "time": elapsed,
                "length": self._calculate_path_length(path2),
                "success": bool(path2 and len(path2) > 1)
            }
            logging.info(f"simple_astar: 成功={bool(path2 and len(path2) > 1)}, 耗时={elapsed:.3f}秒, 路径长度={len(path2) if path2 else 0}")
        except Exception as e:
            logging.error(f"simple_astar 方法失败: {str(e)}")
            self.results[case_key]["simple_astar"] = {
                "path": None,
                "time": 0,
                "length": 0,
                "success": False
            }
        
        # 3. 测试 _fast_astar 方法
        try:
            start_time = time.time()
            path3 = self.planner._fast_astar(start, end)
            elapsed = time.time() - start_time
            
            self.results[case_key]["_fast_astar"] = {
                "path": path3,
                "time": elapsed,
                "length": self._calculate_path_length(path3),
                "success": bool(path3 and len(path3) > 1)
            }
            logging.info(f"_fast_astar: 成功={bool(path3 and len(path3) > 1)}, 耗时={elapsed:.3f}秒, 路径长度={len(path3) if path3 else 0}")
        except Exception as e:
            logging.error(f"_fast_astar 方法失败: {str(e)}")
            self.results[case_key]["_fast_astar"] = {
                "path": None,
                "time": 0,
                "length": 0,
                "success": False
            }
        
        # 4. 测试 _optimized_astar 方法
        try:
            start_time = time.time()
            path4 = self.planner._optimized_astar(start, end, self.test_vehicle)
            elapsed = time.time() - start_time
            
            self.results[case_key]["_optimized_astar"] = {
                "path": path4,
                "time": elapsed,
                "length": self._calculate_path_length(path4),
                "success": bool(path4 and len(path4) > 1)
            }
            logging.info(f"_optimized_astar: 成功={bool(path4 and len(path4) > 1)}, 耗时={elapsed:.3f}秒, 路径长度={len(path4) if path4 else 0}")
        except Exception as e:
            logging.error(f"_optimized_astar 方法失败: {str(e)}")
            self.results[case_key]["_optimized_astar"] = {
                "path": None,
                "time": 0,
                "length": 0,
                "success": False
            }
        
        # 5. 测试 _generate_fallback_path 方法
        try:
            start_time = time.time()
            path5 = self.planner._generate_fallback_path(start, end)
            elapsed = time.time() - start_time
            
            self.results[case_key]["_generate_fallback_path"] = {
                "path": path5,
                "time": elapsed,
                "length": self._calculate_path_length(path5),
                "success": bool(path5 and len(path5) > 1)
            }
            logging.info(f"_generate_fallback_path: 成功={bool(path5 and len(path5) > 1)}, 耗时={elapsed:.3f}秒, 路径长度={len(path5) if path5 else 0}")
        except Exception as e:
            logging.error(f"_generate_fallback_path 方法失败: {str(e)}")
            self.results[case_key]["_generate_fallback_path"] = {
                "path": None,
                "time": 0,
                "length": 0,
                "success": False
            }
        
        # 6. 可选：测试 plan_with_timeout 方法
        if hasattr(self.planner, 'plan_with_timeout'):
            try:
                start_time = time.time()
                path6, _ = self.planner.plan_with_timeout(start, end, self.test_vehicle, timeout=1.0)
                elapsed = time.time() - start_time
                
                self.results[case_key]["plan_with_timeout"] = {
                    "path": path6,
                    "time": elapsed,
                    "length": self._calculate_path_length(path6),
                    "success": bool(path6 and len(path6) > 1)
                }
                logging.info(f"plan_with_timeout: 成功={bool(path6 and len(path6) > 1)}, 耗时={elapsed:.3f}秒, 路径长度={len(path6) if path6 else 0}")
            except Exception as e:
                logging.error(f"plan_with_timeout 方法失败: {str(e)}")
                self.results[case_key]["plan_with_timeout"] = {
                    "path": None,
                    "time": 0,
                    "length": 0,
                    "success": False
                }
    
    def _calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """计算路径长度"""
        if not path or len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            segment_length = math.sqrt(dx*dx + dy*dy)
            total_length += segment_length
            
        return total_length
    
    def _show_results(self):
        """显示测试结果"""
        if not self.results:
            logging.warning("没有测试结果可显示")
            return
        
        # 创建统计结果
        methods = ["plan_path", "simple_astar", "_fast_astar", "_optimized_astar", "_generate_fallback_path"]
        if "plan_with_timeout" in list(self.results.values())[0]:
            methods.append("plan_with_timeout")
            
        stats = {method: {"success": 0, "avg_time": 0.0, "avg_length": 0.0} for method in methods}
        
        # 计算每个方法的统计数据
        for case_key, case_results in self.results.items():
            for method, result in case_results.items():
                if method in stats:
                    if result["success"]:
                        stats[method]["success"] += 1
                        stats[method]["avg_time"] += result["time"]
                        stats[method]["avg_length"] += result["length"]
        
        # 计算平均值
        num_cases = len(self.results)
        for method in stats:
            if stats[method]["success"] > 0:
                stats[method]["avg_time"] /= stats[method]["success"]
                stats[method]["avg_length"] /= stats[method]["success"]
            stats[method]["success_rate"] = stats[method]["success"] / num_cases * 100
        
        # 显示统计结果
        print("\n=== 路径规划方法测试统计 ===")
        print(f"测试用例总数: {num_cases}")
        print("\n方法\t\t成功率\t平均时间(ms)\t平均路径长度")
        print("-" * 60)
        for method in methods:
            success_rate = stats[method]["success_rate"]
            avg_time = stats[method]["avg_time"] * 1000  # 转换为毫秒
            avg_length = stats[method]["avg_length"]
            print(f"{method:<15}\t{success_rate:>5.1f}%\t{avg_time:>8.2f}\t{avg_length:>8.2f}")
        
        # 可视化路径比较
        self._visualize_results()
    
    def _visualize_results(self):
        """可视化比较不同路径规划方法的结果"""
        # 创建全局的多图布局
        test_cases = list(self.results.keys())
        n_cases = len(test_cases)
        n_cols = min(3, n_cases)
        n_rows = (n_cases + n_cols - 1) // n_cols
        
        plt.figure(figsize=(6*n_cols, 5*n_rows))
        
        for i, case_key in enumerate(test_cases):
            case_results = self.results[case_key]
            
            # 创建子图
            plt.subplot(n_rows, n_cols, i+1)
            plt.title(f"测试用例: {case_key}")
            
            # 绘制障碍物
            obstacle_x = [p[0] for p in self.obstacles]
            obstacle_y = [p[1] for p in self.obstacles]
            plt.scatter(obstacle_x, obstacle_y, c='gray', s=10, alpha=0.5, label='障碍物')
            
            # 绘制起点和终点
            start_name, end_name = case_key.split("->")
            start_point = self.test_points[start_name]
            end_point = self.test_points[end_name]
            
            plt.scatter(start_point[0], start_point[1], c='green', s=100, marker='o', label='起点')
            plt.scatter(end_point[0], end_point[1], c='red', s=100, marker='x', label='终点')
            
            # 绘制不同方法生成的路径
            methods = ["plan_path", "simple_astar", "_fast_astar", "_optimized_astar", "_generate_fallback_path"]
            if "plan_with_timeout" in case_results:
                methods.append("plan_with_timeout")
                
            colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
            
            for j, method in enumerate(methods):
                if method in case_results and case_results[method]["success"]:
                    path = case_results[method]["path"]
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    
                    plt.plot(path_x, path_y, '-', color=colors[j % len(colors)], 
                            linewidth=2, alpha=0.7, 
                            label=f"{method} ({len(path)}点)")
            
            # 设置图例和坐标轴
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('X坐标')
            plt.ylabel('Y坐标')
            plt.xlim(0, self.map_size)
            plt.ylim(0, self.map_size)
        
        plt.tight_layout()
        plt.savefig('path_planner_test_results.png', dpi=150)
        plt.show()
        
        logging.info("测试结果可视化完成，已保存到 'path_planner_test_results.png'")
        
        # 另外创建性能比较图
        self._visualize_performance()
    
    def _visualize_performance(self):
        """可视化不同方法的性能指标"""
        methods = ["plan_path", "simple_astar", "_fast_astar", "_optimized_astar", "_generate_fallback_path"]
        if "plan_with_timeout" in list(self.results.values())[0]:
            methods.append("plan_with_timeout")
            
        # 收集性能数据
        success_rates = []
        avg_times = []
        avg_lengths = []
        
        for method in methods:
            success = 0
            total_time = 0.0
            total_length = 0.0
            
            for case_results in self.results.values():
                if method in case_results and case_results[method]["success"]:
                    success += 1
                    total_time += case_results[method]["time"]
                    total_length += case_results[method]["length"]
            
            success_rate = success / len(self.results) * 100
            avg_time = total_time / max(1, success) * 1000  # 转换为毫秒
            avg_length = total_length / max(1, success)
            
            success_rates.append(success_rate)
            avg_times.append(avg_time)
            avg_lengths.append(avg_length)
        
        # 绘制性能对比图
        plt.figure(figsize=(15, 6))
        
        # 1. 成功率对比
        plt.subplot(1, 3, 1)
        plt.bar(methods, success_rates, color='skyblue')
        plt.title('路径规划成功率')
        plt.ylabel('成功率 (%)')
        plt.ylim(0, 110)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. 平均耗时对比
        plt.subplot(1, 3, 2)
        plt.bar(methods, avg_times, color='salmon')
        plt.title('路径规划平均耗时')
        plt.ylabel('耗时 (毫秒)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. 平均路径长度对比
        plt.subplot(1, 3, 3)
        plt.bar(methods, avg_lengths, color='lightgreen')
        plt.title('路径规划平均长度')
        plt.ylabel('路径长度')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('path_planner_performance.png', dpi=150)
        plt.show()
        
        logging.info("性能对比可视化完成，已保存到 'path_planner_performance.png'")

def main():
    """主函数"""
    print("=" * 60)
    print("HybridPathPlanner 路径规划算法测试")
    print("=" * 60)
    
    # 创建测试器并运行测试
    tester = PathPlannerTester(map_size=200)
    tester.run_tests()
    
    print("\n测试完成！结果已保存到当前目录。")

if __name__ == "__main__":
    main()