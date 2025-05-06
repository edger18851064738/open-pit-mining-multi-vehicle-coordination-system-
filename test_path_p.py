"""
混合路径规划器测试脚本
用于测试强化A*算法在露天矿环境中的性能
"""

import sys
import os
import time
import math
import random
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Set
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 导入测试目标
from algorithm.hybrid_path_planner import HybridPathPlanner
from algorithm.reinforced_path_planner import ReinforcedAStar
from algorithm.map_service import MapService
from utils.geo_tools import GeoUtils

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 创建假的MapService类用于测试
class MockMapService:
    """模拟地图服务，用于测试"""
    
    def __init__(self, grid_size=200):
        self.grid_size = grid_size
        self.obstacle_grids = set()
        self.config = type('Config', (), {'grid_size': grid_size})
        
    def is_obstacle(self, point):
        """检查点是否是障碍物"""
        x, y = int(round(point[0])), int(round(point[1]))
        return (x, y) in self.obstacle_grids
        
    def validate_path(self, path):
        """验证路径是否有效"""
        if not path or len(path) < 2:
            return False
        
        # 检查路径中是否有障碍物
        for p in path:
            if self.is_obstacle(p):
                return False
                
        return True

# 创建迷宫式障碍物
def create_maze(map_size=200, wall_width=3):
    """创建迷宫式障碍物"""
    obstacles = set()
    
    # 边界墙
    for i in range(map_size):
        obstacles.add((0, i))
        obstacles.add((map_size-1, i))
        obstacles.add((i, 0))
        obstacles.add((i, map_size-1))
    
    # 水平墙
    h_walls = [
        (20, 40, 100, 40),    # 左侧水平墙
        (90, 80, 160, 80),    # 中间水平墙
        (20, 120, 100, 120),  # 右侧水平墙
        (90, 160, 160, 160)   # 底部水平墙
    ]
    
    # 垂直墙
    v_walls = [
        (40, 20, 40, 80),     # 上方垂直墙
        (80, 40, 80, 120),    # 中间垂直墙
        (120, 20, 120, 80),   # 下方垂直墙
        (160, 90, 160, 160)   # 右侧垂直墙
    ]
    
    # 创建墙壁
    for x1, y1, x2, y2 in h_walls:
        for x in range(x1, x2+1):
            for y in range(y1-wall_width//2, y1+wall_width//2+1):
                obstacles.add((x, y))
    
    for x1, y1, x2, y2 in v_walls:
        for y in range(y1, y2+1):
            for x in range(x1-wall_width//2, x1+wall_width//2+1):
                obstacles.add((x, y))
                
    # 添加随机障碍
    for _ in range(70):
        x = random.randint(1, map_size-2)
        y = random.randint(1, map_size-2)
        
        # 创建小型随机障碍群
        size = random.randint(2, 4)
        for dx in range(-size, size+1):
            for dy in range(-size, size+1):
                nx, ny = x + dx, y + dy
                if 0 < nx < map_size-1 and 0 < ny < map_size-1:
                    # 避开起点和终点区域
                    if not ((nx < 35 and ny < 35) or (nx > map_size-35 and ny > map_size-35)):
                        obstacles.add((nx, ny))
    
    # 确保关键通道畅通
    critical_paths = [
        # 从起点到第一个关键点的路径
        [(x, 30) for x in range(15, 60)],
        # 垂直通道
        [(80, y) for y in range(40, 120)],
        # 终点附近的通道
        [(x, 170) for x in range(160, 190)]
    ]
    
    # 确保关键通道有足够宽度
    for path in critical_paths:
        for point in path:
            x, y = point
            # 清除点周围的障碍物
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    nx, ny = x + dx, y + dy
                    if 0 < nx < map_size-1 and 0 < ny < map_size-1:
                        if (nx, ny) in obstacles:
                            obstacles.remove((nx, ny))
                            
    return obstacles

# 创建随机障碍物
def create_random_obstacles(map_size=200, obstacle_density=0.1):
    """创建随机障碍物"""
    obstacles = set()
    
    # 边界墙
    for i in range(map_size):
        obstacles.add((0, i))
        obstacles.add((map_size-1, i))
        obstacles.add((i, 0))
        obstacles.add((i, map_size-1))
    
    # 随机障碍物
    num_obstacles = int(map_size * map_size * obstacle_density)
    for _ in range(num_obstacles):
        x = random.randint(1, map_size-2)
        y = random.randint(1, map_size-2)
        
        # 避开起点和终点区域
        if (x < 20 and y < 20) or (x > map_size-20 and y > map_size-20):
            continue
            
        obstacles.add((x, y))
    
    return obstacles

# 测试路径规划器
def test_path_planner(map_size=200, obstacle_type='maze', num_tests=10):
    """
    测试混合路径规划器
    
    Args:
        map_size: 地图大小
        obstacle_type: 障碍物类型 ('maze'/'random')
        num_tests: 测试次数
    """
    print(f"\n=== 开始测试混合路径规划器 ({obstacle_type}障碍物, {num_tests}次测试) ===")
    
    # 创建地图服务
    map_service = MockMapService(map_size)
    
    # 创建障碍物
    if obstacle_type == 'maze':
        obstacles = create_maze(map_size)
    else:
        obstacles = create_random_obstacles(map_size)
    
    # 设置障碍物
    map_service.obstacle_grids = obstacles
    
    # 创建路径规划器
    planner = HybridPathPlanner(map_service)
    
    # 打印初始信息
    print(f"地图尺寸: {map_size}x{map_size}")
    print(f"障碍物数量: {len(obstacles)}")
    
    # 设置测试起点和终点
    start_point = (10, 10)
    end_point = (map_size-10, map_size-10)
    
    # 创建随机测试点对
    test_points = [(start_point, end_point)]
    
    for _ in range(num_tests-1):
        # 随机起点和终点（避开障碍物区域）
        while True:
            sx = random.randint(10, map_size-10)
            sy = random.randint(10, map_size-10)
            if not map_service.is_obstacle((sx, sy)):
                start = (sx, sy)
                break
                
        while True:
            ex = random.randint(10, map_size-10)
            ey = random.randint(10, map_size-10)
            if not map_service.is_obstacle((ex, ey)) and (ex, ey) != start:
                end = (ex, ey)
                break
                
        test_points.append((start, end))
    
    # 执行测试
    paths = []
    times = []
    path_lengths = []
    success_flags = []
    num_rows = (num_tests + 2) // 3  # 每行显示3个图
    num_cols = min(3, num_tests)
    plt.figure(figsize=(15, 5 * num_rows))
    
    for i, (start, end) in enumerate(test_points):
        print(f"\n测试 {i+1}/{num_tests}: {start} → {end}")
        
        # 计时并规划路径
        start_time = time.time()
        path = planner.plan_path(start, end)
        elapsed = time.time() - start_time
        
        # 记录结果
        paths.append(path)
        times.append(elapsed)
        
        # 检查路径
        if path:
            path_length = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))
            path_lengths.append(path_length)
            
            # 验证路径不经过障碍物
            is_valid = all(not map_service.is_obstacle(p) for p in path)
            success_flags.append(is_valid)
            
            status = "有效" if is_valid else "无效(穿越障碍物)"
            print(f"路径长度: {len(path)}点, 距离: {path_length:.1f}, 规划时间: {elapsed*1000:.1f}ms - {status}")
        else:
            path_lengths.append(0)
            success_flags.append(False)
            print(f"规划失败! 耗时: {elapsed*1000:.1f}ms")
        
        # 在子图中可视化当前测试结果
        plt.subplot(num_rows, num_cols, i + 1)
        
        # 绘制障碍物
        obstacle_x = [p[0] for p in obstacles]
        obstacle_y = [p[1] for p in obstacles]
        plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', s=10, alpha=0.3, label='障碍物')
        
        # 绘制起点和终点
        plt.scatter(start[0], start[1], c='green', marker='o', s=100, label='起点')
        plt.scatter(end[0], end[1], c='red', marker='o', s=100, label='终点')
        
        # 绘制路径
        if path and len(path) > 1:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, 'b-', linewidth=2, label='规划路径')
        
        plt.title(f"测试 {i+1}")
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.grid(True, alpha=0.3)
        if i == 0:  # 只在第一个子图显示图例
            plt.legend(loc='upper right')
    
    plt.tight_layout()  # 自动调整子图布局
    plt.savefig("all_path_planning_results.png")
    plt.show()
    
    # 统计结果
    success_rate = sum(success_flags) / len(success_flags) * 100
    avg_time = sum(times) / len(times) * 1000  # 转换为毫秒
    avg_length = sum(path_lengths) / sum(1 for l in path_lengths if l > 0) if any(path_lengths) else 0
    
    print("\n=== 测试结果汇总 ===")
    print(f"成功率: {success_rate:.1f}% ({sum(success_flags)}/{len(success_flags)})")
    print(f"平均规划时间: {avg_time:.1f}ms")
    print(f"平均路径长度: {avg_length:.1f}")
    
    # 获取其它性能指标
    stats = planner.get_performance_stats()
    print("\n=== 性能统计 ===")
    print(f"规划总次数: {stats['planning_count']}")
    print(f"成功次数: {stats['success_count']}")
    print(f"失败次数: {stats['failure_count']}")
    print(f"平均规划时间: {stats['avg_planning_time']}")
    
    rl_stats = stats.get('reinforced_learning', {})
    print("\n=== 强化学习统计 ===")
    print(f"已训练轮次: {rl_stats.get('episodes_trained', 0)}")
    print(f"成功路径数: {rl_stats.get('successful_paths', 0)}")
    print(f"Q值状态数: {rl_stats.get('q_values_states', 0)}")
    print(f"经验缓冲区大小: {rl_stats.get('experience_buffer_size', 0)}")
    
    # 可视化第一条测试路径
    visualize_path(planner, obstacles, test_points[0][0], test_points[0][1], paths[0])
    
    return planner, paths, test_points

def visualize_path(planner, obstacles, start, end, path):
    """
    可视化路径规划结果
    
    Args:
        planner: 路径规划器
        obstacles: 障碍物集合
        start: 起点
        end: 终点
        path: 规划的路径
    """
    if not path:
        print("无法可视化空路径")
        return
        
    plt.figure(figsize=(10, 10))
    
    # 绘制障碍物
    obstacle_x = [p[0] for p in obstacles]
    obstacle_y = [p[1] for p in obstacles]
    plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', s=10, alpha=0.3, label='障碍物')
    
    # 绘制起点和终点
    plt.scatter(start[0], start[1], c='green', marker='o', s=100, label='起点')
    plt.scatter(end[0], end[1], c='red', marker='o', s=100, label='终点')
    
    # 绘制路径
    if path and len(path) > 1:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='规划路径')
    
    plt.title("混合路径规划器测试")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # 保存图像
    plt.savefig("path_planning_result.png")
    plt.show()
    
    print("路径规划结果已保存到 path_planning_result.png")

def compare_algorithms(map_size=200, obstacle_type='maze', num_tests=5):
    """
    比较强化A*和普通A*算法
    
    Args:
        map_size: 地图大小
        obstacle_type: 障碍物类型 ('maze'/'random')
        num_tests: 测试次数
    """
    print(f"\n=== 比较强化A*和普通A*算法 ({obstacle_type}障碍物, {num_tests}次测试) ===")
    
    # 创建地图服务
    map_service = MockMapService(map_size)
    
    # 创建障碍物
    if obstacle_type == 'maze':
        obstacles = create_maze(map_size)
    else:
        obstacles = create_random_obstacles(map_size)
    
    # 设置障碍物
    map_service.obstacle_grids = obstacles
    
    # 创建混合路径规划器
    hybrid_planner = HybridPathPlanner(map_service)
    
    # 创建简单的测试A*算法
    class SimpleAStar:
        def __init__(self, obstacles, map_size):
            self.obstacles = obstacles
            self.map_size = map_size
            
        def is_obstacle(self, point):
            x, y = int(round(point[0])), int(round(point[1]))
            return (x, y) in self.obstacles
            
        def plan_path(self, start, end):
            """简单A*算法实现"""
            # 初始化开放集和闭合集
            open_set = []
            open_set_hash = set()
            closed_set = set()
            
            # 添加起始节点到开放集
            start_f = math.dist(start, end)
            heapq.heappush(open_set, (start_f, 0, start))
            open_set_hash.add(start)
            
            # 路径追踪和成本记录
            came_from = {}
            g_score = {start: 0}
            f_score = {start: start_f}
            
            # 主循环
            iterations = 0
            max_iterations = 10000
            
            while open_set and iterations < max_iterations:
                iterations += 1
                
                # 获取f值最低的节点
                current_f, current_g, current = heapq.heappop(open_set)
                open_set_hash.remove(current)
                
                # 检查是否到达目标
                if math.dist(current, end) < 3.0:
                    # 重建路径
                    path = [current]
                    while current in came_from:
                        current = came_from[current]
                        path.append(current)
                    path.reverse()
                    
                    # 添加终点
                    if path[-1] != end:
                        path.append(end)
                        
                    return path
                
                # 添加到闭合集
                closed_set.add(current)
                
                # 检查所有可能的方向
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # 检查边界
                    if not (0 <= neighbor[0] < self.map_size and 0 <= neighbor[1] < self.map_size):
                        continue
                    
                    # 跳过已在闭合集中的节点
                    if neighbor in closed_set:
                        continue
                    
                    # 检查障碍物
                    if self.is_obstacle(neighbor):
                        continue
                    
                    # 检查对角线移动的安全性
                    if dx != 0 and dy != 0:
                        if (self.is_obstacle((current[0], current[1] + dy)) or 
                            self.is_obstacle((current[0] + dx, current[1]))):
                            continue
                    
                    # 计算新的g值
                    move_cost = math.sqrt(dx*dx + dy*dy)
                    tentative_g = g_score[current] + move_cost
                    
                    # 检查是否找到更好的路径
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + math.dist(neighbor, end)
                        
                        # 如果节点不在开放集中，添加它
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
            
            # 搜索失败，返回直线路径
            return [start, end]
    
    # 创建简单A*算法实例
    simple_astar = SimpleAStar(obstacles, map_size)
    
    # 创建随机测试点对
    test_points = []
    for _ in range(num_tests):
        # 随机起点和终点（避开障碍物区域）
        while True:
            sx = random.randint(10, map_size-10)
            sy = random.randint(10, map_size-10)
            if (sx, sy) not in obstacles:
                start = (sx, sy)
                break
                
        while True:
            ex = random.randint(10, map_size-10)
            ey = random.randint(10, map_size-10)
            if (ex, ey) not in obstacles and (ex, ey) != start:
                end = (ex, ey)
                break
                
        test_points.append((start, end))
    
    # 比较结果记录
    hybrid_paths = []
    hybrid_times = []
    hybrid_lengths = []
    
    simple_paths = []
    simple_times = []
    simple_lengths = []
    
    # 执行测试
    for i, (start, end) in enumerate(test_points):
        print(f"\n测试 {i+1}/{num_tests}: {start} → {end}")
        
        # 测试混合规划器
        start_time = time.time()
        hybrid_path = hybrid_planner.plan_path(start, end)
        hybrid_elapsed = time.time() - start_time
        
        hybrid_paths.append(hybrid_path)
        hybrid_times.append(hybrid_elapsed)
        
        if hybrid_path:
            hybrid_length = sum(math.dist(hybrid_path[i], hybrid_path[i+1]) for i in range(len(hybrid_path)-1))
            hybrid_lengths.append(hybrid_length)
            print(f"强化A*: 路径长度 {len(hybrid_path)}点, 距离 {hybrid_length:.1f}, 时间 {hybrid_elapsed*1000:.1f}ms")
        else:
            hybrid_lengths.append(0)
            print(f"强化A*: 规划失败! 时间 {hybrid_elapsed*1000:.1f}ms")
        
        # 测试简单A*
        start_time = time.time()
        simple_path = simple_astar.plan_path(start, end)
        simple_elapsed = time.time() - start_time
        
        simple_paths.append(simple_path)
        simple_times.append(simple_elapsed)
        
        if simple_path:
            simple_length = sum(math.dist(simple_path[i], simple_path[i+1]) for i in range(len(simple_path)-1))
            simple_lengths.append(simple_length)
            print(f"普通A*: 路径长度 {len(simple_path)}点, 距离 {simple_length:.1f}, 时间 {simple_elapsed*1000:.1f}ms")
        else:
            simple_lengths.append(0)
            print(f"普通A*: 规划失败! 时间 {simple_elapsed*1000:.1f}ms")
    
    # 对比统计
    hybrid_avg_time = sum(hybrid_times) / len(hybrid_times) * 1000
    simple_avg_time = sum(simple_times) / len(simple_times) * 1000
    
    hybrid_avg_length = sum(hybrid_lengths) / sum(1 for l in hybrid_lengths if l > 0) if any(hybrid_lengths) else 0
    simple_avg_length = sum(simple_lengths) / sum(1 for l in simple_lengths if l > 0) if any(simple_lengths) else 0
    
    # 计算路径质量比较
    better_paths = 0
    for i in range(len(test_points)):
        if hybrid_lengths[i] > 0 and simple_lengths[i] > 0:
            # 比较路径长度，考虑5%的容差
            if hybrid_lengths[i] < simple_lengths[i] * 0.95:  # 强化A*路径至少短5%
                better_paths += 1
    
    better_percentage = better_paths / num_tests * 100
    
    print("\n=== 算法对比结果 ===")
    print(f"平均规划时间: 强化A*={hybrid_avg_time:.1f}ms, 普通A*={simple_avg_time:.1f}ms")
    print(f"平均路径长度: 强化A*={hybrid_avg_length:.1f}, 普通A*={simple_avg_length:.1f}")
    print(f"强化A*路径质量更好的比例: {better_percentage:.1f}%")
    
    # 可视化比较
    visualize_comparison(obstacles, test_points[0][0], test_points[0][1], 
                         hybrid_paths[0], simple_paths[0])
    
    return hybrid_planner, simple_astar, test_points

def visualize_comparison(obstacles, start, end, hybrid_path, simple_path):
    """
    可视化两种算法的路径比较
    
    Args:
        obstacles: 障碍物集合
        start: 起点
        end: 终点
        hybrid_path: 混合规划器路径
        simple_path: 简单A*路径
    """
    plt.figure(figsize=(12, 10))
    
    # 绘制障碍物
    obstacle_x = [p[0] for p in obstacles]
    obstacle_y = [p[1] for p in obstacles]
    plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', s=10, alpha=0.3, label='障碍物')
    
    # 绘制起点和终点
    plt.scatter(start[0], start[1], c='green', marker='o', s=100, label='起点')
    plt.scatter(end[0], end[1], c='red', marker='o', s=100, label='终点')
    
    # 绘制混合规划器路径
    if hybrid_path and len(hybrid_path) > 1:
        path_x = [p[0] for p in hybrid_path]
        path_y = [p[1] for p in hybrid_path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='强化A*路径')
    
    # 绘制简单A*路径
    if simple_path and len(simple_path) > 1:
        path_x = [p[0] for p in simple_path]
        path_y = [p[1] for p in simple_path]
        plt.plot(path_x, path_y, 'r--', linewidth=2, label='普通A*路径')
    
    plt.title("路径规划算法比较")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # 保存图像
    plt.savefig("algorithm_comparison.png")
    plt.show()
    
    print("算法比较结果已保存到 algorithm_comparison.png")

def test_learning_effect(map_size=200, obstacle_type='maze', num_iterations=5):
    """
    测试强化学习效果
    
    Args:
        map_size: 地图大小
        obstacle_type: 障碍物类型 ('maze'/'random')
        num_iterations: 迭代次数
    """
    print(f"\n=== 测试强化学习效果 ({obstacle_type}障碍物, {num_iterations}次迭代) ===")
    
    # 创建地图服务
    map_service = MockMapService(map_size)
    
    # 创建障碍物
    if obstacle_type == 'maze':
        obstacles = create_maze(map_size)
    else:
        obstacles = create_random_obstacles(map_size)
    
    # 设置障碍物
    map_service.obstacle_grids = obstacles
    
    # 创建混合路径规划器
    planner = HybridPathPlanner(map_service)
    
    # 设置固定的起点和终点进行多次规划
    start_point = (10, 10)
    end_point = (map_size-10, map_size-10)
    
    # 记录每次迭代的结果
    paths = []
    times = []
    path_lengths = []
    
    # 运行多次迭代
    for i in range(num_iterations):
        print(f"\n迭代 {i+1}/{num_iterations}: {start_point} → {end_point}")
        
        # 规划路径
        start_time = time.time()
        path = planner.plan_path(start_point, end_point, force_replan=True)  # 强制重新规划
        elapsed = time.time() - start_time
        
        paths.append(path)
        times.append(elapsed)
        
        # 计算路径长度
        if path:
            path_length = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))
            path_lengths.append(path_length)
            print(f"路径长度: {len(path)}点, 距离: {path_length:.1f}, 时间: {elapsed*1000:.1f}ms")
        else:
            path_lengths.append(0)
            print(f"规划失败! 时间: {elapsed*1000:.1f}ms")
    
    # 获取强化学习指标
    stats = planner.get_performance_stats()
    rl_stats = stats.get('reinforced_learning', {})
    
    print("\n=== 学习效果分析 ===")
    print(f"训练轮次: {rl_stats.get('episodes_trained', 0)}")
    print(f"Q值状态数: {rl_stats.get('q_values_states', 0)}")
    
    # 计算改进效果
    if len(path_lengths) > 1 and path_lengths[0] > 0 and path_lengths[-1] > 0:
        improvement = (path_lengths[0] - path_lengths[-1]) / path_lengths[0] * 100
        print(f"路径长度改进: {improvement:.2f}%")
        
        time_improvement = (times[0] - times[-1]) / times[0] * 100
        print(f"规划时间改进: {time_improvement:.2f}%")
    
    # 可视化学习效果
    visualize_learning(obstacles, start_point, end_point, paths)
    
    return planner, paths

def visualize_learning(obstacles, start, end, paths):
    """
    可视化学习效果
    
    Args:
        obstacles: 障碍物集合
        start: 起点
        end: 终点
        paths: 每次迭代的路径
    """
    plt.figure(figsize=(12, 10))
    
    # 绘制障碍物
    obstacle_x = [p[0] for p in obstacles]
    obstacle_y = [p[1] for p in obstacles]
    plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', s=10, alpha=0.3, label='障碍物')
    
    # 绘制起点和终点
    plt.scatter(start[0], start[1], c='green', marker='o', s=100, label='起点')
    plt.scatter(end[0], end[1], c='red', marker='o', s=100, label='终点')
    
    # 绘制多次迭代的路径
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, path in enumerate(paths):
        if path and len(path) > 1:
            color = colors[i % len(colors)]
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, f'{color}-', linewidth=2, label=f'迭代 {i+1}')
    
    plt.title("强化学习效果")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # 保存图像
    plt.savefig("learning_effect.png")
    plt.close()
    
    print("学习效果可视化已保存到 learning_effect.png")

def test_path_safety(map_size=200, obstacle_type='maze', num_tests=10):
    """
    测试路径安全性，检查路径是否穿过障碍物
    
    Args:
        map_size: 地图大小
        obstacle_type: 障碍物类型 ('maze'/'random')
        num_tests: 测试次数
    """
    print(f"\n=== 测试路径安全性 ({obstacle_type}障碍物, {num_tests}次测试) ===")
    
    # 创建地图服务
    map_service = MockMapService(map_size)
    
    # 创建障碍物
    if obstacle_type == 'maze':
        obstacles = create_maze(map_size)
    else:
        obstacles = create_random_obstacles(map_size, obstacle_density=0.2)  # 提高障碍物密度
    
    # 设置障碍物
    map_service.obstacle_grids = obstacles
    
    # 创建混合路径规划器
    planner = HybridPathPlanner(map_service)
    
    # 创建随机测试点对，但选择更具挑战性的路径
    test_points = []
    for _ in range(num_tests):
        # 随机选择可能会穿过障碍物的起点和终点
        while True:
            sx = random.randint(10, map_size//3)
            sy = random.randint(10, map_size//3)
            if not map_service.is_obstacle((sx, sy)):
                start = (sx, sy)
                break
                
        while True:
            ex = random.randint(2*map_size//3, map_size-10)
            ey = random.randint(2*map_size//3, map_size-10)
            if not map_service.is_obstacle((ex, ey)):
                end = (ex, ey)
                break
                
        test_points.append((start, end))
    
    # 执行测试
    results = []
    for i, (start, end) in enumerate(test_points):
        print(f"\n测试 {i+1}/{num_tests}: {start} → {end}")
        
        # 规划路径
        path = planner.plan_path(start, end)
        
        if not path:
            print("路径规划失败!")
            results.append(False)
            continue
            
        # 检查路径是否安全 (不穿过障碍物)
        is_safe = True
        unsafe_points = []
        
        for j in range(len(path)-1):
            p1 = path[j]
            p2 = path[j+1]
            
            # 检查线段上的所有点
            line_points = planner.reinforced_astar.get_line_points(p1, p2)
            for point in line_points:
                if map_service.is_obstacle(point):
                    is_safe = False
                    unsafe_points.append(point)
        
        results.append(is_safe)
        if is_safe:
            print(f"路径安全! 长度: {len(path)}点")
        else:
            print(f"不安全路径! 发现 {len(unsafe_points)} 个穿过障碍物的点")
    
    # 统计结果
    safety_rate = sum(results) / len(results) * 100
    
    print("\n=== 安全性测试结果 ===")
    print(f"安全路径比例: {safety_rate:.1f}% ({sum(results)}/{len(results)})")
    
    # 可视化最后一次测试
    if test_points:
        path = planner.plan_path(test_points[-1][0], test_points[-1][1])
        visualize_path_safety(obstacles, test_points[-1][0], test_points[-1][1], path)
    
    return planner, results

def visualize_path_safety(obstacles, start, end, path):
    """
    可视化路径安全性测试
    
    Args:
        obstacles: 障碍物集合
        start: 起点
        end: 终点
        path: 规划的路径
    """
    plt.figure(figsize=(12, 10))
    
    # 绘制障碍物
    obstacle_x = [p[0] for p in obstacles]
    obstacle_y = [p[1] for p in obstacles]
    plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', s=10, alpha=0.3, label='障碍物')
    
    # 绘制起点和终点
    plt.scatter(start[0], start[1], c='green', marker='o', s=100, label='起点')
    plt.scatter(end[0], end[1], c='red', marker='o', s=100, label='终点')
    
    # 绘制路径
    if path and len(path) > 1:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='规划路径')
        
        # 检查路径是否安全
        unsafe_points = []
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            
            # Bresenham算法获取线段上的点
            x1, y1 = int(round(p1[0])), int(round(p1[1]))
            x2, y2 = int(round(p2[0])), int(round(p2[1]))
            
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            
            err = dx - dy
            
            while True:
                point = (x1, y1)
                if point in obstacles:
                    unsafe_points.append(point)
                
                if x1 == x2 and y1 == y2:
                    break
                
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x1 += sx
                if e2 < dx:
                    err += dx
                    y1 += sy
        
        # 标记不安全点
        if unsafe_points:
            unsafe_x = [p[0] for p in unsafe_points]
            unsafe_y = [p[1] for p in unsafe_points]
            plt.scatter(unsafe_x, unsafe_y, c='yellow', marker='x', s=100, label='不安全点')
    
    plt.title("路径安全性测试")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # 保存图像
    plt.savefig("path_safety_test.png")
    plt.show()
    
    print("路径安全性测试可视化已保存到 path_safety_test.png")

def main():
    """主函数，运行各种测试"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='混合路径规划器测试')
    parser.add_argument('--map-size', type=int, default=200, help='地图尺寸')
    parser.add_argument('--test-type', choices=['basic', 'compare', 'learning', 'safety', 'all'], 
                       default='basic', help='测试类型')
    parser.add_argument('--obstacle-type', choices=['maze', 'random'], default='maze', 
                       help='障碍物类型')
    parser.add_argument('--num-tests', type=int, default=10, help='测试次数')
    args = parser.parse_args()
    
    # 运行选择的测试
    if args.test_type == 'basic' or args.test_type == 'all':
        test_path_planner(args.map_size, args.obstacle_type, args.num_tests)
        
    if args.test_type == 'compare' or args.test_type == 'all':
        compare_algorithms(args.map_size, args.obstacle_type, args.num_tests)
        
    if args.test_type == 'learning' or args.test_type == 'all':
        test_learning_effect(args.map_size, args.obstacle_type, args.num_tests)
        
    if args.test_type == 'safety' or args.test_type == 'all':
        test_path_safety(args.map_size, args.obstacle_type, args.num_tests)
        
    print("\n所有测试完成!")

if __name__ == "__main__":
    main()