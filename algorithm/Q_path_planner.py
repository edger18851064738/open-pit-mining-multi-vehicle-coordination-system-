import sys
import os
import math
import time
import heapq
import logging
import threading
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Set, Callable
from collections import deque, defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError
from algorithm.map_service import MapService
from matplotlib.patches import Rectangle, Polygon
import matplotlib
import math
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 
# 常量定义
MAX_ITERATIONS = 30000      # 强化A*搜索的最大迭代次数(增加以支持更复杂的地图)
DEFAULT_TIMEOUT = 7.0       # 默认超时时间(秒)(增加以支持更复杂的计算)
EPSILON = 1e-6              # 浮点数比较精度
CACHE_SIZE = 1000           # 缓存大小
CACHE_EXPIRY = 600          # 缓存过期时间(秒)
DEFAULT_LEARNING_RATE = 0.1  # 默认学习率
DEFAULT_DISCOUNT_FACTOR = 0.9  # 默认折扣因子

class PathPlanningError(Exception):
    """路径规划错误基类"""
    pass

class TimeoutError(PathPlanningError):
    """超时错误"""
    pass

class NoPathFoundError(PathPlanningError):
    """无法找到路径错误"""
    pass

class SpatialIndex:
    """空间索引结构，用于加速空间查询"""
    def __init__(self, cell_size=10):
        self.cell_size = cell_size
        self.grid = {}
        
    def add_point(self, point):
        """添加点到索引"""
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        cell_key = (cell_x, cell_y)
        
        if cell_key not in self.grid:
            self.grid[cell_key] = set()
        self.grid[cell_key].add(point)
            
    def add_points(self, points):
        """批量添加点"""
        for point in points:
            self.add_point(point)
            
    def query_point(self, point, radius=0):
        """查询点附近的点"""
        result = set()
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        
        # 计算需要检查的单元格范围
        cell_radius = max(1, int(radius // self.cell_size) + 1)
        
        # 检查相邻单元格
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.grid:
                    for p in self.grid[cell_key]:
                        if radius == 0 or math.dist(point, p) <= radius:
                            result.add(p)
                            
        return result
        
    def clear(self):
        """清空索引"""
        self.grid.clear()

class PathCache:
    """高性能路径缓存系统"""
    def __init__(self, max_size=CACHE_SIZE, expiry=CACHE_EXPIRY):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.expiry = expiry
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key):
        """获取缓存项"""
        with self.lock:
            now = time.time()
            if key in self.cache:
                # 检查是否过期
                if now - self.timestamps[key] <= self.expiry:
                    # 更新时间戳
                    self.timestamps[key] = now
                    self.hits += 1
                    return self.cache[key].copy()  # 返回副本避免修改缓存
                else:
                    # 过期删除
                    del self.cache[key]
                    del self.timestamps[key]
            
            self.misses += 1
            return None
            
    def put(self, key, value):
        """添加缓存项"""
        with self.lock:
            now = time.time()
            
            # 检查容量
            if len(self.cache) >= self.max_size:
                # 删除最旧的项
                oldest_key = min(self.timestamps, key=self.timestamps.get)
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                
            # 添加新项
            self.cache[key] = value.copy()  # 存储副本避免外部修改
            self.timestamps[key] = now
            
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            
    def get_stats(self):
        """获取缓存统计信息"""
        with self.lock:
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses
            }

class ReinforcedAStar:
    """强化A*算法实现，用于学习最优路径"""
    
    def __init__(self, planner, learning_rate=DEFAULT_LEARNING_RATE, discount_factor=DEFAULT_DISCOUNT_FACTOR):
        self.planner = planner
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = {}  # 存储状态-动作对应的Q值
        self.experience_buffer = []  # 经验回放缓冲区
        self.max_buffer_size = 1000  # 最大缓冲区大小
        
        # 地图相关属性
        self.map_size = getattr(planner, 'map_size', 200)
        self.obstacle_grids = getattr(planner, 'obstacle_grids', set())
        
        # 动作空间 - 8个方向
        self.actions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
        
        # 移动成本
        self.move_costs = {
            (0, 1): 1.0, (1, 0): 1.0, (0, -1): 1.0, (-1, 0): 1.0,
            (1, 1): 1.414, (1, -1): 1.414, (-1, 1): 1.414, (-1, -1): 1.414
        }
        
        # 统计和性能指标
        self.episodes_trained = 0
        self.successful_paths = 0
        self.last_path_length = 0
        self.last_training_time = 0
        
    def get_heuristic(self, current, goal):
        """大幅强化版启发式函数"""
        dx, dy = abs(current[0] - goal[0]), abs(current[1] - goal[1])
        
        # 欧几里得距离基础
        d_euclidean = math.sqrt(dx*dx + dy*dy)
        
        # 大幅增加障碍物的影响因子
        obstacles_nearby = self._count_nearby_obstacles(current, radius=3)
        obstacle_factor = 2.0 + (obstacles_nearby * 0.5)  # 大幅增加障碍物影响
        
        # 检查是否在障碍物之间的狭窄通道
        in_corridor = self._is_in_corridor(current)
        corridor_factor = 0.7 if in_corridor else 1.0  # 优先考虑通道
        
        # 混合启发式
        return d_euclidean * obstacle_factor * corridor_factor
    def _is_in_corridor(self, point):
        """检测点是否在障碍物之间的通道中"""
        x, y = point
        
        # 检查四个方向的障碍物情况
        left_blocked = False
        right_blocked = False
        up_blocked = False
        down_blocked = False
        
        # 检查水平方向
        for i in range(1, 4):
            if self._is_obstacle_fast((x-i, y)):
                left_blocked = True
                break
        
        for i in range(1, 4):
            if self._is_obstacle_fast((x+i, y)):
                right_blocked = True
                break
        
        # 检查垂直方向
        for i in range(1, 4):
            if self._is_obstacle_fast((x, y-i)):
                down_blocked = True
                break
        
        for i in range(1, 4):
            if self._is_obstacle_fast((x, y+i)):
                up_blocked = True
                break
        
        # 如果只有一个方向是通畅的，就在通道中
        blocked_count = sum([left_blocked, right_blocked, up_blocked, down_blocked])
        return blocked_count >= 2
    def get_learned_adjustment(self, current, goal):
        """获取从学习中得到的启发式调整值"""
        state = self.get_state_representation(current)
        
        if state in self.q_values:
            # 计算所有可能动作的平均Q值
            action_values = self.q_values[state].values()
            if action_values:
                # 正向调整 - 奖励途经Q值高的区域
                return -0.5 * (sum(action_values) / len(action_values))
        
        return 0  # 默认不调整
    
    def get_state_representation(self, position):
        """
        将位置转换为状态表示
        
        使用网格化的位置作为状态，以避免状态空间过大
        """
        # 使用较粗粒度的网格来降低状态空间大小
        grid_size = 5  # 状态网格大小
        grid_x = int(position[0] // grid_size)
        grid_y = int(position[1] // grid_size)
        
        # 修复错误: 使用字符串格式化而不是字符串拼接
        return f"{grid_x},{grid_y}"  # 使用格式化字符串
    
    def get_action_from_positions(self, current, next_pos):
        """从两个相邻位置计算对应的动作"""
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        # 标准化为单位动作
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
            
        return (dx, dy)
    
    def get_best_action(self, state, available_actions):
        """
        改进的动作选择策略
        
        基于Q值、探索-利用平衡和障碍物感知选择动作
        """
        if not available_actions:
            return None
            
        # 随机探索概率 - 随着训练次数增加而减少
        exploration_prob = max(0.05, 0.3 * math.exp(-0.02 * self.episodes_trained))
        
        # 利用阶段 - 基于Q值选择动作
        if state in self.q_values and random.random() > exploration_prob:
            # 获取可用动作的Q值
            action_values = {a: self.q_values[state].get(a, 0) for a in available_actions}
            
            if action_values:
                # 找出最大Q值
                max_q = max(action_values.values())
                # 找出所有具有最大Q值的动作
                best_actions = [a for a, q in action_values.items() if q == max_q]
                
                # 如果有多个最佳动作，进一步评估
                if len(best_actions) > 1:
                    # 考虑障碍物因素，优先选择避开障碍物的动作
                    safer_actions = []
                    for action in best_actions:
                        dx, dy = action
                        # 状态是字符串格式，需要解析出坐标
                        try:
                            x, y = map(int, state.split(','))
                            next_pos = (x*5 + dx, y*5 + dy)  # 转回实际坐标 (乘以网格大小)
                            
                            # 计算周围障碍物数量
                            obstacles = self._count_nearby_obstacles(next_pos, radius=3)
                            if obstacles < 2:  # 少于2个障碍物的动作优先考虑
                                safer_actions.append(action)
                        except:
                            # 解析失败则保留原动作
                            pass
                    
                    # 如果找到更安全的动作，从中选择，否则使用所有最佳动作
                    return random.choice(safer_actions if safer_actions else best_actions)
                
                # 只有一个最佳动作
                return best_actions[0]
        
        # 探索阶段 - 随机选择动作，但优先考虑安全路径
        # 将动作分为更安全和风险更高的两组
        safe_actions = []
        risky_actions = []
        
        for action in available_actions:
            dx, dy = action
            try:
                # 解析状态字符串获取坐标
                x, y = map(int, state.split(','))
                next_pos = (x*5 + dx, y*5 + dy)  # 转回实际坐标
                
                # 检查该位置周围的障碍物数量
                obstacles = self._count_nearby_obstacles(next_pos, radius=2)
                
                # 根据障碍物数量将动作分类
                if obstacles < 3:
                    safe_actions.append(action)
                else:
                    risky_actions.append(action)
            except:
                # 解析失败则视为普通动作
                risky_actions.append(action)
        
        # 优先从安全动作中选择，如果没有则从风险动作中选择
        if safe_actions:
            return random.choice(safe_actions)
        
        return random.choice(risky_actions) if risky_actions else random.choice(available_actions)
    
    def update_q_values(self, state, action, reward, next_state, done):
        """
        更新Q值表
        
        使用Q-learning更新公式: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        """
        # 确保状态存在于Q值表中
        if state not in self.q_values:
            self.q_values[state] = {}
        
        # 获取当前Q值
        current_q = self.q_values[state].get(action, 0)
        
        # 计算目标Q值
        if done:
            target_q = reward  # 终态
        else:
            # 获取下一个状态的最大Q值
            if next_state in self.q_values and self.q_values[next_state]:
                max_next_q = max(self.q_values[next_state].values())
            else:
                max_next_q = 0
                
            target_q = reward + self.discount_factor * max_next_q
        
        # 更新Q值
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self.q_values[state][action] = new_q
        
        # 将经验添加到缓冲区
        self.experience_buffer.append((state, action, reward, next_state, done))
        
        # 限制缓冲区大小
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)
    
    def replay_experiences(self, batch_size=32):
        """从经验缓冲区中随机采样并更新Q值，以提高学习效率"""
        if len(self.experience_buffer) < batch_size:
            return
            
        # 随机采样经验
        batch = random.sample(self.experience_buffer, batch_size)
        
        for state, action, reward, next_state, done in batch:
            # 应用与update_q_values相同的更新逻辑
            if state not in self.q_values:
                self.q_values[state] = {}
            
            current_q = self.q_values[state].get(action, 0)
            
            if done:
                target_q = reward
            else:
                if next_state in self.q_values and self.q_values[next_state]:
                    max_next_q = max(self.q_values[next_state].values())
                else:
                    max_next_q = 0
                    
                target_q = reward + self.discount_factor * max_next_q
            
            new_q = current_q + self.learning_rate * (target_q - current_q)
            self.q_values[state][action] = new_q
    
    def pathfind(self, start, goal, max_iterations=MAX_ITERATIONS):
        """
        使用强化A*算法进行路径规划
        
        Args:
            start: 起点坐标
            goal: 目标坐标
            max_iterations: 最大迭代次数
                
        Returns:
            List[Tuple[float, float]]: 找到的路径
        """
        logging.debug(f"开始路径规划: 从 {start} 到 {goal}")
        logging.debug(f"障碍物数量: {len(self.obstacle_grids)}")
        
        episode_start_time = time.time()
        
        # 初始化开放集和闭合集
        open_set = []
        open_set_hash = set()
        closed_set = set()
        
        # 添加起始节点到开放集
        start_f = self.get_heuristic(start, goal)
        heapq.heappush(open_set, (start_f, 0, start))
        open_set_hash.add(start)
        
        # 路径追踪和成本记录
        came_from = {}
        g_score = {start: 0}
        f_score = {start: start_f}
        
        # 当前路径的奖励总和(用于学习)
        total_reward = 0
        
        # 记录遍历的路径(用于学习)
        path_states = []
        path_actions = []
        
        # 主循环
        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            
            # 获取f值最低的节点
            current_f, current_g, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            # 记录路径上的状态
            current_state = self.get_state_representation(current)
            path_states.append(current_state)
            
            # 周期性调试输出
            if iterations % 1000 == 0:
                logging.debug(f"A*迭代: {iterations}, 位置: {current}, 距离终点: {math.dist(current, goal)}")
            
            # 检查是否到达目标
            if self._close_enough(current, goal):
                # 重建路径
                path = self._reconstruct_path(came_from, current)
                
                # 如果没有找到终点，添加它
                if path[-1] != goal:
                    path.append(goal)
                
                # 更新统计信息
                self.successful_paths += 1
                self.last_path_length = len(path)
                self.last_training_time = time.time() - episode_start_time
                
                # 学习过程 - 为路径上的每个状态-动作对更新Q值
                self._learn_from_path(path_states, path_actions, path)
                
                logging.info(f"路径规划成功! 迭代次数: {iterations}, 路径长度: {len(path)}")
                return path
            
            # 添加到闭合集
            closed_set.add(current)
            
            # 获取可能的动作
            available_actions = []
            for action in self.actions:
                dx, dy = action
                new_x, new_y = current[0] + dx, current[1] + dy
                new_pos = (new_x, new_y)
                
                # 验证新位置
                if not self._is_valid_position(new_pos):
                    continue
                
                if new_pos in closed_set:
                    continue
                
                # 检查对角线移动时的拐角可通行性
                if dx != 0 and dy != 0:
                    if (self._is_obstacle_fast((current[0], current[1] + dy)) or 
                        self._is_obstacle_fast((current[0] + dx, current[1]))):
                        continue
                
                available_actions.append(action)
            
            # 使用学习的策略选择最佳动作
            if available_actions:
                # 获取最佳动作
                best_action = self.get_best_action(current_state, available_actions)
                
                if best_action:
                    # 记录选择的动作
                    path_actions.append(best_action)
                    
                    # 应用动作
                    dx, dy = best_action
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # 计算移动成本
                    move_cost = self.move_costs.get(best_action, 1.0)
                    
                    # 根据地形和障碍物调整成本
                    adjusted_cost = self._calculate_adjusted_cost(current, neighbor, best_action)
                    
                    # 计算新的g值
                    tentative_g = g_score[current] + adjusted_cost
                    
                    # 检查是否找到更好的路径
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.get_heuristic(neighbor, goal)
                        
                        # 计算即时奖励
                        reward = self._calculate_reward(current, neighbor, goal)
                        total_reward += reward
                        
                        # 更新Q值 - 使用当前状态、动作、奖励和下一个状态
                        next_state = self.get_state_representation(neighbor)
                        done = self._close_enough(neighbor, goal)
                        self.update_q_values(current_state, best_action, reward, next_state, done)
                        
                        # 如果节点不在开放集中，添加它
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
            
        # 如果没有找到路径，尝试学习失败原因
        if iterations >= max_iterations:
            logging.warning(f"强化A*搜索达到最大迭代次数 ({max_iterations})，搜索失败")
        else:
            logging.warning("强化A*搜索失败，开放集为空")
        
        # 尝试从失败中学习
        self._learn_from_failure(path_states, path_actions, goal)
        
        # 更新统计
        self.episodes_trained += 1
        
        # 从经验回放中学习
        self.replay_experiences()
        
        return None
    
    def _learn_from_path(self, states, actions, path):
        """从成功的路径中学习"""
        self.episodes_trained += 1
        
        path_length = len(path)
        
        # 为路径的每个部分计算折扣奖励
        for i in range(len(states) - 1):
            if i >= len(actions):
                break
                
            state = states[i]
            action = actions[i]
            
            # 计算位置到终点的进展
            if i < len(path) - 1:
                # 标准化奖励（接近于0表示到达目标）
                progress_reward = 100.0 * (1.0 - (len(path) - i) / path_length)
                
                # 更新Q值
                if state not in self.q_values:
                    self.q_values[state] = {}
                
                # 增强有效路径上的Q值
                current_q = self.q_values[state].get(action, 0)
                self.q_values[state][action] = current_q + self.learning_rate * progress_reward
        
        # 从经验回放中学习
        self.replay_experiences()
    
    def _learn_from_failure(self, states, actions, goal):
        """从失败的路径中学习"""
        if not states or not actions:
            return
            
        # 对失败路径上的状态-动作对进行负强化
        for i in range(min(len(states) - 1, len(actions))):
            state = states[i]
            action = actions[i]
            
            # 更新Q值为负值，以减少未来选择该动作的概率
            if state not in self.q_values:
                self.q_values[state] = {}
                
            current_q = self.q_values[state].get(action, 0)
            # 对失败路径的惩罚
            self.q_values[state][action] = current_q - self.learning_rate * 10
    
    def _calculate_reward(self, current, next_pos, goal):
        """大幅增强版奖励函数"""
        # 基础奖励
        reward = 1.0
        
        # 接近目标的奖励
        current_dist = math.dist(current, goal)
        next_dist = math.dist(next_pos, goal)
        progress_reward = current_dist - next_dist
        reward += progress_reward * 5.0  # 大幅增加目标吸引力
        
        # 严重惩罚接近障碍物
        obstacles_nearby = self._count_nearby_obstacles(next_pos, radius=2)
        obstacle_penalty = -2.0 * obstacles_nearby  # 大幅增加障碍物惩罚
        reward += obstacle_penalty
        
        # 奖励通道中的移动
        if self._is_in_corridor(next_pos):
            reward += 2.0  # 奖励在通道中的移动
        
        # 到达目标的奖励
        if self._close_enough(next_pos, goal):
            reward += 200.0  # 大幅增加目标奖励
            
        return reward
    
    def _calculate_adjusted_cost(self, current, neighbor, action):
        """改进的成本计算 - 对障碍物更敏感"""
        # 基础移动成本
        base_cost = self.move_costs.get(action, 1.0)
        
        # 障碍物接近度成本 - 加强避障反应
        obstacles_nearby = self._count_nearby_obstacles(neighbor, radius=3)
        # 使用指数函数增加障碍物对成本的影响
        obstacle_cost = 1.0 + 0.3 * obstacles_nearby**1.5
        
        # 检查是否在通道中 - 减少在障碍物之间的狭窄通道中的移动成本
        is_corridor = False
        obstacles_in_range = 0
        narrow_passage_reward = 0
        
        # 检查是否处于狭窄通道
        if obstacles_nearby > 0:
            # 计算8个方向上障碍物的分布
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
            
            direction_has_obstacle = []
            for dx, dy in directions:
                check_pos = (neighbor[0] + dx*2, neighbor[1] + dy*2)
                has_obstacle = 0 <= check_pos[0] < self.map_size and \
                            0 <= check_pos[1] < self.map_size and \
                            self._is_obstacle_fast(check_pos)
                direction_has_obstacle.append(has_obstacle)
                obstacles_in_range += 1 if has_obstacle else 0
            
            # 如果水平或垂直方向上有障碍物，但移动方向是通畅的，降低成本
            if obstacles_in_range >= 2 and \
            ((direction_has_obstacle[0] or direction_has_obstacle[1]) and 
                (direction_has_obstacle[2] or direction_has_obstacle[3])):
                is_corridor = True
                narrow_passage_reward = 0.2  # 鼓励探索通道
        
        # 使用Q值影响成本 - 让算法更多依赖学习到的经验
        q_factor = 1.0
        current_state = self.get_state_representation(current)
        
        if current_state in self.q_values and action in self.q_values[current_state]:
            q_value = self.q_values[current_state][action]
            if q_value > 0:
                # 正Q值大幅降低成本
                q_factor = max(0.4, 1.0 - 0.15 * q_value)  # 更激进的成本降低
            elif q_value < 0:
                # 负Q值大幅增加成本
                q_factor = min(3.0, 1.0 - 0.15 * q_value)  # 更激进的成本增加
        
        # 总成本计算
        total_cost = base_cost * obstacle_cost * q_factor
        
        # 应用通道奖励
        if is_corridor:
            total_cost = max(0.5, total_cost - narrow_passage_reward)
            
        return total_cost
    
    def _reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
            
        return path[::-1]  # 逆序返回路径
    
    def _close_enough(self, point1, point2, threshold=3.0):
        """检查两点是否足够接近"""
        return math.dist(point1, point2) <= threshold
    
    def _is_valid_position(self, pos):
        """检查位置是否有效（在地图范围内且不是障碍物）"""
        x, y = pos
        # 检查边界
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return False
            
        # 检查是否为障碍物
        if self._is_obstacle_fast(pos):
            return False
            
        return True
    
    def _is_obstacle_fast(self, point):
        """强化版快速障碍物检测"""
        # 转换为整数坐标
        x, y = int(round(point[0])), int(round(point[1]))

        # 直接检查点是否在障碍物集合中
        if (x, y) in self.obstacle_grids:
            return True
            
        # 多检查一下周围的点，扩大障碍物的"影响范围"
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in self.obstacle_grids:
                    return True
                    
        return False

    def _count_nearby_obstacles(self, point, radius=3):
        """计算点附近障碍物数量"""
        x, y = point
        count = 0
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = int(x + dx), int(y + dy)
                check_point = (nx, ny)
                
                # 确保点在地图范围内
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if self._is_obstacle_fast(check_point):
                        count += 1
        
        return count

class HybridPathPlanner:
    """
    高性能混合型路径规划器 - 强化学习版
    
    使用强化A*算法进行路径规划，能够从经验中学习，不断改进路径质量。
    特别适用于复杂迷宫环境，提供高效的路径规划能力。
    """
    
    def __init__(self, map_service: MapService):
        """
        初始化路径规划器
        
        Args:
            map_service: 地图服务对象，提供地形和障碍物信息
        """
        # 基础组件
        self.map_service = map_service
        self.dispatch = None  # 由DispatchSystem设置
        
        # 性能监控
        self.load_time = time.time()
        self.planning_count = 0
        self.total_planning_time = 0
        self.success_count = 0
        self.failure_count = 0
        
        # 地图尺寸 - 确保有这个属性
        self.map_size = getattr(map_service, 'grid_size', 200)
        
        # 空间数据
        self.obstacle_index = SpatialIndex(cell_size=20)  # 障碍物空间索引
        self.obstacle_grids = set()                       # 障碍点集合
        
        # 预约系统
        self.reservation_table = {}                      # 路径段预约表
        self.reservation_lock = threading.RLock()        # 预约表锁
        
        # 缓存系统
        self.path_cache = PathCache(max_size=CACHE_SIZE, expiry=CACHE_EXPIRY)
        
        # 强化学习路径规划器
        self.reinforced_astar = ReinforcedAStar(self)
        
        # 加载地图配置
        self._load_map_config()
            
        # 方向数组用于搜索 (8个方向)
        self.directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),   # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
        
        # 预计算的移动成本
        self.move_costs = {
            (0, 1): 1.0, (1, 0): 1.0, (0, -1): 1.0, (-1, 0): 1.0,  # 直线移动成本
            (1, 1): 1.414, (1, -1): 1.414, (-1, 1): 1.414, (-1, -1): 1.414  # 对角线移动成本
        }
        
        # 初始化空间索引
        self._init_spatial_index()
        
        logging.info("强化学习路径规划器初始化完成")
            
    def _load_map_config(self):
        """加载地图配置"""
        try:
            # 尝试读取配置文件
            self.grid_size = 20.0  # 默认网格大小
            self.max_grade = 15.0  # 最大坡度
            self.min_turn_radius = 15.0  # 最小转弯半径
            
            # 如果MapService提供了配置，优先使用
            if hasattr(self.map_service, 'config'):
                config = self.map_service.config
                if hasattr(config, 'grid_size'):
                    self.grid_size = float(config.grid_size)
                if hasattr(config, 'max_grade'):
                    self.max_grade = float(config.max_grade)
                if hasattr(config, 'min_turn_radius'):
                    self.min_turn_radius = float(config.min_turn_radius)
                    
            logging.debug(f"加载地图配置: 网格大小={self.grid_size}, 最大坡度={self.max_grade}, 最小转弯半径={self.min_turn_radius}")
            
        except Exception as e:
            logging.warning(f"加载地图配置失败，使用默认值: {str(e)}")
            self.grid_size = 20.0
            self.max_grade = 15.0
            self.min_turn_radius = 15.0
      
    def plan_path(self, start, end, vehicle=None, force_replan=False):
        """
        路径规划主入口方法
        
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            vehicle: 可选的车辆对象，用于考虑车辆特性
            force_replan: 是否强制重新规划路径（忽略缓存）
            
        Returns:
            List[Tuple[float, float]]: 规划的路径点列表
        """
        # 性能计数
        start_time = time.time()
        self.planning_count += 1
        
        try:
            # 标准化输入坐标
            start = self._validate_point(start)
            end = self._validate_point(end)
            
            # 检查起点和终点是否相同
            if self._points_equal(start, end):
                return [start]
                
            # 创建缓存键
            cache_key = self._create_cache_key(start, end, vehicle)
                    
            # 检查缓存（如果不强制重新规划）
            cached_path = None
            if not force_replan:
                cached_path = self.path_cache.get(cache_key)
                
            if cached_path:
                logging.debug(f"使用缓存路径: {start} -> {end}")
                return cached_path
                
            # 使用强化A*算法
            path = self.reinforced_astar.pathfind(start, end)
            
            # 若强化A*失败，使用备选路径
            if not path or len(path) < 2:
                logging.debug(f"强化A*路径规划失败，使用备选路径: {start} -> {end}")
                path = self._generate_fallback_path(start, end)
                self.failure_count += 1
            else:
                self.success_count += 1
                
            # 路径平滑（如果点数足够）
            if len(path) > 3:
                try:
                    path = self._smooth_path(path)
                except Exception as e:
                    logging.warning(f"路径平滑失败: {str(e)}")
                    
            # 缓存结果
            self.path_cache.put(cache_key, path)
            
            # 记录性能指标
            elapsed = time.time() - start_time
            self.total_planning_time += elapsed
            
            if elapsed > 0.1:  # 记录较慢的规划
                logging.debug(f"路径规划耗时较长: {elapsed:.3f}秒 ({start} -> {end})")
                
            return path
        except Exception as e:
            logging.error(f"路径规划失败: {str(e)}")
            # 最简单的后备方案 - 直接连接起点和终点
            return [start, end]
    
    def _init_spatial_index(self):
        """初始化空间索引结构"""
        self.obstacle_grid_size = 10  # 网格尺寸
        self.obstacle_index_grid = {}
        
        # 建立网格索引
        if hasattr(self, 'obstacle_grids') and self.obstacle_grids:
            for obs_x, obs_y in self.obstacle_grids:
                # 计算网格坐标
                grid_x = obs_x // self.obstacle_grid_size
                grid_y = obs_y // self.obstacle_grid_size
                grid_key = (grid_x, grid_y)
                
                # 添加到网格
                if grid_key not in self.obstacle_index_grid:
                    self.obstacle_index_grid[grid_key] = set()
                
                self.obstacle_index_grid[grid_key].add((obs_x, obs_y))
            
            logging.info(f"空间索引初始化完成: {len(self.obstacle_index_grid)}个网格，{len(self.obstacle_grids)}个障碍点")
    
    def _bresenham_line(self, start, end):
        """
        使用Bresenham算法生成线段上的所有点
        
        Args:
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            
        Returns:
            List[Tuple[int, int]]: 线段上的所有点
        """
        x1, y1 = int(round(start[0])), int(round(start[1]))
        x2, y2 = int(round(end[0])), int(round(end[1]))
        points = []
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        
        err = dx - dy
        
        while True:
            points.append((x1, y1))
            if x1 == x2 and y1 == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
                
        return points
    
    def _smooth_path(self, path):
        """平滑路径并确保不穿过障碍物（改进版）"""
        if len(path) <= 2:
            return path
        
        # 使用道格拉斯-普克算法进行平滑
        result = self._douglas_peucker(path, 2.0)
        
        # 检查平滑后路径的每个相邻点之间是否穿过障碍物
        safe_path = [result[0]]  # 添加起点
        
        for i in range(1, len(result)):
            prev = safe_path[-1]
            curr = result[i]
            
            # 检查连线是否穿过障碍物
            line_points = self._bresenham_line(prev, curr)
            has_obstacle = False
            
            for p in line_points:
                if self._is_obstacle_fast(p):
                    has_obstacle = True
                    break
            
            if has_obstacle:
                # 如果连线穿过障碍物，找出原始路径中的中间点添加到安全路径中
                orig_idx_prev = path.index(prev)
                orig_idx_curr = path.index(curr) if curr in path else len(path) - 1
                
                # 添加原始路径中的点来避开障碍物
                for j in range(orig_idx_prev + 1, orig_idx_curr + 1):
                    if j < len(path):
                        safe_path.append(path[j])
            else:
                # 如果连线没有穿过障碍物，直接添加当前点
                safe_path.append(curr)
        
        # 确保终点添加到路径中
        if safe_path[-1] != path[-1]:
            safe_path.append(path[-1])
        
        return safe_path
    
    def _douglas_peucker(self, points, epsilon):
        """
        道格拉斯-普克算法实现
        
        用于简化路径，保留关键点
        """
        if len(points) <= 2:
            return points
            
        # 找到最远点
        dmax = 0
        index = 0
        start, end = points[0], points[-1]
        
        for i in range(1, len(points) - 1):
            d = self._perpendicular_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d
                
        if dmax > epsilon:
            # 递归处理
            rec1 = self._douglas_peucker(points[:index+1], epsilon)
            rec2 = self._douglas_peucker(points[index:], epsilon)
            
            # 合并结果，避免重复点
            return rec1[:-1] + rec2
        else:
            return [points[0], points[-1]]
            
    def _perpendicular_distance(self, point, line_start, line_end):
        """计算点到线段的垂直距离"""
        if line_start == line_end:
            return math.dist(point, line_start)
            
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 线性代数方法，计算点到直线距离
        num = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
        den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        return num / den if den > 0 else 0
    
    def _is_obstacle_fast(self, point):
        """快速检查点是否为障碍物（使用空间索引优化）"""
        # 转换为整数坐标
        x, y = int(round(point[0])), int(round(point[1]))
        
        # 使用网格索引
        if hasattr(self, 'obstacle_index_grid'):
            grid_x, grid_y = x // self.obstacle_grid_size, y // self.obstacle_grid_size
            grid_key = (grid_x, grid_y)
            
            # 检查网格中是否有障碍物
            if grid_key in self.obstacle_index_grid:
                # 精确检查点是否为障碍物
                if (x, y) in self.obstacle_index_grid[grid_key]:
                    return True
                
            return False
        
        # 回退到基本检查
        return (x, y) in self.obstacle_grids

    def _count_nearby_obstacles(self, point, radius=3):
        """计算指定半径内的障碍物数量"""
        count = 0
        x, y = point
        
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = x + dx, y + dy
                if self._is_obstacle_fast((nx, ny)):
                    count += 1
        
        return count

    def _is_obstacle(self, point):
        """检查点是否为障碍物（改进版）"""
        # 转换为整数坐标（因为障碍物通常以整数坐标存储）
        x, y = int(round(point[0])), int(round(point[1]))
        int_point = (x, y)
        
        # 基本检查
        if int_point in self.obstacle_grids:
            return True
        
        # 使用空间索引进行检查
        if hasattr(self, 'obstacle_index') and self.obstacle_index.grid:
            nearby_points = self.obstacle_index.query_point(int_point, radius=1)  # 检查周围1个单位的点
            if nearby_points:
                return True
        
        # 使用地图服务进行检查
        try:
            if hasattr(self.map_service, 'is_obstacle') and callable(getattr(self.map_service, 'is_obstacle')):
                return self.map_service.is_obstacle(point)
        except Exception as e:
            logging.debug(f"地图服务障碍物检查出错: {str(e)}")
        
        return False
    
    def _generate_fallback_path(self, start, end):
        """
        生成备选路径，确保避开障碍物
        """
        try:
            # 标准化坐标
            start = self._validate_point(start)
            end = self._validate_point(end)
                
            # 直线距离和方向
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # 检查直线路径上的障碍物
            line_points = self._bresenham_line(start, end)
            has_obstacles = False
            
            for point in line_points:
                if self._is_obstacle_fast(point):
                    has_obstacles = True
                    break
            
            # 如果直线路径无障碍，直接返回
            if not has_obstacles:
                return [start, end]
            
            # 有障碍物，尝试更复杂的绕行路径
            
            # 尝试找到多个中间点绕行
            path = [start]
            
            # 尝试几种不同的绕行方案
            detour_attempts = [
                # 尝试上方绕行
                (start[0] + dx/3, start[1] + abs(dx)/2),
                (start[0] + 2*dx/3, start[1] + abs(dx)/2),
                
                # 尝试下方绕行
                (start[0] + dx/3, start[1] - abs(dx)/2),
                (start[0] + 2*dx/3, start[1] - abs(dx)/2),
                
                # 尝试左侧绕行
                (start[0] - abs(dy)/2, start[1] + dy/3),
                (start[0] - abs(dy)/2, start[1] + 2*dy/3),
                
                # 尝试右侧绕行
                (start[0] + abs(dy)/2, start[1] + dy/3),
                (start[0] + abs(dy)/2, start[1] + 2*dy/3),
            ]
            
            # 尝试各种绕行路径
            for i in range(0, len(detour_attempts), 2):
                mid1 = detour_attempts[i]
                mid2 = detour_attempts[i+1]
                
                # 验证中间点不在障碍物上
                if not self._is_obstacle_fast(mid1) and not self._is_obstacle_fast(mid2):
                    # 验证连接线不穿过障碍物
                    line1 = self._bresenham_line(start, mid1)
                    line2 = self._bresenham_line(mid1, mid2)
                    line3 = self._bresenham_line(mid2, end)
                    
                    if (not any(self._is_obstacle_fast(p) for p in line1) and
                        not any(self._is_obstacle_fast(p) for p in line2) and
                        not any(self._is_obstacle_fast(p) for p in line3)):
                        # 找到有效路径
                        return [start, mid1, mid2, end]
            
            # 所有尝试都失败，使用网格搜索寻找简单路径
            # 这是最后的备选方案，不会太高效但确保能找到路径
            grid_size = 20
            best_path = None
            min_length = float('inf')
            
            # 生成一组网格点
            for dx in range(-5, 6, 2):
                for dy in range(-5, 6, 2):
                    # 在起点和终点间寻找一个中点
                    mid_point = (
                        (start[0] + end[0]) / 2 + dx * grid_size,
                        (start[1] + end[1]) / 2 + dy * grid_size
                    )
                    
                    # 检查中点是否在障碍物上
                    if self._is_obstacle_fast(mid_point):
                        continue
                        
                    # 检查从起点到中点的路径
                    line1 = self._bresenham_line(start, mid_point)
                    if any(self._is_obstacle_fast(p) for p in line1):
                        continue
                        
                    # 检查从中点到终点的路径
                    line2 = self._bresenham_line(mid_point, end)
                    if any(self._is_obstacle_fast(p) for p in line2):
                        continue
                    
                    # 找到一条可行路径
                    path_length = len(line1) + len(line2)
                    if path_length < min_length:
                        min_length = path_length
                        best_path = [start, mid_point, end]
            
            # 如果找到了可行路径，返回
            if best_path:
                return best_path
                
            # 最终方案：大范围绕行
            perimeter_size = max(abs(dx), abs(dy)) * 2
            corners = [
                (start[0] - perimeter_size, start[1] - perimeter_size),
                (start[0] - perimeter_size, end[1] + perimeter_size),
                (end[0] + perimeter_size, end[1] + perimeter_size),
                (end[0] + perimeter_size, start[1] - perimeter_size)
            ]
            
            # 找到至少两个不在障碍物上的角点
            valid_corners = [c for c in corners if not self._is_obstacle_fast(c)]
            
            if len(valid_corners) >= 2:
                # 选择第一个和最后一个有效角点
                return [start, valid_corners[0], valid_corners[-1], end]
                
            # 所有尝试都失败，只能返回直线路径
            logging.warning(f"无法找到避开障碍物的路径 {start} -> {end}，返回直线路径")
            return [start, end]
            
        except Exception as e:
            logging.error(f"备选路径生成失败: {str(e)}")
            # 最简单的后备方案 - 直接连接起点和终点
            return [start, end]

    def _create_cache_key(self, start, end, vehicle):
        """创建缓存键"""
        if vehicle:
            # 包含车辆属性的缓存键
            return (
                "path",
                start,
                end,
                getattr(vehicle, 'turning_radius', 0), 
                getattr(vehicle, 'min_hardness', 0),
                getattr(vehicle, 'current_load', 0)
            )
        else:
            # 基本缓存键
            return ("base_path", start, end)
            
    def _validate_point(self, point):
        """验证并标准化坐标点"""
        if isinstance(point, tuple) and len(point) >= 2:
            return (float(point[0]), float(point[1]))
        elif hasattr(point, 'as_tuple'):
            return point.as_tuple()
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            return (float(point.x), float(point.y))
        elif isinstance(point, (list, np.ndarray)) and len(point) >= 2:
            return (float(point[0]), float(point[1]))
        else:
            # 无效点警告
            logging.warning(f"无效的坐标点: {point}，使用(0,0)")
            return (0.0, 0.0)
    
    def _points_equal(self, p1, p2, tolerance=EPSILON):
        """检查两点是否相等(考虑浮点误差)"""
        return (abs(p1[0] - p2[0]) < tolerance and 
                abs(p1[1] - p2[1]) < tolerance)
                
    def plan_with_timeout(self, start, end, vehicle=None, timeout=DEFAULT_TIMEOUT):
        """
        带超时控制的路径规划方法
        
        Args:
            start: 起点坐标
            end: 终点坐标
            vehicle: 可选的车辆对象
            timeout: 超时时间(秒)
            
        Returns:
            Tuple[List[Tuple], float]: (路径, 耗时)
        """
        import queue
        import threading
        
        # 使用队列安全地传递结果
        result_queue = queue.Queue()
        
        def planning_worker():
            try:
                # 执行路径规划
                path_result = self.plan_path(start, end, vehicle)
                result_queue.put(path_result)
            except Exception as e:
                # 捕获异常并放入队列
                logging.error(f"路径规划线程出错: {str(e)}")
                result_queue.put(Exception(f"路径规划出错: {str(e)}"))
        
        # 启动规划线程
        planning_thread = threading.Thread(target=planning_worker, daemon=True)
        start_time = time.time()
        planning_thread.start()
        
        # 等待线程完成或超时
        planning_thread.join(timeout)
        
        # 计算耗时
        elapsed = time.time() - start_time
        
        # 获取结果
        if planning_thread.is_alive():
            # 超时情况，生成备用路径
            logging.warning(f"路径规划超时 ({timeout}秒)，生成备用路径")
            path = self._generate_fallback_path(start, end)
            return path, elapsed
        else:
            try:
                # 检查队列中的结果
                result = result_queue.get(block=False)
                if isinstance(result, Exception):
                    # 出错，生成备用路径
                    logging.warning(f"路径规划失败: {str(result)}，生成备用路径")
                    path = self._generate_fallback_path(start, end)
                    return path, elapsed
                elif not result or len(result) < 2:
                    # 无效结果，生成备用路径
                    logging.warning(f"无效路径，生成备用路径")
                    path = self._generate_fallback_path(start, end)
                    return path, elapsed
                else:
                    # 规划成功
                    logging.info(f"路径规划成功！路径包含 {len(result)} 个点，用时 {elapsed:.3f} 秒")
                    return result, elapsed
            except queue.Empty:
                # 队列为空，生成备用路径
                logging.warning(f"路径规划过程异常终止，生成备用路径")
                path = self._generate_fallback_path(start, end)
                return path, elapsed
    
    def clear_caches(self):
        """清除所有缓存数据"""
        self.path_cache.clear()
        
    def get_performance_stats(self):
        """
        获取性能统计
        
        Returns:
            Dict: 性能统计信息
        """
        runtime = time.time() - self.load_time
        avg_planning_time = self.total_planning_time / self.planning_count if self.planning_count > 0 else 0
        success_rate = self.success_count / self.planning_count if self.planning_count > 0 else 0
        
        # 获取强化学习相关统计
        rl_stats = {
            'episodes_trained': self.reinforced_astar.episodes_trained,
            'successful_paths': self.reinforced_astar.successful_paths,
            'last_path_length': self.reinforced_astar.last_path_length,
            'last_training_time': f"{self.reinforced_astar.last_training_time*1000:.2f}毫秒",
            'q_values_states': len(self.reinforced_astar.q_values),
            'experience_buffer_size': len(self.reinforced_astar.experience_buffer)
        }
        
        return {
            'runtime': f"{runtime:.1f}秒",
            'planning_count': self.planning_count,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': f"{success_rate*100:.1f}%",
            'avg_planning_time': f"{avg_planning_time*1000:.2f}毫秒",
            'cache_stats': self.path_cache.get_stats(),
            'obstacle_count': len(self.obstacle_grids),
            'reservations': len(self.reservation_table),
            'reinforced_learning': rl_stats
        }
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    
    # 设置日志级别
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("===== 强化A*算法 - 迷宫场景测试 =====")
    
    # 创建一个模拟的MapService
    class MockMapService:
        def __init__(self):
            self.grid_size = 200
            self.obstacle_grids = set()
            
        def is_obstacle(self, point):
            x, y = int(round(point[0])), int(round(point[1]))
            return (x, y) in self.obstacle_grids
    
    # 创建迷宫型障碍物
    def create_maze(map_size=200, wall_width=3):
        """创建简化版迷宫式障碍物，复杂度降低约30%"""
        obstacles = set()
        
        # 边界墙
        for i in range(map_size):
            obstacles.add((0, i))
            obstacles.add((map_size-1, i))
            obstacles.add((i, 0))
            obstacles.add((i, map_size-1))
        
        # 水平墙 - 减少长度和数量
        h_walls = [
            (20, 40, 100, 40),    # 缩短第一个水平墙
            (90, 80, 160, 80),    # 缩短第二个水平墙
            (20, 120, 100, 120),  # 缩短第三个水平墙
            (90, 160, 160, 160)   # 保持第四个水平墙
        ]
        
        # 垂直墙 - 减少长度和数量
        v_walls = [
            (40, 20, 40, 80),     # 缩短第一个垂直墙
            (80, 40, 80, 120),    # 保持第二个垂直墙
            (120, 20, 120, 80),   # 缩短第三个垂直墙
            (160, 90, 160, 160)   # 缩短第四个垂直墙
        ]
        
        # 创建墙壁 - 减小墙壁宽度
        for x1, y1, x2, y2 in h_walls:
            for x in range(x1, x2+1):
                for y in range(y1-wall_width//2, y1+wall_width//2+1):
                    obstacles.add((x, y))
        
        for x1, y1, x2, y2 in v_walls:
            for y in range(y1, y2+1):
                for x in range(x1-wall_width//2, x1+wall_width//2+1):
                    obstacles.add((x, y))
                    
        # 添加随机障碍 - 减少数量约30%
        import random
        for _ in range(70):  # 从100减少到70
            x = random.randint(1, map_size-2)
            y = random.randint(1, map_size-2)
            
            # 创建小型随机障碍群
            size = random.randint(2, 4)  # 从2-5减少到2-4
            for dx in range(-size, size+1):
                for dy in range(-size, size+1):
                    nx, ny = x + dx, y + dy
                    if 0 < nx < map_size-1 and 0 < ny < map_size-1:
                        # 增大避开起点和终点区域的范围
                        if not ((nx < 35 and ny < 35) or (nx > map_size-35 and ny > map_size-35)):
                            obstacles.add((nx, ny))
        
        # 确保关键通道畅通
        # 识别并清除可能导致死锁的障碍物
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
                # 清除点周围的障碍物，创建更宽的通道
                for dx in range(-3, 4):  # 通道宽度增加到7
                    for dy in range(-3, 4):
                        nx, ny = x + dx, y + dy
                        if 0 < nx < map_size-1 and 0 < ny < map_size-1:
                            if (nx, ny) in obstacles:
                                obstacles.remove((nx, ny))
                                
        return obstacles
    
    # 设置地图尺寸和起点终点
    map_size = 200
    start_point = (10, 10)
    end_point = (190, 190)
    
    # 创建地图和障碍物
    map_service = MockMapService()
    map_service.obstacle_grids = create_maze(map_size)
    
    # 创建规划器
    planner = HybridPathPlanner(map_service)
    planner.map_size = map_size
    planner.obstacle_grids = map_service.obstacle_grids
    
    print(f"地图尺寸: {map_size}x{map_size}")
    print(f"障碍物数量: {len(planner.obstacle_grids)}")
    print(f"起点: {start_point}, 终点: {end_point}")
    
    # 执行多次路径规划，观察学习效果
    num_iterations = 50
    paths = []
    times = []
    
    for i in range(num_iterations):
        print(f"\n第 {i+1}/{num_iterations} 次路径规划:")
        
        start_time = time.time()
        path = planner.plan_path(start_point, end_point, force_replan=True)
        elapsed = time.time() - start_time
        
       
        
    # 清除缓存，确保每次都重新规划
        planner.path_cache.clear()
        
        # 添加微小的随机扰动，确保每次规划都有不同路径
        slight_offset = random.randint(-2, 2)
        modified_start = (start_point[0] + slight_offset, start_point[1] + slight_offset)
        
        start_time = time.time()
        path = planner.plan_path(modified_start, end_point)
        elapsed = time.time() - start_time
        
        paths.append(path)
        times.append(elapsed)
        
        print(f"路径长度: {len(path)}")
        print(f"规划耗时: {elapsed*1000:.2f}毫秒")
        
        # 输出强化学习统计信息
        rl_stats = planner.reinforced_astar
        print(f"已训练轮次: {rl_stats.episodes_trained}")
        print(f"成功路径数: {rl_stats.successful_paths}")
        print(f"Q值状态数: {len(rl_stats.q_values)}")
        print(f"经验缓冲区大小: {len(rl_stats.experience_buffer)}")
    
    # 可视化结果
    plt.figure(figsize=(12, 10))
    
    # 绘制障碍物
    obstacle_x = [p[0] for p in planner.obstacle_grids]
    obstacle_y = [p[1] for p in planner.obstacle_grids]
    plt.scatter(obstacle_x, obstacle_y, c='gray', marker='s', s=10, alpha=0.6, label='障碍物')
    
    # 绘制起点和终点
    plt.scatter(start_point[0], start_point[1], c='green', marker='o', s=150, label='起点')
    plt.scatter(end_point[0], end_point[1], c='red', marker='o', s=150, label='终点')
    
    # 绘制路径
    colors = ['blue', 'cyan', 'magenta', 'yellow', 'lime']
    for i, path in enumerate(paths):
        if path:
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            plt.plot(x_coords, y_coords, c=colors[i % len(colors)], linewidth=2, 
                    label=f'第{i+1}次规划 ({len(path)}点, {times[i]*1000:.0f}ms)')
    
    plt.title("强化A* - 迷宫场景路径规划测试")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 绘制第二个图：性能比较
    plt.figure(figsize=(10, 6))
    
    # 路径长度比较
    plt.subplot(1, 2, 1)
    path_lengths = [len(p) for p in paths]
    plt.plot(range(1, len(path_lengths)+1), path_lengths, 'o-', c='blue')
    plt.title("路径长度随学习进展变化")
    plt.xlabel("迭代次数")
    plt.ylabel("路径点数")
    plt.grid(True, alpha=0.3)
    
    # 规划时间比较
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(times)+1), [t*1000 for t in times], 'o-', c='red')
    plt.title("规划时间随学习进展变化")
    plt.xlabel("迭代次数")
    plt.ylabel("耗时(毫秒)")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n测试完成！")
    print(f"最终性能统计: {planner.get_performance_stats()}")