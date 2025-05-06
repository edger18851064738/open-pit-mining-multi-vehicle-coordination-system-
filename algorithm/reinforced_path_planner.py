"""
强化学习A*算法路径规划器
为露天矿多车协同调度系统提供高效路径规划能力
"""

import sys
import os
import math
import time
import heapq
import logging
import threading
import random
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError

# 常量定义
MAX_ITERATIONS = 10000      # 强化A*搜索的最大迭代次数
DEFAULT_TIMEOUT = 5.0       # 默认超时时间(秒)
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

class PathCache:
    """高性能路径缓存系统，用于存储和快速访问已计算的路径"""
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
    """强化A*算法实现，提供智能路径规划功能"""
    
    def __init__(self, obstacle_grids=None, map_size=200, learning_rate=DEFAULT_LEARNING_RATE, discount_factor=DEFAULT_DISCOUNT_FACTOR):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = {}  # 存储状态-动作对应的Q值
        self.experience_buffer = []  # 经验回放缓冲区
        self.max_buffer_size = 500  # 最大缓冲区大小
        self.collision_history = {}  # 记录碰撞历史
        
        # 地图相关属性
        self.map_size = map_size
        self.obstacle_grids = set() if obstacle_grids is None else set(obstacle_grids)
        
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
        self.metrics = {
            'consecutive_collisions': 0,
            'total_collisions': 0,
            'successful_navigations': 0
        }

    def get_heuristic(self, current, goal):
        """启发式函数 - 欧几里得距离加障碍物因子"""
        dx, dy = abs(current[0] - goal[0]), abs(current[1] - goal[1])
        
        # 欧几里得距离
        d_euclidean = math.sqrt(dx*dx + dy*dy)
        
        # 计算障碍物影响因子
        obstacles_nearby = self.count_nearby_obstacles(current, radius=5)
        obstacle_factor = 1.0 + (obstacles_nearby * 0.5)
        
        return d_euclidean * obstacle_factor

    def get_learned_adjustment(self, current, goal):
        """获取从学习中得到的启发式调整值"""
        state = self.get_state_representation(current)
        
        if state in self.q_values:
            # 计算所有可能动作的平均Q值
            action_values = self.q_values[state].values()
            if action_values:
                # 正向调整 - 奖励途经Q值高的区域
                return -1.0 * (sum(action_values) / len(action_values))
        
        return 0  # 默认不调整
    
    def get_state_representation(self, position):
        """将位置转换为状态表示"""
        # 使用网格化的位置作为状态，以避免状态空间过大
        grid_size = 3  # 状态网格大小
        grid_x = int(position[0] // grid_size)
        grid_y = int(position[1] // grid_size)
        
        return f"{grid_x},{grid_y}"
    
    def get_best_action(self, state, available_actions):
        """动作选择策略"""
        if not available_actions:
            return None
            
        # 随机探索概率 - 随着训练次数增加而减少
        exploration_prob = max(0.05, 0.3 * math.exp(-0.01 * self.episodes_trained))
        
        # 如果碰到障碍物太多，临时增加探索概率
        if self.metrics['consecutive_collisions'] > 3:
            exploration_prob += 0.2
            self.metrics['consecutive_collisions'] = 0  # 重置
        
        # 利用阶段 - 基于Q值选择动作
        if state in self.q_values and random.random() > exploration_prob:
            # 获取可用动作的Q值
            action_values = {a: self.q_values[state].get(a, 0) for a in available_actions}
            
            if action_values:
                # 找出最大Q值及对应动作
                max_q = max(action_values.values())
                best_actions = [a for a, q in action_values.items() if q == max_q]
                return random.choice(best_actions)
        
        # 探索阶段 - 随机选择动作
        return random.choice(available_actions)
    
    def update_q_values(self, state, action, reward, next_state, done):
        """Q值更新"""
        if state not in self.q_values:
            self.q_values[state] = {}
        
        current_q = self.q_values[state].get(action, 0)
        
        # 计算目标Q值
        if done:
            target_q = reward
        else:
            # 获取下一个状态的最大Q值
            if next_state in self.q_values and self.q_values[next_state]:
                max_next_q = max(self.q_values[next_state].values())
            else:
                max_next_q = 0
                
            target_q = reward + self.discount_factor * max_next_q
        
        # 修改学习率根据奖励情况
        adaptive_learning_rate = self.learning_rate
        if reward < 0:  # 负面奖励时提高学习率
            adaptive_learning_rate = min(0.5, self.learning_rate * 2.0)
        
        # 更新Q值
        new_q = current_q + adaptive_learning_rate * (target_q - current_q)
        self.q_values[state][action] = new_q

    def replay_experiences(self, batch_size=16):
        """从经验缓冲区中随机采样并更新Q值"""
        if len(self.experience_buffer) < batch_size:
            return
            
        # 随机采样经验
        batch = random.sample(self.experience_buffer, batch_size)
        
        for state, action, reward, next_state, done in batch:
            self.update_q_values(state, action, reward, next_state, done)

    def pathfind(self, start, goal, max_iterations=MAX_ITERATIONS):
        """强化A*路径规划主方法"""
        logging.info(f"开始强化A*路径规划: 从 {start} 到 {goal}")
        min_dist_to_goal = float('inf')
        no_improvement_count = 0
        # 验证起点和终点不在障碍物上
        if self.is_obstacle(start):
            start = self.find_safe_point_near(start)
            if not start:
                logging.error("无法找到安全的起点")
                return None
                    
        if self.is_obstacle(goal):
            goal = self.find_safe_point_near(goal)
            if not goal:
                logging.error("无法找到安全的终点")
                return None
                
        # 检查起点终点是否有直接路径
        direct_path = self.check_direct_path(start, goal)
        if direct_path:
            logging.info(f"发现直接路径，跳过A*搜索")
            self.successful_paths += 1
            self.metrics['successful_navigations'] += 1
            return direct_path
                
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
        
        # 记录遍历的路径(用于学习)
        path_states = []
        path_actions = []
        
        # 重置本次搜索的连续碰撞计数
        self.metrics['consecutive_collisions'] = 0
        
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
            curr_dist_to_goal = math.dist(current, goal)
            if curr_dist_to_goal < min_dist_to_goal:
                min_dist_to_goal = curr_dist_to_goal
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            # 检查是否到达目标
            if self.close_enough(current, goal):
                # 重建路径
                path = self.reconstruct_path(came_from, current)
                
                # 如果没有找到终点，添加它
                if path[-1] != goal:
                    path.append(goal)
                
                # 更新统计信息
                self.successful_paths += 1
                self.metrics['successful_navigations'] += 1
                
                # 学习过程 - 为路径上的每个状态-动作对更新Q值
                self.learn_from_path(path_states, path_actions, path)
                
                logging.info(f"路径规划成功! 迭代次数: {iterations}, 路径长度: {len(path)}")
                
                # 路径优化
                if len(path) > 3:
                    path = self.post_process_path(path)
                
                return path
            
            # 添加到闭合集
            closed_set.add(current)
            
            # 获取可能的动作
            available_actions = []
            for action in self.actions:
                dx, dy = action
                new_pos = (current[0] + dx, current[1] + dy)
                
                # 验证新位置
                if not self.is_valid_position(new_pos):
                    continue
                
                if new_pos in closed_set:
                    continue
                
                # 检查对角线移动的安全性
                if dx != 0 and dy != 0:
                    if (self.is_obstacle((current[0], current[1] + dy)) or 
                        self.is_obstacle((current[0] + dx, current[1]))):
                        continue
                    
                    # 额外检查对角线路径本身
                    midpoint = ((current[0] + new_pos[0])/2, (current[1] + new_pos[1])/2)
                    if self.is_obstacle(midpoint):
                        continue
                
                # 检查是否是障碍物
                if self.is_obstacle(new_pos):
                    self.metrics['consecutive_collisions'] += 1
                    self.metrics['total_collisions'] += 1
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
                    
                    # 计算新的g值
                    tentative_g = g_score[current] + move_cost
                    
                    # 检查是否找到更好的路径或新节点
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        
                        # 计算启发式值 + 学习调整
                        h_value = self.get_heuristic(neighbor, goal)
                        learned_adjustment = self.get_learned_adjustment(neighbor, goal)
                        f_score[neighbor] = tentative_g + h_value + learned_adjustment
                        
                                        # 计算即时奖励
                        reward = self.calculate_reward(current, neighbor, goal)
                        
                        # 更新Q值
                        next_state = self.get_state_representation(neighbor)
                        done = self.close_enough(neighbor, goal)
                        self.update_q_values(current_state, best_action, reward, next_state, done)
                        
                        # 将经验添加到回放缓冲区
                        if len(self.experience_buffer) >= self.max_buffer_size:
                            self.experience_buffer.pop(0)
                        self.experience_buffer.append((current_state, best_action, reward, next_state, done))
                        
                        # 如果节点不在开放集中，添加它
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
            
            # 定期从经验回放中学习
            if iterations % 100 == 0 and len(self.experience_buffer) > 16:
                self.replay_experiences()
        
        # 如果没有找到路径
        self.learn_from_failure(path_states, path_actions, goal)
        self.episodes_trained += 1
        
        # 尝试使用备用算法
        backup_path = self.find_backup_path(start, goal)
        if backup_path:
            logging.info(f"A*搜索失败，使用备用算法找到路径，长度: {len(backup_path)}")
            return backup_path
        
        return None

    def find_safe_point_near(self, point, max_radius=10):
        """查找点附近的安全点(非障碍物)"""
        x, y = point
        
        for radius in range(1, max_radius + 1):
            # 检查圆周上的点
            num_points = max(8, int(radius * 8))
            for i in range(num_points):
                angle = 2 * math.pi * i / num_points
                nx = x + radius * math.cos(angle)
                ny = y + radius * math.sin(angle)
                
                # 确保在地图内
                if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                    if not self.is_obstacle((nx, ny)):
                        return (nx, ny)
        
        return None  # 找不到安全点

    def check_direct_path(self, start, goal):
        """检查起点和终点之间是否有直接路径"""
        # 使用Bresenham算法检查直线路径上是否有障碍物
        points = self.get_line_points(start, goal)
        
        # 检查线上所有点是否无障碍
        for point in points:
            if self.is_obstacle(point):
                return None  # 有障碍物，返回None
        
        # 如果直线路径无障碍，直接返回
        return [start, goal]

    def get_line_points(self, start, end):
        """使用Bresenham算法获取线段上的所有点"""
        points = []
        x1, y1 = int(round(start[0])), int(round(start[1]))
        x2, y2 = int(round(end[0])), int(round(end[1]))
        
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
        
    def post_process_path(self, path):
        """路径后处理 - 平滑和简化"""
        if len(path) <= 2:
            return path
        
        # 使用Douglas-Peucker算法简化路径
        simplified = self.simplify_path(path, tolerance=1.5)
        
        # 确保简化后的路径安全
        safe_path = [simplified[0]]
        
        for i in range(1, len(simplified)):
            prev = safe_path[-1]
            curr = simplified[i]
            
            # 检查线段安全性
            line_points = self.get_line_points(prev, curr)
            if all(not self.is_obstacle(p) for p in line_points):
                safe_path.append(curr)
            else:
                # 线段不安全，寻找安全替代路径
                midpoint = ((prev[0] + curr[0])/2, (prev[1] + curr[1])/2)
                if not self.is_obstacle(midpoint):
                    safe_path.append(midpoint)
                    safe_path.append(curr)
                else:
                    # 如找不到安全中点，使用原始路径的点
                    try:
                        orig_idx_prev = path.index(prev)
                        orig_idx_curr = path.index(curr)
                        
                        for j in range(orig_idx_prev + 1, orig_idx_curr + 1):
                            if j < len(path):
                                safe_path.append(path[j])
                    except ValueError:
                        # 处理找不到索引的情况
                        safe_path.append(curr)
        
        return safe_path

    def simplify_path(self, points, tolerance=1.5):
        """Douglas-Peucker算法简化路径"""
        if len(points) <= 2:
            return points
        
        # 找到偏差最大的点
        max_distance = 0
        index = 0
        end = len(points) - 1
        
        # 计算每个点到首尾连线的距离
        for i in range(1, end):
            distance = self.perpendicular_distance(points[i], points[0], points[end])
            if distance > max_distance:
                max_distance = distance
                index = i
        
        # 如果最大距离大于容差，递归简化
        if max_distance > tolerance:
            # 递归简化两部分
            first_part = self.simplify_path(points[:index+1], tolerance)
            second_part = self.simplify_path(points[index:], tolerance)
            
            # 合并结果，避免重复中间点
            return first_part[:-1] + second_part
        else:
            # 距离足够小，返回首尾两点
            return [points[0], points[end]]

    def perpendicular_distance(self, point, line_start, line_end):
        """计算点到线段的垂直距离"""
        if line_start == line_end:
            return math.dist(point, line_start)
        
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 线性代数计算点到直线距离
        num = abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1)
        den = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        
        if den == 0:
            return 0
            
        return num / den

    def find_backup_path(self, start, goal):
        """备用路径寻找算法 - 简化版BFS"""
        # 网格大小
        grid_size = 10
        start_grid = (int(start[0] // grid_size), int(start[1] // grid_size))
        goal_grid = (int(goal[0] // grid_size), int(goal[1] // grid_size))
        
        # 使用队列进行BFS
        queue = deque([(start_grid, [start])])  # (当前网格, 路径)
        visited = {start_grid}
        
        # 移动方向: 上下左右 + 对角线
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        # BFS主循环
        while queue:
            current, path = queue.popleft()
            
            # 检查是否到达目标
            if current == goal_grid:
                # 添加终点
                if path[-1] != goal:
                    path.append(goal)
                return path
            
            # 尝试所有方向
            for dx, dy in directions:
                next_grid = (current[0] + dx, current[1] + dy)
                
                # 检查是否在地图范围内
                if not (0 <= next_grid[0] < self.map_size//grid_size and 
                        0 <= next_grid[1] < self.map_size//grid_size):
                    continue
                
                # 检查是否已访问
                if next_grid in visited:
                    continue
                
                # 转换为实际坐标
                next_pos = (next_grid[0] * grid_size, next_grid[1] * grid_size)
                
                # 检查是否是障碍物
                if self.is_obstacle(next_pos):
                    continue
                
                # 添加到队列和已访问集合
                visited.add(next_grid)
                new_path = path + [next_pos]
                queue.append((next_grid, new_path))
        
        # 搜索失败 - 生成简单绕行路径
        midpoint = ((start[0] + goal[0])/2, (start[1] + goal[1])/2)
        offset = max(abs(goal[0] - start[0]), abs(goal[1] - start[1])) / 2
        
        # 尝试几个可能的中间点
        for dx, dy in [(offset, offset), (-offset, offset), (offset, -offset), (-offset, -offset)]:
            mid = (midpoint[0] + dx, midpoint[1] + dy)
            if not self.is_obstacle(mid):
                # 检查路径安全性
                if (not any(self.is_obstacle(p) for p in self.get_line_points(start, mid)) and
                    not any(self.is_obstacle(p) for p in self.get_line_points(mid, goal))):
                    return [start, mid, goal]
                    
        # 所有尝试失败，返回直接路径
        return [start, goal]
        
    def learn_from_path(self, states, actions, path):
        """从成功路径中学习"""
        self.episodes_trained += 1
        
        # 确保有足够的数据
        if len(states) <= 1 or len(actions) == 0:
            return
            
        path_length = len(path)
        
        # 为路径的每个部分计算奖励
        for i in range(len(states) - 1):
            if i >= len(actions):
                break
                
            state = states[i]
            action = actions[i]
            
            # 计算位置到终点的进展
            if i < len(path) - 1:
                # 归一化奖励
                progress_reward = 100.0 * (1.0 - (len(path) - i) / path_length)
                
                # 更新Q值
                if state not in self.q_values:
                    self.q_values[state] = {}
                
                # 增强有效路径上的Q值
                current_q = self.q_values[state].get(action, 0)
                self.q_values[state][action] = current_q + self.learning_rate * progress_reward
        
        # 从经验回放中学习
        self.replay_experiences()
    
    def learn_from_failure(self, states, actions, goal):
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
    def smooth_path(self, path, smoothing_factor=0.5):
        """使用贝塞尔曲线平滑路径"""
        if len(path) <= 2:
            return path
            
        smoothed = [path[0]]  # 保留起点
        
        # 对中间段进行平滑
        for i in range(1, len(path)-1):
            prev = path[i-1]
            current = path[i]
            next_point = path[i+1]
            
            # 计算平滑点
            smooth_x = current[0] + smoothing_factor * (0.5 * (prev[0] + next_point[0]) - current[0])
            smooth_y = current[1] + smoothing_factor * (0.5 * (prev[1] + next_point[1]) - current[1])
            
            # 确保平滑点不在障碍物上
            smooth_point = (smooth_x, smooth_y)
            if not self.is_obstacle(smooth_point):
                smoothed.append(smooth_point)
            else:
                smoothed.append(current)  # 使用原始点
        
        smoothed.append(path[-1])  # 保留终点
        return smoothed    
    def calculate_reward(self, current, next_pos, goal):
        """奖励函数"""
        # 基础奖励
        reward = 1.0
        
        # 接近目标的奖励
        current_dist = math.dist(current, goal)
        next_dist = math.dist(next_pos, goal)
        progress_reward = current_dist - next_dist
        reward += progress_reward * 5.0
        
        # 检查障碍物
        obstacles_nearby = self.count_nearby_obstacles(next_pos, radius=3)
        
        # 障碍物惩罚
        if obstacles_nearby > 0:
            obstacle_penalty = -20.0 * obstacles_nearby
            reward += obstacle_penalty
        else:
            reward += 5.0
        
        # 碰撞惩罚
        if self.is_obstacle(next_pos):
            reward -= 100.0
        
        # 到达目标的奖励
        if self.close_enough(next_pos, goal):
            reward += 100.0
            
        # 重复访问同一位置的惩罚
        collision_key = (int(next_pos[0]), int(next_pos[1]))
        if collision_key in self.collision_history:
            reward -= 10.0 * self.collision_history[collision_key]

        # 在遇到障碍物时更新历史
        if self.is_obstacle(next_pos):
            if collision_key in self.collision_history:
                self.collision_history[collision_key] += 1
            else:
                self.collision_history[collision_key] = 1
                
        return reward
    def reconstruct_path(self, came_from, current):
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
            
        return path[::-1]  # 逆序返回路径
        
    def close_enough(self, point1, point2, threshold=3.0):
        """检查两点是否足够接近"""
        return math.dist(point1, point2) <= threshold
        
    def is_valid_position(self, pos):
        """检查位置是否有效（在地图范围内）"""
        x, y = pos
        # 检查边界
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return False
            
        return True
        
    def is_obstacle(self, point):
        """障碍物检测 - 增强版"""
        # 转换为整数坐标
        x, y = int(round(point[0])), int(round(point[1]))
        
        # 检查点是否在障碍物集合中
        if (x, y) in self.obstacle_grids:
            return True
            
        # 增强版检测 - 检查周围小范围区域
        for dx, dy in [(0,0), (0,1), (1,0), (0,-1), (-1,0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in self.obstacle_grids:
                return True
                
        return False
        
    def count_nearby_obstacles(self, point, radius=3):
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
                    if self.is_obstacle(check_point):
                        count += 1
        
        return count
class MultiLevelCache:
    """多级缓存系统，支持精确匹配和近似匹配"""
    def __init__(self, max_size=1000, expiry=600):
        self.exact_cache = {}  # 精确匹配缓存
        self.approx_cache = {}  # 近似匹配缓存
        self.timestamps = {}  # 缓存时间戳
        self.max_size = max_size  # 最大缓存项数
        self.expiry = expiry  # 过期时间(秒)
        self.lock = threading.RLock()  # 线程锁
        
        # 统计信息
        self.exact_hits = 0
        self.approx_hits = 0
        self.misses = 0
        
    def get(self, key, tolerance=3.0):
        """
        获取缓存项
        
        Args:
            key: 缓存键
            tolerance: 近似匹配的容差
            
        Returns:
            (path, is_exact): 路径和是否为精确匹配
        """
        with self.lock:
            now = time.time()
            
            # 精确查找
            if key in self.exact_cache:
                # 检查是否过期
                if now - self.timestamps[key] <= self.expiry:
                    # 更新时间戳
                    self.timestamps[key] = now
                    self.exact_hits += 1
                    return self.exact_cache[key].copy(), True
                else:
                    # 过期删除
                    del self.exact_cache[key]
                    del self.timestamps[key]
            
            # 近似查找
            if len(key) >= 3:  # 确保键包含起点和终点
                start, end = key[1], key[2]
                
                for cached_key, path in list(self.approx_cache.items()):
                    # 确保键格式一致
                    if len(cached_key) < 3:
                        continue
                        
                    cached_start, cached_end = cached_key[1], cached_key[2]
                    
                    if (math.dist(start, cached_start) < tolerance and 
                        math.dist(end, cached_end) < tolerance):
                        
                        # 检查过期
                        if cached_key in self.timestamps and now - self.timestamps[cached_key] <= self.expiry:
                            # 更新时间戳
                            self.timestamps[cached_key] = now
                            self.approx_hits += 1
                            return path.copy(), False
                        elif cached_key in self.timestamps:
                            # 过期删除
                            del self.approx_cache[cached_key]
                            del self.timestamps[cached_key]
            
            self.misses += 1
            return None, False
            
    def put(self, key, value, is_exact=True):
        """
        添加缓存项
        
        Args:
            key: 缓存键
            value: 路径值
            is_exact: 是否为精确匹配
        """
        with self.lock:
            now = time.time()
            
            # 检查容量
            total_size = len(self.exact_cache) + len(self.approx_cache)
            if total_size >= self.max_size:
                # 删除最旧的项
                if self.timestamps:
                    oldest_key = min(self.timestamps, key=self.timestamps.get)
                    
                    if oldest_key in self.exact_cache:
                        del self.exact_cache[oldest_key]
                    elif oldest_key in self.approx_cache:
                        del self.approx_cache[oldest_key]
                        
                    del self.timestamps[oldest_key]
                    
            # 添加新项
            target_cache = self.exact_cache if is_exact else self.approx_cache
            target_cache[key] = value.copy()  # 存储副本避免外部修改
            self.timestamps[key] = now
            
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.exact_cache.clear()
            self.approx_cache.clear()
            self.timestamps.clear()
            
            # 重置统计信息
            self.exact_hits = 0
            self.approx_hits = 0
            self.misses = 0
            
    def get_stats(self):
        """获取缓存统计信息"""
        with self.lock:
            total_hits = self.exact_hits + self.approx_hits
            total_requests = total_hits + self.misses
            
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            exact_hit_rate = self.exact_hits / total_requests if total_requests > 0 else 0
            approx_hit_rate = self.approx_hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.exact_cache) + len(self.approx_cache),
                'exact_size': len(self.exact_cache),
                'approx_size': len(self.approx_cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'exact_hit_rate': exact_hit_rate,
                'approx_hit_rate': approx_hit_rate,
                'exact_hits': self.exact_hits,
                'approx_hits': self.approx_hits,
                'misses': self.misses
            }
    
    def contains(self, key, tolerance=3.0):
        """
        检查键是否在缓存中
        
        Args:
            key: 缓存键
            tolerance: 近似匹配的容差
            
        Returns:
            bool: 是否存在于缓存中
        """
        path, _ = self.get(key, tolerance)
        return path is not None
    
    def update_obstacle(self, obstacle_point, radius=10.0):
        """
        障碍物更新时，清理受影响的缓存路径
        
        Args:
            obstacle_point: 障碍物坐标
            radius: 影响半径
        """
        with self.lock:
            # 检查精确缓存和近似缓存
            affected_keys = []
            
            # 检查所有缓存中的路径
            for cache_dict in [self.exact_cache, self.approx_cache]:
                for key, path in list(cache_dict.items()):
                    # 检查路径是否经过障碍物附近
                    for point in path:
                        if math.dist(point, obstacle_point) < radius:
                            affected_keys.append(key)
                            break
            
            # 删除受影响的缓存
            for key in affected_keys:
                if key in self.exact_cache:
                    del self.exact_cache[key]
                elif key in self.approx_cache:
                    del self.approx_cache[key]
                    
                if key in self.timestamps:
                    del self.timestamps[key]