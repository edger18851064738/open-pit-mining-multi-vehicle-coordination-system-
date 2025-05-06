"""
强化学习A*算法路径规划器 - 增强版
为露天矿多车协同调度系统提供更高效的路径规划能力

主要改进:
1. 增强的障碍物感知能力
2. 改进的Q值模型和状态表示
3. 动态路径评估与修正
4. 与CBS(冲突基于搜索)的深度集成
5. 多车协作学习能力
"""

import sys
import os
import math
import time
import heapq
import logging
import threading
import random
import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Set, Optional, Union, Any

# 常量定义
MAX_ITERATIONS = 10000      # 强化A*搜索的最大迭代次数
DEFAULT_TIMEOUT = 5.0       # 默认超时时间(秒)
EPSILON = 1e-6              # 浮点数比较精度
CACHE_SIZE = 1000           # 缓存大小
CACHE_EXPIRY = 600          # 缓存过期时间(秒)
DEFAULT_LEARNING_RATE = 0.1  # 默认学习率
DEFAULT_DISCOUNT_FACTOR = 0.9  # 默认折扣因子
TRAINING_EPISODES = 1000    # 训练回合数
REPLAY_BUFFER_SIZE = 2000   # 经验回放缓冲区大小

class ReinforcedAStar:
    """强化学习A*算法实现，提供智能路径规划功能"""
    
    def __init__(self, obstacle_grids=None, map_size=200, 
                 learning_rate=DEFAULT_LEARNING_RATE, 
                 discount_factor=DEFAULT_DISCOUNT_FACTOR,
                 enable_multiagent_learning=True):
        
        # 强化学习参数
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = {}  # 存储状态-动作对应的Q值
        self.multi_agent_q_values = {}  # 多智能体Q值 {(vehicle_id, state): {action: value}}
        self.enable_multiagent_learning = enable_multiagent_learning
        
        # 经验回放机制增强
        self.experience_buffer = []  # 经验回放缓冲区
        self.max_buffer_size = REPLAY_BUFFER_SIZE
        self.prioritized_replay = True  # 启用优先级经验回放
        self.replay_priorities = []  # 经验回放优先级
        
        # 环境模型与状态表示
        self.obstacle_memory = {}  # 状态-障碍物关联记录 {state: obstacle_count}
        self.collision_history = {}  # 记录碰撞历史
        self.state_visit_count = defaultdict(int)  # 状态访问计数
        self.action_success_rate = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # {state: {action: [success, total]}}
        
        # 地图相关属性
        self.map_size = map_size
        self.obstacle_grids = set() if obstacle_grids is None else set(obstacle_grids)
        self.obstacle_gradients = {}  # 存储障碍物梯度场
        self.dynamic_obstacles = {}  # 动态障碍物 {location: timestamp}
        
        # 动作空间 - 8个方向
        self.actions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),  # 上下左右
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
        ]
        
        # 移动成本 - 更精细的成本模型
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
            'successful_navigations': 0,
            'training_time': 0.0,
            'avg_q_values': 0.0,
            'path_efficiency': 0.0  # 实际路径长度/最短距离的比率
        }
        
        # 算法调优参数
        self.exploration_strategy = 'epsilon_decay'  # 可选: 'epsilon_decay', 'ucb', 'thompson'
        self.multi_step_learning = 3  # 多步学习
        self.double_q_learning = True  # 双Q学习
        self.exploration_bonus = True  # 探索奖励
        self.use_neural_network = False  # 神经网络替代Q表
        
        # 与CBS集成支持
        self.conflict_constraints = {}  # 冲突约束 {vehicle_id: [(time, location)]}
        self.other_vehicle_paths = {}  # 其他车辆的路径 {vehicle_id: path}
        self.vehicle_id = None  # 当前车辆ID
        
        # 障碍物感知增强
        self._build_obstacle_gradients()
        
        # 模型保存和加载
        self.model_path = os.path.join(os.path.dirname(__file__), "models", "reinforced_astar_model.pkl")
        
        # 尝试加载预训练模型
        self._try_load_model()

    def _build_obstacle_gradients(self):
        """构建障碍物梯度场，用于更精确的避障"""
        # 清空梯度场
        self.obstacle_gradients = np.zeros((self.map_size, self.map_size))
        
        # 为每个障碍物创建影响梯度
        for obstacle in self.obstacle_grids:
            x, y = obstacle
            
            # 确保坐标在地图范围内
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                # 添加障碍物点
                self.obstacle_gradients[int(x), int(y)] = 1.0
                
                # 创建周围梯度 (梯度随距离衰减)
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        nx, ny = int(x + dx), int(y + dy)
                        if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                            # 计算距离和影响值 (高斯衰减)
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist > 0:  # 避免除以零
                                influence = math.exp(-0.3 * dist)
                                # 叠加梯度 (取最大值)
                                self.obstacle_gradients[nx, ny] = max(
                                    self.obstacle_gradients[nx, ny], 
                                    influence
                                )
        
        logging.debug(f"障碍物梯度场已构建，大小: {self.obstacle_gradients.shape}")

    def get_state_representation(self, position, vehicle_id=None, goal=None):
        """增强的状态表示方法"""
        # 基础状态表示 - 网格化位置
        grid_size = 3  # 状态网格大小
        grid_x = int(position[0] // grid_size)
        grid_y = int(position[1] // grid_size)
        
        # 如果有目标，增加方向信息
        if goal:
            # 计算到目标的方向 (8个方向区间)
            dx = goal[0] - position[0]
            dy = goal[1] - position[1]
            angle = math.atan2(dy, dx)
            direction = int(((angle + math.pi) / (2 * math.pi) * 8) % 8)
            
            # 计算到目标的距离区间 (近、中、远)
            distance = math.sqrt(dx*dx + dy*dy)
            distance_zone = min(2, int(distance / 50))
            
            # 合并信息到状态表示
            base_state = f"{grid_x},{grid_y},{direction},{distance_zone}"
        else:
            base_state = f"{grid_x},{grid_y}"
        
        # 多智能体情况下，加入车辆ID
        if vehicle_id is not None and self.enable_multiagent_learning:
            return f"{vehicle_id}:{base_state}"
        
        return base_state

    def get_heuristic(self, current, goal):
        """增强的启发式函数 - 结合欧几里得距离和障碍物梯度"""
        dx, dy = abs(current[0] - goal[0]), abs(current[1] - goal[1])
        
        # 欧几里得距离基础
        d_euclidean = math.sqrt(dx*dx + dy*dy)
        
        # 计算障碍物影响因子
        obstacles_nearby = self.count_nearby_obstacles(current, radius=5)
        obstacle_factor = 1.0 + (obstacles_nearby * 0.5)
        
        # 从障碍物梯度场获取额外障碍信息
        try:
            cx, cy = int(current[0]), int(current[1])
            if 0 <= cx < self.map_size and 0 <= cy < self.map_size:
                gradient_value = self.obstacle_gradients[cx, cy]
                # 障碍物梯度越高，启发值越大
                gradient_factor = 1.0 + (gradient_value * 5.0)
            else:
                gradient_factor = 1.0
        except (IndexError, ValueError):
            gradient_factor = 1.0
        
        # 计算最终启发值
        return d_euclidean * obstacle_factor * gradient_factor

    def get_learned_adjustment(self, current, goal, vehicle_id=None):
        """从学习中获取启发式调整值"""
        state = self.get_state_representation(current, vehicle_id, goal)
        
        # 获取适用的Q值表
        q_table = self.multi_agent_q_values.get((vehicle_id, state), {}) if vehicle_id and self.enable_multiagent_learning else self.q_values.get(state, {})
        
        if q_table:
            # 计算动作的平均Q值
            action_values = list(q_table.values())
            if action_values:
                # 正向调整 - 奖励途经Q值高的区域
                average_q = sum(action_values) / len(action_values)
                max_q = max(action_values)
                
                # 记录平均Q值用于性能指标
                if self.episodes_trained > 0:
                    self.metrics['avg_q_values'] = (self.metrics['avg_q_values'] * (self.episodes_trained - 1) + average_q) / self.episodes_trained
                
                # 正向调整 - 动态调整系数
                adjustment_strength = min(5.0, 0.5 * math.log(self.episodes_trained + 1))
                return -1.0 * adjustment_strength * max_q
        
        # 默认调整：探索奖励
        if self.exploration_bonus and self.state_visit_count[state] == 0:
            return -3.0  # 探索新状态的奖励
            
        return 0  # 默认不调整

    def get_best_action(self, state, available_actions, vehicle_id=None, goal=None):
        """改进的动作选择策略"""
        if not available_actions:
            return None
        
        # 获取适用的Q值表
        q_table = self.multi_agent_q_values.get((vehicle_id, state), {}) if vehicle_id and self.enable_multiagent_learning else self.q_values.get(state, {})
        
        # 确定探索策略和探索概率
        if self.exploration_strategy == 'epsilon_decay':
            # 随机探索概率 - 随着训练次数增加而减少
            exploration_prob = max(0.05, 0.3 * math.exp(-0.01 * self.episodes_trained))
            
            # 如果碰到障碍物太多，临时增加探索概率
            if self.metrics['consecutive_collisions'] > 3:
                exploration_prob += 0.2
                self.metrics['consecutive_collisions'] = 0  # 重置
                
            # 利用阶段 - 基于Q值选择动作
            if q_table and random.random() > exploration_prob:
                # 获取可用动作的Q值
                action_values = {a: q_table.get(a, 0) for a in available_actions}
                
                if action_values:
                    # 找出最大Q值及对应动作
                    max_q = max(action_values.values())
                    best_actions = [a for a, q in action_values.items() if q == max_q]
                    return random.choice(best_actions)
        
        elif self.exploration_strategy == 'ucb':
            # UCB探索 (Upper Confidence Bound)
            if q_table:
                ucb_values = {}
                for action in available_actions:
                    # 计算UCB值 = Q值 + C * sqrt(ln(总访问次数) / 动作访问次数)
                    q_value = q_table.get(action, 0)
                    action_count = self.action_success_rate[state][action][1] + 1  # 避免除以零
                    total_count = sum(sr[1] for sr in self.action_success_rate[state].values()) + len(available_actions)
                    
                    exploration_term = 2.0 * math.sqrt(math.log(total_count) / action_count)
                    ucb_values[action] = q_value + exploration_term
                
                # 选择UCB值最高的动作
                if ucb_values:
                    max_ucb = max(ucb_values.values())
                    best_actions = [a for a, v in ucb_values.items() if v == max_ucb]
                    return random.choice(best_actions)
        
        elif self.exploration_strategy == 'thompson':
            # 汤普森采样 (Thompson Sampling)
            if q_table:
                sampled_values = {}
                for action in available_actions:
                    # 获取动作的成功/失败计数
                    success, total = self.action_success_rate[state][action]
                    # 避免除以零
                    total = max(1, total)
                    success = min(success, total)
                    
                    # Beta分布采样
                    alpha = success + 1
                    beta = total - success + 1
                    sampled_value = random.betavariate(alpha, beta)
                    sampled_values[action] = sampled_value
                
                # 选择采样值最高的动作
                if sampled_values:
                    max_val = max(sampled_values.values())
                    best_actions = [a for a, v in sampled_values.items() if v == max_val]
                    return random.choice(best_actions)
        
        # 探索阶段或其他情况 - 随机选择动作
        # 如果有目标，偏好朝着目标方向的动作
        if goal:
            dx, dy = goal[0] - state[0], goal[1] - state[1]
            # 计算每个动作与目标方向的点积
            direction_scores = {}
            for action in available_actions:
                # 点积衡量方向相似度
                direction_score = action[0]*dx + action[1]*dy
                direction_scores[action] = direction_score
            
            # 概率性地选择朝向目标的动作
            if random.random() < 0.7:  # 70%的概率选择更好的方向
                max_score = max(direction_scores.values())
                best_directions = [a for a, s in direction_scores.items() if s == max_score]
                return random.choice(best_directions)
        
        return random.choice(available_actions)

    def update_q_values(self, state, action, reward, next_state, done, vehicle_id=None):
        """Q值更新，支持多智能体学习"""
        # 确定使用哪个Q值表
        if vehicle_id is not None and self.enable_multiagent_learning:
            if (vehicle_id, state) not in self.multi_agent_q_values:
                self.multi_agent_q_values[(vehicle_id, state)] = {}
            q_table = self.multi_agent_q_values[(vehicle_id, state)]
        else:
            if state not in self.q_values:
                self.q_values[state] = {}
            q_table = self.q_values[state]
        
        current_q = q_table.get(action, 0)
        
        # 计算目标Q值
        if done:
            target_q = reward
        else:
            # 获取下一个状态的Q值表
            if vehicle_id is not None and self.enable_multiagent_learning:
                next_q_table = self.multi_agent_q_values.get((vehicle_id, next_state), {})
            else:
                next_q_table = self.q_values.get(next_state, {})
                
            # 下一个状态的最大Q值
            max_next_q = max(next_q_table.values()) if next_q_table else 0
            
            # 使用双Q学习
            if self.double_q_learning and random.random() < 0.5:
                # 选择最大Q值的动作，但使用另一套评估
                max_action = max(next_q_table.items(), key=lambda x: x[1])[0] if next_q_table else None
                if max_action:
                    # 使用当前Q值表评估该动作
                    max_next_q = q_table.get(max_action, 0)
            
            # 考虑多步学习
            if self.multi_step_learning > 1:
                # 实际中需要存储多步轨迹，这里简化处理
                target_q = reward + self.discount_factor * max_next_q
            else:
                target_q = reward + self.discount_factor * max_next_q
        
        # 修改学习率根据奖励情况
        adaptive_learning_rate = self.learning_rate
        if reward < 0:  # 负面奖励时提高学习率
            adaptive_learning_rate = min(0.5, self.learning_rate * 2.0)
        
        # 更新Q值
        new_q = current_q + adaptive_learning_rate * (target_q - current_q)
        q_table[action] = new_q
        
        # 更新动作成功率
        success = reward > 0
        success_count, total_count = self.action_success_rate[state][action]
        if success:
            self.action_success_rate[state][action][0] += 1
        self.action_success_rate[state][action][1] += 1
        
        # 更新状态访问计数
        self.state_visit_count[state] += 1

    def replay_experiences(self, batch_size=16):
        """从经验缓冲区中随机采样并更新Q值"""
        if len(self.experience_buffer) < batch_size:
            return
        
        # 选择经验回放策略
        if self.prioritized_replay and self.replay_priorities:
            # 优先级采样
            priorities = np.array(self.replay_priorities)
            probs = priorities / priorities.sum()
            indices = np.random.choice(len(self.experience_buffer), batch_size, p=probs, replace=False)
            batch = [self.experience_buffer[i] for i in indices]
        else:
            # 随机采样
            batch = random.sample(self.experience_buffer, batch_size)
        
        # 更新每个经验的Q值
        for i, (state, action, reward, next_state, done, vehicle_id) in enumerate(batch):
            # 旧Q值
            old_q = self.q_values.get(state, {}).get(action, 0)
            
            # 更新Q值
            self.update_q_values(state, action, reward, next_state, done, vehicle_id)
            
            # 如果使用优先级经验回放，更新优先级
            if self.prioritized_replay and i < len(indices):
                # 优先级基于TD误差
                if state in self.q_values and action in self.q_values[state]:
                    new_q = self.q_values[state][action]
                    td_error = abs(new_q - old_q)
                    self.replay_priorities[indices[i]] = td_error + 0.01  # 避免优先级为零

    def add_experience(self, state, action, reward, next_state, done, vehicle_id=None):
        """添加经验到回放缓冲区"""
        experience = (state, action, reward, next_state, done, vehicle_id)
        
        # 管理缓冲区大小
        if len(self.experience_buffer) >= self.max_buffer_size:
            # 如果使用优先级回放，移除优先级最低的经验
            if self.prioritized_replay and self.replay_priorities:
                min_idx = np.argmin(self.replay_priorities)
                self.experience_buffer.pop(min_idx)
                self.replay_priorities.pop(min_idx)
                # 添加新经验和默认优先级(1.0)
                self.experience_buffer.append(experience)
                self.replay_priorities.append(1.0)
            else:
                # 先进先出
                self.experience_buffer.pop(0)
                self.experience_buffer.append(experience)
        else:
            # 直接添加
            self.experience_buffer.append(experience)
            if self.prioritized_replay:
                self.replay_priorities.append(1.0)  # 默认优先级

    def pathfind(self, start, goal, vehicle_id=None, max_iterations=MAX_ITERATIONS, constraints=None):
        """强化A*路径规划主方法，支持多智能体和约束"""
        logging.info(f"开始{vehicle_id or ''}强化A*路径规划: 从 {start} 到 {goal}")
        self.vehicle_id = vehicle_id  # 记录当前车辆ID
        
        path_start_time = time.time()
        min_dist_to_goal = float('inf')
        no_improvement_count = 0
        
        # 保存约束信息
        if constraints:
            self.conflict_constraints[vehicle_id] = constraints
            
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
            
            # 路径效率计算
            actual_length = sum(math.dist(direct_path[i], direct_path[i+1]) for i in range(len(direct_path)-1))
            straight_length = math.dist(start, goal)
            if straight_length > 0:
                efficiency = straight_length / actual_length
                self.metrics['path_efficiency'] = (self.metrics['path_efficiency'] * self.successful_paths + efficiency) / (self.successful_paths + 1)
            
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
            current_state = self.get_state_representation(current, vehicle_id, goal)
            path_states.append(current_state)
            
            # 监控到目标的距离进展
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
                
                # 路径效率计算
                actual_length = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1))
                straight_length = math.dist(start, goal)
                if straight_length > 0:
                    efficiency = straight_length / actual_length
                    self.metrics['path_efficiency'] = (self.metrics['path_efficiency'] * self.successful_paths + efficiency) / (self.successful_paths + 1)
                
                # 路径后处理
                if len(path) > 3:
                    path = self.post_process_path(path)
                
                # 记录规划耗时
                self.metrics['training_time'] += time.time() - path_start_time
                
                return path
            
            # 如果没有进展且尝试次数过多，采取备选策略
            if no_improvement_count > self.map_size / 2:  # 如距离地图尺寸一半还没进展，考虑备选策略
                logging.debug(f"搜索进度停滞，尝试随机探索")
                no_improvement_count = 0
                
                # 随机跳跃到新区域
                for _ in range(10):  # 尝试10次
                    jump_x = current[0] + random.randint(-20, 20)
                    jump_y = current[1] + random.randint(-20, 20)
                    jump_point = (jump_x, jump_y)
                    
                    # 确保点在地图范围内且不是障碍物
                    if (0 <= jump_x < self.map_size and 0 <= jump_y < self.map_size and 
                        not self.is_obstacle(jump_point)):
                        # 向开放集添加这个点，但给予较高的g值
                        g_jump = g_score[current] + 100  # 较高的跳跃成本
                        f_jump = g_jump + self.get_heuristic(jump_point, goal)
                        heapq.heappush(open_set, (f_jump, g_jump, jump_point))
                        open_set_hash.add(jump_point)
                        came_from[jump_point] = current
                        g_score[jump_point] = g_jump
                        f_score[jump_point] = f_jump
                        break