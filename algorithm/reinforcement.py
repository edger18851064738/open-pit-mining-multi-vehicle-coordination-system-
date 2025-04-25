import numpy as np
import heapq
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle, Polygon
import matplotlib
import math
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']  # 优先使用的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False 
class ReinforcedAStar:
    def __init__(self, grid_map, learning_rate=0.1, discount_factor=0.9):
        self.grid_map = grid_map
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_values = {}  # 存储状态-动作对应的Q值
        
    def get_heuristic(self, current, goal):
        # 基础启发式函数 - 曼哈顿距离
        base_h = abs(current[0] - goal[0]) + abs(current[1] - goal[1])
        
        # 从Q值表中获取学习到的启发式调整
        state = self.get_state_representation(current)
        learned_adjustment = self.get_learned_adjustment(state)
        
        # 组合基础启发式和学习调整
        return base_h + learned_adjustment
    
    def get_state_representation(self, position):
        # 将位置转换为状态表示
        # 可以包含位置、周围障碍物分布等信息
        return str(position)
    
    def get_learned_adjustment(self, state):
        # 从Q值表中获取该状态的平均学习调整
        if state in self.q_values:
            return np.mean(list(self.q_values[state].values()))
        return 0
    
    def path_planning(self, start, goal):
        # A*路径规划的主要逻辑
        open_set = [(self.get_heuristic(start, goal), 0, start)]
        came_from = {}
        g_score = {str(start): 0}
        
        while open_set:
            # 获取f值最小的节点
            _, cost_so_far, current = heapq.heappop(open_set)
            
            if current == goal:
                # 找到路径，返回并更新Q值
                path = self.reconstruct_path(came_from, current)
                self.update_q_values(path, goal)
                return path
            
            # 扩展当前节点
            for next_node in self.get_neighbors(current):
                tentative_g_score = g_score[str(current)] + 1
                
                if str(next_node) not in g_score or tentative_g_score < g_score[str(next_node)]:
                    came_from[str(next_node)] = current
                    g_score[str(next_node)] = tentative_g_score
                    f_score = tentative_g_score + self.get_heuristic(next_node, goal)
                    heapq.heappush(open_set, (f_score, tentative_g_score, next_node))
        
        # 没有找到路径
        return None
    
    def get_neighbors(self, node):
        # 获取节点的相邻节点
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 四连通
        neighbors = []
        
        for dx, dy in directions:
            nx, ny = node[0] + dx, node[1] + dy
            if 0 <= nx < len(self.grid_map) and 0 <= ny < len(self.grid_map[0]) and self.grid_map[nx][ny] == 0:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def reconstruct_path(self, came_from, current):
        # 重建路径
        path = [current]
        while str(current) in came_from:
            current = came_from[str(current)]
            path.append(current)
        
        return path[::-1]  # 反转路径使其从起点到终点
    
    def update_q_values(self, path, goal):
        # 基于找到的路径更新Q值
        if not path:
            return
        
        # 计算路径奖励
        path_length = len(path) - 1
        reward = 100 - 0.1 * path_length  # 路径越短奖励越高
        
        # 为路径中的每个状态-动作对更新Q值
        for i in range(len(path) - 1):
            current = path[i]
            next_node = path[i + 1]
            
            state = self.get_state_representation(current)
            action = (next_node[0] - current[0], next_node[1] - current[1])
            
            if state not in self.q_values:
                self.q_values[state] = {}
            
            if action not in self.q_values[state]:
                self.q_values[state][action] = 0
            
            # Q-learning更新公式
            target = reward if next_node == goal else reward + self.discount_factor * self.get_best_q_value(next_node)
            self.q_values[state][action] += self.learning_rate * (target - self.q_values[state][action])
    
    def get_best_q_value(self, state):
        # 获取状态的最佳Q值
        state_str = self.get_state_representation(state)
        if state_str in self.q_values and self.q_values[state_str]:
            return max(self.q_values[state_str].values())
        return 0
    
    def train(self, start, goal, episodes=100):
        # 训练强化A*
        paths = []
        for episode in range(episodes):
            path = self.path_planning(start, goal)
            paths.append(path)
            print(f"Episode {episode+1}: Path length = {len(path)-1 if path else 'No path found'}")
        return paths

if __name__ == "__main__":
    # 创建一个示例网格地图 (0=可通行, 1=障碍物)
    grid_size = 20
    grid_map = np.zeros((grid_size, grid_size), dtype=int)
    
    # 添加一些障碍物 - 创建迷宫样式的环境
    # 水平墙
    grid_map[5, 2:15] = 1
    grid_map[10, 5:18] = 1
    grid_map[15, 2:15] = 1
    
    # 垂直墙
    grid_map[2:15, 5] = 1
    grid_map[5:18, 10] = 1
    grid_map[2:15, 15] = 1
    
    # 设置起点和终点
    start = (2, 2)
    goal = (18, 18)
    
    # 初始化强化A*算法
    ras = ReinforcedAStar(grid_map, learning_rate=0.2, discount_factor=0.9)
    
    # 训练算法
    print("开始训练强化A*算法...")
    num_episodes = 20
    paths = ras.train(start, goal, episodes=num_episodes)
    
    # 可视化训练结果
    plt.figure(figsize=(12, 10))
    
    # 绘制网格和障碍物
    plt.imshow(grid_map, cmap='binary')
    
    # 绘制起点和终点
    plt.plot(start[1], start[0], 'go', markersize=10, label='起点')
    plt.plot(goal[1], goal[0], 'ro', markersize=10, label='终点')
    
    # 绘制第一条路径
    if paths[0]:
        path_x = [node[1] for node in paths[0]]
        path_y = [node[0] for node in paths[0]]
        plt.plot(path_x, path_y, 'b-', linewidth=1, alpha=0.5, label='初始路径')
    
    # 绘制最后一条路径
    if paths[-1]:
        path_x = [node[1] for node in paths[-1]]
        path_y = [node[0] for node in paths[-1]]
        plt.plot(path_x, path_y, 'g-', linewidth=2, label='最终路径')
    
    plt.title(f'强化A*算法路径规划 - 训练 {num_episodes} 次')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # 显示图像
    plt.show()
    
    # 显示效果统计
    print("\n训练效果统计:")
    print(f"初始路径长度: {len(paths[0])-1}")
    print(f"最终路径长度: {len(paths[-1])-1}")
    path_improvement = (len(paths[0]) - len(paths[-1])) / len(paths[0]) * 100
    print(f"路径长度改进: {path_improvement:.2f}%")
    
    # 模拟复杂地图场景测试
    print("\n在不同起点和终点上测试算法...")
    new_start = (18, 2)
    new_goal = (2, 18)
    
    # 测试规划
    start_time = time.time()
    test_path = ras.path_planning(new_start, new_goal)
    end_time = time.time()
    
    # 显示测试结果
    print(f"测试路径长度: {len(test_path)-1}")
    print(f"规划耗时: {(end_time - start_time)*1000:.2f} 毫秒")
    
    # 可视化测试结果
    plt.figure(figsize=(12, 10))
    plt.imshow(grid_map, cmap='binary')
    plt.plot(new_start[1], new_start[0], 'go', markersize=10, label='新起点')
    plt.plot(new_goal[1], new_goal[0], 'ro', markersize=10, label='新终点')
    
    if test_path:
        path_x = [node[1] for node in test_path]
        path_y = [node[0] for node in test_path]
        plt.plot(path_x, path_y, 'g-', linewidth=2, label='测试路径')
    
    plt.title('强化A*算法在新起点和终点上的测试')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()