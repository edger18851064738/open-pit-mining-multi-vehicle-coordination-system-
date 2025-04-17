import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from config.paths import PROJECT_ROOT
from typing import List, Dict
from collections import defaultdict
from algorithm.dispatch_service_v1 import DispatchService
from models.vehicle import MiningVehicle
from models.task import TransportTask

class QMixNetwork(nn.Module):
    """QMIX神经网络模型"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 调整输入维度以适应批量处理
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class QMixTrainer:
    """QMIX训练器"""
    def __init__(self, agent_num, state_dim, action_dim):
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 初始化调度服务
        from algorithm.path_planner import HybridPathPlanner
        from algorithm.map_service import MapService
        map_service = MapService()
        planner = HybridPathPlanner(map_service)
        self.dispatch_service = DispatchService(planner, map_service, use_qmix=True)
        
        # 初始化网络
        self.agent_network = QMixNetwork(state_dim, 512, action_dim)
        self.mixer_network = QMixNetwork(agent_num * action_dim, 512, 1)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.agent_network.parameters()) + 
            list(self.mixer_network.parameters()),
            lr=0.001,
            weight_decay=1e-4
        )
        
        # 经验回放缓冲区
        self.replay_buffer = []
        self.buffer_size = 50000
        self.batch_size = 256
        
        # 冲突惩罚参数
        self.collision_penalty = -10.0
        self.route_deviation_penalty = -5.0
        
    def store_experience(self, states, actions, rewards, next_states, dones):
        """存储经验到回放缓冲区"""
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
        self.replay_buffer.append({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        })
    
    def train_step(self, batch_size=None):
        """执行一步训练"""
        batch_size = batch_size or self.batch_size
        if len(self.replay_buffer) < batch_size:
            return
            
        # 从缓冲区采样
        batch = random.sample(self.replay_buffer, batch_size)
        
        # 获取当前环境状态
        vehicles = self.dispatch_service.get_vehicles()
        tasks = self.dispatch_service.get_tasks()
        
        # 准备数据
        states = torch.stack([item['states'].clone().detach().float() if torch.is_tensor(item['states']) 
                             else torch.tensor(item['states'], dtype=torch.float32) for item in batch])
        actions = torch.stack([item['actions'].clone().detach().long() if torch.is_tensor(item['actions']) 
                              else torch.tensor(item['actions'], dtype=torch.long) for item in batch])
        # 处理奖励值，确保正确处理张量和标量
        rewards = []
        for item in batch:
            if torch.is_tensor(item['rewards']):
                if item['rewards'].dim() == 0:  # 标量张量
                    rewards.append(item['rewards'].item())
                else:  # 多维张量
                    rewards.extend(item['rewards'].tolist())
            else:
                rewards.append(item['rewards'])
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        next_states = torch.stack([item['next_states'].clone().detach().float() if torch.is_tensor(item['next_states']) 
                                  else torch.tensor(item['next_states'], dtype=torch.float32) for item in batch])
        
        # 处理dones标志，确保正确处理张量和标量
        dones = []
        for item in batch:
            if torch.is_tensor(item['dones']):
                if item['dones'].dim() == 0:  # 标量张量
                    dones.append(item['dones'].item())
                else:  # 多维张量
                    dones.extend(item['dones'].tolist())
            else:
                dones.append(item['dones'])
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # 计算Q值
        q_values = self.agent_network(states)
        # 确保动作索引维度与Q值维度匹配
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        elif actions.dim() == 2 and actions.size(1) == 1:
            actions = actions.squeeze(1)
            actions = actions.unsqueeze(1)
        
        # 调整动作张量维度以匹配Q值
        if actions.dim() == 1:
            actions = actions.unsqueeze(-1)  # 从[batch]变为[batch,1]
        elif actions.dim() == 2 and actions.size(1) == 1:
            actions = actions.unsqueeze(-1)  # 从[batch,1]变为[batch,1]
            
        # 确保q_values和actions维度一致
        if q_values.dim() == 3 and actions.dim() == 2:
            actions = actions.unsqueeze(1)  # 从[batch,action]变为[batch,1,action]
            
        # 检查维度是否匹配
        if q_values.dim() != actions.dim():
            raise ValueError(f"Q值维度{q_values.dim()}与动作维度{actions.dim()}不匹配")
            
        selected_q_values = q_values.gather(-1, actions)
        
        # 调整rewards维度以匹配selected_q_values
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        elif rewards.dim() == 2 and rewards.size(1) == 1:
            rewards = rewards.unsqueeze(-1)
            
        # 确保rewards维度与selected_q_values匹配
        if rewards.size(0) != selected_q_values.size(0):
            rewards = rewards.view(selected_q_values.size(0), -1, 1)
            
        # 调整rewards维度与selected_q_values完全匹配
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        elif rewards.dim() == 2 and rewards.size(1) == 1:
            rewards = rewards.unsqueeze(-1)
            
        # 确保rewards与selected_q_values形状一致
        if rewards.size(0) != selected_q_values.size(0):
            rewards = rewards.expand(selected_q_values.size(0), -1, -1)
        
        # 最终维度检查
        if rewards.size() != selected_q_values.size():
            rewards = rewards.view_as(selected_q_values)
            
        # 检查维度是否匹配
        if selected_q_values.size() != rewards.size():
            raise ValueError(f"selected_q_values维度{selected_q_values.size()}与rewards维度{rewards.size()}不匹配")
            
        # 计算损失
        loss = torch.mean((selected_q_values - rewards) ** 2)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

if __name__ == "__main__":
    """训练示例"""
    trainer = QMixTrainer(agent_num=5, state_dim=10, action_dim=4)
    
    # 更真实的测试场景
    for episode in range(500):
        # 模拟真实车辆状态
        vehicle_states = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 空闲车辆
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 装载中
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 运输中
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 卸载中
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]   # 充电中
        ], dtype=torch.float32)
        
        # 模拟真实任务分配
        actions = torch.tensor([0, 1, 2, 3, 0], dtype=torch.long)  # 0:空闲 1:装载 2:运输 3:卸载
        
        # 基于状态的奖励
        rewards = torch.tensor([
            0.0,  # 空闲无奖励
            1.0,  # 装载奖励
            0.5,  # 运输奖励
            2.0,  # 卸载奖励
            -0.1  # 充电惩罚
        ], dtype=torch.float32)
        
        # 模拟状态转移
        next_states = torch.tensor([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 空闲->装载
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 装载->运输
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 运输->卸载
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 卸载->充电
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # 充电->空闲
        ], dtype=torch.float32)
        
        # 模拟任务完成情况
        dones = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32)  # 只有卸载任务完成
        
        trainer.store_experience(vehicle_states, actions, rewards, next_states, dones)
        loss = trainer.train_step()
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss}")
            
            # 打印当前策略效果
            with torch.no_grad():
                test_state = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
                q_values = trainer.agent_network(test_state)
                print(f"Test Q-values for idle vehicle: {q_values}")