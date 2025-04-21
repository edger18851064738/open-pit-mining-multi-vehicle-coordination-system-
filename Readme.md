露天矿多车协同调度系统

项目概述
本项目是一个基于Python的露天矿多车协同调度系统，旨在解决露天矿场景下多车辆的路径规划和调度问题。系统通过先进的冲突检测和解决算法，实现了车辆间的高效协同，避免碰撞并优化整体运输效率。
系统采用模块化设计，核心调度算法与底层功能分离，便于灵活扩展不同的调度策略（如CBS、QMIX、模仿学习等）。同时提供了强大的可视化工具，支持实时监控和调试。
功能特性

多车协同调度: 支持多辆车辆同时执行不同任务，自动检测和解决冲突
高效路径规划: 实现了基于A*的路径规划算法，支持障碍物避开和路径平滑
冲突检测与解决: 基于CBS(Conflict-Based Search)算法检测和解决路径冲突
车辆任务管理: 完善的任务分配和状态管理机制
实时可视化: 使用PyQtGraph实现高性能可视化，支持路径规划和冲突解决过程的直观展示
测试与评估: 内置集成测试框架，支持多场景测试和性能评估

系统架构
项目采用分层架构设计，主要包含以下模块:
├── algorithm/          # 核心算法模块
│   ├── cbs.py          # 基于冲突的搜索算法
│   ├── map_service.py  # 地图服务
│   ├── optimized_path_planner.py  # 优化的路径规划器
│
├── models/             # 业务模型
│   ├── vehicle.py      # 车辆模型
│   ├── task.py         # 任务模型
│
├── utils/              # 工具类
│   ├── geo_tools.py    # 地理坐标工具
│   ├── path_tools.py   # 路径处理工具
│   ├── show_PQ.py      # 可视化工具
│
├── tests/              # 测试模块
│   ├── test_whole.py   # 集成测试框架
│   ├── test_PQ.py      # 路径规划测试
│
├── master.py           # 主程序入口
├── config.ini          # 配置文件
└── README.md           # 项目说明
安装与配置
环境要求

Python 3.8+
Anaconda(推荐)或其他Python环境

依赖安装
bash# 创建并激活环境
conda create -n mining_env python=3.9
conda activate mining_env

# 安装依赖
pip install numpy matplotlib networkx scipy configparser pyqt5 pyqtgraph
配置文件
在项目根目录下创建config.ini文件:
ini[MAP]
grid_size = 200
virtual_origin_x = 0
virtual_origin_y = 0
safe_radius = 30
obstacle_density = 0.15
使用方法
运行主程序
bashpython master.py
主程序将启动多车调度模拟，在控制台输出调度过程和状态信息。
路径规划测试与可视化
bashpython tests/test_PQ.py
这将启动路径规划可视化工具，提供图形界面展示路径规划过程和结果。
集成测试
bashpython tests/test_whole.py
运行完整的集成测试框架，测试多种场景下系统的性能和稳定性。
核心算法
路径规划
系统使用优化的A*算法进行路径规划，主要特点：

支持障碍物检测和避开
路径平滑处理，生成更自然的行驶路径
性能优化，包括路径缓存和空间索引

冲突检测与解决
基于CBS(Conflict-Based Search)算法实现冲突检测与解决：

检测两种冲突类型：节点冲突(同一位置)和边冲突(路径交叉)
基于车辆优先级的冲突解决策略
自动路径重规划，避开冲突区域

系统特点

高效性: 优化的路径规划和调度算法，确保快速响应
可扩展性: 模块化设计允许轻松添加新的调度算法和功能
鲁棒性: 完善的异常处理和备选方案，确保系统稳定性
可视化: 强大的可视化功能，支持实时监控和调试

未来规划

实现更多调度算法(QMIX, 模仿学习+强化学习)
增强系统性能，支持更大规模场景
添加更多车辆特性和约束条件
开发更完善的用户界面