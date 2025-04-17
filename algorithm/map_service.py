import sys
import os
import logging
import configparser
import math
import time
import networkx as nx
import osmnx as ox
from typing import Dict, List, Tuple, Set, Optional
from threading import RLock



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from utils.geo_tools import GeoUtils
from utils.path_tools import PathProcessor, PathOptimizationError
from scipy import spatial
import numpy as np
from utils.bezier import Bernstein

import random
logging.basicConfig(level=logging.INFO)

class MiningMapConfig:
    """矿山专用地图配置"""
    def __init__(self):
        # 基础路网配置
        self.grid_nodes = 50   # 网格密度
        self.config = configparser.ConfigParser()
        self.config.read(os.path.join(PROJECT_ROOT, 'config.ini'))
        self.grid_size = int(self.config.get('MAP', 'grid_size', fallback='200'))
        self.safe_radius = int(self.config.get('MAP', 'safe_radius', fallback='30'))
        self.obstacle_density = float(self.config.get('MAP', 'obstacle_density', fallback='0.15'))
        
        # 关键点坐标 (基于网格尺寸的百分比布局)
        edge_padding = self.grid_size * 0.05  # 5%边距
        self.key_points = {
            'parking': (self.grid_size*0.75, self.grid_size/2),      # 中间右侧
            'unload': (self.grid_size - edge_padding, self.grid_size - edge_padding), # 右上角
            'load1': (edge_padding, self.grid_size - edge_padding), # 左上角
            'load2': (edge_padding, edge_padding), # 左下角
            'load3': (self.grid_size - edge_padding, edge_padding)  # 右下角
        }
        
    def get(self, section, option, fallback=None):
        """模拟configparser的get方法"""
        return self.config.get(section, option, fallback=fallback)


class MapLoadError(Exception):
    """地图加载异常"""
    pass

class MapService:
    """矿山地图核心服务"""
    def __init__(self, path_processor: PathProcessor = None):
        self.geo_utils = GeoUtils()
        self.config = MiningMapConfig()
        # 修正参数传递方式（移除grid_size参数）
        self.path_processor = path_processor or PathProcessor(GeoUtils())
        self.grid_size = self.config.grid_size
        self.grid_nodes = self.config.grid_nodes
        self.safe_radius = self.config.safe_radius
        self.obstacle_density = self.config.obstacle_density
        self._obstacle_nodes = set()  # 提前初始化障碍节点集合
        self.virtual_origin = self._get_virtual_origin()
        try:
            self.road_network = self._generate_mining_grid()
        except Exception as e:
            logging.critical(f"地图初始化失败: {str(e)}")
            self.road_network = nx.Graph()

    def coordinate_validator(func):
        """坐标标准化装饰器"""
        def wrapper(self, *args, **kwargs):
            args_list = list(args)
            # 转换前两个位置参数中的坐标
            for i, arg in enumerate(args_list):
                if isinstance(arg, tuple) and len(arg) == 2:
                    args_list[i] = self.geo_utils.metres_to_grid(*arg)
            # 转换关键字参数中的坐标
            for k, v in kwargs.items():
                if isinstance(v, tuple) and len(v) == 2:
                    kwargs[k] = self.geo_utils.metres_to_grid(*v)
            return func(self, *args_list, **kwargs)
        return wrapper
        # 修正参数传递方式（移除grid_size参数）
        self.path_processor = path_processor or PathProcessor(
                    GeoUtils()  # 不再手动传递grid_size
                )
        self.grid_size = self.config.grid_size
        self.grid_nodes = self.config.grid_nodes
        self.safe_radius = self.config.safe_radius
        self.obstacle_density = self.config.obstacle_density
        self._obstacle_nodes = set()  # 提前初始化障碍节点集合
        self.virtual_origin = self._get_virtual_origin()        
        try:
            self.road_network = self._generate_mining_grid()
        except Exception as e:
            logging.critical(f"地图初始化失败: {str(e)}")
            self.road_network = nx.Graph()
    def _get_virtual_origin(self) -> Tuple[float, float]:
        """获取虚拟坐标系原点（使用停车场坐标）"""
        return (
            self.config.key_points['parking'][0],
            self.config.key_points['parking'][1]
        )
    def _create_node_attr(self, node_id: int) -> dict:
        """创建节点属性（统一实现）"""
        node_spacing = self.config.grid_size / (self.config.grid_nodes - 1)
        x = (node_id % self.grid_nodes) * node_spacing
        y = (node_id // self.grid_nodes) * node_spacing
        
        # 修改坐标转换调用方式
        virtual_x, virtual_y = self.path_processor.geo_utils.metres_to_grid(x, y)
        
        return {
            'x': x,
            'y': y,
            'passable': True,
            'node_type': 'regular',
            'hardness': random.uniform(2.5, 5.0),
            'grade': random.uniform(0, 10)
        }


    def _mark_safe_zones(self, graph: nx.Graph):
        """使用传入的graph对象代替self.road_network"""
        for name, coord in self.config.key_points.items():
            # 使用临时坐标转换方法
            center_node = self._temp_coord_to_node(coord, graph)
            # 确保关键点所在节点可通行
            graph.nodes[center_node]['passable'] = True
            graph.nodes[center_node]['node_type'] = 'key_point'
            
            # 保护周围区域
            for node in graph.nodes():
                node_pos = (graph.nodes[node]['x'], graph.nodes[node]['y'])
                if node != center_node and math.dist(node_pos, coord) <= self.safe_radius:
                    graph.nodes[node]['passable'] = False
                    graph.nodes[node]['node_type'] = 'protected'
                    self._obstacle_nodes.add(node)

    def _temp_coord_to_node(self, coord: Tuple[float, float], graph: nx.Graph) -> int:
        """临时坐标转换方法，用于路网生成阶段"""
        valid_nodes = [(n, data) for n, data in graph.nodes(data=True) if data['passable']]
        if not valid_nodes:
            raise MapLoadError("没有可通行的节点用于坐标转换")
            
        kd_tree = spatial.KDTree([(data['x'], data['y']) for n, data in valid_nodes])
        _, index = kd_tree.query(coord)
        return valid_nodes[index][0]

    def _mark_obstacles(self, graph: nx.Graph):
        """生成随机障碍物（避开安全区）"""
        for node in graph.nodes():
            if (graph.nodes[node]['node_type'] == 'regular' 
                and random.random() < self.obstacle_density):
                graph.nodes[node]['passable'] = False
                self._obstacle_nodes.add(node)

    @coordinate_validator
    def coord_to_node(self, coord: Tuple[float, float]) -> int:
        """坐标转节点（核心方法）"""
        valid_nodes = [
            (n, data) for n, data in self.road_network.nodes(data=True)
            if data['passable']
        ]
        kd_tree = spatial.KDTree([(data['x'], data['y']) for n, data in valid_nodes])
        _, index = kd_tree.query(coord)
        return valid_nodes[index][0]

    def _generate_mining_grid(self) -> nx.Graph:
        """生成矿山专用路网"""
        G = nx.grid_2d_graph(self.grid_nodes, self.grid_nodes)
        G = nx.convert_node_labels_to_integers(G)
        
        # 设置节点属性
        nx.set_node_attributes(G, {
            node: self._create_node_attr(node)
            for node in G.nodes()
        })
        
        self._mark_safe_zones(G)  # 先标记安全区
        self._mark_obstacles(G)   # 再生成障碍物
        return G



    def get_keypoint_nodes(self) -> Dict[str, int]:  # 补全缺失的方法
        """获取关键点对应节点"""
        return {
            name: self.coord_to_node(coord)
            for name, coord in self.config.key_points.items()
        }

    def get_operation_nodes(self) -> Dict[str, List[int]]:
        """获取运营节点拓扑关系"""
        key_nodes = self.get_keypoint_nodes()
        return {
            'parking': [key_nodes['parking']],
            'loading': [key_nodes[f'load{i}'] for i in range(1,4)],
            'unloading': [key_nodes['unload']],
            'charging': [key_nodes['parking']]  # 停车场即充电站
        }
    
    @coordinate_validator
    def find_nearest_loading(self, coord: Tuple[float, float]) -> int:
        """寻找最近可用装载点"""
        loading_nodes = self.get_operation_nodes()['loading']
        valid_nodes = [(n, self.road_network.nodes[n]) for n in loading_nodes 
                      if self.road_network.nodes[n]['passable']]
        
        if not valid_nodes:
            raise MapLoadError("没有可用的装载点")
        
        kd_tree = spatial.KDTree([(n[1]['x'], n[1]['y']) for n in valid_nodes])
        _, index = kd_tree.query(coord)
        return valid_nodes[index][0]
        
    @coordinate_validator
    def get_terrain_hardness(self, x: float, y: float) -> float:
        """获取指定坐标的地形硬度"""
        node = self.coord_to_node((x, y))
        return self.road_network.nodes[node]['hardness']

    def plan_vehicle_route(self, start: int, end: int) -> List[Tuple[float, float]]:
        """规划车辆完整路线"""
        try:
            path = nx.shortest_path(self.road_network, start, end, weight='grade_cost')
            return self._convert_path_coordinates(path)
        except nx.NetworkXNoPath:
            raise PathOptimizationError(f"无法找到路径 {start} -> {end}")

    def _convert_path_coordinates(self, path: List[int]) -> List[Tuple]:
        """使用geo_tools转换坐标"""
        return [self.path_processor.geo_utils.grid_to_metres(  # 修改访问路径
            self.road_network.nodes[n]['x'],
            self.road_network.nodes[n]['y']
        ) for n in path]
        
    def plan_route(self, start: Tuple[float, float], end: Tuple[float, float], vehicle_type: str = 'empty') -> Dict:
        """规划路径（集成path_planner）
        
        Args:
            start: 起点坐标
            end: 终点坐标
            vehicle_type: 车辆类型（empty或loaded）
            
        Returns:
            Dict: 包含path和distance的字典
        """
        try:
            # 导入path_planner
            from algorithm.path_planner import HybridPathPlanner
            
            # 创建规划器实例
            planner = HybridPathPlanner(self)
            
            # 创建简化的车辆配置
            vehicle_config = {
                'turning_radius': 10.0,
                'min_hardness': 2.5,
                'current_load': 0 if vehicle_type == 'empty' else 50000
            }
            
            # 使用path_planner优化路径
            from algorithm.path_planner import MiningVehicle as PlannerVehicle
            vehicle = PlannerVehicle("temp", vehicle_config)
            path = planner.optimize_path(start, end, vehicle)
            
            # 计算路径距离
            distance = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else 0
            
            return {
                'path': path,
                'distance': distance
            }
        except Exception as e:
            logging.error(f"路径规划失败: {str(e)}")
            raise PathOptimizationError(f"路径规划失败: {str(e)}") from e

if __name__ == "__main__":
    """地图服务自测试模块"""
    import matplotlib.pyplot as plt
    
    def visualize_enhanced(graph, key_nodes, operation_nodes):
        """增强版可视化：显示道路属性和运营节点"""
        plt.figure(figsize=(15, 12))
        
        # 绘制所有节点（按坡度着色）
        all_x = [data['x'] for _, data in graph.nodes(data=True)]
        all_y = [data['y'] for _, data in graph.nodes(data=True)]
        grades = [data['grade'] if data['passable'] else 0 
                for _, data in graph.nodes(data=True)]
        
        plt.scatter(all_x, all_y, c=grades, cmap='YlOrRd', s=8, alpha=0.6)
        plt.colorbar(label='Road Grade (%)')

        # 标注关键运营节点
        node_types = {
            'parking': ('P', 'blue'),
            'loading': ('L', 'green'),
            'unloading': ('U', 'red'),
            'charging': ('C', 'purple')
        }
        
        for cat, nodes in operation_nodes.items():
            for n in nodes:
                x = graph.nodes[n]['x']
                y = graph.nodes[n]['y']
                plt.scatter(x, y, s=120, marker='D',  # 从\boxslash改为菱形
                        c=node_types[cat][1], edgecolors='black')
                plt.text(x+8, y+8, f"{node_types[cat][0]}{n}", 
                        fontsize=9, weight='bold')

        plt.title("Enhanced Mining Map Visualization")
        plt.xlabel("X Coordinate (m)")
        plt.ylabel("Y Coordinate (m)")
        plt.grid(True, alpha=0.3)
        plt.show()

    try:
        print("=== 开始地图服务增强自检 ===")
        # 显式传递配置参数（参数传递给GeoUtils）
        service = MapService(
            path_processor=PathProcessor(GeoUtils())
        )
        
        # 验证关键配置参数
        cfg = service.config
        expected_parking_x = cfg.grid_size * 0.05
        actual_parking_x = cfg.key_points['parking'][0]
        assert abs(actual_parking_x - expected_parking_x) < 1, "停车场X坐标异常"
        
        # 运营节点拓扑验证
        ops = service.get_operation_nodes()
        
        assert len(ops['loading']) == 3, "装载点数量应为3"
        assert ops['parking'] == ops['charging'], "充电站应与停车场位置一致"
        
        # 路径规划集成测试
        test_route = service.plan_vehicle_route(
            start=ops['parking'][0],
            end=ops['loading'][0]  # 直接使用节点ID
        )
        
        # 添加调试信息输出
        print(f"\n[节点拓扑调试信息]")
        print(f"停车场节点: {ops['parking']}")
        print(f"装载节点: {ops['loading']}")
        print(f"测试路径节点数: {len(test_route)}")
        assert len(test_route) > 2, "停车场到装载点应有有效路径"
        
        # 显示增强统计
        nodes_data = service.road_network.nodes(data=True)
        avg_hardness = sum(d['hardness'] for _,d in nodes_data if d['passable'])/len(nodes_data)
        max_grade = max(d['grade'] for _,d in nodes_data if d['passable'])
        
        print(f"\n[道路特性统计]")
        print(f"平均地形硬度: {avg_hardness:.1f} | 最大坡度: {max_grade:.1f}%")
        
        # 增强可视化
        visualize_enhanced(service.road_network, 
                          service.get_keypoint_nodes(),
                          service.get_operation_nodes())

    except Exception as e:
        print(f"!!! 自检失败: {str(e)}")
    finally:
        print("=== 增强自检结束 ===")