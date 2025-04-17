import sys
import os
import logging
import configparser
import math
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
from threading import RLock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from utils.geo_tools import GeoUtils
from utils.path_tools import PathProcessor, PathOptimizationError
from scipy import spatial
import numpy as np
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
        
        # 关键点坐标 (基于网格尺寸的百分比布局) - 修复坐标计算
        edge_padding = self.grid_size * 0.05  # 5%边距
        self.key_points = {
            'parking': (self.grid_size * 0.75, self.grid_size * 0.5),  # 修正为乘法，确保是中间右侧
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
        
        # 修正初始化顺序，避免重复初始化
        self.path_processor = path_processor or PathProcessor(self.geo_utils)
        self.grid_size = self.config.grid_size
        self.grid_nodes = self.config.grid_nodes
        self.safe_radius = self.config.safe_radius
        self.obstacle_density = self.config.obstacle_density
        self._obstacle_nodes = set()  # 提前初始化障碍节点集合
        self.virtual_origin = self._get_virtual_origin()
        
        try:
            self.road_network = self._generate_mining_grid()
            logging.info(f"矿山路网初始化成功: {self.grid_nodes}x{self.grid_nodes} 网格")
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
        
        if not valid_nodes:
            raise MapLoadError("没有可通行的节点")
            
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
        
        # 为边添加权重
        for u, v in G.edges():
            # 基于坡度和硬度计算边权重
            hardness_u = G.nodes[u]['hardness']
            hardness_v = G.nodes[v]['hardness']
            grade_u = G.nodes[u]['grade']
            grade_v = G.nodes[v]['grade']
            
            # 坡度成本（上坡更高）
            grade_cost = 1.0 + abs(grade_u - grade_v) / 10.0
            if grade_v > grade_u:  # 上坡
                grade_cost *= 1.5
                
            # 硬度成本（较软的地面更难通过）
            hardness_cost = 1.0 + (5.0 - min(hardness_u, hardness_v)) / 5.0
            
            # 设置边权重
            G[u][v]['weight'] = 1.0
            G[u][v]['grade_cost'] = grade_cost
            G[u][v]['hardness_cost'] = hardness_cost
            G[u][v]['combined_cost'] = grade_cost * hardness_cost
        
        self._mark_safe_zones(G)  # 先标记安全区
        self._mark_obstacles(G)   # 再生成障碍物
        return G

    def get_keypoint_nodes(self) -> Dict[str, int]:
        """获取关键点对应节点"""
        result = {}
        for name, coord in self.config.key_points.items():
            try:
                node = self.coord_to_node(coord)
                result[name] = node
                logging.debug(f"关键点 {name} 对应节点 {node}, 坐标 {coord}")
            except Exception as e:
                logging.error(f"获取关键点 {name} 节点失败: {str(e)}")
                
        if not result:
            raise MapLoadError("无法获取任何关键点节点")
            
        return result

    def get_operation_nodes(self) -> Dict[str, List[int]]:
        """获取运营节点拓扑关系"""
        key_nodes = self.get_keypoint_nodes()
        
        # 确保所有必需的关键点存在
        required_keys = ['parking', 'load1', 'load2', 'load3', 'unload']
        for key in required_keys:
            if key not in key_nodes:
                raise MapLoadError(f"缺少必需的关键点: {key}")
        
        # 构建运营节点字典
        return {
            'parking': [key_nodes['parking']],
            'loading': [key_nodes[f'load{i}'] for i in range(1, 4) if f'load{i}' in key_nodes],
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
            # 确保两个节点都在图中且可通行
            if start not in self.road_network or end not in self.road_network:
                raise PathOptimizationError(f"节点不在路网中: {start} 或 {end}")
                
            if not self.road_network.nodes[start]['passable'] or not self.road_network.nodes[end]['passable']:
                raise PathOptimizationError(f"节点不可通行: {start} 或 {end}")
                
            # 尝试使用不同的权重计算最短路径
            try:
                path = nx.shortest_path(self.road_network, start, end, weight='grade_cost')
            except nx.NetworkXNoPath:
                # 备选方案：使用普通权重
                path = nx.shortest_path(self.road_network, start, end, weight='weight')
                
            return self._convert_path_coordinates(path)
        except nx.NetworkXNoPath:
            raise PathOptimizationError(f"无法找到路径 {start} -> {end}")
        except Exception as e:
            raise PathOptimizationError(f"路径规划失败: {str(e)}")

    def _convert_path_coordinates(self, path: List[int]) -> List[Tuple]:
        """使用geo_tools转换坐标"""
        return [
            (self.road_network.nodes[n]['x'], self.road_network.nodes[n]['y'])
            for n in path
        ]
        
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
            
            # 创建车辆配置
            vehicle_config = {
                'turning_radius': 10.0,
                'min_hardness': 2.5,
                'current_load': 0 if vehicle_type == 'empty' else 50000
            }
            
            # 优化路径
            try:
                # 先尝试使用完整的车辆对象
                from models.vehicle import MiningVehicle as PlannerVehicle
                vehicle = PlannerVehicle("temp", self, vehicle_config)
                path = planner.plan_path(start, end, vehicle)
            except Exception as e:
                logging.warning(f"HybridPathPlanner规划失败: {str(e)}, 尝试备选方案")
                # 如果失败，尝试直接连接起点和终点
                path = [start, end]
            
            # 计算路径距离
            distance = sum(math.dist(path[i], path[i+1]) for i in range(len(path)-1)) if len(path) > 1 else 0
            
            return {
                'path': path,
                'distance': distance
            }
        except Exception as e:
            logging.error(f"路径规划失败: {str(e)}")
            # 返回直线路径作为备用
            return {
                'path': [start, end],
                'distance': math.dist(start, end)
            }
            
    def validate_path(self, path: List[Tuple]) -> bool:
        """验证路径有效性"""
        if not path or len(path) < 2:
            return False
            
        # 检查路径中点的可通行性
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i+1]
            
            try:
                start_node = self.coord_to_node(start)
                end_node = self.coord_to_node(end)
                
                # 检查节点可通行性
                if not self.road_network.nodes[start_node]['passable'] or not self.road_network.nodes[end_node]['passable']:
                    return False
                    
                # 检查节点连通性
                if not nx.has_path(self.road_network, start_node, end_node):
                    return False
            except Exception:
                # 任何异常都视为路径无效
                return False
                
        return True


