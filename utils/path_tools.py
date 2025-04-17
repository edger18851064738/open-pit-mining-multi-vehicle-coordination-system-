import sys
import os
import logging
import numpy as np
import networkx as nx
from scipy.interpolate import splprep, splev
from typing import List, Tuple, Optional
from functools import lru_cache
import math

from utils.geo_tools import GeoUtils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
logging.basicConfig(level=logging.INFO)

class PathOptimizationError(Exception):
    """路径优化异常基类"""
    pass

class HybridPathPlanner:
    """通用混合路径规划器（虚拟坐标系专用）"""
    
    def __init__(self, geo_utils: GeoUtils):
        self.geo_utils = geo_utils
        self._path_cache = lru_cache(maxsize=100)(self._path_cache_logic)

    def smooth_path(self, path: List[Tuple[float, float]],
                   network: Optional[nx.Graph] = None) -> List[Tuple[float, float]]:
        """路径平滑入口（适配虚拟网格）"""
        try:
            return self._path_cache(tuple(path), network)
        except PathOptimizationError as e:
            logging.warning(f"路径缓存失效: {str(e)}")
            return self._bezier_smoothing(path)

    @lru_cache(maxsize=100)
    def _path_cache_logic(self, path: tuple, network: nx.Graph) -> List[Tuple[float, float]]:
        """带缓存的路径处理核心逻辑"""
        if network:
            try:
                return self._road_network_smoothing(list(path), network)
            except Exception as e:
                raise PathOptimizationError(f"路网平滑失败: {str(e)}")
        return self._bezier_smoothing(list(path))

    def _road_network_smoothing(self, path: List[Tuple[float, float]],
                              network: nx.Graph) -> List[Tuple[float, float]]:
        """基于路网的路径平滑"""
        road_path = []
        for i in range(len(path)-1):
            try:
                start_node = self._find_nearest_node(path[i], network)
                end_node = self._find_nearest_node(path[i+1], network)
                segment = nx.shortest_path(network, start_node, end_node, weight='length')
                road_path.extend([(network.nodes[n]['x'], network.nodes[n]['y']) for n in segment])
            except nx.NetworkXNoPath:
                road_path.extend([path[i], path[i+1]])
        return road_path

    def _find_nearest_node(self, point: Tuple[float, float], 
                          network: nx.Graph) -> int:
        """查找最近路网节点（通用方法）"""
        return min(network.nodes(data=True),
                 key=lambda n: (n[1]['x']-point[0])**2 + (n[1]['y']-point[1])**2)[0]

    def _bezier_smoothing(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """贝塞尔曲线平滑（保持坐标顺序）"""
        if len(path) < 3:
            return path

        x = [p[0] for p in path]
        y = [p[1] for p in path]
        
        try:
            tck, _ = splprep([x, y], s=0, k=min(3, len(path)-1))
            u_new = np.linspace(0, 1, 100)
            x_new, y_new = splev(u_new, tck)
            return list(zip(x_new, y_new))
        except Exception as e:
            logging.error(f"平滑失败: {str(e)}")
            return path

    def optimize_path(self, start: Tuple[float, float], 
                     end: Tuple[float, float],
                     graph: nx.Graph) -> List[Tuple[float, float]]:
        """A*路径规划核心方法"""
        try:
            start_node = self._find_nearest_node(start, graph)
            end_node = self._find_nearest_node(end, graph)
            
            return nx.astar_path(
                graph,
                start_node,
                end_node,
                heuristic=lambda u, v: math.hypot(
                    graph.nodes[v]['x']-graph.nodes[u]['x'],
                    graph.nodes[v]['y']-graph.nodes[u]['y']
                ),
                weight='length'
            )
        except nx.NetworkXNoPath:
            raise PathOptimizationError("路径不可达")
        except Exception as e:
            raise PathOptimizationError(f"规划错误: {str(e)}")

class PathProcessor(HybridPathPlanner):
    """路径处理器（扩展基础功能）"""
    
    def calculate_path_length(self, path: List[Tuple[float, float]]) -> float:
        """
        基于网格坐标系的路径长度计算
        Args:
            path: 路径点列表，每个点为(grid_x, grid_y)网格坐标
        Returns:
            总长度（米），保留两位小数
        """
        total = 0.0
        for i in range(len(path)-1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            total += math.hypot(dx, dy) * self.geo_utils.grid_size
        return round(total, 2)

if __name__ == "__main__":
    # 更新测试用例（使用虚拟坐标系坐标）
    geo_util = GeoUtils()
    processor = PathProcessor(geo_util)
    
    # 测试坐标转换为虚拟网格坐标（示例值）
    test_path = [
        (900, 500),  # parking
        (850, 400),  # 路径点1
        (800, 100)   # load1
    ]
    
    # 平滑测试
    smoothed = processor.smooth_path(test_path)
    print(f"平滑后路径点数: {len(smoothed)}")  # 应该输出100
    

    print(f"路径长度: {processor.calculate_path_length(test_path)}米")