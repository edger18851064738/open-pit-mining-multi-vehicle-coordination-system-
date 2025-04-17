import logging
import math
import requests
import configparser
from typing import Tuple, Optional, List
import networkx as nx
from functools import lru_cache
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
class GeoUtils:
    """通用地理坐标工具类（不包含业务逻辑）"""
    _instance = None
    
    def __new__(cls, config_path: str = None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.config = configparser.ConfigParser()
            cls._instance._config_path = config_path
            cls._instance._init_config()
            cls._instance._haversine_impl = cls._instance._haversine_impl
            cls._instance.haversine = lru_cache(maxsize=1000)(cls._instance._haversine_impl)
        return cls._instance

    def __init__(self, config_path: str = None):
        """单例模式保持空初始化"""
        pass

    def _init_config(self):
        """通用虚拟坐标配置"""
        try:
            config_path = self._config_path or os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config.ini'
            )
            
            self.config.read(config_path)
            self.grid_size = int(self.config.get('MAP', 'grid_size', fallback='1000'))
            self.origin = (
                int(self.config.getfloat('MAP', 'virtual_origin_x', fallback='0')),
                int(self.config.getfloat('MAP', 'virtual_origin_y', fallback='0'))
            )
            logging.info(f"坐标工具初始化 | 网格尺寸:{self.grid_size} 原点:{self.origin}")

        except (FileNotFoundError, configparser.Error) as e:
            logging.warning(f"使用默认坐标配置: {str(e)}")
            self.grid_size = 100
            self.origin = (0, 0)

    def _haversine_impl(self, coord1: Tuple[int, int], coord2: Tuple[int, int]) -> float:
        """通用距离计算（单位：米）"""
        dx = coord2[0] - coord1[0]
        dy = coord2[1] - coord1[1]
        return math.sqrt(dx**2 + dy**2) * self.grid_size

    def grid_to_metres(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """
        标准网格坐标转米坐标
        Args:
            grid_x: 网格X坐标
            grid_y: 网格Y坐标
        Returns:
            元组(米坐标X, 米坐标Y)
        """
        return (
            (grid_x - self.origin[0]) * self.grid_size,
            (grid_y - self.origin[1]) * self.grid_size
        )

    def metres_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        标准米坐标转网格坐标
        Args:
            x: 米坐标X
            y: 米坐标Y
        Returns:
            元组(网格X坐标, 网格Y坐标)
        """
        return (
            int(round(x / self.grid_size)) + self.origin[0],
            int(round(y / self.grid_size)) + self.origin[1]
        )

    def validate_coordinate_system(self) -> bool:
        """通用坐标系验证"""
        test_point = (100, 200)
        try:
            metres = self.grid_to_metres(*test_point)
            restored = self.metres_to_grid(*metres)
            return restored == test_point
        except Exception as e:
            logging.error(f"坐标系验证失败: {str(e)}")
            return False

    def reload_config(self):
        """动态配置重载"""
        self._init_config()
        logging.info("坐标配置已重载")
        
    @staticmethod
    def bresenham_line(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Bresenham直线算法实现
        :param start: 起点坐标(x1,y1)
        :param end: 终点坐标(x2,y2)
        :return: 直线上的所有点坐标列表
        """
        x1, y1 = start
        x2, y2 = end
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

if __name__ == "__main__":
    # 通用测试用例
    util = GeoUtils()
    print(f"网格转换测试: (50,50) → {util.grid_to_metres(50, 50)}")
    print(f"距离计算测试: {util.haversine((0,0), (3,4)):.1f}米")
    print(f"坐标系验证: {'通过' if util.validate_coordinate_system() else '失败'}")