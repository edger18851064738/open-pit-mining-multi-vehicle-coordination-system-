# 新增类型提示和配置项
from typing import Dict, Any
import os
import configparser

class PathConfig:
    """路径规划专用配置"""
    CACHE_EXPIRE = 3600
    MAX_RETRY = 3  # 新增重试机制配置

class MapConfig:
    """地图服务配置"""
    GRID_SIZE = 20.0      # 与path_planner中的grid_size保持一致
    MAX_STEER = 30       
    MAX_GRADE = 15       # 新增最大坡度限制（%）
    MIN_TURN_RADIUS = 15 # 新增最小转弯半径（米）

class VehicleConfig:
    """车辆参数配置"""
    MAX_SPEED = 15.0    
    FUEL_RATE = 0.015   
    MIN_HARDNESS = 2.5   # 新增地面硬度要求
    LOAD_THRESHOLD = 0.8 # 新增负载安全系数

class AppConfig:
    """统一配置入口（增强版）"""
    def __init__(self, config: Dict[str, Any] = None):
        # 设置默认值
        self.path = PathConfig()
        self.map = MapConfig()
        self.vehicle = VehicleConfig()
        
        # 从字典加载配置
        if config:
            self._load_from_dict(config)
            
        # 从环境变量覆盖配置
        self._load_from_env()

    def _load_from_dict(self, config: Dict[str, Any]):
        """从字典加载配置"""
        # 示例配置项映射逻辑
        if 'grid_size' in config:
            self.map.GRID_SIZE = float(config['grid_size'])

    def _load_from_env(self):
        """从环境变量加载配置"""
        if 'MAX_GRADE' in os.environ:
            self.map.MAX_GRADE = float(os.environ['MAX_GRADE'])

# 修改AppConfig的load方法
    @classmethod
    def load(cls, path: str = 'e:/mine/config.ini'):
        """从INI文件加载配置（增强版）"""
        config = configparser.ConfigParser(interpolation=None)  # 禁用插值解析
        config.optionxform = str  # 保持键的大小写
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                config.read_file(f)
        except UnicodeDecodeError:
            with open(path, 'r', encoding='gbk') as f:
                config.read_file(f)
        except FileNotFoundError:
            logging.warning("配置文件未找到，使用默认配置")
            return cls(None)
        
        # 安全获取配置值并处理百分号
# 修改safe_get函数处理注释
    def safe_get(section, option, default):
        raw_value = config.get(section, option, fallback=str(default))
        # 分割实际值和注释
        clean_value = raw_value.split(';')[0].split('#')[0].strip()
        # 移除百分号和其他非数字字符
        numeric_value = ''.join(filter(lambda x: x.isdigit() or x in ('.', '-'), clean_value))
        return float(numeric_value) if numeric_value else float(default)