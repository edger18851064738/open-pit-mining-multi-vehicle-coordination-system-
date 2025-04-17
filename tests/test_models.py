import os
import sys
import unittest
from unittest.mock import Mock
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict
from utils.geo_tools import GeoUtils
from models.vehicle import MiningVehicle
# 在顶部添加缺失导入
import logging
import os
import sys
from threading import RLock
# 添加项目根目录到系统路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.vehicle import MiningVehicle, FuelExhaustedError
from models.task import TransportTask

class VehicleTestCase(unittest.TestCase):
    def setUp(self):
        self.mock_map = Mock()
        self.vehicle = MiningVehicle(1, self.mock_map, {
            'fuel_consumption_rate': 0.02,
            'maintenance_thresholds': {'mileage': 5000}
        })

    def test_fuel_consumption(self):
        initial_fuel = self.vehicle.current_fuel
        self.vehicle.update_position((31.63, 118.95))
        self.assertLess(self.vehicle.current_fuel, initial_fuel)

    def test_concurrent_task_error(self):
        self.vehicle.assign_task('t1', [])
        with self.assertRaises(ConcurrentTaskError):
            self.vehicle.assign_task('t2', [])

class TaskTestCase(unittest.TestCase):
    def test_status_flow(self):
        task = TransportTask("t1", [(31.63, 118.95)], 1000)
        self.assertTrue(task._validate_status_transition('assigned'))
        self.assertFalse(task._validate_status_transition('completed'))