"""
Enhanced Integration Solution for Multi-Vehicle Collaborative Dispatch System
===============================================================

This module provides improved integration between components of the mining vehicle
dispatch system, with focus on:

1. Improved error handling and system resilience
2. Enhanced conflict detection and resolution
3. Centralized configuration management 
4. Optimized path planning with fallback strategies
5. Deadlock detection and prevention
"""
import sys
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import logging
from typing import Dict, List, Tuple, Optional, Set, Union, Any, TYPE_CHECKING
import math
import threading
import time
import configparser
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dispatch_integration")

# Import local modules
from models.vehicle import MiningVehicle, VehicleState, TransportStage
from models.task import TransportTask
from algorithm.map_service import MapService
from algorithm.path_planner import HybridPathPlanner
from algorithm.dispatch_service import TransportScheduler, ConflictBasedSearch, DispatchSystem
from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError


class DispatchSystemError(Exception):
    """Base exception class for dispatch system errors"""
    pass


class PathPlanningError(DispatchSystemError):
    """Exception raised for path planning errors"""
    pass


class ConfigurationError(DispatchSystemError):
    """Exception raised for configuration errors"""
    pass


class IntegratedConfig:
    """Centralized configuration management for the dispatch system"""
    
    def __init__(self, config_path=None):
        """Initialize configuration with optional config path"""
        self.config = configparser.ConfigParser()
        self._load_defaults()
        
        # Load from file if available
        if config_path and os.path.exists(config_path):
            try:
                self.config.read(config_path)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load configuration: {str(e)}")
    
    def _load_defaults(self):
        """Load default configuration values"""
        # MAP section
        self.config['MAP'] = {
            'grid_size': '200',
            'grid_nodes': '50',
            'safe_radius': '30',
            'obstacle_density': '0.15',
            'data_type': 'virtual',
            'virtual_origin_x': '0',
            'virtual_origin_y': '0',
            'max_grade': '15.0',
            'min_turn_radius': '15.0'
        }
        
        # DISPATCH section
        self.config['DISPATCH'] = {
            'loading_points': '[(-100,50), (0,150), (100,50)]',
            'unloading_point': '(0,-100)',
            'parking_area': '(200,200)',
            'max_charging_vehicles': '2',
            'scheduling_interval': '2.0',
            'conflict_resolution_method': 'priority',
            'task_assignment_method': 'nearest'
        }
        
        # VEHICLE section
        self.config['VEHICLE'] = {
            'default_speed': '5.0',
            'default_capacity': '50',
            'default_hardness': '2.5',
            'default_turning_radius': '10.0',
            'battery_capacity': '100.0',
            'power_consumption': '2.0',
            'maintenance_interval': '500'
        }
        
        # SIMULATION section
        self.config['SIMULATION'] = {
            'simulation_mode': 'True',
            'num_vehicles': '5',
            'simulation_speed': '2.0',
            'task_generation_rate': '0.2',
            'visualization_interval': '0.5',
            'scheduling_interval': '2.0',
            'random_seed': '42'
        }
        
        # INTEGRATION section (new)
        self.config['INTEGRATION'] = {
            'monitor_interval': '60',
            'path_cache_size': '1000',
            'path_cache_ttl': '600',
            'conflict_check_interval': '0.5',
            'deadlock_threshold': '3',
            'performance_log_interval': '300'
        }
        
        # LOGGING section
        self.config['LOGGING'] = {
            'level': 'INFO',
            'console_output': 'True',
            'file_output': 'True',
            'log_file': 'dispatch.log',
            'rotate_logs': 'True',
            'max_file_size': '10485760',
            'backup_count': '5'
        }
    
    def get(self, section: str, key: str, fallback: Any = None) -> str:
        """Get string configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def getint(self, section: str, key: str, fallback: Optional[int] = None) -> int:
        """Get integer configuration value"""
        return self.config.getint(section, key, fallback=fallback)
    
    def getfloat(self, section: str, key: str, fallback: Optional[float] = None) -> float:
        """Get float configuration value"""
        return self.config.getfloat(section, key, fallback=fallback)
    
    def getboolean(self, section: str, key: str, fallback: Optional[bool] = None) -> bool:
        """Get boolean configuration value"""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def getlist(self, section: str, key: str, fallback: Optional[List] = None) -> List:
        """Get list configuration value (parses string lists)"""
        try:
            value = self.config.get(section, key)
            return eval(value)
        except (SyntaxError, NameError, configparser.NoOptionError, configparser.NoSectionError):
            return fallback if fallback is not None else []
    
    def gettuple(self, section: str, key: str, fallback: Optional[Tuple] = None) -> Tuple:
        """Get tuple configuration value (parses string tuples)"""
        try:
            value = self.config.get(section, key)
            return eval(value)
        except (SyntaxError, NameError, configparser.NoOptionError, configparser.NoSectionError):
            return fallback if fallback is not None else ()
    
    def save(self, path: str) -> bool:
        """Save current configuration to file"""
        try:
            with open(path, 'w') as file:
                self.config.write(file)
            logger.info(f"Configuration saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False


class SpatialIndex:
    """Spatial indexing for efficient conflict detection"""
    
    def __init__(self, cell_size=20):
        """Initialize spatial index with given cell size"""
        self.cell_size = cell_size
        self.point_grid = defaultdict(set)  # (cell_x, cell_y) -> set of points
        self.segment_grid = defaultdict(list)  # (cell_x, cell_y) -> list of (vid, segment)
        
    def clear(self):
        """Clear all spatial data"""
        self.point_grid.clear()
        self.segment_grid.clear()
        
    def add_point(self, point, data=None):
        """Add a point to the spatial index"""
        cell_x, cell_y = self._get_cell(point)
        if data:
            self.point_grid[(cell_x, cell_y)].add((point, data))
        else:
            self.point_grid[(cell_x, cell_y)].add(point)
        
    def add_segment(self, segment, vid):
        """Add a line segment to the spatial index"""
        cells = self._get_segment_cells(segment)
        for cell in cells:
            self.segment_grid[cell].append((vid, segment))
            
    def query_point(self, point, radius=0):
        """Query points near the given point"""
        results = set()
        center_cell = self._get_cell(point)
        
        # Determine cell radius
        cell_radius = max(1, math.ceil(radius / self.cell_size))
        
        # Search neighboring cells
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell = (center_cell[0] + dx, center_cell[1] + dy)
                if cell in self.point_grid:
                    for p in self.point_grid[cell]:
                        if radius == 0 or self._distance(point, p if isinstance(p, tuple) and len(p) == 2 else p[0]) <= radius:
                            results.add(p)
        
        return results
        
    def query_nearby_segments(self, segment):
        """Query segments that might intersect with the given segment"""
        cells = self._get_segment_cells(segment)
        results = []
        seen_vids = set()
        
        for cell in cells:
            if cell in self.segment_grid:
                for vid, seg in self.segment_grid[cell]:
                    # Avoid duplicate segments from same vehicle
                    if vid not in seen_vids:
                        results.append((vid, seg))
                        seen_vids.add(vid)
        
        return results
        
    def _get_cell(self, point):
        """Get the cell coordinates for a point"""
        return (int(point[0] // self.cell_size), int(point[1] // self.cell_size))
        
    def _get_segment_cells(self, segment):
        """Get all cells that a segment passes through"""
        (x1, y1), (x2, y2) = segment
        
        # Get the cells of the endpoints
        cell1 = self._get_cell((x1, y1))
        cell2 = self._get_cell((x2, y2))
        
        # If they're in the same cell, just return that cell
        if cell1 == cell2:
            return [cell1]
            
        # Otherwise, use Bresenham's algorithm to get cells
        cells = set([cell1, cell2])
        
        # Compute Bresenham line
        dx = abs(cell2[0] - cell1[0])
        dy = abs(cell2[1] - cell1[1])
        sx = 1 if cell1[0] < cell2[0] else -1
        sy = 1 if cell1[1] < cell2[1] else -1
        err = dx - dy
        
        x, y = cell1
        while (x, y) != cell2:
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
            cells.add((x, y))
            
        return list(cells)
        
    def _distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


class IntegratedDispatchSystem:
    """
    Enhanced integration layer for the mining dispatch system components.
    
    This class wraps the standard DispatchSystem with improved error handling,
    component integration, and monitoring capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None, 
                map_service: Optional[MapService] = None, 
                planner: Optional[HybridPathPlanner] = None):
        """Initialize the integrated dispatch system with proper component connections."""
        # Load configuration
        self.config = IntegratedConfig(config_path)
        
        # Configure logging based on config
        self._configure_logging()
        
        # Create core components if not provided
        self.map_service = map_service or MapService()
        self.planner = planner or HybridPathPlanner(self.map_service)
        
        # Create a safe wrapper for path planning
        self._patch_path_planner()
        
        # Initialize the dispatch system
        self.dispatch = DispatchSystem(self.planner, self.map_service)
        
        # Ensure the planner has a dispatch reference
        self.planner.dispatch = self.dispatch
        
        # Initialize conflict resolver
        self.conflict_resolver = EnhancedConflictResolution(self)
        
        # Enhance the scheduling cycle
        self._enhance_scheduling_cycle()
        
        # Monitor locks
        self.monitor_lock = threading.RLock()
        
        # System state
        self.is_running = False
        self.dispatch_thread = None
        self.monitor_thread = None
        self.performance_thread = None
        
        # Monitoring stats
        self.stats = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'tasks_assigned': 0,
            'tasks_completed': 0,
            'path_planning_failures': 0,
            'deadlocks_detected': 0,
            'deadlocks_resolved': 0,
            'system_start_time': None,
            'last_monitor_time': None
        }
        
        logger.info("Integrated dispatch system initialized")

    def _configure_logging(self):
        """Configure logging based on configuration"""
        log_level_str = self.config.get('LOGGING', 'level', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
            
        # Console handler
        console_output = self.config.getboolean('LOGGING', 'console_output', True)
        if console_output:
            console = logging.StreamHandler()
            console.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console.setFormatter(formatter)
            root_logger.addHandler(console)
            
        # File handler
        file_output = self.config.getboolean('LOGGING', 'file_output', True)
        if file_output:
            log_file = self.config.get('LOGGING', 'log_file', 'dispatch.log')
            rotate_logs = self.config.getboolean('LOGGING', 'rotate_logs', True)
            
            if rotate_logs:
                from logging.handlers import RotatingFileHandler
                max_size = self.config.getint('LOGGING', 'max_file_size', 10485760)
                backup_count = self.config.getint('LOGGING', 'backup_count', 5)
                
                file_handler = RotatingFileHandler(
                    log_file, maxBytes=max_size, backupCount=backup_count
                )
            else:
                file_handler = logging.FileHandler(log_file)
                
            file_handler.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

    def _patch_path_planner(self):
        """Create a safer version of path planning with fallback options."""
        # Save the original plan_path method
        if not hasattr(self.planner, 'original_plan_path'):
            self.planner.original_plan_path = self.planner.plan_path
        
        # Replace with our enhanced version
        def safe_plan_path(start, end, vehicle=None):
            """Enhanced path planning with fallbacks for error resilience."""
            try:
                # Try the original method first
                return self.planner.original_plan_path(start, end, vehicle)
            except Exception as e:
                logger.warning(f"Original path planning failed: {str(e)}, using fallback")
                self.stats['path_planning_failures'] += 1
                
                # Try basic A* without vehicle constraints
                try:
                    if hasattr(self.planner, '_fast_astar'):
                        path = self.planner._fast_astar(start, end)
                        if path and len(path) > 1:
                            logger.info("Used _fast_astar fallback for path planning")
                            return path
                except Exception as e:
                    logger.warning(f"Fast A* fallback failed: {str(e)}")
                
                # Simple straight line path as final fallback
                if isinstance(start, tuple) and isinstance(end, tuple):
                    # Generate intermediate points for smoother path
                    dx = (end[0] - start[0]) / 5
                    dy = (end[1] - start[1]) / 5
                    path = [start]
                    for i in range(1, 5):
                        path.append((start[0] + dx * i, start[1] + dy * i))
                    path.append(end)
                    logger.info("Used straight line fallback for path planning")
                    return path
                return [start, end]
        
        # Apply the patch
        self.planner.plan_path = safe_plan_path
        logger.info("Path planner patched with safe fallback")
    
    def _enhance_scheduling_cycle(self):
        """Enhance the scheduling cycle with additional functionality"""
        # Save original method
        original_scheduling_cycle = self.dispatch.scheduling_cycle
        
        # Enhanced version
        def enhanced_scheduling_cycle():
            """Enhanced scheduling cycle with integrated conflict detection."""
            try:
                # Run the original cycle first
                original_scheduling_cycle()
                
                # Then run our enhanced conflict detection
                conflicts_resolved = self.conflict_resolver.detect_and_resolve_all_conflicts()
                
                # Update stats
                if conflicts_resolved > 0:
                    self.stats['conflicts_detected'] += conflicts_resolved
                    self.stats['conflicts_resolved'] += conflicts_resolved
                    logger.info(f"Resolved {conflicts_resolved} conflicts in enhanced cycle")
                    
                # Check for deadlocks
                deadlocks = self.conflict_resolver.deadlock_detector.check_persistent_deadlocks()
                if deadlocks:
                    self.stats['deadlocks_detected'] += len(deadlocks)
                    
                    # Attempt to resolve deadlocks
                    for vid in deadlocks:
                        if self._resolve_deadlock(vid):
                            self.stats['deadlocks_resolved'] += 1
                            
            except Exception as e:
                logger.error(f"Enhanced scheduling cycle error: {str(e)}")
        
        # Apply the enhancement
        self.dispatch.scheduling_cycle = enhanced_scheduling_cycle
        logger.info("Enhanced scheduling cycle applied with dynamic conflict resolution")

    def _resolve_deadlock(self, vehicle_id) -> bool:
        """Attempt to resolve a deadlock for a specific vehicle"""
        try:
            if isinstance(vehicle_id, str) and vehicle_id.isdigit():
                vehicle_id = int(vehicle_id)
                
            if vehicle_id not in self.dispatch.vehicles:
                return False
                
            vehicle = self.dispatch.vehicles[vehicle_id]
            
            # Strategy 1: If vehicle has a task, reset its path
            if vehicle.current_task and hasattr(vehicle, 'current_path'):
                # Replan from current position to task endpoint
                end_point = vehicle.current_task.end_point
                new_path = self.planner.plan_path(vehicle.current_location, end_point, vehicle)
                
                if new_path and len(new_path) > 1:
                    # Add some randomness to help break the deadlock
                    mid_point = (
                        (vehicle.current_location[0] + end_point[0]) / 2 + (20 * (0.5 - random.random())),
                        (vehicle.current_location[1] + end_point[1]) / 2 + (20 * (0.5 - random.random()))
                    )
                    
                    # Try to route through the midpoint
                    path1 = self.planner.plan_path(vehicle.current_location, mid_point)
                    path2 = self.planner.plan_path(mid_point, end_point)
                    
                    if path1 and path2 and len(path1) > 1 and len(path2) > 1:
                        combined_path = path1[:-1] + path2
                        vehicle.assign_path(combined_path)
                        logger.info(f"Resolved deadlock for vehicle {vehicle_id} with rerouting through random midpoint")
                        return True
                    else:
                        vehicle.assign_path(new_path)
                        logger.info(f"Resolved deadlock for vehicle {vehicle_id} with direct rerouting")
                        return True
            
            # Strategy 2: If vehicle is stuck, perform emergency stop then resume
            if vehicle.state != VehicleState.EMERGENCY_STOP:
                # Emergency stop
                if hasattr(vehicle, 'perform_emergency_stop'):
                    vehicle.perform_emergency_stop()
                    
                    # Resume after a short delay
                    def delayed_resume():
                        time.sleep(2)  # 2 second delay
                        if hasattr(vehicle, 'resume_operation'):
                            vehicle.resume_operation()
                            logger.info(f"Resumed vehicle {vehicle_id} after emergency stop")
                    
                    # Start in background thread
                    threading.Thread(target=delayed_resume, daemon=True).start()
                    logger.info(f"Resolved deadlock for vehicle {vehicle_id} with emergency stop")
                    return True
            
            logger.warning(f"Failed to resolve deadlock for vehicle {vehicle_id}")
            return False
        except Exception as e:
            logger.error(f"Deadlock resolution error for vehicle {vehicle_id}: {str(e)}")
            return False

    def register_vehicle(self, vehicle: MiningVehicle):
        """Register a new vehicle with the dispatch system."""
        # Initialize required attributes if missing
        if not hasattr(vehicle, 'current_path'):
            vehicle.current_path = []
        if not hasattr(vehicle, 'path_index'):
            vehicle.path_index = 0
        
        # Ensure metrics dict exists
        if not hasattr(vehicle, 'metrics'):
            vehicle.metrics = {
                'tasks_completed': 0,
                'total_distance': 0.0,
                'waiting_time': 0.0,
                'conflicts': 0
            }
        
        # Add to dispatch system
        self.dispatch.vehicles[vehicle.vehicle_id] = vehicle
        logger.info(f"Vehicle {vehicle.vehicle_id} registered with dispatch system")

    def add_task(self, task: TransportTask):
        """Add a transport task to the dispatch queue."""
        # Task validation
        if not hasattr(task, 'is_completed'):
            task.is_completed = False
        if not hasattr(task, 'assigned_to'):
            task.assigned_to = None
        
        # Add to dispatch queue
        self.dispatch.add_task(task)
        self.stats['tasks_assigned'] += 1
        logger.info(f"Task {task.task_id} added to dispatch queue")

    def start_dispatch_service(self, dispatch_interval: int = None):
        """Start the automated dispatch service."""
        if self.is_running:
            logger.warning("Dispatch service is already running")
            return
        
        # Use configured interval if not specified
        if dispatch_interval is None:
            dispatch_interval = self.config.getfloat('DISPATCH', 'scheduling_interval', 2.0)
        
        self.is_running = True
        self.stats['system_start_time'] = datetime.now()
        self.stats['last_monitor_time'] = datetime.now()
        
        # Start dispatch thread
        def dispatch_loop():
            while self.is_running:
                try:
                    self.dispatch.scheduling_cycle()
                    time.sleep(dispatch_interval)
                except Exception as e:
                    logger.error(f"Error in dispatch cycle: {str(e)}")
                    time.sleep(1)  # Shorter sleep on error
        
        self.dispatch_thread = threading.Thread(target=dispatch_loop, daemon=True)
        self.dispatch_thread.start()
        
        # Start monitoring thread
        def monitor_loop():
            monitor_interval = self.config.getint('INTEGRATION', 'monitor_interval', 60)
            while self.is_running:
                try:
                    self._update_system_stats()
                    time.sleep(monitor_interval)
                except Exception as e:
                    logger.error(f"Error in monitoring: {str(e)}")
                    time.sleep(5)  # Shorter sleep on error
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start performance logging thread
        def performance_log_loop():
            log_interval = self.config.getint('INTEGRATION', 'performance_log_interval', 300)
            while self.is_running:
                try:
                    self._log_performance_metrics()
                    time.sleep(log_interval)
                except Exception as e:
                    logger.error(f"Error in performance logging: {str(e)}")
                    time.sleep(30)  # Shorter sleep on error
        
        self.performance_thread = threading.Thread(target=performance_log_loop, daemon=True)
        self.performance_thread.start()
        
        logger.info(f"Dispatch service started with {dispatch_interval}s cycle")

    def stop_dispatch_service(self):
        """Stop the automated dispatch service."""
        if not self.is_running:
            logger.warning("Dispatch service is not running")
            return
        
        self.is_running = False
        
        # Wait for threads to terminate
        threads = [
            (self.dispatch_thread, "dispatch"),
            (self.monitor_thread, "monitor"),
            (self.performance_thread, "performance")
        ]
        
        for thread, name in threads:
            if thread:
                logger.debug(f"Waiting for {name} thread to terminate...")
                thread.join(timeout=5)
                if thread.is_alive():
                    logger.warning(f"{name.capitalize()} thread did not terminate within timeout")
            
        logger.info("Dispatch service stopped")

    def _update_system_stats(self):
        """Update system monitoring statistics."""
        with self.monitor_lock:
            # Calculate time since last update
            now = datetime.now()
            time_since_last = (now - self.stats['last_monitor_time']).total_seconds()
            self.stats['last_monitor_time'] = now
            
            # Count vehicles by state
            vehicle_states = {}
            for v in self.dispatch.vehicles.values():
                state_name = v.state.name if hasattr(v.state, 'name') else str(v.state)
                vehicle_states[state_name] = vehicle_states.get(state_name, 0) + 1
            
            # Count completed tasks since last update
            completed_count = len(self.dispatch.completed_tasks)
            if 'prev_completed_count' in self.stats:
                new_completed = completed_count - self.stats['prev_completed_count']
                if new_completed > 0:
                    self.stats['tasks_completed'] += new_completed
            self.stats['prev_completed_count'] = completed_count
            
            # Update stats
            self.stats['vehicle_states'] = vehicle_states
            self.stats['active_tasks'] = len(self.dispatch.active_tasks)
            self.stats['queued_tasks'] = len(self.dispatch.task_queue)
            self.stats['completed_task_count'] = completed_count
            self.stats['uptime_seconds'] = (now - self.stats['system_start_time']).total_seconds()
            
            # Calculate task throughput (tasks per minute)
            if time_since_last > 0:
                throughput = (new_completed * 60) / time_since_last
                self.stats['task_throughput'] = throughput
            
            logger.debug(f"System stats updated: Active tasks={self.stats['active_tasks']}, "
                         f"Queued tasks={self.stats['queued_tasks']}, "
                         f"Vehicle states={vehicle_states}")

    def _log_performance_metrics(self):
        """Log detailed performance metrics"""
        with self.monitor_lock:
            # Get path planner metrics
            path_metrics = {}
            if hasattr(self.planner, 'get_performance_stats'):
                try:
                    path_metrics = self.planner.get_performance_stats()
                except Exception as e:
                    logger.warning(f"Failed to get path planner metrics: {str(e)}")
            
            # Log comprehensive performance data
            uptime = self.stats['uptime_seconds']
            uptime_hours = uptime / 3600
            
            logger.info(
                f"Performance Report - Uptime: {uptime_hours:.2f}h | "
                f"Tasks: {self.stats['tasks_completed']}/{self.stats['tasks_assigned']} | "
                f"Conflicts: {self.stats['conflicts_detected']} detected, {self.stats['conflicts_resolved']} resolved | "
                f"Deadlocks: {self.stats['deadlocks_detected']} detected, {self.stats['deadlocks_resolved']} resolved | "
                f"Path planning failures: {self.stats['path_planning_failures']}"
            )
            
            # Log additional planning metrics if available
            if path_metrics:
                logger.info(
                    f"Path Planning Metrics - "
                    f"Avg planning time: {path_metrics.get('avg_planning_time', 'N/A')} | "
                    f"Cache hit rate: {path_metrics.get('cache_stats', {}).get('hit_rate', 0):.2f} | "
                    f"Obstacle count: {path_metrics.get('obstacle_count', 0)}"
                )

    def get_system_status(self) -> Dict:
        """Get the current system status."""
        with self.monitor_lock:
            # Ensure active_tasks and completed_tasks exist
            active_tasks = getattr(self.dispatch, 'active_tasks', {})
            completed_tasks = getattr(self.dispatch, 'completed_tasks', {})
            
            # Build detailed vehicle status information
            vehicle_details = {}
            for vid, v in self.dispatch.vehicles.items():
                # Basic info
                vehicle_info = {
                    'state': v.state.name if hasattr(v.state, 'name') else str(v.state),
                    'location': v.current_location,
                    'has_task': v.current_task is not None,
                }
                
                # Add transport stage if available
                if hasattr(v, 'transport_stage') and v.transport_stage:
                    vehicle_info['transport_stage'] = (
                        v.transport_stage.name 
                        if hasattr(v.transport_stage, 'name') 
                        else str(v.transport_stage)
                    )
                
                # Add current load if available
                if hasattr(v, 'current_load'):
                    vehicle_info['current_load'] = v.current_load
                    if hasattr(v, 'max_capacity'):
                        vehicle_info['load_percentage'] = (
                            v.current_load / v.max_capacity * 100 
                            if v.max_capacity > 0 
                            else 0
                        )
                
                # Add task info if available
                if v.current_task:
                    vehicle_info['task'] = {
                        'id': v.current_task.task_id,
                        'type': getattr(v.current_task, 'task_type', 'unknown'),
                        'priority': getattr(v.current_task, 'priority', 0),
                        'progress': getattr(v.current_task, 'progress', 0),
                    }
                    
                # Add metrics if available
                if hasattr(v, 'metrics'):
                    vehicle_info['metrics'] = v.metrics
                
                vehicle_details[vid] = vehicle_info
                        
            return {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'vehicles': vehicle_details,
                'active_tasks': list(active_tasks.keys()) if active_tasks else [],
                'completed_tasks': list(completed_tasks.keys()) if completed_tasks else [],
                'system_health': self._get_system_health()
            }
            
    def _get_system_health(self) -> Dict:
        """Get system health indicators"""
        health = {
            'status': 'healthy',
            'components': {
                'dispatch': 'operational',
                'path_planning': 'operational',
                'task_management': 'operational',
                'conflict_resolution': 'operational'
            },
            'warnings': [],
            'errors': []
        }
        
        # Check for path planning issues
        if self.stats.get('path_planning_failures', 0) > 10:
            health['components']['path_planning'] = 'degraded'
            health['warnings'].append('High rate of path planning failures')
            health['status'] = 'degraded'
        
        # Check for conflict resolution issues
        conflict_resolution_rate = (
            self.stats.get('conflicts_resolved', 0) / 
            max(1, self.stats.get('conflicts_detected', 0))
        )
        if conflict_resolution_rate < 0.8 and self.stats.get('conflicts_detected', 0) > 5:
            health['components']['conflict_resolution'] = 'degraded'
            health['warnings'].append('Low conflict resolution success rate')
            health['status'] = 'degraded'
        
        # Check for deadlock issues
        if self.stats.get('deadlocks_detected', 0) > 5:
            health['warnings'].append('High number of deadlocks detected')
            if self.stats.get('deadlocks_resolved', 0) / max(1, self.stats.get('deadlocks_detected', 0)) < 0.5:
                health['components']['conflict_resolution'] = 'critical'
                health['errors'].append('Poor deadlock resolution performance')
                health['status'] = 'critical'
        
        # Check overall system performance
        if self.stats.get('tasks_completed', 0) == 0 and self.stats.get('tasks_assigned', 0) > 5:
            health['components']['task_management'] = 'critical'
            health['errors'].append('No tasks being completed despite assignments')
            health['status'] = 'critical'
            
        return health

    def direct_dispatch(self, vehicle_id: int, destination: Tuple[float, float]):
        """Directly dispatch a vehicle to a specific location."""
        try:
            self.dispatch.dispatch_vehicle_to(vehicle_id, destination)
            self.stats['tasks_assigned'] += 1
            return True
        except Exception as e:
            logger.error(f"Direct dispatch failed: {str(e)}")
            return False

    def refresh_paths(self):
        """Force refresh of all vehicle paths to resolve conflicts."""
        try:
            # Collect current vehicle paths
            all_paths = {}
            for vid, vehicle in self.dispatch.vehicles.items():
                if hasattr(vehicle, 'current_path') and vehicle.current_path:
                    all_paths[str(vid)] = vehicle.current_path
            
            if not all_paths:
                logger.info("No paths to refresh")
                return
            
            # Resolve conflicts
            resolved_paths = self.dispatch.cbs.resolve_conflicts(all_paths)
            
            # Update vehicle paths
            with self.dispatch.vehicle_lock:
                for vid_str, path in resolved_paths.items():
                    try:
                        vid = int(vid_str) if vid_str.isdigit() else vid_str
                        if path and vid in self.dispatch.vehicles:
                            self.dispatch.vehicles[vid].assign_path(path)
                            logger.debug(f"Refreshed path for vehicle {vid}")
                    except Exception as e:
                        logger.error(f"Path refresh failed for vehicle {vid}: {str(e)}")
                        
            logger.info(f"Refreshed paths for {len(resolved_paths)} vehicles")
            return True
        except Exception as e:
            logger.error(f"Path refresh failed: {str(e)}")
            return False

    def get_vehicle_recommendations(self, task_id: str) -> List[int]:
        """Get recommended vehicles for a specific task based on current state."""
        if task_id not in self.dispatch.active_tasks and task_id not in self.dispatch.task_queue:
            logger.warning(f"Task {task_id} not found")
            return []
        
        # Find the task
        task = None
        for t in self.dispatch.task_queue:
            if t.task_id == task_id:
                task = t
                break
        
        if not task:
            for tid, t in self.dispatch.active_tasks.items():
                if tid == task_id:
                    task = t
                    break
        
        if not task:
            return []
        
        # Rank vehicles by suitability
        ranked_vehicles = []
        for vid, vehicle in self.dispatch.vehicles.items():
            # Skip vehicles with tasks
            if vehicle.current_task:
                continue
                
            # Calculate distance to task start
            try:
                distance = math.hypot(
                    vehicle.current_location[0] - task.start_point[0],
                    vehicle.current_location[1] - task.start_point[1]
                )
                
                # Calculate a score (lower is better)
                score = distance
                
                # Adjust for vehicle state
                if vehicle.state == VehicleState.IDLE:
                    score *= 0.8  # Prefer idle vehicles
                    
                ranked_vehicles.append((vid, score))
            except Exception as e:
                logger.error(f"Recommendation calculation failed: {str(e)}")
                
        # Sort by score (lower is better)
        ranked_vehicles.sort(key=lambda x: x[1])
        
        # Return vehicle IDs
        return [vid for vid, _ in ranked_vehicles[:3]]  # Top 3 recommendations

    def clear_completed_tasks(self):
        """Clear completed tasks to free up memory."""
        count = len(self.dispatch.completed_tasks)
        self.dispatch.completed_tasks.clear()
        logger.info(f"Cleared {count} completed tasks")
        return count


class EnhancedConflictResolution:
    """
    Enhanced conflict resolution algorithm with improved prioritization
    and deadlock prevention.
    """
    
    def __init__(self, dispatch_system: IntegratedDispatchSystem):
        """Initialize with a reference to the dispatch system."""
        self.dispatch = dispatch_system.dispatch
        self.integrated_system = dispatch_system
        self.vehicle_priorities = {}
        self.deadlock_detector = DeadlockDetector()
        self.last_check_time = time.time()
        self.spatial_index = SpatialIndex(cell_size=20)
        
        # Configuration parameters
        config = dispatch_system.config
        self.conflict_check_interval = config.getfloat(
            'INTEGRATION', 'conflict_check_interval', 0.5
        )
        
    def detect_and_resolve_all_conflicts(self) -> int:
        """Detect and resolve all conflicts in the system."""
        # Only run periodically to improve performance
        current_time = time.time()
        if current_time - self.last_check_time < self.conflict_check_interval:
            return 0
            
        self.last_check_time = current_time
        
        # Collect all vehicle paths with current positions
        all_paths = {}
        for vid, vehicle in self.dispatch.vehicles.items():
            if (hasattr(vehicle, 'current_path') and vehicle.current_path and 
                hasattr(vehicle, 'path_index')):
                # Only consider the remaining path from current position
                remaining_path = vehicle.current_path[vehicle.path_index:]
                if len(remaining_path) > 1:
                    all_paths[str(vid)] = remaining_path
        
        if not all_paths:
            return 0
            
        # Update spatial index for efficient conflict detection
        self._update_spatial_index(all_paths)
            
        # Enhanced conflict detection with spatial indexing
        conflicts = self._detect_conflicts_optimized(all_paths)
        
        if not conflicts:
            return 0
            
        # Check for potential deadlocks
        deadlocked_vehicles = self.deadlock_detector.check_deadlocks(
            conflicts, self.dispatch.vehicles
        )
        
        # Update priorities with deadlock information
        for vid in deadlocked_vehicles:
            self.vehicle_priorities[vid] = -1  # Lowest priority to break deadlocks
            self.integrated_system.stats['deadlocks_detected'] += 1
        
        # Resolve conflicts with priorities
        resolved_count = self._resolve_conflicts_with_spatial_awareness(conflicts, all_paths)
        
        return resolved_count
        
    def _update_spatial_index(self, paths):
        """Update spatial index with current path segments"""
        # Clear previous index
        self.spatial_index.clear()
        
        # Add all path segments to the index
        for vid, path in paths.items():
            for i in range(len(path) - 1):
                segment = (path[i], path[i+1])
                self.spatial_index.add_segment(segment, vid)
                
    def _detect_conflicts_optimized(self, paths):
        """
        Optimized conflict detection using spatial indexing
        """
        conflicts = []
        checked_pairs = set()
        
        # Check each path against potentially conflicting paths
        for vid1, path1 in paths.items():
            vehicle1 = self._get_vehicle(vid1)
            speed1 = getattr(vehicle1, 'max_speed', 5.0)
            
            # Check each segment of the path
            for i in range(len(path1) - 1):
                segment1 = (path1[i], path1[i+1])
                
                # Get potentially conflicting segments
                nearby_segments = self.spatial_index.query_nearby_segments(segment1)
                
                # Check for conflicts with nearby segments
                for vid2, segment2 in nearby_segments:
                    # Skip self-conflicts
                    if vid1 == vid2:
                        continue
                        
                    # Skip already checked pairs
                    pair_key = tuple(sorted([str(vid1), str(vid2)]))
                    if pair_key in checked_pairs:
                        continue
                    checked_pairs.add(pair_key)
                    
                    # Get second vehicle info
                    vehicle2 = self._get_vehicle(vid2)
                    speed2 = getattr(vehicle2, 'max_speed', 5.0)
                    path2 = paths.get(str(vid2), [])
                    
                    if not path2:
                        continue
                    
                    # Check for position conflicts at similar indices
                    min_len = min(len(path1), len(path2))
                    for t in range(min_len):
                        if path1[t] == path2[t]:
                            # If vehicles would be at the same point at similar times
                            time1 = t / speed1
                            time2 = t / speed2
                            
                            if abs(time1 - time2) < 1.0:
                                conflicts.append({
                                    'type': 'position',
                                    'time_index': t,
                                    'position': path1[t],
                                    'vehicle1': vid1,
                                    'vehicle2': vid2,
                                    'time1': time1,
                                    'time2': time2
                                })
                    
                    # Find segment conflicts
                    if self._segments_intersect(segment1, segment2):
                        # Find indices of segments in paths
                        idx1 = i
                        idx2 = next((j for j in range(len(path2)-1) 
                                    if (path2[j], path2[j+1]) == segment2), 0)
                        
                        # Calculate time estimates
                        time1 = idx1 / speed1
                        time2 = idx2 / speed2
                        
                        if abs(time1 - time2) < 1.0:  # Within 1 time unit
                            conflicts.append({
                                'type': 'crossing',
                                'segment1': segment1,
                                'segment2': segment2,
                                'vehicle1': vid1,
                                'vehicle2': vid2,
                                'time1': time1,
                                'time2': time2
                            })
        
        logger.debug(f"Detected {len(conflicts)} conflicts using optimized detection")
        return conflicts

    def _segments_intersect(self, seg1, seg2) -> bool:
        """Check if two line segments intersect."""
        (x1, y1), (x2, y2) = seg1
        (x3, y3), (x4, y4) = seg2
        
        # Calculate direction vectors
        dx1 = x2 - x1
        dy1 = y2 - y1
        dx2 = x4 - x3
        dy2 = y4 - y3
        
        # Calculate the determinant
        det = dx1 * dy2 - dy1 * dx2
        
        # Lines are parallel if det is zero
        if abs(det) < 1e-6:
            return False
            
        # Calculate parameters for intersection point
        t1 = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / det
        t2 = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / det
        
        # Check if intersection point is within both segments
        return 0 <= t1 <= 1 and 0 <= t2 <= 1
    
    def _get_vehicle(self, vid):
        """Get vehicle by string or int ID."""
        try:
            if isinstance(vid, str) and vid.isdigit():
                vid = int(vid)
            return self.dispatch.vehicles.get(vid)
        except:
            return None
            
    def _get_vehicle_priority(self, vid) -> int:
        """Get priority for a vehicle with enhancements for special cases."""
        # Check manual override first
        if vid in self.vehicle_priorities:
            return self.vehicle_priorities[vid]
            
        vehicle = self._get_vehicle(vid)
        if not vehicle:
            return 5  # Default priority
            
        # State-based priorities (lower is higher priority)
        priorities = {
            VehicleState.UNLOADING: 1,  # Highest priority
            VehicleState.PREPARING: 2,
            TransportStage.TRANSPORTING: 2,  # Loaded vehicles have priority
            TransportStage.APPROACHING: 3,
            TransportStage.RETURNING: 4,
            VehicleState.IDLE: 5  # Lowest priority
        }
        
        # Get base priority from state
        if vehicle.state == VehicleState.EN_ROUTE and hasattr(vehicle, 'transport_stage'):
            priority = priorities.get(vehicle.transport_stage, 3)
        else:
            priority = priorities.get(vehicle.state, 3)
            
        # Consider task priority if vehicle has a task
        if hasattr(vehicle, 'current_task') and vehicle.current_task:
            task_priority = getattr(vehicle.current_task, 'priority', 1)
            # Higher task priority (1-5) reduces vehicle priority score
            priority_modifier = max(0, 1 - task_priority/10.0)
            priority -= priority_modifier
            
        # Consider current load - loaded vehicles get higher priority
        if hasattr(vehicle, 'current_load') and vehicle.current_load > 0:
            # Boost priority based on load percentage
            if hasattr(vehicle, 'max_capacity') and vehicle.max_capacity > 0:
                load_pct = vehicle.current_load / vehicle.max_capacity
                if load_pct > 0.5:  # More than half loaded
                    priority -= 0.5  # Boost priority
        
        return priority
            
    def _resolve_conflicts_with_spatial_awareness(self, conflicts, paths) -> int:
        """
        Resolve conflicts with spatial awareness and multiple strategies.
        
        Returns the number of conflicts resolved.
        """
        resolved_count = 0
        new_paths = paths.copy()
        
        # Group conflicts by vehicle pairs for efficient resolution
        vehicle_conflicts = defaultdict(list)
        for conflict in conflicts:
            pair = tuple(sorted([conflict['vehicle1'], conflict['vehicle2']]))
            vehicle_conflicts[pair].append(conflict)
        
        logger.debug(f"Resolving conflicts for {len(vehicle_conflicts)} vehicle pairs")
        
        # Process conflicts by vehicle pair
        for pair, conflict_list in vehicle_conflicts.items():
            vid1, vid2 = pair
            prio1 = self._get_vehicle_priority(vid1)
            prio2 = self._get_vehicle_priority(vid2)
            
            # Determine which vehicle should be replanned
            replan_vid = vid2 if prio1 < prio2 else vid1
            
            # Log the decision
            logger.debug(f"Resolving conflicts between {vid1}(prio:{prio1:.1f}) and {vid2}(prio:{prio2:.1f})")
            logger.debug(f"Vehicle {replan_vid} will be replanned for {len(conflict_list)} conflicts")
            
            # Attempt to replan the path for the lower priority vehicle
            if self._replan_vehicle_path(replan_vid, conflict_list, new_paths):
                resolved_count += len(conflict_list)
                
        return resolved_count
        
    def _replan_vehicle_path(self, vid, conflicts, paths) -> bool:
        """Replan path for a specific vehicle to avoid conflicts"""
        try:
            # Get vehicle and verify it has a task
            vehicle = self._get_vehicle(vid)
            if not vehicle or not vehicle.current_task:
                logger.warning(f"Vehicle {vid} not found or has no task for replanning")
                return False
                
            # Get the end point from the current task
            end_point = vehicle.current_task.end_point
            
            # Create a map of points to avoid
            avoid_points = set()
            
            # Add conflict points to avoid
            for conflict in conflicts:
                if conflict['type'] == 'position':
                    pos = conflict['position']
                    # Add points around the conflict (padding)
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            avoid_points.add((pos[0] + dx, pos[1] + dy))
                elif conflict['type'] == 'crossing':
                    # Add both segments to avoid
                    for segment in [conflict['segment1'], conflict['segment2']]:
                        for point in segment:
                            for dx in range(-2, 3):
                                for dy in range(-2, 3):
                                    avoid_points.add((point[0] + dx, point[1] + dy))
            
            # Temporarily add avoidance points to obstacles
            original_obstacles = set()
            if hasattr(self.dispatch.planner, 'obstacle_grids'):
                original_obstacles = self.dispatch.planner.obstacle_grids.copy()
                self.dispatch.planner.obstacle_grids.update(avoid_points)
            
            try:
                # Try several strategies for path planning
                
                # Strategy 1: Direct replanning
                new_path = self.dispatch.planner.plan_path(
                    vehicle.current_location,
                    end_point,
                    vehicle
                )
                
                # If direct replanning fails or path is too short, try with midpoint
                if not new_path or len(new_path) < 2:
                    # Strategy 2: Use random midpoint
                    mid_x = (vehicle.current_location[0] + end_point[0]) / 2
                    mid_y = (vehicle.current_location[1] + end_point[1]) / 2
                    # Add randomness to break symmetry
                    mid_point = (
                        mid_x + random.uniform(-20, 20),
                        mid_y + random.uniform(-20, 20)
                    )
                    
                    # Try path through midpoint
                    path1 = self.dispatch.planner.plan_path(vehicle.current_location, mid_point)
                    path2 = self.dispatch.planner.plan_path(mid_point, end_point)
                    
                    if path1 and path2 and len(path1) > 1 and len(path2) > 1:
                        new_path = path1[:-1] + path2  # Combine paths, avoid duplicate midpoint
                    
                # If all else fails, try direct line with intermediate points
                if not new_path or len(new_path) < 2:
                    # Strategy 3: Simple line with intermediate points
                    dx = (end_point[0] - vehicle.current_location[0]) / 5
                    dy = (end_point[1] - vehicle.current_location[1]) / 5
                    
                    new_path = [vehicle.current_location]
                    for i in range(1, 5):
                        new_path.append((
                            vehicle.current_location[0] + dx * i,
                            vehicle.current_location[1] + dy * i
                        ))
                    new_path.append(end_point)
                
                # Assign the new path if valid
                if new_path and len(new_path) > 1:
                    vehicle.assign_path(new_path)
                    logger.info(f"Successfully replanned path for vehicle {vid} to avoid conflicts")
                    return True
                else:
                    logger.warning(f"Failed to generate valid path for vehicle {vid}")
                    return False
                    
            finally:
                # Restore original obstacles
                if hasattr(self.dispatch.planner, 'obstacle_grids'):
                    self.dispatch.planner.obstacle_grids = original_obstacles
                    
        except Exception as e:
            logger.error(f"Path replanning failed for vehicle {vid}: {str(e)}")
            return False


class DeadlockDetector:
    """
    Deadlock detection algorithm to prevent vehicles from getting stuck.
    """
    
    def __init__(self):
        """Initialize the deadlock detector."""
        self.deadlock_history = {}  # Track potential deadlocks over time
        self.resolution_count = {}  # Track how many times a vehicle has been replanned
        self.last_positions = {}    # Last known positions of vehicles
        self.stuck_time = {}        # Time vehicles have been at the same position
        
    def check_deadlocks(self, conflicts, vehicles) -> Set[int]:
        """
        Check for potential deadlocks in the system based on conflicts.
        
        Returns a set of vehicle IDs that should be prioritized for replanning
        to break deadlocks.
        """
        # Build a dependency graph from conflicts
        dependency_graph = {}
        
        for conflict in conflicts:
            vid1 = conflict['vehicle1']
            vid2 = conflict['vehicle2']
            
            # Convert to integers if possible
            if isinstance(vid1, str) and vid1.isdigit():
                vid1 = int(vid1)
            if isinstance(vid2, str) and vid2.isdigit():
                vid2 = int(vid2)
                
            if vid1 not in dependency_graph:
                dependency_graph[vid1] = set()
            if vid2 not in dependency_graph:
                dependency_graph[vid2] = set()
                
            # Add dependencies based on conflict
            dependency_graph[vid1].add(vid2)
            dependency_graph[vid2].add(vid1)
            
        # Find cycles (potential deadlocks)
        deadlocked_vehicles = self._find_cycles(dependency_graph)
        
        # Update deadlock history
        current_time = time.time()
        for vid in deadlocked_vehicles:
            if vid not in self.deadlock_history:
                self.deadlock_history[vid] = []
            
            self.deadlock_history[vid].append(current_time)
            
            # Clean up old entries (more than 5 minutes old)
            self.deadlock_history[vid] = [t for t in self.deadlock_history[vid] 
                                        if current_time - t < 300]
        
        # Also check for physically stuck vehicles (not moving)
        stuck_vehicles = self._check_for_stuck_vehicles(vehicles)
        deadlocked_vehicles.update(stuck_vehicles)
        
        return deadlocked_vehicles
    
    def _check_for_stuck_vehicles(self, vehicles) -> Set[int]:
        """Detect vehicles that aren't making progress"""
        stuck_vehicles = set()
        current_time = time.time()
        
        for vid, vehicle in vehicles.items():
            # Skip vehicles that aren't en route
            if vehicle.state != VehicleState.EN_ROUTE:
                continue
                
            current_pos = vehicle.current_location
            
            # Check if position has changed
            if vid in self.last_positions:
                last_pos = self.last_positions[vid]
                
                # If position hasn't changed significantly
                if math.dist(current_pos, last_pos) < 1.0:
                    # Update stuck time
                    if vid not in self.stuck_time:
                        self.stuck_time[vid] = current_time
                    
                    # If stuck for more than 10 seconds
                    if current_time - self.stuck_time[vid] > 10:
                        stuck_vehicles.add(vid)
                else:
                    # Reset stuck time if moving
                    self.stuck_time.pop(vid, None)
            
            # Update last position
            self.last_positions[vid] = current_pos
            
        return stuck_vehicles
                
    def _find_cycles(self, graph) -> Set:
        """Find cycles in dependency graph using DFS"""
        visited = set()
        rec_stack = set()
        cycle_nodes = set()
        
        def dfs(node, parent=None):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor == parent:
                    continue
                    
                if neighbor not in visited:
                    if dfs(neighbor, node):
                        cycle_nodes.add(neighbor)
                        return True
                elif neighbor in rec_stack:
                    cycle_nodes.add(neighbor)
                    cycle_nodes.add(node)
                    return True
            
            rec_stack.remove(node)
            return False
            
        for node in graph:
            if node not in visited:
                dfs(node)
                
        return cycle_nodes
        
    def check_persistent_deadlocks(self) -> Set[int]:
        """
        Check for persistent deadlocks that have occurred multiple times.
        
        Returns a set of vehicle IDs involved in persistent deadlocks.
        """
        persistent_deadlocks = set()
        
        # Count deadlock occurrences
        for vid, timestamps in self.deadlock_history.items():
            if len(timestamps) >= 3:  # Detected at least 3 times
                persistent_deadlocks.add(vid)
                
                # Increment resolution count
                self.resolution_count[vid] = self.resolution_count.get(vid, 0) + 1
                
        # Also check physical stuck vehicles
        for vid, stuck_since in self.stuck_time.items():
            if time.time() - stuck_since > 30:  # Stuck for 30+ seconds
                persistent_deadlocks.add(vid)
                
        return persistent_deadlocks


# Main integration function
def integrate_dispatch_system(config_path=None):
    """
    Create and configure a fully integrated dispatch system with all components.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        IntegratedDispatchSystem: Ready-to-use dispatch system
    """
    # Create the integrated system
    system = IntegratedDispatchSystem(config_path)
    
    # Configure system based on need
    if config_path:
        logger.info(f"Integrated system configured using {config_path}")
    else:
        logger.info("Integrated system configured with default settings")
    
    return system


# Example usage demonstration
def run_example():
    """
    Run a simple demonstration of the integrated system.
    """
    # Create the integrated system
    system = integrate_dispatch_system()
    
    # Create test vehicles
    vehicles = [
        MiningVehicle(
            vehicle_id=i,
            map_service=system.map_service,
            config={
                'current_location': (100+i*50, 100+i*30),
                'max_capacity': 50,
                'max_speed': 5 + i*2,
                'base_location': (200, 200),
                'turning_radius': 10.0
            }
        )
        for i in range(1, 4)
    ]
    
    # Register vehicles
    for vehicle in vehicles:
        system.register_vehicle(vehicle)
    
    # Create test tasks
    tasks = [
        TransportTask(
            task_id=f"TASK-{i:02d}",
            start_point=(0, i*50),
            end_point=(300, i*30),
            task_type="loading" if i % 2 == 0 else "unloading",
            priority=i
        )
        for i in range(1, 4)
    ]
    
    # Add tasks
    for task in tasks:
        system.add_task(task)
    
    # Start the dispatch service
    system.start_dispatch_service(dispatch_interval=5)
    
    # Run for a short period
    try:
        for i in range(5):
            status = system.get_system_status()
            print(f"System status at {status['timestamp']}:")
            print(f"  Active tasks: {len(status['active_tasks'])}")
            print(f"  Queued tasks: {status['stats']['queued_tasks']}")
            print(f"  Vehicle states: {status['stats'].get('vehicle_states', {})}")
            print(f"  System health: {status['system_health']['status']}")
            
            # Sleep between status updates
            time.sleep(5)
            
            # Add another task halfway through
            if i == 2:
                new_task = TransportTask(
                    task_id="TASK-EXTRA",
                    start_point=(50, 50),
                    end_point=(250, 250),
                    task_type="manual",
                    priority=5
                )
                system.add_task(new_task)
                print("Added extra task: TASK-EXTRA")
    finally:
        # Stop the service
        system.stop_dispatch_service()
        print("Dispatch service stopped")


def run_performance_test(duration=60, num_vehicles=10, task_rate=0.2):
    """
    Run a performance test of the integrated system.
    
    Args:
        duration: Test duration in seconds
        num_vehicles: Number of vehicles to simulate
        task_rate: Tasks per second generation rate
    """
    print(f"Starting performance test: {num_vehicles} vehicles, {duration}s duration")
    
    # Create system
    system = integrate_dispatch_system()
    
    # Create vehicles
    vehicles = []
    for i in range(1, num_vehicles + 1):
        # Create vehicles with random initial positions
        vehicle = MiningVehicle(
            vehicle_id=i,
            map_service=system.map_service,
            config={
                'current_location': (
                    random.uniform(0, 300), 
                    random.uniform(0, 300)
                ),
                'max_capacity': random.uniform(40, 60),
                'max_speed': random.uniform(4, 10),
                'base_location': (200, 200),
                'status': VehicleState.IDLE
            }
        )
        vehicles.append(vehicle)
        system.register_vehicle(vehicle)
    
    # Generate initial tasks
    dispatch_config = system.dispatch._load_config()
    loading_points = dispatch_config['loading_points']
    unloading_point = dispatch_config['unloading_point']
    
    for i in range(1, num_vehicles):
        task = TransportTask(
            task_id=f"PERF-{i:03d}",
            start_point=random.choice(loading_points),
            end_point=unloading_point,
            task_type="loading",
            priority=random.randint(1, 3)
        )
        system.add_task(task)
    
    # Start the service
    system.start_dispatch_service(dispatch_interval=1.0)
    
    # Run task generation loop
    start_time = time.time()
    task_counter = num_vehicles
    
    try:
        while time.time() - start_time < duration:
            # Generate random tasks based on rate
            if random.random() < task_rate:
                task_counter += 1
                task_type = random.choice(["loading", "unloading"])
                
                if task_type == "loading":
                    start_point = random.choice(loading_points)
                    end_point = unloading_point
                else:
                    start_point = unloading_point
                    end_point = dispatch_config['parking_area']
                
                task = TransportTask(
                    task_id=f"PERF-{task_counter:03d}",
                    start_point=start_point,
                    end_point=end_point,
                    task_type=task_type,
                    priority=random.randint(1, 3)
                )
                system.add_task(task)
            
            # Sleep for a short time
            time.sleep(0.1)
            
            # Print status every 5 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 5 == 0 and elapsed % 5 < 0.1:
                status = system.get_system_status()
                print(f"Status at {elapsed:.1f}s: "
                      f"Active:{len(status['active_tasks'])} "
                      f"Queued:{status['stats']['queued_tasks']} "
                      f"Completed:{status['stats']['tasks_completed']}")
    
    finally:
        # Get final stats
        final_status = system.get_system_status()
        
        # Stop the service
        system.stop_dispatch_service()
        
        # Print performance report
        print("\nPerformance Test Results:")
        print(f"Duration: {duration} seconds")
        print(f"Vehicles: {num_vehicles}")
        print(f"Tasks assigned: {system.stats['tasks_assigned']}")
        print(f"Tasks completed: {system.stats['tasks_completed']}")
        print(f"Completion rate: {system.stats['tasks_completed']/max(1, system.stats['tasks_assigned']):.1%}")
        print(f"Conflicts detected: {system.stats['conflicts_detected']}")
        print(f"Conflicts resolved: {system.stats['conflicts_resolved']}")
        print(f"Resolution rate: {system.stats['conflicts_resolved']/max(1, system.stats['conflicts_detected']):.1%}")
        print(f"Deadlocks detected: {system.stats['deadlocks_detected']}")
        print(f"Deadlocks resolved: {system.stats['deadlocks_resolved']}")
        
        if hasattr(system.planner, 'get_performance_stats'):
            path_stats = system.planner.get_performance_stats()
            print("\nPath Planner Performance:")
            print(f"Avg planning time: {path_stats.get('avg_planning_time', 'N/A')}")
            print(f"Path cache hit rate: {path_stats.get('cache_stats', {}).get('hit_rate', 0):.1%}")
            
        print(f"\nFinal system health: {final_status['system_health']['status']}")
        if final_status['system_health']['warnings']:
            print("Warnings:")
            for warning in final_status['system_health']['warnings']:
                print(f"  - {warning}")
        if final_status['system_health']['errors']:
            print("Errors:")
            for error in final_status['system_health']['errors']:
                print(f"  - {error}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mining Vehicle Dispatch System Integration Tests")
    parser.add_argument('--test', choices=['example', 'performance'], default='example',
                      help='Test to run')
    parser.add_argument('--duration', type=int, default=60,
                      help='Test duration in seconds (for performance test)')
    parser.add_argument('--vehicles', type=int, default=10,
                      help='Number of vehicles (for performance test)')
    parser.add_argument('--task-rate', type=float, default=0.2,
                      help='Task generation rate (for performance test)')
    parser.add_argument('--config', type=str, default=None,
                      help='Path to config file')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    if args.test == 'example':
        print("Running example test...")
        run_example()
    elif args.test == 'performance':
        print("Running performance test...")
        run_performance_test(
            duration=args.duration,
            num_vehicles=args.vehicles,
            task_rate=args.task_rate
        )
