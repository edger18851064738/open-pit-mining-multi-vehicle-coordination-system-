"""
Integration example of Hybrid Path Planner with Conflict-Based Search (CBS)
for multi-vehicle coordination in open-pit mining environment
"""

import os
import sys
import math
import time
import logging
import threading
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import matplotlib.pyplot as plt
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import required modules (assuming they are in the correct path)
from algorithm.cbs import ConflictBasedSearch
from models.vehicle import MiningVehicle, VehicleState
from models.task import TransportTask
from algorithm.dispatch_service import DispatchSystem
from algorithm.map_service import MapService
from utils.geo_tools import GeoUtils

# Import the Hybrid Path Planner we defined
# Note: In real implementation, you would import this from the appropriate module
from hybrid_path_planner import HybridPathPlanner

class EnhancedCBS(ConflictBasedSearch):
    """
    Enhanced Conflict-Based Search that works with the Hybrid Path Planner
    """
    
    def __init__(self, path_planner):
        """
        Initialize Enhanced CBS
        
        Args:
            path_planner: The path planner object (HybridPathPlanner)
        """
        super().__init__(path_planner)
        
        # Store reference to path planner
        self.planner = path_planner
        
        # Enhanced conflict detection parameters
        self.vehicle_safety_margin = 1.5  # Safety margin around vehicles
        self.time_horizon = 5.0  # Time horizon for conflict prediction
        self.path_interpolation = True  # Enable path interpolation for smoother conflict checks
    
    def find_conflicts(self, paths: Dict[str, List[Tuple]]) -> List[Dict]:
        """
        Enhanced conflict detection that considers vehicle kinematics
        
        Args:
            paths: Dictionary mapping vehicle IDs to paths
            
        Returns:
            List of detected conflicts
        """
        # Use parent implementation as a base
        conflicts = super().find_conflicts(paths)
        
        # Apply enhanced conflict detection if needed
        if self.path_interpolation:
            # Interpolate paths to make them smoother and more granular
            interpolated_paths = {}
            for vid, path in paths.items():
                if len(path) > 1:
                    interpolated_paths[vid] = self._interpolate_path(path)
                else:
                    interpolated_paths[vid] = path
            
            # Detect conflicts using interpolated paths
            enhanced_conflicts = self._detect_enhanced_conflicts(interpolated_paths)
            
            # Merge conflicts while removing duplicates
            for conflict in enhanced_conflicts:
                if conflict not in conflicts:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _interpolate_path(self, path: List[Tuple]) -> List[Tuple]:
        """
        Interpolate a path to get more points for smoother conflict detection
        
        Args:
            path: Original path with points (x, y, theta)
            
        Returns:
            Interpolated path with more points
        """
        if len(path) < 2:
            return path
            
        interpolated_path = []
        num_segments = max(5, len(path))  # At least 5 points per segment
        
        for i in range(len(path) - 1):
            start = path[i]
            end = path[i+1]
            
            # Linear interpolation for now (could be replaced with spline)
            for j in range(num_segments):
                t = j / num_segments
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
                
                # Interpolate heading angle
                angle_diff = end[2] - start[2]
                # Normalize angle difference
                while angle_diff > math.pi:
                    angle_diff -= 2 * math.pi
                while angle_diff < -math.pi:
                    angle_diff += 2 * math.pi
                
                theta = start[2] + t * angle_diff
                interpolated_path.append((x, y, theta))
        
        # Add final point
        interpolated_path.append(path[-1])
        
        return interpolated_path
    
    def _detect_enhanced_conflicts(self, paths: Dict[str, List[Tuple]]) -> List[Dict]:
        """
        Detect conflicts with enhanced algorithm considering vehicle shapes
        
        Args:
            paths: Dictionary of interpolated paths
            
        Returns:
            List of detected conflicts
        """
        conflicts = []
        path_items = list(paths.items())
        
        # Check conflicts between all path pairs
        for i in range(len(path_items)):
            vid1, path1 = path_items[i]
            for j in range(i+1, len(path_items)):
                vid2, path2 = path_items[j]
                
                # Get minimum of both path lengths
                min_len = min(len(path1), len(path2))
                if min_len <= 1:
                    continue
                
                # Check potential collisions at each time step
                for t in range(min_len):
                    if self._vehicles_collide(path1[t], path2[t], 
                                              self.vehicle_safety_margin):
                        # Conflict found
                        conflicts.append({
                            "type": "vertex",
                            "time": t,
                            "location": ((path1[t][0] + path2[t][0]) / 2, 
                                         (path1[t][1] + path2[t][1]) / 2),
                            "agent1": vid1,
                            "agent2": vid2
                        })
                        break
        
        return conflicts
    
    def _vehicles_collide(self, state1, state2, safety_margin=0.0):
        """
        Check if two vehicles would collide at given states
        
        Args:
            state1: State of vehicle 1 (x, y, theta)
            state2: State of vehicle 2 (x, y, theta)
            safety_margin: Additional safety distance
            
        Returns:
            bool: Whether vehicles would collide
        """
        # Get vehicle dimensions
        vehicle_length = self.planner.vehicle_length
        vehicle_width = self.planner.vehicle_width
        
        # Simple distance check first (optimization)
        x1, y1, _ = state1
        x2, y2, _ = state2
        distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        # Quick rejection: If centers too far apart, no collision
        if distance > (vehicle_length + safety_margin):
            return False
        
        # Detailed collision check needed
        # Create bounding rectangles for both vehicles
        rect1 = self._vehicle_rectangle(state1)
        rect2 = self._vehicle_rectangle(state2)
        
        # Check rectangle intersection
        return self._rectangles_intersect(rect1, rect2, safety_margin)
    
    def _vehicle_rectangle(self, state):
        """
        Generate rectangle corners for vehicle at given state
        
        Args:
            state: Vehicle state (x, y, theta)
            
        Returns:
            list: Four corners of vehicle rectangle [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        x, y, theta = state
        vehicle_length = self.planner.vehicle_length
        vehicle_width = self.planner.vehicle_width
        
        # Calculate half-dimensions
        half_length = vehicle_length / 2
        half_width = vehicle_width / 2
        
        # Calculate corners relative to center
        corners_rel = [
            (half_length, half_width),   # Front right
            (half_length, -half_width),  # Front left
            (-half_length, -half_width), # Rear left
            (-half_length, half_width)   # Rear right
        ]
        
        # Transform corners to world coordinates
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        corners_world = []
        for cx, cy in corners_rel:
            wx = x + cx * cos_theta - cy * sin_theta
            wy = y + cx * sin_theta + cy * cos_theta
            corners_world.append((wx, wy))
        
        return corners_world
    
    def _rectangles_intersect(self, rect1, rect2, margin=0.0):
        """
        Check if two rectangles intersect using Separating Axis Theorem
        
        Args:
            rect1: First rectangle corners
            rect2: Second rectangle corners
            margin: Safety margin to add around rectangles
            
        Returns:
            bool: Whether rectangles intersect
        """
        # Get edge vectors for both rectangles
        edges = []
        for rect in [rect1, rect2]:
            for i in range(4):
                edge = (
                    rect[(i+1)%4][0] - rect[i][0],
                    rect[(i+1)%4][1] - rect[i][1]
                )
                edges.append(edge)
        
        # Get perpendicular axes to check
        axes = []
        for edge in edges:
            axis = (-edge[1], edge[0])  # Perpendicular
            # Normalize
            length = math.sqrt(axis[0]**2 + axis[1]**2)
            if length > 0:
                axis = (axis[0]/length, axis[1]/length)
                axes.append(axis)
        
        # Check for separation along each axis
        for axis in axes:
            # Project each rectangle onto the axis
            min1, max1 = self._project_rectangle(rect1, axis)
            min2, max2 = self._project_rectangle(rect2, axis)
            
            # Add safety margin
            min1 -= margin
            max1 += margin
            min2 -= margin
            max2 += margin
            
            # Check for separation
            if max1 < min2 or max2 < min1:
                return False  # Separated axis found - no collision
        
        # No separating axis found - collision
        return True
    
    def _project_rectangle(self, rect, axis):
        """
        Project rectangle onto axis
        
        Args:
            rect: Rectangle corners
            axis: Axis to project onto
            
        Returns:
            (min, max): Minimum and maximum projections
        """
        projections = []
        for corner in rect:
            projection = corner[0] * axis[0] + corner[1] * axis[1]
            projections.append(projection)
        
        return min(projections), max(projections)
    
    def _replan_path(self, vehicle_id, conflict_location=None, conflict_time=None, max_retries=3):
        """
        Replan path for specified vehicle to avoid conflict point
        
        Args:
            vehicle_id: Vehicle ID to replan for
            conflict_location: Conflict location to avoid
            conflict_time: Time of conflict
            max_retries: Maximum retries for planning
            
        Returns:
            List[Tuple]: New path or None if planning failed
        """
        logging.debug(f"Replanning path for vehicle {vehicle_id}")
        
        try:
            # Convert vehicle_id to integer if it's a string
            if isinstance(vehicle_id, str) and vehicle_id.isdigit():
                vehicle_id = int(vehicle_id)
                
            # Ensure dispatch exists and the vehicle exists
            if not hasattr(self.planner, 'dispatch'):
                logging.error("Planner has no dispatch attribute")
                return None
                
            if vehicle_id not in self.planner.dispatch.vehicles:
                logging.error(f"Vehicle ID not found: {vehicle_id}")
                return None
                
            vehicle = self.planner.dispatch.vehicles[vehicle_id]
            
            # Determine start and end points
            start_point = vehicle.current_location
            
            # Determine target location
            if vehicle.current_task and hasattr(vehicle.current_task, 'end_point'):
                end_point = vehicle.current_task.end_point
            elif hasattr(vehicle, 'current_path') and vehicle.current_path:
                end_point = vehicle.current_path[-1]
            else:
                logging.warning(f"Cannot determine target location for vehicle {vehicle_id}")
                return None
            
            # If conflict location provided, create a detour
            if conflict_location:
                # Calculate detour point
                import random
                offset = 20 + random.random() * 10  # 20-30 units offset
                
                # Determine detour direction based on path direction
                if abs(end_point[0] - start_point[0]) > abs(end_point[1] - start_point[1]):
                    # Horizontal path, vertical offset
                    detour_point = (
                        conflict_location[0], 
                        conflict_location[1] + offset * (1 if random.random() > 0.5 else -1)
                    )
                else:
                    # Vertical path, horizontal offset
                    detour_point = (
                        conflict_location[0] + offset * (1 if random.random() > 0.5 else -1),
                        conflict_location[1]
                    )
                
                # Convert points to vehicle states (x, y, theta)
                if len(start_point) == 2:
                    # Default heading if not available
                    if hasattr(vehicle, 'heading'):
                        theta = vehicle.heading
                    else:
                        # Estimate heading from start to detour
                        dx = detour_point[0] - start_point[0]
                        dy = detour_point[1] - start_point[1]
                        theta = math.atan2(dy, dx)
                    
                    start_state = (start_point[0], start_point[1], theta)
                else:
                    start_state = start_point
                
                # Estimate heading for detour point
                dx = end_point[0] - detour_point[0]
                dy = end_point[1] - detour_point[1]
                detour_theta = math.atan2(dy, dx)
                detour_state = (detour_point[0], detour_point[1], detour_theta)
                
                # Estimate heading for end point
                if len(end_point) == 2:
                    end_state = (end_point[0], end_point[1], detour_theta)
                else:
                    end_state = end_point
                
                # Two-stage planning
                path1 = self.planner.plan_path(start_state, detour_state, vehicle)
                path2 = self.planner.plan_path(detour_state, end_state, vehicle)
                
                if path1 and path2:
                    # Combine paths, removing duplicate junction point
                    return path1[:-1] + path2
            
            # Direct planning
            # Create start and end states with headings
            if len(start_point) == 2:
                # Default heading if not available
                if hasattr(vehicle, 'heading'):
                    theta = vehicle.heading
                else:
                    # Estimate heading from start to end
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]
                    theta = math.atan2(dy, dx)
                
                start_state = (start_point[0], start_point[1], theta)
            else:
                start_state = start_point
                
            if len(end_point) == 2:
                # Set end heading in the same direction as approach
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]
                end_theta = math.atan2(dy, dx)
                end_state = (end_point[0], end_point[1], end_theta)
            else:
                end_state = end_point
            
            # Use hybrid path planner
            return self.planner.plan_path(start_state, end_state, vehicle)
            
        except Exception as e:
            logging.error(f"Replanning path error: {str(e)}")
            return None
    
    def resolve_conflicts(self, paths: Dict[str, List[Tuple]]) -> Dict[str, List[Tuple]]:
        """
        Resolve path conflicts using enhanced priority-based strategy
        
        Args:
            paths: Dictionary of vehicle paths
            
        Returns:
            Dict[str, List[Tuple]]: Conflict-free paths
        """
        if not paths:
            return paths
            
        try:
            start_time = time.time()
            new_paths = paths.copy()
            
            # Get all conflicts
            conflicts = self.find_conflicts(paths)
            
            if not conflicts:
                return new_paths
                
            # Sort conflicts by time
            conflicts.sort(key=lambda x: x["time"])
            
            # Handle each conflict
            for i, conflict in enumerate(conflicts):
                conflict_type = conflict["type"]
                conflict_time = conflict["time"]
                conflict_location = conflict["location"]
                vehicle1 = conflict["agent1"]
                vehicle2 = conflict["agent2"]
                
                logging.debug(f"Handling conflict {i+1}/{len(conflicts)}: type={conflict_type}, time={conflict_time}")
                
                # Get vehicle priorities
                prio1 = self._get_vehicle_priority(vehicle1)
                prio2 = self._get_vehicle_priority(vehicle2)
                
                logging.debug(f"Vehicle priorities: {vehicle1}={prio1:.1f}, {vehicle2}={prio2:.1f}")
                
                # Determine conflict resolution strategy
                if abs(prio1 - prio2) < 0.5:
                    # Similar priority, randomly select one vehicle to replan
                    import random
                    vehicle_to_replan = vehicle1 if random.random() < 0.5 else vehicle2
                    logging.debug(f"Similar priorities, randomly selected vehicle {vehicle_to_replan} for replanning")
                elif prio1 < prio2:
                    # Lower number means higher priority, keep vehicle1's path
                    vehicle_to_replan = vehicle2
                    logging.debug(f"Vehicle {vehicle1} has higher priority, replanning vehicle {vehicle2}")
                else:
                    vehicle_to_replan = vehicle1
                    logging.debug(f"Vehicle {vehicle2} has higher priority, replanning vehicle {vehicle1}")
                
                # Replan path
                new_path = self._replan_path(vehicle_to_replan, conflict_location, conflict_time)
                if new_path:
                    new_paths[vehicle_to_replan] = new_path
                    self.stats['conflicts_resolved'] += 1
                    
                    # Cache resolved conflict to avoid reprocessing
                    conflict_key = (vehicle1, vehicle2, conflict_time)
                    self.conflict_cache[conflict_key] = time.time()
                else:
                    logging.warning(f"Failed to replan path for vehicle {vehicle_to_replan}, conflict unresolved")
            
            # Clean expired conflict cache (older than 30 seconds)
            current_time = time.time()
            expired_keys = [k for k, v in self.conflict_cache.items() if current_time - v > 30]
            for k in expired_keys:
                del self.conflict_cache[k]
            
            # Update statistics
            self.stats['total_resolution_time'] += (time.time() - start_time)
                
            return new_paths
            
        except Exception as e:
            logging.error(f"Conflict resolution error: {str(e)}")
            # Return original paths on error
            return paths

class EnhancedDispatchSystem(DispatchSystem):
    """
    Enhanced Dispatch System that integrates with Hybrid Path Planner
    and Enhanced CBS for conflict resolution
    """
    
    def __init__(self, path_planner: HybridPathPlanner, map_service: MapService):
        """
        Initialize Enhanced Dispatch System
        
        Args:
            path_planner: HybridPathPlanner instance
            map_service: MapService instance
        """
        # Initialize parent class
        super().__init__(path_planner, map_service)
        
        # Replace CBS with enhanced version
        self.cbs = EnhancedCBS(path_planner)
        
        # Set path planner's dispatch reference
        self.path_planner = path_planner
        self.path_planner.dispatch = self
        
        # Additional performance metrics
        self.metrics.update({
            'planning_failures': 0,
            'avg_planning_time': 0,
            'total_planning_time': 0,
            'path_smoothness': 0
        })
        
        logging.info("Enhanced Dispatch System initialized with Hybrid Path Planner")
    
    def _assign_tasks_to_idle_vehicles(self):
        """
        Enhanced task assignment that uses hybrid path planning
        """
        # Get all idle vehicles
        idle_vehicles = [v for v in self.vehicles.values() 
                        if v.state == VehicleState.IDLE and not v.current_task]
        
        if not idle_vehicles or not self.task_queue:
            return
        
        for vehicle in idle_vehicles:
            if not self.task_queue:
                break
                
            # Get next task from queue
            task = self.task_queue.popleft()
            
            # Plan path using hybrid planner
            try:
                start_time = time.time()
                
                # Prepare start and end states with headings
                start_point = vehicle.current_location
                end_point = task.end_point
                
                # Estimate heading if needed
                if len(start_point) == 2:
                    # Estimate heading from start to end
                    dx = end_point[0] - start_point[0]
                    dy = end_point[1] - start_point[1]
                    start_theta = math.atan2(dy, dx)
                    start_state = (start_point[0], start_point[1], start_theta)
                else:
                    start_state = start_point
                    
                if len(end_point) == 2:
                    # Set end heading
                    end_theta = start_theta if 'start_theta' in locals() else 0
                    end_state = (end_point[0], end_point[1], end_theta)
                else:
                    end_state = end_point
                
                # Plan path using hybrid planner
                path = self.path_planner.plan_path(start_state, end_state, vehicle)
                
                planning_time = time.time() - start_time
                self.metrics['total_planning_time'] += planning_time
                self.metrics['planning_count'] += 1
                self.metrics['avg_planning_time'] = (
                    self.metrics['total_planning_time'] / self.metrics['planning_count']
                )
                
                if path and len(path) > 1:
                    # Assign task to vehicle
                    vehicle.assign_task(task)
                    vehicle.assign_path(path)
                    
                    # Update task status
                    self.active_tasks[task.task_id] = task
                    
                    # Update path record
                    self.vehicle_paths[str(vehicle.vehicle_id)] = path
                    
                    # Calculate path smoothness (simple metric)
                    smoothness = self._calculate_path_smoothness(path)
                    self.metrics['path_smoothness'] = (
                        (self.metrics['path_smoothness'] * (self.metrics['planning_count'] - 1) + smoothness) / 
                        self.metrics['planning_count']
                    )
                    
                    logging.info(f"Assigned task {task.task_id} to vehicle {vehicle.vehicle_id}, path length: {len(path)}")
                else:
                    # Path planning failed, return task to queue
                    logging.warning(f"Path planning failed for task {task.task_id}, returning to queue")
                    self.task_queue.append(task)
                    self.metrics['planning_failures'] += 1
            except Exception as e:
                logging.error(f"Task assignment error: {str(e)}")
                # Return task to queue on error
                self.task_queue.append(task)
                self.metrics['planning_failures'] += 1
    
    def _calculate_path_smoothness(self, path):
        """
        Calculate path smoothness as a quality metric
        
        Args:
            path: Path points list
            
        Returns:
            float: Smoothness score (higher is smoother)
        """
        if len(path) < 3:
            return 1.0  # Perfect smoothness for short paths
        
        # Calculate angle changes at each waypoint
        angle_changes = []
        
        for i in range(1, len(path) - 1):
            prev = path[i-1]
            curr = path[i]
            next_p = path[i+1]
            
            # Calculate vectors
            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (next_p[0] - curr[0], next_p[1] - curr[1])
            
            # Calculate magnitudes
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            # Skip near-zero vectors
            if mag1 < 1e-3 or mag2 < 1e-3:
                continue
                
            # Calculate dot product
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            
            # Calculate cosine of angle
            cos_angle = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
            
            # Convert to angle in radians
            angle = math.acos(cos_angle)
            angle_changes.append(angle)
        
        if not angle_changes:
            return 1.0
            
        # Calculate smoothness (1.0 for straight line, lower for sharp turns)
        avg_angle_change = sum(angle_changes) / len(angle_changes)
        smoothness = 1.0 - avg_angle_change / math.pi
        
        return max(0.0, min(1.0, smoothness))

# Example test function
def test_enhanced_dispatch_system():
    """
    Test the enhanced dispatch system with hybrid path planner
    """
    # Initialize map service
    map_service = MapService()
    
    # Add some obstacles to the map for testing
    obstacles = []
    
    # Create vertical obstacle
    for y in range(50, 70):
        for x in range(40, 60):
            obstacles.append((x, y))
    
    # Create horizontal obstacle
    for y in range(100, 120):
        for x in range(80, 120):
            obstacles.append((x, y))
    
    # Initialize hybrid path planner with obstacles
    planner = HybridPathPlanner(map_service)
    planner.obstacle_grids = set(obstacles)
    
    # Initialize enhanced dispatch system
    dispatch = EnhancedDispatchSystem(planner, map_service)
    
    # Create test vehicles
    vehicle1 = MiningVehicle(
        vehicle_id=1,
        map_service=map_service,
        config={
            'current_location': (10, 10),
            'max_capacity': 50,
            'max_speed': 6.0,
            'turning_radius': 5.0
        }
    )
    
    vehicle2 = MiningVehicle(
        vehicle_id=2,
        map_service=map_service,
        config={
            'current_location': (150, 10),
            'max_capacity': 50,
            'max_speed': 5.0,
            'turning_radius': 8.0
        }
    )
    
    # Add vehicles to dispatch system
    dispatch.add_vehicle(vehicle1)
    dispatch.add_vehicle(vehicle2)
    
    # Create test tasks
    task1 = TransportTask(
        task_id="T1",
        start_point=(10, 10),
        end_point=(80, 150),
        task_type="transport",
        priority=1
    )
    
    task2 = TransportTask(
        task_id="T2",
        start_point=(150, 10),
        end_point=(20, 150),
        task_type="transport",
        priority=2
    )
    
    # Add tasks to dispatch system
    dispatch.add_task(task1)
    dispatch.add_task(task2)
    
    # Run dispatch cycle
    print("Running dispatch cycle...")
    dispatch.scheduling_cycle()
    
    # Check results
    print("\nDispatch Status:")
    status = dispatch.get_status()
    print(f"Vehicles: {status['vehicles']}")
    print(f"Tasks: {status['tasks']}")
    print(f"Metrics: {status['metrics']}")
    
    # Check path conflicts
    print("\nChecking path conflicts...")
    vehicle_paths = {}
    for vid, vehicle in dispatch.vehicles.items():
        if hasattr(vehicle, 'current_path') and vehicle.current_path:
            vehicle_paths[str(vid)] = vehicle.current_path
    
    conflicts = dispatch.cbs.find_conflicts(vehicle_paths)
    print(f"Detected conflicts: {len(conflicts)}")
    
    # Resolve conflicts if any
    if conflicts:
        print("Resolving conflicts...")
        resolved_paths = dispatch.cbs.resolve_conflicts(vehicle_paths)
        
        # Count modified paths
        changed_paths = 0
        for vid_str, new_path in resolved_paths.items():
            if new_path != vehicle_paths.get(vid_str, []):
                changed_paths += 1
        
        print(f"Resolved conflicts by modifying {changed_paths} paths")
    
    # Visualize results
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw obstacles
        obstacle_x = [p[0] for p in obstacles]
        obstacle_y = [p[1] for p in obstacles]
        ax.scatter(obstacle_x, obstacle_y, c='gray', marker='s', alpha=0.5, s=10, label='Obstacles')
        
        # Draw vehicles and paths
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (vid, vehicle) in enumerate(dispatch.vehicles.items()):
            # Draw vehicle
            ax.scatter(
                vehicle.current_location[0], 
                vehicle.current_location[1], 
                c=colors[i % len(colors)], 
                marker='o', 
                s=100, 
                label=f'Vehicle {vid}'
            )
            
            # Draw path if exists
            if hasattr(vehicle, 'current_path') and vehicle.current_path:
                path = vehicle.current_path
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                
                ax.plot(path_x, path_y, c=colors[i % len(colors)], linestyle='-', linewidth=2)
                
                # Draw vehicle orientation along path
                for j in range(0, len(path), max(1, len(path)//10)):
                    x, y, theta = path[j]
                    dx = math.cos(theta) * 3  # Arrow length
                    dy = math.sin(theta) * 3
                    ax.arrow(x, y, dx, dy, head_width=2, head_length=2, fc=colors[i % len(colors)], ec=colors[i % len(colors)])
        
        # Set plot properties
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)
        ax.set_title('Enhanced Dispatch System with Hybrid Path Planning')
        ax.legend()
        ax.grid(True)
        
        plt.savefig('enhanced_dispatch_test.png')
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")
    
    return dispatch

if __name__ == "__main__":
    # Run test function
    test_enhanced_dispatch_system()