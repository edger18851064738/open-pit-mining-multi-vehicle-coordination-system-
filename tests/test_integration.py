"""
Integration Solution for Multi-Vehicle Collaborative Dispatch System
===============================================================

This module provides improved integration between components of the mining vehicle
dispatch system, focusing on:

1. Path planning integration
2. Conflict resolution algorithm improvements
3. Vehicle state management
4. Task scheduling coordination
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import logging
from typing import Dict, List, Tuple, Optional, Set
import math
import threading
import time
from datetime import datetime

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


class IntegratedDispatchSystem:
    """
    Enhanced integration layer for the mining dispatch system components.
    
    This class wraps the standard DispatchSystem with improved error handling,
    component integration, and monitoring capabilities.
    """
    
    def __init__(self, map_service: Optional[MapService] = None, 
                planner: Optional[HybridPathPlanner] = None):
        """Initialize the integrated dispatch system with proper component connections."""
        # Create core components if not provided
        self.map_service = map_service or MapService()
        self.planner = planner or HybridPathPlanner(self.map_service)
        
        # Create a safe wrapper for path planning
        self._patch_path_planner()
        
        # Initialize the dispatch system
        self.dispatch = DispatchSystem(self.planner, self.map_service)
        
        # Ensure the planner has a dispatch reference
        self.planner.dispatch = self.dispatch
        
        # Monitor locks
        self.monitor_lock = threading.RLock()
        
        # System state
        self.is_running = False
        self.dispatch_thread = None
        self.monitor_thread = None
        
        # Monitoring stats
        self.stats = {
            'conflicts_detected': 0,
            'conflicts_resolved': 0,
            'tasks_assigned': 0,
            'tasks_completed': 0,
            'path_planning_failures': 0,
            'system_start_time': None
        }
        
        logger.info("Integrated dispatch system initialized")

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
                # Simple straight line path as fallback
                if isinstance(start, tuple) and isinstance(end, tuple):
                    # Generate intermediate points for smoother path
                    dx = (end[0] - start[0]) / 5
                    dy = (end[1] - start[1]) / 5
                    path = [start]
                    for i in range(1, 5):
                        path.append((start[0] + dx * i, start[1] + dy * i))
                    path.append(end)
                    return path
                return [start, end]
        
        # Apply the patch
        self.planner.plan_path = safe_plan_path
        logger.info("Path planner patched with safe fallback")

    def register_vehicle(self, vehicle: MiningVehicle):
        """Register a new vehicle with the dispatch system."""
        # Initialize required attributes if missing
        if not hasattr(vehicle, 'current_path'):
            vehicle.current_path = []
        if not hasattr(vehicle, 'path_index'):
            vehicle.path_index = 0
        
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
        logger.info(f"Task {task.task_id} added to dispatch queue")

    def start_dispatch_service(self, dispatch_interval: int = 30):
        """Start the automated dispatch service."""
        if self.is_running:
            logger.warning("Dispatch service is already running")
            return
        
        self.is_running = True
        self.stats['system_start_time'] = datetime.now()
        
        # Start dispatch thread
        def dispatch_loop():
            while self.is_running:
                try:
                    self.dispatch.scheduling_cycle()
                    logger.info(f"Dispatch cycle completed. Active tasks: {len(self.dispatch.active_tasks)}")
                except Exception as e:
                    logger.error(f"Error in dispatch cycle: {str(e)}")
                time.sleep(dispatch_interval)
        
        self.dispatch_thread = threading.Thread(target=dispatch_loop, daemon=True)
        self.dispatch_thread.start()
        
        # Start monitoring thread
        def monitor_loop():
            while self.is_running:
                try:
                    self._update_system_stats()
                except Exception as e:
                    logger.error(f"Error in monitoring: {str(e)}")
                time.sleep(60)  # Update stats every minute
                
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Dispatch service started with {dispatch_interval}s cycle")

    def stop_dispatch_service(self):
        """Stop the automated dispatch service."""
        if not self.is_running:
            logger.warning("Dispatch service is not running")
            return
        
        self.is_running = False
        
        # Wait for threads to terminate
        if self.dispatch_thread:
            self.dispatch_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        logger.info("Dispatch service stopped")

    def _update_system_stats(self):
        """Update system monitoring statistics."""
        with self.monitor_lock:
            # Count vehicles by state
            vehicle_states = {}
            for v in self.dispatch.vehicles.values():
                state_name = v.state.name if hasattr(v.state, 'name') else str(v.state)
                vehicle_states[state_name] = vehicle_states.get(state_name, 0) + 1
            
            # Update stats
            self.stats['vehicle_states'] = vehicle_states
            self.stats['active_tasks'] = len(self.dispatch.active_tasks)
            self.stats['queued_tasks'] = len(self.dispatch.task_queue)
            self.stats['uptime_seconds'] = (datetime.now() - self.stats['system_start_time']).total_seconds()
            
            logger.debug(f"System stats updated: {self.stats}")

    def get_system_status(self) -> Dict:
        """Get the current system status."""
        with self.monitor_lock:
            # 确保active_tasks和completed_tasks存在
            active_tasks = getattr(self.dispatch, 'active_tasks', {})
            completed_tasks = getattr(self.dispatch, 'completed_tasks', {})
            
            return {
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats.copy(),
                'vehicles': {
                    vid: {
                        'state': v.state.name if hasattr(v.state, 'name') else str(v.state),
                        'location': v.current_location,
                        'has_task': v.current_task is not None
                    } for vid, v in self.dispatch.vehicles.items()
                },
                'active_tasks': list(active_tasks.keys()) if active_tasks else [],
                'completed_tasks': list(completed_tasks.keys()) if completed_tasks else []
            }

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
        self.vehicle_priorities = {}
        self.deadlock_detection = DeadlockDetector()
        
    def detect_and_resolve_all_conflicts(self) -> int:
        """Detect and resolve all conflicts in the system."""
        # Collect all vehicle paths
        all_paths = {}
        for vid, vehicle in self.dispatch.vehicles.items():
            if hasattr(vehicle, 'current_path') and vehicle.current_path:
                all_paths[str(vid)] = vehicle.current_path
        
        if not all_paths:
            return 0
            
        # Enhanced conflict detection with our improved algorithm
        conflicts = self._detect_conflicts_with_timeframes(all_paths)
        
        if not conflicts:
            return 0
            
        # Check for potential deadlocks
        deadlocked_vehicles = self.deadlock_detection.check_deadlocks(
            conflicts, self.dispatch.vehicles
        )
        
        # Update priorities with deadlock information
        for vid in deadlocked_vehicles:
            self.vehicle_priorities[vid] = -1  # Lowest priority to break deadlocks
        
        # Resolve conflicts
        resolved_count = self._resolve_conflicts_enhanced(conflicts, all_paths)
        
        return resolved_count
        
    def _detect_conflicts_with_timeframes(self, paths: Dict[str, List[Tuple]]) -> List[Dict]:
        """
        Enhanced conflict detection that considers vehicle speed and timeframes.
        """
        conflicts = []
        path_items = list(paths.items())
        
        for i in range(len(path_items)):
            vid1, path1 = path_items[i]
            vehicle1 = self._get_vehicle(vid1)
            speed1 = getattr(vehicle1, 'max_speed', 5.0)
            
            for j in range(i+1, len(path_items)):
                vid2, path2 = path_items[j]
                vehicle2 = self._get_vehicle(vid2)
                speed2 = getattr(vehicle2, 'max_speed', 5.0)
                
                # Calculate minimum path length
                min_len = min(len(path1), len(path2))
                if min_len < 2:
                    continue
                
                # Check direct position conflicts
                for t in range(min_len):
                    point1 = path1[t]
                    point2 = path2[t]
                    
                    # Calculate time estimates based on speeds
                    time1 = t / speed1
                    time2 = t / speed2
                    
                    # If vehicles would be at the same point at similar times
                    if point1 == point2 and abs(time1 - time2) < 0.5:
                        conflicts.append({
                            'type': 'position',
                            'time_index': t,
                            'position': point1,
                            'vehicle1': vid1,
                            'vehicle2': vid2,
                            'time1': time1,
                            'time2': time2
                        })
                
                # Check crossing paths
                for t1 in range(min_len-1):
                    segment1 = (path1[t1], path1[t1+1])
                    for t2 in range(min_len-1):
                        segment2 = (path2[t2], path2[t2+1])
                        
                        if self._segments_intersect(segment1, segment2):
                            # Calculate time estimates
                            time1 = t1 / speed1
                            time2 = t2 / speed2
                            
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
            
        # State-based priorities
        priorities = {
            VehicleState.UNLOADING: 1,  # Highest priority
            VehicleState.PREPARING: 2,
            TransportStage.TRANSPORTING: 3,
            TransportStage.APPROACHING: 4,
            VehicleState.IDLE: 5  # Lowest priority
        }
        
        # Determine priority based on state
        if vehicle.state == VehicleState.EN_ROUTE and hasattr(vehicle, 'transport_stage'):
            priority = priorities.get(vehicle.transport_stage, 5)
        else:
            priority = priorities.get(vehicle.state, 5)
            
        # Consider task priority if vehicle has a task
        if hasattr(vehicle, 'current_task') and vehicle.current_task:
            task_priority = getattr(vehicle.current_task, 'priority', 1)
            priority = min(priority, 5 - task_priority)  # Adjust for task priority
            
        return priority
            
    def _resolve_conflicts_enhanced(self, conflicts, paths) -> int:
        """
        Resolve conflicts with enhanced logic for better traffic management.
        
        Returns the number of conflicts resolved.
        """
        resolved_count = 0
        new_paths = paths.copy()
        
        # Group conflicts by vehicle pairs to address all conflicts for a pair at once
        vehicle_conflicts = {}
        for conflict in conflicts:
            pair = tuple(sorted([conflict['vehicle1'], conflict['vehicle2']]))
            if pair not in vehicle_conflicts:
                vehicle_conflicts[pair] = []
            vehicle_conflicts[pair].append(conflict)
        
        # Process conflicts by vehicle pair
        for pair, conflict_list in vehicle_conflicts.items():
            vid1, vid2 = pair
            prio1 = self._get_vehicle_priority(vid1)
            prio2 = self._get_vehicle_priority(vid2)
            
            # Determine which vehicle should be replanned
            replan_vid = vid2 if prio1 <= prio2 else vid1
            
            # Log the conflict prioritization
            logger.debug(f"Resolving conflicts between {vid1}(prio:{prio1}) and {vid2}(prio:{prio2})")
            logger.debug(f"Vehicle {replan_vid} will be replanned for {len(conflict_list)} conflicts")
            
            # Attempt to replan the path for the lower priority vehicle
            try:
                if replan_vid == vid1:
                    original_vid = vid1
                    vehicle = self._get_vehicle(vid1)
                    other_vehicle = self._get_vehicle(vid2)
                else:
                    original_vid = vid2
                    vehicle = self._get_vehicle(vid2)
                    other_vehicle = self._get_vehicle(vid2)
                
                if vehicle and vehicle.current_task:
                    # Get the end point from the current task
                    end_point = vehicle.current_task.end_point
                    
                    # Get the other vehicle's path to avoid
                    avoid_path = new_paths[str(vid1 if replan_vid == vid2 else vid2)]
                    
                    # Create a padded avoidance area around the conflict points
                    avoid_points = set()
                    for conflict in conflict_list:
                        if conflict['type'] == 'position':
                            pos = conflict['position']
                            # Add points around the conflict
                            for dx in range(-1, 2):
                                for dy in range(-1, 2):
                                    avoid_points.add((pos[0] + dx, pos[1] + dy))
                        elif conflict['type'] == 'crossing':
                            # Add both segments to avoid
                            for segment in [conflict['segment1'], conflict['segment2']]:
                                for point in segment:
                                    for dx in range(-1, 2):
                                        for dy in range(-1, 2):
                                            avoid_points.add((point[0] + dx, point[1] + dy))
                    
                    # Attempt path replanning
                    original_obstacles = self.dispatch.planner.obstacle_grids.copy()
                    
                    # Temporarily add avoidance points to obstacles
                    self.dispatch.planner.obstacle_grids.update(avoid_points)
                    
                    try:
                        # Replan path
                        new_path = self.dispatch.planner.plan_path(
                            vehicle.current_location,
                            end_point,
                            vehicle
                        )
                        
                        if new_path and len(new_path) > 1:
                            new_paths[str(original_vid)] = new_path
                            vehicle.assign_path(new_path)
                            resolved_count += len(conflict_list)
                            logger.info(f"Successfully replanned path for vehicle {original_vid}")
                        else:
                            logger.warning(f"Path replanning returned empty path for vehicle {original_vid}")
                    finally:
                        # Restore original obstacles
                        self.dispatch.planner.obstacle_grids = original_obstacles
                else:
                    logger.warning(f"Vehicle {original_vid} not found or has no task")
            except Exception as e:
                logger.error(f"Path replanning failed: {str(e)}")
        
        return resolved_count


class DeadlockDetector:
    """
    Deadlock detection algorithm to prevent vehicles from getting stuck.
    """
    
    def __init__(self):
        """Initialize the deadlock detector."""
        self.deadlock_history = {}  # Track potential deadlocks over time
        self.resolution_count = {}  # Track how many times a vehicle has been replanned
    
    def check_deadlocks(self, conflicts, vehicles) -> Set[int]:
        """
        Check for potential deadlocks in the system.
        
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
        
        # Update history
        current_time = time.time()
        for vid in deadlocked_vehicles:
            if vid not in self.deadlock_history:
                self.deadlock_history[vid] = []
            
            self.deadlock_history[vid].append(current_time)
            
            # Clean up old entries (more than 5 minutes old)
            self.deadlock_history[vid] = [t for t in self.deadlock_history[vid] 
                                        if current_time - t < 300]
        
        # Identify persistent deadlocks (detected multiple times in short period)
        persistent_deadlocks = set()
        for vid, timestamps in self.deadlock_history.items():
            if len(timestamps) >= 3:  # Detected at least 3 times
                persistent_deadlocks.add(vid)
                
                # Increment resolution count
                self.resolution_count[vid] = self.resolution_count.get(vid, 0) + 1
                
        return persistent_deadlocks
                
    def _find_cycles(self, graph) -> Set:
        """Find cycles in dependency graph using DFS."""
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


# Main integration function to patch and connect system components
def integrate_dispatch_system():
    """
    Create and configure a fully integrated dispatch system with all components.
    
    Returns:
        IntegratedDispatchSystem: Ready-to-use dispatch system
    """
    # Create the map service
    map_service = MapService()
    
    # Create the path planner
    planner = HybridPathPlanner(map_service)
    
    # Create the integrated system
    system = IntegratedDispatchSystem(map_service, planner)
    
    # Create enhanced conflict resolution
    conflict_resolver = EnhancedConflictResolution(system)
    
    # Set up hooks for dynamic conflict resolution
    original_scheduling_cycle = system.dispatch.scheduling_cycle
    
    def enhanced_scheduling_cycle():
        """Enhanced scheduling cycle with integrated conflict detection."""
        # Run the original cycle first
        original_scheduling_cycle()
        
        # Then run our enhanced conflict detection
        conflicts_resolved = conflict_resolver.detect_and_resolve_all_conflicts()
        
        # Update stats
        if conflicts_resolved > 0:
            system.stats['conflicts_detected'] += conflicts_resolved
            system.stats['conflicts_resolved'] += conflicts_resolved
            logger.info(f"Resolved {conflicts_resolved} conflicts in enhanced cycle")
    
    # Apply the enhancement
    system.dispatch.scheduling_cycle = enhanced_scheduling_cycle
    logger.info("Enhanced scheduling cycle applied with dynamic conflict resolution")
    
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
            print(f"  Vehicle states: {status['stats']['vehicle_states']}")
            
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


if __name__ == "__main__":
    # Run the example
    run_example()