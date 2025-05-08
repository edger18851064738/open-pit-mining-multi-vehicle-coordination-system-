import os
import sys
import math
import time
import logging
import heapq
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict, deque
from threading import RLock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.geo_tools import GeoUtils
from utils.path_tools import PathOptimizationError
from algorithm.map_service import MapService

# Constants
MAX_ITERATIONS = 10000     # Maximum A* search iterations
DEFAULT_TIMEOUT = 5.0      # Default timeout in seconds
EPSILON = 1e-6             # Floating point comparison precision
CACHE_SIZE = 1000          # Path cache size
CACHE_EXPIRY = 600         # Cache expiry time in seconds

class PathPlanningError(Exception):
    """Base exception for path planning errors"""
    pass

class TimeoutError(PathPlanningError):
    """Timeout exception"""
    pass

class NoPathFoundError(PathPlanningError):
    """No path found exception"""
    pass

class HybridAStarNode:
    """Hybrid A* node class"""
    
    def __init__(self, x, y, theta, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta  # Heading angle (radians)
        self.g = g          # Cost from start to current node
        self.h = h          # Estimated cost from current node to goal
        self.f = g + h      # Total cost
        self.parent = parent
        
    def __lt__(self, other):
        """Override less than operator for priority queue"""
        return self.f < other.f
    
    def __eq__(self, other):
        """Override equality operator"""
        if isinstance(other, HybridAStarNode):
            return (self.x == other.x and 
                    self.y == other.y and 
                    abs(self.theta - other.theta) < 0.1)  # Angles within 0.1 radians are considered equal
        return False
    
    def __hash__(self):
        """Define hash method for use in sets"""
        # Discretize continuous space for hashing
        grid_size = 0.5  
        angle_grid = 0.05
        x_grid = int(self.x / grid_size)
        y_grid = int(self.y / grid_size)
        theta_grid = int(self.theta / angle_grid)
        return hash((x_grid, y_grid, theta_grid))


class SpatialIndex:
    """Spatial index for efficient spatial queries"""
    
    def __init__(self, cell_size=10):
        self.cell_size = cell_size
        self.grid = {}
        
    def add_point(self, point):
        """Add point to index"""
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        cell_key = (cell_x, cell_y)
        
        if cell_key not in self.grid:
            self.grid[cell_key] = set()
        self.grid[cell_key].add(point)
            
    def add_points(self, points):
        """Add multiple points to index"""
        for point in points:
            self.add_point(point)
            
    def query_point(self, point, radius=0):
        """Query points near the given point"""
        result = set()
        cell_x = int(point[0] // self.cell_size)
        cell_y = int(point[1] // self.cell_size)
        
        # Calculate cell range to check
        cell_radius = max(1, int(radius // self.cell_size) + 1)
        
        # Check neighboring cells
        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                cell_key = (cell_x + dx, cell_y + dy)
                if cell_key in self.grid:
                    for p in self.grid[cell_key]:
                        if radius == 0 or math.dist(point, p) <= radius:
                            result.add(p)
                            
        return result
        
    def clear(self):
        """Clear the index"""
        self.grid.clear()


class PathCache:
    """High-performance path cache"""
    
    def __init__(self, max_size=CACHE_SIZE, expiry=CACHE_EXPIRY):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.expiry = expiry
        self.lock = RLock()
        self.hits = 0
        self.misses = 0
        
    def get(self, key):
        """Get cache entry"""
        with self.lock:
            now = time.time()
            if key in self.cache:
                # Check expiry
                if now - self.timestamps[key] <= self.expiry:
                    # Update timestamp
                    self.timestamps[key] = now
                    self.hits += 1
                    return self.cache[key].copy()  # Return copy to avoid modifying cache
                else:
                    # Expired, remove
                    del self.cache[key]
                    del self.timestamps[key]
            
            self.misses += 1
            return None
            
    def put(self, key, value):
        """Add cache entry"""
        with self.lock:
            now = time.time()
            
            # Check capacity
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.timestamps, key=self.timestamps.get)
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
                
            # Add new entry
            self.cache[key] = value.copy()  # Store copy to avoid external modification
            self.timestamps[key] = now
            
    def clear(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            
    def get_stats(self):
        """Get cache statistics"""
        with self.lock:
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'hits': self.hits,
                'misses': self.misses
            }


class HybridPathPlanner:
    """
    Hybrid Path Planner with A* and Reed-Shepp curves
    
    Features:
    1. Hybrid A* search algorithm
    2. Fallback path generation
    3. Timeout control
    4. Path smoothing
    5. Performance statistics
    """
    
    def __init__(self, map_service: MapService):
        """
        Initialize the path planner
        
        Args:
            map_service: Map service providing terrain and obstacle information
        """
        # Base components
        self.map_service = map_service
        self.dispatch = None  # Set by DispatchSystem
        
        # Performance monitoring
        self.load_time = time.time()
        self.planning_count = 0
        self.total_planning_time = 0
        self.success_count = 0
        self.failure_count = 0
        
        # Map size
        self.map_size = getattr(map_service, 'grid_size', 200)
        
        # Spatial data
        self.obstacle_index = SpatialIndex(cell_size=20)
        self.obstacle_grids = set()
        
        # Caching system
        self.path_cache = PathCache(max_size=CACHE_SIZE, expiry=CACHE_EXPIRY)
        
        # Path planning parameters
        self.vehicle_length = 5.0
        self.vehicle_width = 2.0
        self.turning_radius = 5.0
        self.step_size = 0.8
        self.grid_resolution = 0.3
        
        # Direction array for A* search (8 directions)
        self.directions = [
            (0, 1), (1, 0), (0, -1), (-1, 0),   # Cardinal directions
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # Diagonal directions
        ]
        
        # Movement costs
        self.move_costs = {
            (0, 1): 1.0, (1, 0): 1.0, (0, -1): 1.0, (-1, 0): 1.0,  # Cardinal movement costs
            (1, 1): 1.414, (1, -1): 1.414, (-1, 1): 1.414, (-1, -1): 1.414  # Diagonal movement costs
        }
        
        # Steering angles
        self.steering_angles = np.linspace(-0.7, 0.7, 15)
        
        # Wheel base (60% of vehicle length)
        self.wheel_base = self.vehicle_length * 0.6
        
        # Step sizes for different scenarios
        self.step_sizes = [self.step_size, self.step_size*0.5]
        
        # Path smoothing parameters
        self.path_smoothing = True
        self.smoothing_factor = 0.5
        self.smoothing_iterations = 10
        
        # RS curve parameters
        self.rs_step_size = 0.2
        self.use_rs_heuristic = True
        self.analytic_expansion_step = 5
        
        logging.info("Hybrid Path Planner initialization complete")
        
    def plan_path(self, start, end, vehicle=None):
        """
        Plan a path from start to end
        
        Args:
            start: Start coordinates (x, y)
            end: End coordinates (x, y)
            vehicle: Optional vehicle object
            
        Returns:
            List[Tuple[float, float]]: Planned path
        """
        # Performance counting
        start_time = time.time()
        self.planning_count += 1
        
        try:
            # Normalize input coordinates
            start = self._validate_point(start)
            end = self._validate_point(end)
            
            # Check if start and end are the same
            if self._points_equal(start, end):
                return [start]
                
            # Create cache key
            cache_key = self._create_cache_key(start, end, vehicle)
                    
            # Check cache
            cached_path = self.path_cache.get(cache_key)
            if cached_path:
                logging.debug(f"Using cached path: {start} -> {end}")
                return cached_path
            
            # Extract vehicle parameters if available
            if vehicle:
                if hasattr(vehicle, 'turning_radius'):
                    self.turning_radius = vehicle.turning_radius
                if hasattr(vehicle, 'current_location'):
                    current_theta = 0
                    if len(vehicle.current_location) > 2:
                        current_theta = vehicle.current_location[2]
                    start = (start[0], start[1], current_theta)
                if isinstance(end, tuple) and len(end) == 2:
                    # If end has no orientation, compute a reasonable default
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    end_theta = math.atan2(dy, dx) if (abs(dx) > 0.001 or abs(dy) > 0.001) else 0
                    end = (end[0], end[1], end_theta)
                
            # Determine which path planning method to use
            # For simple paths or when performance is critical, use A*
            if self._should_use_astar(start, end, vehicle):
                path = self._astar(start, end, vehicle)
            else:
                # For more complex maneuvers, use Hybrid A* with RS curves
                path = self._hybrid_astar(start, end, vehicle)
            
            # If planning failed, use fallback path
            if not path or len(path) < 2:
                logging.debug(f"Path planning failed, using fallback path: {start} -> {end}")
                path = self._generate_fallback_path(start, end)
                self.failure_count += 1
            else:
                self.success_count += 1
                
            # Smooth path if it's long enough
            if len(path) > 3 and self.path_smoothing:
                try:
                    path = self._smooth_path(path)
                except Exception as e:
                    logging.warning(f"Path smoothing failed: {str(e)}")
                    
            # Cache result
            self.path_cache.put(cache_key, path)
            
            # Record performance metrics
            elapsed = time.time() - start_time
            self.total_planning_time += elapsed
            
            if elapsed > 0.1:  # Log slow planning
                logging.debug(f"Path planning took significant time: {elapsed:.3f}s ({start} -> {end})")
                
            return path
            
        except Exception as e:
            logging.error(f"Path planning failed: {str(e)}")
            # Simplest fallback - direct connection
            return [start, end]
            
    def _should_use_astar(self, start, end, vehicle=None):
        """Decide whether to use A* or Hybrid A* based on situation"""
        # Calculate straight-line distance
        if isinstance(start, tuple) and len(start) > 2 and isinstance(end, tuple) and len(end) > 2:
            distance = math.dist(start[:2], end[:2])
        else:
            try:
                distance = math.dist(start, end)
            except:
                return True  # Default to A* if distance calculation fails
                
        # For short distances, use A*
        if distance < 20:
            return True
            
        # For long distances, use Hybrid A*
        if distance > 50:
            return False
            
        # Check if there's a clear path
        if hasattr(self, '_is_clear_path') and self._is_clear_path(start, end):
            return True
            
        # Default to Hybrid A* for complex scenarios
        return False
    
    def _astar(self, start, end, vehicle=None):
        """A* search algorithm implementation"""
        try:
            # Adjust start and end to avoid obstacles
            start_is_obstacle = self._is_obstacle_fast(start[:2] if len(start) > 2 else start)
            end_is_obstacle = self._is_obstacle_fast(end[:2] if len(end) > 2 else end)
            
            if start_is_obstacle:
                adjusted_start = self._find_nearest_non_obstacle(
                    start[:2] if len(start) > 2 else start, 
                    max_radius=10
                )
                if adjusted_start:
                    logging.debug(f"Adjusted start to: {adjusted_start}")
                    if len(start) > 2:
                        start = (adjusted_start[0], adjusted_start[1], start[2])
                    else:
                        start = adjusted_start
                else:
                    logging.warning("Cannot find non-obstacle start point")
                    return None
            
            if end_is_obstacle:
                adjusted_end = self._find_nearest_non_obstacle(
                    end[:2] if len(end) > 2 else end,
                    max_radius=10
                )
                if adjusted_end:
                    logging.debug(f"Adjusted end to: {adjusted_end}")
                    if len(end) > 2:
                        end = (adjusted_end[0], adjusted_end[1], end[2])
                    else:
                        end = adjusted_end
                else:
                    logging.warning("Cannot find non-obstacle end point")
                    return None
            
            # Extract 2D coordinates for A*
            start_point = start[:2] if len(start) > 2 else start
            end_point = end[:2] if len(end) > 2 else end
            
            # Initialize priority queue with start node
            open_set = [(0, 0, start_point)]  # (f, g, point)
            open_set_hash = {start_point}
            
            # Initialize cost dictionaries
            g_score = {start_point: 0}
            f_score = {start_point: self._heuristic(start_point, end_point)}
            
            # Path tracking
            came_from = {}
            
            # A* main loop
            iterations = 0
            while open_set and iterations < MAX_ITERATIONS:
                iterations += 1
                
                # Get node with lowest f-score
                current_f, current_g, current = heapq.heappop(open_set)
                open_set_hash.remove(current)
                
                # Check if reached goal
                if self._close_enough(current, end_point, threshold=2.0):
                    path = self._reconstruct_path(came_from, current, end_point)
                    logging.debug(f"A* search successful: {iterations} iterations, path length: {len(path)}")
                    return path
                
                # Explore neighbors
                for dx, dy in self.directions:
                    neighbor = (current[0] + dx, current[1] + dy)
                    
                    # Check if neighbor is valid
                    if not (0 <= neighbor[0] < self.map_size and 0 <= neighbor[1] < self.map_size):
                        continue
                    
                    # Check if neighbor is obstacle
                    if self._is_obstacle_fast(neighbor):
                        continue
                    
                    # For diagonal movement, check if corners are passable
                    if dx != 0 and dy != 0:
                        if self._is_obstacle_fast((current[0] + dx, current[1])) or \
                           self._is_obstacle_fast((current[0], current[1] + dy)):
                            continue
                    
                    # Calculate tentative g-score
                    move_cost = self.move_costs.get((dx, dy), 1.0)
                    tentative_g = g_score[current] + move_cost
                    
                    # If we found a better path to neighbor
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        # Record best path
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, end_point)
                        
                        # Add to open set if not already there
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
            
            # Search failed
            logging.warning(f"A* search failed after {iterations} iterations")
            return None
            
        except Exception as e:
            logging.error(f"A* search error: {str(e)}")
            return None
            
    def _hybrid_astar(self, start, end, vehicle=None):
        """Hybrid A* search with Reeds-Shepp curves"""
        try:
            logging.debug(f"Starting Hybrid A* search: {start} -> {end}")
            
            # Make sure start and end have orientation
            if isinstance(start, tuple) and len(start) < 3:
                start = (start[0], start[1], 0)
            if isinstance(end, tuple) and len(end) < 3:
                # Compute reasonable end orientation
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                end_theta = math.atan2(dy, dx) if (abs(dx) > 0.001 or abs(dy) > 0.001) else 0
                end = (end[0], end[1], end_theta)
            
            # Initialize the distance map (used for improved heuristic)
            self._initialize_distance_map(end)
            
            # First try using RS curve directly
            rs_length, rs_path = self._get_reeds_shepp_path(start, end)
            if rs_path and len(rs_path) > 1 and rs_length < float('inf'):
                logging.debug("Direct Reed-Shepp path found!")
                return rs_path
            
            # If direct path failed, use Hybrid A*
            logging.debug("Reed-Shepp direct connection failed, using Hybrid A* search")
            
            # Create start node
            start_node = HybridAStarNode(
                start[0], start[1], start[2], 0, 
                self._improved_heuristic(HybridAStarNode(*start), end)
            )
            
            # Initialize priority queue and visited set
            open_set = []
            heapq.heappush(open_set, start_node)
            closed_set = set()  # Set of visited discretized states
            
            # Store best node and minimum distance
            best_node = start_node
            best_distance = self._improved_heuristic(start_node, end)
            
            iterations = 0
            expansion_count = 0
            
            while open_set and iterations < MAX_ITERATIONS:
                iterations += 1
                expansion_count += 1
                
                # Get current node with lowest cost
                current = heapq.heappop(open_set)
                
                # Check if reached goal
                if self._is_goal_reached(current, end, tolerance=3.0, angle_tolerance=0.5):
                    logging.debug(f"Hybrid A* search successful: {iterations} iterations")
                    path = self._reconstruct_hybrid_path(current, end)
                    return path
                
                # Update best node if closer to goal
                current_distance = self._improved_heuristic(current, end)
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_node = current
                
                # Mark current node as visited
                state_key = self._discretize_state(current.x, current.y, current.theta)
                if state_key in closed_set:
                    continue
                closed_set.add(state_key)
                
                # Try RS curve connection every few expansions
                if expansion_count % self.analytic_expansion_step == 0:
                    current_state = (current.x, current.y, current.theta)
                    rs_length, rs_path = self._get_reeds_shepp_path(current_state, end)
                    
                    if rs_path and len(rs_path) > 1 and rs_length < float('inf'):
                        # Found a valid RS curve to goal
                        logging.debug(f"Found Reed-Shepp connection at iteration {iterations}")
                        # Complete path: current path + RS curve
                        hybrid_path = self._reconstruct_hybrid_path(current, None)[:-1] + rs_path
                        return hybrid_path
                
                # Get neighbor nodes
                neighbors = self._get_neighbors(current)
                
                for neighbor in neighbors:
                    # Compute heuristic cost
                    neighbor.h = self._improved_heuristic(neighbor, end)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    # Check if state is already visited
                    neighbor_state = self._discretize_state(neighbor.x, neighbor.y, neighbor.theta)
                    if neighbor_state in closed_set:
                        continue
                    
                    # Add to open set
                    heapq.heappush(open_set, neighbor)
            
            # If max iterations reached but no path found, return path to closest node
            logging.warning(f"Hybrid A* search failed after {iterations} iterations. Returning best partial path.")
            logging.debug(f"Best path distance to goal: {best_distance:.2f}")
            
            if best_node is not start_node:
                partial_path = self._reconstruct_hybrid_path(best_node, None)
                
                # Try connecting best node to goal using RS curve
                best_state = (best_node.x, best_node.y, best_node.theta)
                rs_length, rs_path = self._get_reeds_shepp_path(best_state, end)
                
                if rs_path and len(rs_path) > 1 and rs_length < float('inf'):
                    logging.debug("Using Reed-Shepp curve to complete partial path")
                    final_path = partial_path[:-1] + rs_path
                    return final_path
                
                return partial_path
            
            return None
            
        except Exception as e:
            logging.error(f"Hybrid A* search error: {str(e)}")
            return None
    
    def _initialize_distance_map(self, goal):
        """Initialize distance map from goal for improved heuristic"""
        # Skip if goal doesn't have position
        if not goal or len(goal) < 2:
            return
            
        # Extract goal position
        goal_x, goal_y = int(goal[0]), int(goal[1])
        
        # Simple BFS to calculate distances
        distance_map = {}
        queue = [(goal_x, goal_y, 0)]  # (x, y, distance)
        visited = set([(goal_x, goal_y)])
        
        # BFS traversal directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), 
                      (1, 1), (-1, 1), (1, -1), (-1, -1)]
        
        while queue:
            x, y, dist = queue.pop(0)
            
            # Store distance
            distance_map[(x, y)] = dist
            
            # Process neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                # Skip if out of bounds
                if not (0 <= nx < self.map_size and 0 <= ny < self.map_size):
                    continue
                    
                # Skip if visited
                if (nx, ny) in visited:
                    continue
                    
                # Skip if obstacle
                if self._is_obstacle_fast((nx, ny)):
                    continue
                
                # Calculate movement cost (diagonal is √2)
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                
                # Add to queue and mark as visited
                queue.append((nx, ny, dist + move_cost))
                visited.add((nx, ny))
        
        # Store distance map
        self.distance_map = distance_map
    
    def _improved_heuristic(self, node, goal):
        """
        Improved heuristic combining Euclidean distance, distance map, 
        angle difference, and RS curve length
        """
        # Extract goal coordinates
        goal_x, goal_y = goal[0], goal[1]
        goal_theta = goal[2] if len(goal) > 2 else 0
        
        # Basic Euclidean distance
        dx = node.x - goal_x
        dy = node.y - goal_y
        euclidean_dist = math.sqrt(dx*dx + dy*dy)
        
        # Angle difference penalty
        angle_diff = abs(node.theta - goal_theta)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        angle_penalty = angle_diff * self.turning_radius * 0.4
        
        # RS curve heuristic (if enabled)
        rs_dist = float('inf')
        if self.use_rs_heuristic and euclidean_dist < 30.0:
            rs_path = self._get_reeds_shepp_path(
                (node.x, node.y, node.theta),
                goal
            )
            if rs_path and rs_path[0] < float('inf'):
                rs_dist = rs_path[0]
        
        # Use distance map if available
        if hasattr(self, 'distance_map') and self.distance_map:
            node_pos = (int(node.x), int(node.y))
            if node_pos in self.distance_map:
                distance_value = self.distance_map[node_pos]
                
                # Combine all heuristic values
                if rs_dist < float('inf'):
                    # If RS curve available, give it high weight
                    return 0.1 * distance_value + 0.3 * euclidean_dist + 0.2 * angle_penalty + 0.4 * rs_dist
                else:
                    # Otherwise use distance map and Euclidean distance
                    return 0.2 * distance_value + 0.5 * euclidean_dist + 0.3 * angle_penalty
        
        # If distance map not available but RS curve is
        if rs_dist < float('inf'):
            return 0.6 * rs_dist + 0.4 * (euclidean_dist + angle_penalty)
        
        # Basic heuristic
        return euclidean_dist + angle_penalty
    
    def _get_neighbors(self, node):
        """Get all possible successor nodes"""
        neighbors = []
        
        # Try different step sizes
        for step_size in self.step_sizes:
            # Try different steering angles
            for steering_angle in self.steering_angles:
                # Calculate next state using bicycle model
                x, y, theta = self._bicycle_model(
                    node.x, node.y, node.theta, 
                    steering_angle, step_size
                )
                
                # Check if new state is valid
                if self._is_state_valid(x, y, theta):
                    # Calculate movement cost (with steering penalty)
                    turn_penalty = abs(steering_angle) * 3.0
                    move_cost = step_size * (1.0 + turn_penalty)
                    
                    # Add extra penalty for large steering angle changes
                    if node.parent is not None:
                        prev_steering = self._estimate_steering_angle(
                            node.parent.x, node.parent.y, 
                            node.x, node.y, node.theta
                        )
                        steering_change = abs(prev_steering - steering_angle)
                        move_cost += steering_change * 2.5
                    
                    new_node = HybridAStarNode(x, y, theta, node.g + move_cost, 0, node)
                    neighbors.append(new_node)
        
        # Add in-place rotation operations
        for delta_theta in [-0.2, 0.2]:
            new_theta = (node.theta + delta_theta) % (2 * math.pi)
            # Cost for in-place rotation
            turn_cost = 0.8 * abs(delta_theta)
            
            if self._is_state_valid(node.x, node.y, new_theta):
                new_node = HybridAStarNode(node.x, node.y, new_theta, node.g + turn_cost, 0, node)
                neighbors.append(new_node)
        
                    
        return neighbors
    
    def _estimate_steering_angle(self, x1, y1, x2, y2, theta):
        """Estimate steering angle from (x1,y1) to (x2,y2)"""
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate movement direction
        move_angle = math.atan2(dy, dx)
        
        # Calculate angle difference
        angle_diff = move_angle - theta
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # Convert to steering angle
        return math.atan2(angle_diff, self.wheel_base)
        
    def _bicycle_model(self, x, y, theta, steering_angle, distance):
        """
        Calculate next state using bicycle motion model
        
        Args:
            x, y, theta: Current position and heading
            steering_angle: Steering angle (radians)
            distance: Movement distance
            
        Returns:
            tuple: New position and heading (x_new, y_new, theta_new)
        """
        # Check for small steering angle (nearly straight movement)
        if abs(steering_angle) < 1e-3:
            next_x = x + distance * math.cos(theta)
            next_y = y + distance * math.sin(theta)
            next_theta = theta
            return (next_x, next_y, next_theta)
        
        # Calculate turning radius
        turning_radius = self.wheel_base / math.tan(abs(steering_angle))
        
        # Calculate rotation center
        if steering_angle > 0:  # Left turn
            cx = x - turning_radius * math.sin(theta)
            cy = y + turning_radius * math.cos(theta)
            turn_direction = 1
        else:  # Right turn
            cx = x + turning_radius * math.sin(theta)
            cy = y - turning_radius * math.cos(theta)
            turn_direction = -1
            
        # Calculate rotation angle
        beta = distance / turning_radius
        
        # Calculate new heading
        next_theta = (theta + turn_direction * beta) % (2 * math.pi)
        
        # Calculate new position
        next_x = cx + turning_radius * math.sin(next_theta)
        next_y = cy - turning_radius * math.cos(next_theta)
        
        return (next_x, next_y, next_theta)
    
    def _is_goal_reached(self, node, goal, tolerance=3.0, angle_tolerance=0.5):
        """Check if goal is reached"""
        # Extract goal position
        goal_x, goal_y = goal[0], goal[1]
        goal_theta = goal[2] if len(goal) > 2 else 0
        
        # Check position distance
        dx = node.x - goal_x
        dy = node.y - goal_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check angle difference
        angle_diff = abs(node.theta - goal_theta)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        
        # Check if within tolerance
        return distance <= tolerance and angle_diff <= angle_tolerance
    
    def _discretize_state(self, x, y, theta):
        """Discretize state for visited nodes tracking"""
        # Use grid resolution to discretize position
        x_grid = int(x / self.grid_resolution)
        y_grid = int(y / self.grid_resolution)
        
        # Discretize angle (36 bins, every 10 degrees)
        theta_grid = int(theta / (math.pi / 18)) % 36
        
        return (x_grid, y_grid, theta_grid)
    
    def _reconstruct_hybrid_path(self, node, end=None):
        """Reconstruct path from end node"""
        path = []
        current = node
        
        # Trace path from end to start
        while current is not None:
            path.append((current.x, current.y, current.theta))
            current = current.parent
            
        # Reverse path to get start to end order
        path.reverse()
        
        # Add final goal point if provided
        if end and (len(path) == 0 or path[-1][:2] != end[:2]):
            if len(end) > 2:
                path.append(end)
            else:
                # If end doesn't have orientation, use last node's orientation
                path.append((end[0], end[1], path[-1][2] if path else 0))
        
        # Apply path smoothing
        if self.path_smoothing and len(path) > 2:
            path = self._smooth_path(path)
        
        return path
        
    def _reconstruct_path(self, came_from, current, end):
        """Reconstruct path from A* search"""
        path = [current]
        
        # Trace path from end to start
        while current in came_from:
            current = came_from[current]
            path.append(current)
            
        # Reverse path to get start to end order
        path.reverse()
        
        # Add final goal point if not already included
        if path[-1] != end:
            path.append(end)
            
        return path
    
    def _smooth_path(self, path):
        """Smooth path using combination of approaches"""
        if len(path) <= 2:
            return path
        
        # First pass: use Douglas-Peucker algorithm
        smoothed_path = self._douglas_peucker(path, 2.0)
        
        # Second pass: apply relaxation smoothing
        if len(smoothed_path) > 3:
            smoothed_path = self._relaxation_smoothing(smoothed_path)
        
        # Ensure path safety by checking for obstacle collisions
        safe_path = [smoothed_path[0]]
        
        for i in range(1, len(smoothed_path)):
            prev = safe_path[-1]
            curr = smoothed_path[i]
            
            # Check for obstacles between points
            if self._is_path_segment_valid(prev, curr):
                safe_path.append(curr)
            else:
                # Find original path points between these two points
                orig_idx_prev = -1
                orig_idx_curr = -1
                
                for j, point in enumerate(path):
                    if self._points_equal(point, prev):
                        orig_idx_prev = j
                    if self._points_equal(point, curr):
                        orig_idx_curr = j
                
                if orig_idx_prev >= 0 and orig_idx_curr >= 0:
                    # Add original path points to navigate around obstacles
                    for j in range(orig_idx_prev+1, orig_idx_curr):
                        safe_path.append(path[j])
                    safe_path.append(curr)
                else:
                    # Fallback: just add the current point
                    safe_path.append(curr)
        
        return safe_path
    
    def _douglas_peucker(self, points, epsilon):
        """Douglas-Peucker simplification algorithm"""
        if len(points) <= 2:
            return points
        
        # Find point with maximum distance
        dmax = 0
        index = 0
        start, end = points[0], points[-1]
        
        for i in range(1, len(points) - 1):
            d = self._perpendicular_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance > epsilon, recursively simplify
        if dmax > epsilon:
            # Recursive call
            rec1 = self._douglas_peucker(points[:index+1], epsilon)
            rec2 = self._douglas_peucker(points[index:], epsilon)
            
            # Concatenate results
            return rec1[:-1] + rec2
        else:
            return [points[0], points[-1]]
    
    def _perpendicular_distance(self, point, line_start, line_end):
        """Calculate perpendicular distance from point to line"""
        # Extract coordinates (handle both 2D and 3D points)
        px = point[0]
        py = point[1]
        
        line_start_x = line_start[0]
        line_start_y = line_start[1]
        
        line_end_x = line_end[0]
        line_end_y = line_end[1]
        
        # Line length
        line_length = math.sqrt(
            (line_end_x - line_start_x)**2 + 
            (line_end_y - line_start_y)**2
        )
        
        if line_length < 1e-6:
            return math.sqrt((px - line_start_x)**2 + (py - line_start_y)**2)
        
        # Calculate distance
        numerator = abs(
            (line_end_y - line_start_y) * px - 
            (line_end_x - line_start_x) * py + 
            line_end_x * line_start_y - 
            line_end_y * line_start_x
        )
        
        return numerator / line_length
    
    def _relaxation_smoothing(self, path):
        """Path smoothing using relaxation technique"""
        if len(path) <= 2:
            return path
        
        # Copy original path
        smoothed_path = path.copy()
        
        # Perform multiple iterations of smoothing
        for _ in range(self.smoothing_iterations):
            # Create new path starting with original endpoints
            new_path = [smoothed_path[0]]
            
            # Smooth intermediate points
            for i in range(1, len(smoothed_path) - 1):
                prev = smoothed_path[i-1]
                curr = smoothed_path[i]
                next_p = smoothed_path[i+1]
                
                # Weighted average for position
                x = curr[0] * (1 - self.smoothing_factor) + (prev[0] + next_p[0]) * self.smoothing_factor / 2
                y = curr[1] * (1 - self.smoothing_factor) + (prev[1] + next_p[1]) * self.smoothing_factor / 2
                
                # Calculate new heading
                if len(curr) > 2:
                    # For 3D points with orientation
                    dx = next_p[0] - prev[0]
                    dy = next_p[1] - prev[1]
                    
                    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                        theta = math.atan2(dy, dx)
                    else:
                        theta = curr[2]
                    
                    new_point = (x, y, theta)
                else:
                    # For 2D points
                    new_point = (x, y)
                
                # Only add point if it's valid and doesn't collide with obstacles
                if self._is_state_valid(x, y, theta if len(curr) > 2 else 0):
                    new_path.append(new_point)
                else:
                    # Keep original point if smoothed point is invalid
                    new_path.append(curr)
            
            # Add final point
            new_path.append(smoothed_path[-1])
            smoothed_path = new_path
        
        return smoothed_path
    
    def _is_path_segment_valid(self, p1, p2):
        """Check if path segment is valid (no obstacles)"""
        # Extract 2D coordinates
        p1_2d = p1[:2] if len(p1) > 2 else p1
        p2_2d = p2[:2] if len(p2) > 2 else p2
        
        # Sample points along the line
        line_points = self._bresenham_line(p1_2d, p2_2d)
        
        # Check if any point is an obstacle
        for point in line_points:
            if self._is_obstacle_fast(point):
                return False
        
        return True
    
    def _bresenham_line(self, start, end):
        """Generate points along a line using Bresenham's algorithm"""
        x1, y1 = int(round(start[0])), int(round(start[1]))
        x2, y2 = int(round(end[0])), int(round(end[1]))
        
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
    
    def _is_state_valid(self, x, y, theta):
        """Check if state is valid (not in collision)"""
        # Basic check - vehicle center point
        if self._is_obstacle_fast((x, y)):
            return False
        
        # Vehicle dimensions
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        
        # Vehicle corners
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        corners = [
            (half_length, half_width),   # Front right
            (half_length, -half_width),  # Front left
            (-half_length, half_width),  # Rear right
            (-half_length, -half_width)  # Rear left
        ]
        
        # Check all corners
        for corner in corners:
            corner_x = x + corner[0] * cos_theta - corner[1] * sin_theta
            corner_y = y + corner[0] * sin_theta + corner[1] * cos_theta
            
            if self._is_obstacle_fast((corner_x, corner_y)):
                return False
        
        return True
    
    def _is_obstacle_fast(self, point):
        """Quick check if point is an obstacle"""
        # Convert to integers
        x, y = int(round(point[0])), int(round(point[1]))
        
        # Check map boundaries
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return True
        
        # Check obstacle grids
        if (x, y) in self.obstacle_grids:
            return True
        
        # Check map service
        if hasattr(self.map_service, 'is_obstacle') and callable(getattr(self.map_service, 'is_obstacle')):
            return self.map_service.is_obstacle(point)
        
        return False
    
    def _find_nearest_non_obstacle(self, point, max_radius=5):
        """Find nearest non-obstacle point"""
        # Check if point itself is not an obstacle
        if not self._is_obstacle_fast(point):
            return point
        
        # Search in expanding radius
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    # Only check points at radius r
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    
                    candidate = (point[0] + dx, point[1] + dy)
                    if not self._is_obstacle_fast(candidate):
                        return candidate
        
        return None
    
    def _heuristic(self, p1, p2):
        """Basic heuristic function (Euclidean distance)"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _points_equal(self, p1, p2, tolerance=EPSILON):
        """Check if points are equal (with tolerance)"""
        # Handle both 2D and 3D points
        p1_2d = p1[:2] if len(p1) > 2 else p1
        p2_2d = p2[:2] if len(p2) > 2 else p2
        
        return (abs(p1_2d[0] - p2_2d[0]) < tolerance and 
                abs(p1_2d[1] - p2_2d[1]) < tolerance)
    
    def _close_enough(self, p1, p2, threshold=3.0):
        """Check if points are close enough"""
        return math.dist(p1, p2) <= threshold
    
    def _validate_point(self, point):
        """Validate and normalize point coordinates"""
        if isinstance(point, tuple) and len(point) >= 2:
            if len(point) > 2:
                return (float(point[0]), float(point[1]), float(point[2]))
            return (float(point[0]), float(point[1]))
        elif hasattr(point, 'x') and hasattr(point, 'y'):
            if hasattr(point, 'theta'):
                return (float(point.x), float(point.y), float(point.theta))
            return (float(point.x), float(point.y))
        elif isinstance(point, (list, np.ndarray)) and len(point) >= 2:
            if len(point) > 2:
                return (float(point[0]), float(point[1]), float(point[2]))
            return (float(point[0]), float(point[1]))
        else:
            # Invalid point warning
            logging.warning(f"Invalid point: {point}, using (0,0)")
            return (0.0, 0.0)
    
    def _create_cache_key(self, start, end, vehicle):
        """Create cache key for path caching"""
        if vehicle:
            # Include vehicle properties in cache key
            return (
                "path",
                start,
                end,
                getattr(vehicle, 'turning_radius', 0), 
                getattr(vehicle, 'min_hardness', 0),
                getattr(vehicle, 'current_load', 0)
            )
        else:
            # Basic cache key
            return ("base_path", start, end)
    
    def _generate_fallback_path(self, start, end):
        """Generate fallback path when planning fails"""
        # Normalize coordinates
        start_2d = start[:2] if len(start) > 2 else start
        end_2d = end[:2] if len(end) > 2 else end
        
        # Direct connection
        if not self._is_obstacle_fast(start_2d) and not self._is_obstacle_fast(end_2d):
            # Check if direct line doesn't pass through obstacles
            line_points = self._bresenham_line(start_2d, end_2d)
            if not any(self._is_obstacle_fast(p) for p in line_points):
                if len(start) > 2 or len(end) > 2:
                    # Preserve orientation information
                    if len(start) > 2 and len(end) > 2:
                        return [start, end]
                    elif len(start) > 2:
                        return [start, (end[0], end[1], start[2])]
                    else:
                        return [(start[0], start[1], end[2]), end]
                else:
                    return [start, end]
        
        # Try different waypoints to avoid obstacles
        waypoints = []
        
        # Calculate direction vector
        dx = end_2d[0] - start_2d[0]
        dy = end_2d[1] - start_2d[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1e-6:
            return [start, end]
        
        # Normalize direction
        dx /= distance
        dy /= distance
        
        # Try perpendicular offsets
        perpendicular_x = -dy
        perpendicular_y = dx
        
        # Try different offset distances
        offsets = [0.3, 0.5, 0.7]
        for t in [0.33, 0.67]:  # Try at 1/3 and 2/3 of the path
            for offset_scale in offsets:
                offset = distance * offset_scale
                
                # Try both perpendicular directions
                for direction in [1, -1]:
                    waypoint_x = start_2d[0] + t * dx * distance + direction * perpendicular_x * offset
                    waypoint_y = start_2d[1] + t * dy * distance + direction * perpendicular_y * offset
                    
                    waypoint = (waypoint_x, waypoint_y)
                    
                    # Check if waypoint is valid
                    if not self._is_obstacle_fast(waypoint):
                        # Check if connections are valid
                        if not any(self._is_obstacle_fast(p) for p in self._bresenham_line(start_2d, waypoint)) and \
                           not any(self._is_obstacle_fast(p) for p in self._bresenham_line(waypoint, end_2d)):
                            waypoints.append(waypoint)
        
        # Find the best waypoint (closest to the direct line)
        if waypoints:
            best_waypoint = min(waypoints, key=lambda w: self._perpendicular_distance(w, start_2d, end_2d))
            
            # Create path with the waypoint
            if len(start) > 2 or len(end) > 2:
                # Handle orientation
                if len(start) > 2 and len(end) > 2:
                    mid_theta = (start[2] + end[2]) / 2
                    return [start, (best_waypoint[0], best_waypoint[1], mid_theta), end]
                elif len(start) > 2:
                    return [start, (best_waypoint[0], best_waypoint[1], start[2]), (end[0], end[1], start[2])]
                else:
                    return [(start[0], start[1], end[2]), (best_waypoint[0], best_waypoint[1], end[2]), end]
            else:
                return [start, best_waypoint, end]
        
        # If all else fails, return direct connection
        if len(start) > 2 or len(end) > 2:
            if len(start) > 2 and len(end) > 2:
                return [start, end]
            elif len(start) > 2:
                return [start, (end[0], end[1], start[2])]
            else:
                return [(start[0], start[1], end[2]), end]
        else:
            return [start, end]
    
    #------------ Reed-Shepp curves related functions ------------#
    
    def _get_reeds_shepp_path(self, start, goal):
        """
        Calculate shortest Reed-Shepp path
        
        Args:
            start: Start state (x, y, theta)
            goal: Goal state (x, y, theta)
            
        Returns:
            (path_length, path): Path length and path points
        """
        try:
            # Convert global coordinates to local coordinate system with start as origin
            goal_x, goal_y = goal[0], goal[1]
            goal_theta = goal[2] if len(goal) > 2 else 0
            
            start_x, start_y = start[0], start[1]
            start_theta = start[2] if len(start) > 2 else 0
            
            # Calculate relative position
            dx = goal_x - start_x
            dy = goal_y - start_y
            
            # Rotate to align with start orientation
            c = math.cos(start_theta)
            s = math.sin(start_theta)
            
            local_x = dx * c + dy * s
            local_y = -dx * s + dy * c
            
            # Relative orientation
            local_theta = self._normalize_angle(goal_theta - start_theta)
            
            # Scale by turning radius
            scaled_x = local_x / self.turning_radius
            scaled_y = local_y / self.turning_radius
            
            # Compute all possible RS curves
            paths = self._compute_rs_curves(scaled_x, scaled_y, local_theta)
            
            if not paths:
                return float('inf'), None
            
            # Sort by path length
            paths.sort(key=lambda p: p[0])
            
            # Check if shortest path is valid
            for path_length, path_type, controls in paths:
                # Convert RS curve to global coordinates
                path = self._generate_rs_path(start, path_type, controls)
                
                # Check path validity
                if self._check_rs_path_validity(path):
                    return path_length * self.turning_radius, path
            
            # No valid path found
            return float('inf'), None
            
        except Exception as e:
            logging.error(f"Reed-Shepp path calculation error: {str(e)}")
            return float('inf'), None
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _compute_rs_curves(self, x, y, phi):
        """
        Compute all possible Reed-Shepp curves
        
        Args:
            x, y: Target position in local coordinates (normalized)
            phi: Target orientation (relative to start)
            
        Returns:
            list: All possible paths (length, type, controls)
        """
        paths = []
        
        # CSC paths (Curve-Straight-Curve)
        self._add_csc_paths(x, y, phi, paths)
        
        # CCC paths (Curve-Curve-Curve)
        self._add_ccc_paths(x, y, phi, paths)
        
        return paths
    
    def _add_csc_paths(self, x, y, phi, paths):
        """Add CSC type paths"""
        # LSL (Left-Straight-Left)
        t, u, v = self._compute_lsl(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LSL', (t, u, v)))
        
        # RSR (Right-Straight-Right)
        t, u, v = self._compute_rsr(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RSR', (t, u, v)))
        
        # LSR (Left-Straight-Right)
        t, u, v = self._compute_lsr(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LSR', (t, u, v)))
        
        # RSL (Right-Straight-Left)
        t, u, v = self._compute_rsl(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RSL', (t, u, v)))
    
    def _add_ccc_paths(self, x, y, phi, paths):
        """Add CCC type paths"""
        # LRL (Left-Right-Left)
        t, u, v = self._compute_lrl(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LRL', (t, u, v)))
        
        # RLR (Right-Left-Right)
        t, u, v = self._compute_rlr(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RLR', (t, u, v)))
    
    def _compute_lsl(self, x, y, phi):
        """Compute Left-Straight-Left path parameters"""
        # Center of the first circle
        cx = 0
        cy = 1
        
        # Center of the last circle
        dx = x - math.sin(phi)
        dy = y - 1 + math.cos(phi)
        
        # Distance between circle centers
        d = math.sqrt(dx*dx + dy*dy)
        
        if d < 1e-10:
            return None, None, None
        
        # Path parameters
        theta = math.atan2(dy, dx)
        alpha = math.acos(1/d)  # Half angle of the tangent
        
        t = self._normalize_angle(theta + alpha)
        u = self._normalize_angle(phi - theta - alpha)
        v = d * math.sin(alpha)
        
        return t, v, u
    
    def _compute_rsr(self, x, y, phi):
        """Compute Right-Straight-Right path parameters"""
        # Mirror of LSL
        t, u, v = self._compute_lsl(x, -y, -phi)
        if t is not None:
            return -t, u, -v
        return None, None, None
    
    def _compute_lsr(self, x, y, phi):
        """Compute Left-Straight-Right path parameters"""
        # Centers of the circles
        cx1 = 0
        cy1 = 1
        cx2 = x + math.sin(phi)
        cy2 = y - 1 - math.cos(phi)
        
        # Distance between circle centers
        dx = cx2 - cx1
        dy = cy2 - cy1
        d = math.sqrt(dx*dx + dy*dy)
        
        if d < 2:
            return None, None, None
        
        # Path parameters
        theta = math.atan2(dy, dx)
        alpha = math