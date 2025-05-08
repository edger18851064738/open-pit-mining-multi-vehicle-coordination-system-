import os
import sys
import math
import time
import logging
import threading
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
# Import your existing modules
from models.vehicle import MiningVehicle, VehicleState, TransportStage



# Hybrid A* Node class for path planning
class HybridAStarNode:
    """Hybrid A* algorithm node class"""
    
    def __init__(self, x, y, theta, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta  # Heading angle (radians)
        self.g = g          # Actual cost from start to current node
        self.h = h          # Estimated cost from current node to goal
        self.f = g + h      # Total cost
        self.parent = parent
        
    def __lt__(self, other):
        """Override less than operator for priority queue"""
        return self.f < other.f
    
    def __eq__(self, other):
        """Override equals operator"""
        if isinstance(other, HybridAStarNode):
            return (self.x == other.x and 
                    self.y == other.y and 
                    abs(self.theta - other.theta) < 0.1)  # Angles within 0.1 radians are considered equal
        return False
    
    def __hash__(self):
        """Define hash method for use in sets"""
        # Discretize continuous space for hashing
        grid_size = 0.5  # Smaller discretization interval
        angle_grid = 0.05  # Smaller angle discretization interval
        x_grid = int(self.x / grid_size)
        y_grid = int(self.y / grid_size)
        theta_grid = int(self.theta / angle_grid)
        return hash((x_grid, y_grid, theta_grid))

class HybridPathPlanner:
    """
    Enhanced Hybrid A* Path Planner with Reeds-Shepp curves
    
    This planner combines:
    1. A* search for global path planning
    2. Vehicle kinematic constraints for realistic motion
    3. Reeds-Shepp curves for optimized paths
    4. Path smoothing for drivable trajectories
    """
    
    def __init__(self, env=None, vehicle_length=5.0, vehicle_width=2.0, 
                 turning_radius=5.0, step_size=0.8, grid_resolution=0.3):
        """
        Initialize the hybrid path planner
        
        Args:
            env: Environment object or map service
            vehicle_length: Vehicle length
            vehicle_width: Vehicle width
            turning_radius: Minimum turning radius
            step_size: Step size for movement
            grid_resolution: Grid resolution for acceleration
        """
        self.env = env
        self.vehicle_length = vehicle_length
        self.vehicle_width = vehicle_width
        self.turning_radius = turning_radius
        self.step_size = step_size
        self.grid_resolution = grid_resolution
        
        # Use more steering angles for better planning flexibility
        self.steering_angles = np.linspace(-0.7, 0.7, 15)
        
        # Vehicle motion model parameters - wheelbase is 60% of vehicle length
        self.wheel_base = vehicle_length * 0.6
        
        # Add different step size options for narrower areas
        self.step_sizes = [step_size, step_size*0.5]
        
        # Initialize distance map
        self.distance_map = None
        self.compute_distance_map = False  # Control whether to compute distance map
        
        # Path smoothing parameters
        self.path_smoothing = True  # Enable path smoothing
        self.smoothing_factor = 0.5  # Smoothing factor
        self.smoothing_iterations = 10  # Smoothing iterations
        
        # RS curve parameters
        self.rs_step_size = 0.2  # RS curve discretization step
        self.use_rs_heuristic = True  # Use RS curves as heuristic
        self.analytic_expansion_step = 5  # Try using RS curves to connect to target every N expansions
        
        # Initialize map dimensions if not from environment
        if env is None:
            self.width = 200  # Default map width
            self.height = 200  # Default map height
            self.grid = np.zeros((self.width, self.height))  # Empty grid
        else:
            if hasattr(env, 'width') and hasattr(env, 'height'):
                self.width = env.width
                self.height = env.height
                self.grid = env.grid if hasattr(env, 'grid') else np.zeros((self.width, self.height))
            else:
                # Try to get dimensions from map_service
                self.width = getattr(env, 'grid_size', 200)
                self.height = getattr(env, 'grid_size', 200)
                self.grid = np.zeros((self.width, self.height))
        
        # Support for map_service interface
        self.obstacle_grids = set()
        self.dispatch = None  # Will be set by DispatchSystem
                
    def plan_path(self, start, end, vehicle=None, max_iterations=5000):
        """
        Plan a path from start to end positions
        
        Args:
            start: Start position (x, y, theta) or vehicle
            end: End position (x, y, theta)
            vehicle: Optional vehicle object with constraints
            max_iterations: Maximum iterations for planning
            
        Returns:
            List of path points [(x, y, theta), ...] or None if no path found
        """
        # Normalize inputs
        if isinstance(start, MiningVehicle):
            vehicle = start
            vehicle_pos = vehicle.current_location
            if hasattr(vehicle, 'heading'):
                theta = vehicle.heading
            else:
                theta = 0.0  # Default heading if not available
            start = (vehicle_pos[0], vehicle_pos[1], theta)
        elif len(start) == 2:
            # If only x,y provided, add default heading
            start = (start[0], start[1], 0.0)
            
        if len(end) == 2:
            # If only x,y provided, add default heading
            end = (end[0], end[1], 0.0)
        
        # First try using RS curves directly
        rs_length, rs_path = self.get_reeds_shepp_path(start, end)
        if rs_path and len(rs_path) > 1 and rs_length < float('inf') and self.check_rs_path_validity(rs_path):
            logging.info("Using Reeds-Shepp curve for direct path")
            return rs_path
            
        # If RS direct path fails, use hybrid A* algorithm
        logging.info("Reeds-Shepp direct path failed, using Hybrid A* search")
        
        # Create start node
        start_node = HybridAStarNode(start[0], start[1], start[2], 0, 
                                     self.improved_heuristic(HybridAStarNode(*start), end))
        
        # Initialize open set (priority queue) and closed set
        open_set = []
        import heapq
        heapq.heappush(open_set, start_node)
        closed_set = set()  # Store visited discretized states
        
        # Store best node and minimum distance
        best_node = start_node
        best_distance = self.improved_heuristic(start_node, end)
        
        expansion_count = 0
        
        for iterations in range(max_iterations):
            if not open_set:
                break
                
            expansion_count += 1
            
            # Get current lowest cost node
            current = heapq.heappop(open_set)
            
            # Check if goal reached
            if self.is_goal_reached(current, end):
                logging.info(f"Path found in {iterations} iterations")
                # Reconstruct and smooth path
                path = self.reconstruct_path(current)
                return self.normalize_path(path)
            
            # Try RS curves periodically to connect to goal
            if expansion_count % self.analytic_expansion_step == 0:
                current_state = (current.x, current.y, current.theta)
                rs_length, rs_path = self.get_reeds_shepp_path(current_state, end)
                
                if rs_path and len(rs_path) > 1 and rs_length < float('inf'):
                    # Build complete path: start to current + RS path
                    hybrid_path = self.reconstruct_path(current)[:-1] + rs_path
                    
                    # Validate path
                    if self.validate_path(hybrid_path):
                        logging.info(f"Found Reeds-Shepp connection at iteration {iterations}")
                        return self.normalize_path(hybrid_path)
            
            # Update best node
            current_distance = self.improved_heuristic(current, end)
            if current_distance < best_distance:
                best_distance = current_distance
                best_node = current
            
            # Mark current node as visited
            state_key = self.discretize_state(current.x, current.y, current.theta)
            if state_key in closed_set:
                continue
            closed_set.add(state_key)
            
            # Get all possible successor nodes
            neighbors = self.get_neighbors(current)
            
            for neighbor in neighbors:
                # Calculate heuristic cost
                neighbor.h = self.improved_heuristic(neighbor, end)
                neighbor.f = neighbor.g + neighbor.h
                
                # Check if already visited
                neighbor_state = self.discretize_state(neighbor.x, neighbor.y, neighbor.theta)
                if neighbor_state in closed_set:
                    continue
                
                # Add to open set
                heapq.heappush(open_set, neighbor)
        
        # If max iterations reached but no path found, return partial path
        logging.warning(f"No complete path found in {max_iterations} iterations. Returning best partial path.")
        
        if best_node is not start_node:
            partial_path = self.reconstruct_path(best_node)
            
            # Try connecting best node to goal with RS curve
            best_state = (best_node.x, best_node.y, best_node.theta)
            rs_length, rs_path = self.get_reeds_shepp_path(best_state, end)
            
            if rs_path and len(rs_path) > 1 and rs_length < float('inf'):
                logging.info("Using Reeds-Shepp curve to complete the last part of path")
                final_path = partial_path[:-1] + rs_path
                
                # Validate path
                if self.validate_path(final_path):
                    return self.normalize_path(final_path)
            
            return self.normalize_path(partial_path)
        
        # If all fails, return a simple direct path
        return [start, end]
    
    def improved_heuristic(self, node, goal):
        """
        Improved heuristic function combining Euclidean distance, angle difference and RS curve length
        
        Args:
            node: Current node
            goal: Goal position (x, y, theta)
            
        Returns:
            float: Estimated cost
        """
        # Basic Euclidean distance
        dx = node.x - goal[0]
        dy = node.y - goal[1]
        euclidean_dist = math.sqrt(dx*dx + dy*dy)
        
        # Angle difference penalty
        angle_diff = abs(node.theta - goal[2])
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)
        angle_penalty = angle_diff * self.turning_radius * 0.4
        
        # RS curve heuristic (if enabled)
        rs_dist = float('inf')
        if self.use_rs_heuristic and euclidean_dist < 30.0:  # Only use when close to goal to save computation
            rs_path = self.get_reeds_shepp_path(
                (node.x, node.y, node.theta),
                (goal[0], goal[1], goal[2])
            )
            if rs_path:
                rs_dist = rs_path[0]  # RS curve length
        
        # If RS curve available, give it high weight
        if not math.isinf(rs_dist):
            return 0.6 * rs_dist + 0.4 * (euclidean_dist + angle_penalty)
        
        # Basic heuristic
        return euclidean_dist + angle_penalty
    
    def get_neighbors(self, node):
        """
        Get all possible successor nodes
        
        Args:
            node: Current node
            
        Returns:
            list: Possible successor nodes
        """
        neighbors = []
        
        # Try different step sizes
        for step_size in self.step_sizes:
            # Try different steering angles
            for steering_angle in self.steering_angles:
                # Calculate next state using bicycle model
                x, y, theta = self.bicycle_model(node.x, node.y, node.theta, 
                                               steering_angle, step_size)
                
                # Check if new position is valid
                if self.is_state_valid(x, y, theta):
                    # Calculate movement cost (with steering penalty)
                    turn_penalty = abs(steering_angle) * 3.0
                    move_cost = step_size * (1.0 + turn_penalty)
                    
                    # Add steering change penalty
                    if node.parent is not None:
                        prev_steering = self.estimate_steering_angle(node.parent.x, node.parent.y, 
                                                                   node.x, node.y, node.theta)
                        steering_change = abs(prev_steering - steering_angle)
                        move_cost += steering_change * 2.5
                    
                    new_node = HybridAStarNode(x, y, theta, node.g + move_cost, 0, node)
                    neighbors.append(new_node)
        
        # Add in-place rotation options
        for delta_theta in [-0.2, 0.2]:
            new_theta = (node.theta + delta_theta) % (2 * math.pi)
            # In-place rotation cost
            turn_cost = 0.8 * abs(delta_theta)
            
            if self.is_state_valid(node.x, node.y, new_theta):
                new_node = HybridAStarNode(node.x, node.y, new_theta, node.g + turn_cost, 0, node)
                neighbors.append(new_node)
        
        return neighbors
    
    def estimate_steering_angle(self, x1, y1, x2, y2, theta):
        """Estimate steering angle from (x1,y1) to (x2,y2)"""
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate movement direction
        move_angle = math.atan2(dy, dx)
        
        # Calculate steering angle difference
        angle_diff = move_angle - theta
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # Simplify to steering angle (assuming bicycle model)
        return math.atan2(angle_diff, self.wheel_base)
        
    def bicycle_model(self, x, y, theta, steering_angle, distance):
        """
        Use bicycle motion model to calculate next state
        
        Args:
            x, y, theta: Current position and heading
            steering_angle: Steering angle (radians)
            distance: Movement distance
            
        Returns:
            tuple: New position and heading (x_new, y_new, theta_new)
        """
        # Move in segments, checking for collisions at each segment
        num_segments = 5  # Divide movement into 5 segments for collision checking
        segment_distance = distance / num_segments
        current_x, current_y, current_theta = x, y, theta
        
        for _ in range(num_segments):
            # If steering angle close to zero, move in straight line
            if abs(steering_angle) < 1e-3:
                next_x = current_x + segment_distance * math.cos(current_theta)
                next_y = current_y + segment_distance * math.sin(current_theta)
                next_theta = current_theta
            else:
                # Turning radius
                turning_radius = self.wheel_base / math.tan(abs(steering_angle))
                # Rotation center
                if steering_angle > 0:  # Left turn
                    cx = current_x - turning_radius * math.sin(current_theta)
                    cy = current_y + turning_radius * math.cos(current_theta)
                    turn_direction = 1
                else:  # Right turn
                    cx = current_x + turning_radius * math.sin(current_theta)
                    cy = current_y - turning_radius * math.cos(current_theta)
                    turn_direction = -1
                    
                # Calculate rotation angle
                beta = segment_distance / turning_radius
                # Calculate new heading
                next_theta = (current_theta + turn_direction * beta) % (2 * math.pi)
                # Calculate new position
                next_x = cx + turning_radius * math.sin(next_theta)
                next_y = cy - turning_radius * math.cos(next_theta)
            
            # Check if intermediate point is valid
            if not self.is_intermediate_state_valid(current_x, current_y, next_x, next_y):
                # If invalid, return current state
                return current_x, current_y, current_theta
            
            # Update current state
            current_x, current_y, current_theta = next_x, next_y, next_theta
        
        return current_x, current_y, current_theta

    def is_intermediate_state_valid(self, x1, y1, x2, y2, samples=10):
        """
        Check if path between two points is valid
        
        Args:
            x1, y1: Start coordinates
            x2, y2: End coordinates
            samples: Number of check points
            
        Returns:
            bool: Whether path is valid
        """
        for i in range(samples):
            t = (i + 1) / (samples + 1)  # Skip start and end points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Check if point is valid
            if not self.is_point_valid(x, y):
                return False
        
        return True
    
    def is_point_valid(self, x, y):
        """Check if a single point is valid"""
        # Check if point is within map bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
            
        # Check if point is on obstacle
        # Handle both grid-based and set-based obstacle representations
        if hasattr(self, 'grid') and hasattr(self.grid, 'shape'):
            if self.grid[int(x), int(y)] == 1:
                return False
        elif hasattr(self, 'obstacle_grids'):
            if (int(x), int(y)) in self.obstacle_grids:
                return False
            
        # If map_service is available, use its obstacle check
        if self.env and hasattr(self.env, 'is_obstacle'):
            if self.env.is_obstacle((x, y)):
                return False
                
        return True

    def is_state_valid(self, x, y, theta):
        """
        Check if state is valid (including vehicle shape)
        
        Args:
            x, y, theta: Position and heading
            
        Returns:
            bool: Whether state is valid
        """
        # Basic check - vehicle center point is within map bounds
        if not (0 <= x < self.width and 0 <= y < self.height):
            return False
        
        # Check if vehicle center point is on obstacle
        if not self.is_point_valid(x, y):
            return False
        
        # More precise collision detection - check vehicle outline
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        # Vehicle rectangle four corner points relative coordinates
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        corners = [
            (half_length, half_width),   # Front right
            (half_length, -half_width),  # Front left
            (-half_length, half_width),  # Rear right
            (-half_length, -half_width)  # Rear left
        ]
        
        # Check corners
        for corner in corners:
            # Rotate and translate corner to world coordinates
            corner_x = x + corner[0] * cos_theta - corner[1] * sin_theta
            corner_y = y + corner[0] * sin_theta + corner[1] * cos_theta
            
            # Check if point is valid
            if not self.is_point_valid(corner_x, corner_y):
                return False
        
        # Enhanced: Sample more points along vehicle outline
        num_samples = 10  # More sampling points per edge
        
        # Front and rear edges
        for i in range(num_samples):
            ratio = i / (num_samples - 1)
            # Front edge
            front_x = x + half_length * cos_theta + (2 * ratio - 1) * half_width * sin_theta
            front_y = y + half_length * sin_theta - (2 * ratio - 1) * half_width * cos_theta
            
            # Rear edge
            back_x = x - half_length * cos_theta + (2 * ratio - 1) * half_width * sin_theta
            back_y = y - half_length * sin_theta - (2 * ratio - 1) * half_width * cos_theta
            
            # Check front and rear edge sample points
            for point_x, point_y in [(front_x, front_y), (back_x, back_y)]:
                if not self.is_point_valid(point_x, point_y):
                    return False
        
        # Left and right edges
        for i in range(num_samples):
            ratio = i / (num_samples - 1)
            # Right edge
            right_x = x + (2 * ratio - 1) * half_length * cos_theta + half_width * sin_theta
            right_y = y + (2 * ratio - 1) * half_length * sin_theta - half_width * cos_theta
            
            # Left edge
            left_x = x + (2 * ratio - 1) * half_length * cos_theta - half_width * sin_theta
            left_y = y + (2 * ratio - 1) * half_length * sin_theta + half_width * cos_theta
            
            # Check left and right edge sample points
            for point_x, point_y in [(right_x, right_y), (left_x, left_y)]:
                if not self.is_point_valid(point_x, point_y):
                    return False
        
        return True

    def is_goal_reached(self, node, goal, tolerance=3.0, angle_tolerance=0.5):
        """Check if goal is reached
        
        Args:
            node: Current node
            goal: Goal position (x, y, theta)
            tolerance: Position error tolerance
            angle_tolerance: Angle error tolerance (radians)
            
        Returns:
            bool: Whether goal is reached
        """
        # Calculate position distance
        dx = node.x - goal[0]
        dy = node.y - goal[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Calculate angle difference
        angle_diff = abs(node.theta - goal[2])
        # Normalize angle difference to [0, pi]
        angle_diff = min(angle_diff, 2*math.pi - angle_diff) 
        
        # Check if within tolerance
        return distance <= tolerance and angle_diff <= angle_tolerance

    def discretize_state(self, x, y, theta):
        """Discretize state for visited state checking
        
        Args:
            x, y, theta: State
            
        Returns:
            tuple: Discretized state
        """
        # Use smaller grid resolution for more precision
        x_grid = int(x / self.grid_resolution)
        y_grid = int(y / self.grid_resolution)
        # Discretize angle to 36 intervals (every 10 degrees)
        theta_grid = int(theta / (math.pi / 18)) % 36
        
        return (x_grid, y_grid, theta_grid)

    def reconstruct_path(self, node):
        """Reconstruct complete path from end node
        
        Args:
            node: End node
            
        Returns:
            list: Path points list, each point is (x, y, theta)
        """
        path = []
        current = node
        
        # Trace back from end to start
        while current is not None:
            path.append((current.x, current.y, current.theta))
            current = current.parent
            
        # Reverse path to start to end
        path = path[::-1]
        
        # Apply path smoothing if enabled
        if self.path_smoothing and len(path) > 2:
            path = self.smooth_path(path)
        
        return path
    
    def smooth_path(self, path, iterations=None):
        """
        Smooth the planned path using iterative smoothing algorithm
        
        Args:
            path: Original path points list
            iterations: Smoothing iterations, if None use default value
            
        Returns:
            smoothed_path: Smoothed path
        """
        if iterations is None:
            iterations = self.smoothing_iterations
            
        if len(path) <= 2:
            return path
            
        # Copy original path
        smoothed_path = path.copy()
        
        # Perform multiple iterations of smoothing
        for _ in range(iterations):
            # Temporary store for new path (not including start and end)
            new_path = [smoothed_path[0]]
            
            # Smooth middle points
            for i in range(1, len(smoothed_path) - 1):
                prev = smoothed_path[i-1]
                curr = smoothed_path[i]
                next_p = smoothed_path[i+1]
                
                # Smooth x and y coordinates
                x = curr[0] * (1 - self.smoothing_factor) + (prev[0] + next_p[0]) * self.smoothing_factor / 2
                y = curr[1] * (1 - self.smoothing_factor) + (prev[1] + next_p[1]) * self.smoothing_factor / 2
                
                # Smooth angle - consider movement direction
                dx = next_p[0] - prev[0]
                dy = next_p[1] - prev[1]
                
                # If there's a significant direction change, use forward direction as angle
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    theta = math.atan2(dy, dx)
                else:
                    # Otherwise keep original angle
                    theta = curr[2]
                
                # Ensure new position doesn't go through walls
                if self.is_state_valid(x, y, theta) and self.is_intermediate_state_valid(prev[0], prev[1], x, y) and self.is_intermediate_state_valid(x, y, next_p[0], next_p[1]):
                    new_path.append((x, y, theta))
                else:
                    # If new position is invalid, keep original position
                    new_path.append(curr)
            
            # Add endpoint
            new_path.append(smoothed_path[-1])
            smoothed_path = new_path
        
        return smoothed_path
    
    def validate_path(self, path):
        """
        Validate if path is valid (no walls)
        
        Args:
            path: Path points list
            
        Returns:
            bool: Whether path is valid
        """
        if len(path) < 2:
            return False
            
        for i in range(len(path) - 1):
            x1, y1 = path[i][0], path[i][1]
            x2, y2 = path[i+1][0], path[i+1][1]
            
            if not self.is_intermediate_state_valid(x1, y1, x2, y2, samples=10):
                return False
        
        return True
    
    # Reeds-Shepp curves methods
    def get_reeds_shepp_path(self, start, goal):
        """
        Calculate shortest path from start to end using Reeds-Shepp curves
        
        Args:
            start: Start state (x, y, theta)
            goal: Goal state (x, y, theta)
            
        Returns:
            (path_length, path): Path length and path points list, (inf, None) if no path found
        """
        # Convert global coordinates to local coordinate system with start as origin
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        
        # Calculate rotation matrix to rotate start heading to x-axis
        c = math.cos(start[2])
        s = math.sin(start[2])
        
        # Transform goal point to local coordinates
        local_x = dx * c + dy * s
        local_y = -dx * s + dy * c
        
        # Goal heading in local coordinate system
        local_theta = self.normalize_angle(goal[2] - start[2])
        
        # Calculate normalized distance, all distances in units of turning radius
        scaled_x = local_x / self.turning_radius
        scaled_y = local_y / self.turning_radius
        
        # Calculate all possible RS curve paths
        paths = self.compute_rs_curves(scaled_x, scaled_y, local_theta)
        
        if not paths:
            return float('inf'), None
        
        # Sort by path length
        paths.sort(key=lambda p: p[0])
        
        # Check if shortest path is feasible
        for path_length, path_type, controls in paths:
            # Convert RS curve to actual path in global coordinates
            path = self.generate_rs_path(start, path_type, controls)
            
            # Check if path is valid
            if self.check_rs_path_validity(path):
                return path_length * self.turning_radius, path
        
        # If no feasible path found
        return float('inf'), None
    
    def normalize_angle(self, angle):
        """Normalize angle to [-π, π] range"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def compute_rs_curves(self, x, y, phi):
        """
        Calculate all possible Reeds-Shepp curves
        
        Args:
            x, y: Target position in local coordinate system (normalized)
            phi: Target heading (relative to start heading)
            
        Returns:
            list: All possible paths, each element is (length, type, control_params)
        """
        # Store all possible paths
        paths = []
        
        # Calculate CSC paths (Curve-Straight-Curve)
        self.CSC(x, y, phi, paths)
        
        # Calculate CCC paths (Curve-Curve-Curve)
        self.CCC(x, y, phi, paths)
        
        return paths
    
    def CSC(self, x, y, phi, paths):
        """Calculate all CSC type paths"""
        # Left-Straight-Left (LSL)
        t, u, v = self.LSL(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LSL', (t, u, v)))
        
        # Right-Straight-Right (RSR)
        t, u, v = self.RSR(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RSR', (t, u, v)))
        
        # Left-Straight-Right (LSR)
        t, u, v = self.LSR(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LSR', (t, u, v)))
        
        # Right-Straight-Left (RSL)
        t, u, v = self.RSL(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RSL', (t, u, v)))
    
    def CCC(self, x, y, phi, paths):
        """Calculate all CCC type paths"""
        # Left-Right-Left (LRL)
        t, u, v = self.LRL(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'LRL', (t, u, v)))
        
        # Right-Left-Right (RLR)
        t, u, v = self.RLR(x, y, phi)
        if t is not None and abs(t) > 1e-10 and abs(u) > 1e-10 and abs(v) > 1e-10:
            paths.append((abs(t) + abs(u) + abs(v), 'RLR', (t, u, v)))
    
    def LSL(self, x, y, phi):
        """Calculate Left-Straight-Left (LSL) path parameters"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # Calculate circle center to target vector
        cx = -math.sin(phi)
        cy = 1.0 - math.cos(phi)
        
        # Calculate u (straight segment length)
        u = math.sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy))
        
        if u < 1e-10:
            return None, None, None
        
        # Calculate t and v (two arc angles)
        theta = math.atan2(y-cy, x-cx)
        t = self.normalize_angle(theta)
        v = self.normalize_angle(phi - t)
        
        return t, u, v
    
    def RSR(self, x, y, phi):
        """Calculate Right-Straight-Right (RSR) path parameters"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # Calculate circle center to target vector
        cx = math.sin(phi)
        cy = -1.0 + math.cos(phi)
        
        # Calculate u (straight segment length)
        u = math.sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy))
        
        if u < 1e-10:
            return None, None, None
        
        # Calculate t and v (two arc angles)
        theta = math.atan2(y-cy, x-cx)
        t = self.normalize_angle(-theta)
        v = self.normalize_angle(-phi + t)
        
        return t, u, v
    
    def LSR(self, x, y, phi):
        """Calculate Left-Straight-Right (LSR) path parameters"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # Opposite angle by subtracting from π
        phi = self.normalize_angle(math.pi - phi)
        
        # LSR path target point needs coordinate transformation
        x_prime = x * math.cos(phi) + y * math.sin(phi)
        y_prime = x * math.sin(phi) - y * math.cos(phi)
        
        # Calculate parameters
        u = math.sqrt(x_prime*x_prime + (y_prime-2)*(y_prime-2))
        
        if u < 2.0:
            return None, None, None
        
        theta = math.atan2(y_prime-2, x_prime)
        t = self.normalize_angle(theta)
        v = self.normalize_angle(t + phi)
        
        return t, u, v
    
    def RSL(self, x, y, phi):
        """Calculate Right-Straight-Left (RSL) path parameters"""
        # Due to symmetry, RSL is essentially a mirror of LSR
        u = 0.0
        t = 0.0
        v = 0.0
        
        # Opposite angle by subtracting from π
        phi = self.normalize_angle(math.pi - phi)
        
        # RSL path target point needs coordinate transformation
        x_prime = x * math.cos(phi) - y * math.sin(phi)
        y_prime = x * math.sin(phi) + y * math.cos(phi)
        
        # Calculate parameters
        u = math.sqrt(x_prime*x_prime + (y_prime+2)*(y_prime+2))
        
        if u < 2.0:
            return None, None, None
        
        theta = math.atan2(y_prime+2, -x_prime)
        t = self.normalize_angle(theta)
        v = self.normalize_angle(-t - phi)
        
        return t, u, v
    
    def LRL(self, x, y, phi):
        """Calculate Left-Right-Left (LRL) path parameters"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # Calculate parameters
        u = math.sqrt(x*x + y*y)
        
        # Check bounds for u
        if u < 4.0 or u/4.0 > 1.0:
            return None, None, None
        
        alpha = math.atan2(y, x)
        beta = math.acos(u/4.0)
        
        t = self.normalize_angle(alpha + beta)
        u = self.normalize_angle(math.pi - 2*beta)
        v = self.normalize_angle(phi - t - u)
        
        return t, u, v
    
    def RLR(self, x, y, phi):
        """Calculate Right-Left-Right (RLR) path parameters"""
        u = 0.0
        t = 0.0
        v = 0.0
        
        # Calculate parameters
        u = math.sqrt(x*x + y*y)
        
        # Check bounds for u
        if u < 4.0 or u/4.0 > 1.0:
            return None, None, None
        
        alpha = math.atan2(y, x)
        beta = math.acos(u/4.0)
        
        t = self.normalize_angle(alpha - beta)
        u = self.normalize_angle(-math.pi + 2*beta)
        v = self.normalize_angle(phi - t - u)
        
        return t, u, v
    
    def generate_rs_path(self, start, path_type, controls):
        """
        Generate actual path points from Reeds-Shepp curve type and control parameters
        
        Args:
            start: Start state (x, y, theta)
            path_type: Path type, like 'LSL', 'RSR', etc.
            controls: Control parameters (t, u, v)
            
        Returns:
            list: Path points list
        """
        t, u, v = controls
        path = []
        
        # Add start point
        x, y, theta = start[0], start[1], start[2]
        path.append((x, y, theta))
        
        # Generate path points
        step_size = self.rs_step_size  # Path discretization step
        
        # Generate based on path type
        if path_type == 'LSL':
            # First segment: left turn
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
            
            # Second segment: straight
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'S')
                path.append((x, y, theta))
            
            # Third segment: left turn
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
        
        elif path_type == 'RSR':
            # First segment: right turn
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
            
            # Second segment: straight
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'S')
                path.append((x, y, theta))
            
            # Third segment: right turn
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
        
        elif path_type == 'LSR':
            # First segment: left turn
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
            
            # Second segment: straight
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'S')
                path.append((x, y, theta))
            
            # Third segment: right turn
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
        
        elif path_type == 'RSL':
            # First segment: right turn
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
            
            # Second segment: straight
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'S')
                path.append((x, y, theta))
            
            # Third segment: left turn
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
        
        elif path_type == 'LRL':
            # First segment: left turn
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
            
            # Second segment: right turn
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
            
            # Third segment: left turn
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
        
        elif path_type == 'RLR':
            # First segment: right turn
            for i in range(1, int(abs(t) / step_size) + 1):
                s = min(i * step_size, abs(t)) * (1 if t >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
            
            # Second segment: left turn
            for i in range(1, int(abs(u) / step_size) + 1):
                s = min(i * step_size, abs(u)) * (1 if u >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'L')
                path.append((x, y, theta))
            
            # Third segment: right turn
            for i in range(1, int(abs(v) / step_size) + 1):
                s = min(i * step_size, abs(v)) * (1 if v >= 0 else -1)
                x, y, theta = self.move_along_curve(x, y, theta, s, 'R')
                path.append((x, y, theta))
        
        return path
    def normalize_path(self, path):
        """标准化路径点，确保所有点都是二维(x,y)点"""
        if not path:
            return path
            
        # 如果路径点是三维的(x,y,theta)，转换为二维(x,y)
        normalized_path = []
        for point in path:
            if isinstance(point, tuple) and len(point) > 2:
                normalized_path.append((point[0], point[1]))
            else:
                normalized_path.append(point)
                
        return normalized_path    
    def move_along_curve(self, x, y, theta, arc_length, curve_type):
        """
        Move along specified type of curve by specified arc length
        
        Args:
            x, y, theta: Current position and heading
            arc_length: Arc length to move
            curve_type: Curve type ('L' left turn, 'R' right turn, 'S' straight)
            
        Returns:
            tuple: New position and heading (x_new, y_new, theta_new)
        """
        if curve_type == 'S':  # Straight
            return (x + arc_length * math.cos(theta),
                    y + arc_length * math.sin(theta),
                    theta)
        
        elif curve_type == 'L':  # Left turn
            return (x + self.turning_radius * (math.sin(theta + arc_length) - math.sin(theta)),
                    y - self.turning_radius * (math.cos(theta + arc_length) - math.cos(theta)),
                    self.normalize_angle(theta + arc_length))
        
        elif curve_type == 'R':  # Right turn
            return (x - self.turning_radius * (math.sin(theta - arc_length) - math.sin(theta)),
                    y + self.turning_radius * (math.cos(theta - arc_length) - math.cos(theta)),
                    self.normalize_angle(theta - arc_length))
    
    def check_rs_path_validity(self, path):
        """
        Check if Reeds-Shepp path is valid (no collision)
        
        Args:
            path: Path points list
            
        Returns:
            bool: Whether path is valid
        """
        if not path or len(path) < 2:
            return False
            
        # Check if each point is valid
        for x, y, theta in path:
            if not self.is_state_valid(x, y, theta):
                return False
        
        # Check if line between adjacent points is valid
        for i in range(len(path) - 1):
            x1, y1 = path[i][0], path[i][1]
            x2, y2 = path[i+1][0], path[i+1][1]
            
            if not self.is_intermediate_state_valid(x1, y1, x2, y2):
                return False
        
        return True