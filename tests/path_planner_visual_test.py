
"""
Enhanced Hybrid Path Planner Test Script

This script tests and visualizes the Hybrid A* path planner with Reeds-Shepp curves
for vehicle path planning in open-pit mining environments.

Features:
1. Complex obstacle environments 
2. Multiple vehicle path planning
3. Visualization of vehicle kinematics constraints
4. Comparison with basic path planning
5. Enhanced visualization and animation
"""

import os
import sys
import time
import math
import random
import logging
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
import argparse

# Add project root directory to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import required modules 
try:
    from algorithm.hybrid_path_planner import HybridPathPlanner
    from algorithm.map_service import MapService
    from algorithm.cbs import ConflictBasedSearch
    from utils.geo_tools import GeoUtils
    from models.vehicle import MiningVehicle, VehicleState
    from enhanced_visualization import EnhancedVisualization
    
    # Check if matplotlib is available for visualization
    try:
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        logging.warning("Matplotlib not found. Visualization will be disabled.")
        HAS_MATPLOTLIB = False
        
    logging.info("Successfully imported required modules")
except ImportError as e:
    logging.error(f"Failed to import modules: {str(e)}")
    sys.exit(1)

class MineEnvironment:
    """Simple mine environment for testing path planning"""
    
    def __init__(self, width=200, height=200):
        """Initialize mine environment"""
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height), dtype=int)  # 0 = free, 1 = obstacle
        self.loading_points = []
        self.unloading_points = []
        self.vehicles = {}
        
        # Create default points
        self.add_loading_point((30, 170))  # Example loading point
        self.add_unloading_point((170, 30))  # Example unloading point
    
    def add_loading_point(self, point):
        """Add loading point"""
        self.loading_points.append(point)
    
    def add_unloading_point(self, point):
        """Add unloading point"""
        self.unloading_points.append(point)
    
    def add_vehicle(self, vehicle_id, position, target=None, status='idle'):
        """Add vehicle to environment"""
        if isinstance(position, tuple) and len(position) == 2:
            # Add default orientation
            position = (position[0], position[1], 0)
            
        self.vehicles[vehicle_id] = {
            'position': position,
            'target': target,
            'status': status,
            'path': [],
            'load': 0,  # Current load (0 = empty)
            'max_load': 100,  # Maximum load capacity
            'color': [random.random(), random.random(), random.random()]  # Random color
        }
    
    def is_obstacle(self, point):
        """Check if point is an obstacle"""
        x, y = int(point[0]), int(point[1])
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x, y] == 1
        return True  # Out of bounds is considered obstacle
    
    def create_rectangular_obstacle(self, x1, y1, x2, y2):
        """Create rectangular obstacle"""
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[x, y] = 1
    
    def create_circular_obstacle(self, center_x, center_y, radius):
        """Create circular obstacle"""
        for x in range(int(center_x - radius), int(center_x + radius) + 1):
            for y in range(int(center_y - radius), int(center_y + radius) + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                        self.grid[x, y] = 1

def create_test_environment():
    """Create a complex test environment with various obstacles"""
    env = MineEnvironment(width=200, height=200)
    
    # Create complex obstacles
    # 1. Central mountains/pit
    env.create_circular_obstacle(100, 100, 30)
    
    # 2. Access road barriers
    env.create_rectangular_obstacle(50, 40, 70, 160)  # Vertical barrier
    env.create_rectangular_obstacle(130, 40, 150, 160)  # Vertical barrier
    env.create_rectangular_obstacle(50, 40, 150, 60)  # Horizontal barrier
    env.create_rectangular_obstacle(50, 140, 150, 160)  # Horizontal barrier
    
    # 3. Some scattered small obstacles
    for _ in range(5):
        x = random.randint(20, 180)
        y = random.randint(20, 180)
        radius = random.randint(5, 10)
        env.create_circular_obstacle(x, y, radius)
    
    # Add loading and unloading points
    env.add_loading_point((30, 30))  # Bottom left loading
    env.add_loading_point((30, 170))  # Top left loading
    env.add_unloading_point((170, 170))  # Top right unloading
    env.add_unloading_point((170, 30))  # Bottom right unloading
    
    # Create a open road around the obstacles
    # Clear paths connecting key points
    def clear_path(start, end, width=10):
        """Create clear path between points"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 1:
            return
            
        # Unit vector
        dx /= distance
        dy /= distance
        
        # Create path
        for t in range(int(distance) + 1):
            x = int(start[0] + t * dx)
            y = int(start[1] + t * dy)
            
            # Clear area around path
            for ox in range(-width//2, width//2 + 1):
                for oy in range(-width//2, width//2 + 1):
                    nx, ny = x + ox, y + oy
                    if 0 <= nx < env.width and 0 <= ny < env.height:
                        env.grid[nx, ny] = 0
    
    # Connect loading and unloading points
    for load_point in env.loading_points:
        for unload_point in env.unloading_points:
            clear_path(load_point, unload_point, width=15)
            
    # Add more complex paths
    waypoints = [
        (100, 30),   # Bottom middle
        (170, 100),  # Right middle
        (100, 170),  # Top middle
        (30, 100)    # Left middle
    ]
    
    # Connect waypoints in sequence
    for i in range(len(waypoints)):
        start = waypoints[i]
        end = waypoints[(i+1) % len(waypoints)]
        clear_path(start, end, width=15)
    
    return env

def test_basic_path_planning(vis_mode="save", use_enhanced_vis=True):
    """
    Test basic path planning for comparison
    
    Args:
        vis_mode: 'save', 'show', or 'both'
        use_enhanced_vis: Whether to use enhanced visualization
    """
    print("\n=== Testing Basic Path Planning ===")
    
    # Create environment
    env = create_test_environment()
    
    # Create simple planner based on map service
    map_service = MapService()
    map_service.obstacle_grids = set([(x, y) for x in range(env.width) for y in range(env.height) if env.grid[x, y] == 1])
    
    # Create vehicles
    env.add_vehicle(1, env.loading_points[0], env.unloading_points[0])
    env.add_vehicle(2, env.loading_points[1], env.unloading_points[1])
    
    # Plan paths using basic planner
    start_time = time.time()
    
    for vid, vehicle in env.vehicles.items():
        start = vehicle['position']
        target = vehicle['target']
        
        # Use map_service's plan_route method
        route = map_service.plan_route(start[:2], target[:2])
        
        if route and 'path' in route:
            # Convert to list of (x, y, theta) points
            path = []
            for i, point in enumerate(route['path']):
                # Calculate theta for each point
                if i < len(route['path']) - 1:
                    next_point = route['path'][i+1]
                    dx = next_point[0] - point[0]
                    dy = next_point[1] - point[1]
                    theta = math.atan2(dy, dx)
                else:
                    # Last point - use previous theta
                    theta = path[-1][2] if path else 0
                
                path.append((point[0], point[1], theta))
            
            vehicle['path'] = path
            print(f"Vehicle {vid} path planned with {len(path)} points")
        else:
            print(f"Could not plan path for vehicle {vid}")
    
    planning_time = time.time() - start_time
    print(f"Basic path planning completed in {planning_time:.2f} seconds")
    
    # Visualize environment
    if HAS_MATPLOTLIB:
        if use_enhanced_vis:
            # Use enhanced visualization
            visualizer = EnhancedVisualization(figsize=(12, 10))
            fig, ax, info_panel = visualizer.setup_figure("Basic Path Planning")
            
            # Draw environment
            visualizer.draw_environment(env)
            
            # Draw vehicles and paths
            for vid, vehicle in env.vehicles.items():
                # Draw vehicle
                position = vehicle['position']
                load_percent = (vehicle['load'] / vehicle['max_load']) * 100 if vehicle['max_load'] > 0 else 0
                visualizer.draw_vehicle(
                    position, 5.0, 2.5, 
                    color=vehicle['color'], 
                    load_percent=load_percent, 
                    label=f"V{vid}"
                )
                
                # Draw path
                if 'path' in vehicle and vehicle['path']:
                    visualizer.draw_path(
                        vehicle['path'], 
                        color=vehicle['color'], 
                        label=f"Vehicle {vid} Path"
                    )
            
            # Show legend and add status info
            visualizer.show_legend()
            visualizer.update_info_panel({
                "Algorithm": "Basic A* Path Planning",
                "Planning Time": f"{planning_time:.2f} seconds",
                "Vehicles": f"{len(env.vehicles)}",
                "Paths": f"{sum(1 for v in env.vehicles.values() if 'path' in v and v['path'])}/{len(env.vehicles)}"
            })
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                visualizer.save_figure("basic_path_planning.png")
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
        else:
            # Use original visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Draw obstacles
            obstacle_mask = env.grid.T == 1
            ax.imshow(obstacle_mask, cmap='gray', alpha=0.5, extent=(0, env.width, 0, env.height), origin='lower')
            
            # Draw loading/unloading points
            for i, point in enumerate(env.loading_points):
                ax.scatter(point[0], point[1], c='green', marker='o', s=100)
                ax.text(point[0]+5, point[1]+5, f"Loading {i+1}", fontsize=12)
            
            for i, point in enumerate(env.unloading_points):
                ax.scatter(point[0], point[1], c='red', marker='s', s=100)
                ax.text(point[0]+5, point[1]+5, f"Unloading {i+1}", fontsize=12)
            
            # Draw vehicles and paths
            for vid, vehicle in env.vehicles.items():
                position = vehicle['position']
                color = vehicle['color']
                
                # Draw vehicle as a circle
                ax.scatter(position[0], position[1], c=[color], marker='o', s=100)
                ax.text(position[0], position[1], f"V{vid}", fontsize=12, ha='center', va='center', color='white')
                
                # Draw path
                if 'path' in vehicle and vehicle['path']:
                    path = vehicle['path']
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    
                    ax.plot(path_x, path_y, c=color, linestyle='-', linewidth=2, label=f"Vehicle {vid}")
            
            # Set axis properties
            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_title("Basic Path Planning")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                plt.savefig("basic_path_planning.png")
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
    
    return env, planning_time

def test_hybrid_path_planning(vis_mode="save", use_enhanced_vis=True):
    """
    Test hybrid path planning with vehicle kinematics
    
    Args:
        vis_mode: 'save', 'show', or 'both'
        use_enhanced_vis: Whether to use enhanced visualization
    """
    print("\n=== Testing Hybrid Path Planning ===")
    
    # Create environment
    env = create_test_environment()
    
    # Create hybrid path planner
    map_service = MapService()
    
    # Set obstacle grid
    obstacle_grid = set([(x, y) for x in range(env.width) for y in range(env.height) if env.grid[x, y] == 1])
    
    # Create hybrid planner
    planner = HybridPathPlanner(env)
    planner.vehicle_length = 6.0  # Length in grid units
    planner.vehicle_width = 3.0   # Width in grid units
    planner.turning_radius = 8.0  # Minimum turning radius
    planner.step_size = 0.8       # Step size for motion
    planner.obstacle_grids = obstacle_grid
    
    # Create vehicles
    env.add_vehicle(1, (env.loading_points[0][0], env.loading_points[0][1], 0), 
                   (env.unloading_points[0][0], env.unloading_points[0][1], 0))
    env.add_vehicle(2, (env.loading_points[1][0], env.loading_points[1][1], 0),
                   (env.unloading_points[1][0], env.unloading_points[1][1], 0))
    
    # Plan paths using hybrid planner
    start_time = time.time()
    
    for vid, vehicle in env.vehicles.items():
        start = vehicle['position']
        target = vehicle['target']
        
        # Plan path using hybrid planner
        path = planner.plan_path(start, target)
        
        if path and len(path) > 1:
            vehicle['path'] = path
            print(f"Vehicle {vid} path planned with {len(path)} points")
        else:
            print(f"Could not plan path for vehicle {vid}")
    
    planning_time = time.time() - start_time
    print(f"Hybrid path planning completed in {planning_time:.2f} seconds")
    
    # Visualize environment
    if HAS_MATPLOTLIB:
        if use_enhanced_vis:
            # Use enhanced visualization
            visualizer = EnhancedVisualization(figsize=(12, 10))
            fig, ax, info_panel = visualizer.setup_figure("Hybrid Path Planning")
            
            # Draw environment
            visualizer.draw_environment(env)
            
            # Draw vehicles and paths
            for vid, vehicle in env.vehicles.items():
                # Set vehicle color based on load
                if vid == 1:
                    color = [0.2, 0.6, 0.8]  # Blue
                else:
                    color = [0.8, 0.3, 0.3]  # Red
                vehicle['color'] = color
                
                # Draw vehicle
                position = vehicle['position']
                visualizer.draw_vehicle(
                    position, 
                    planner.vehicle_length, 
                    planner.vehicle_width, 
                    color=color, 
                    label=f"V{vid}"
                )
                
                # Draw path with orientation indicators
                if 'path' in vehicle and vehicle['path']:
                    visualizer.draw_path(
                        vehicle['path'], 
                        color=color, 
                        show_orientations=True, 
                        label=f"Vehicle {vid} Path"
                    )
                    
                    # Draw target point
                    target = vehicle['target']
                    ax.scatter(
                        target[0], target[1], 
                        c=[color], marker='x', s=100, 
                        linewidth=2, zorder=15
                    )
            
            # Show legend and add status info
            visualizer.show_legend()
            visualizer.update_info_panel({
                "Algorithm": "Hybrid A* Path Planning",
                "Planning Time": f"{planning_time:.2f} seconds",
                "Turning Radius": f"{planner.turning_radius} units",
                "Vehicle Size": f"{planner.vehicle_length}x{planner.vehicle_width}",
                "Paths Created": f"{sum(1 for v in env.vehicles.values() if 'path' in v and v['path'])}/{len(env.vehicles)}"
            })
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                visualizer.save_figure("hybrid_path_planning.png")
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
        else:
            # Original visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Draw obstacles
            obstacle_mask = env.grid.T == 1
            ax.imshow(obstacle_mask, cmap='gray', alpha=0.5, extent=(0, env.width, 0, env.height), origin='lower')
            
            # Draw loading/unloading points
            for i, point in enumerate(env.loading_points):
                ax.scatter(point[0], point[1], c='green', marker='o', s=100)
                ax.text(point[0]+5, point[1]+5, f"Loading {i+1}", fontsize=12)
            
            for i, point in enumerate(env.unloading_points):
                ax.scatter(point[0], point[1], c='red', marker='s', s=100)
                ax.text(point[0]+5, point[1]+5, f"Unloading {i+1}", fontsize=12)
            
            # Draw vehicles and paths
            for vid, vehicle in env.vehicles.items():
                position = vehicle['position']
                
                # Set vehicle color
                if vid == 1:
                    color = [0.2, 0.6, 0.8]  # Blue
                else:
                    color = [0.8, 0.3, 0.3]  # Red
                vehicle['color'] = color
                
                # Draw vehicle with orientation
                x, y, theta = position
                
                # Create vehicle rectangle
                length = planner.vehicle_length
                width = planner.vehicle_width
                dx = length / 2
                dy = width / 2
                corners = [
                    (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy), (-dx, -dy)
                ]
                
                # Rotate and translate corners
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                corners_rotated = []
                
                for cx, cy in corners:
                    rx = x + cx * cos_t - cy * sin_t
                    ry = y + cx * sin_t + cy * cos_t
                    corners_rotated.append((rx, ry))
                
                # Draw polygon
                xs, ys = zip(*corners_rotated)
                ax.fill(xs, ys, color=color, alpha=0.7)
                
                # Draw direction indicator
                front_x = x + dx * 0.8 * cos_t
                front_y = y + dx * 0.8 * sin_t
                ax.arrow(x, y, front_x-x, front_y-y, head_width=width*0.3, head_length=length*0.2, 
                        fc='black', ec='black', alpha=0.8)
                
                # Add vehicle label
                ax.text(x, y, f"V{vid}", fontsize=10, ha='center', va='center', color='white', weight='bold')
                
                # Draw path
                if 'path' in vehicle and vehicle['path']:
                    path = vehicle['path']
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    
                    # Draw path line
                    ax.plot(path_x, path_y, c=color, linestyle='-', linewidth=2, label=f"Vehicle {vid}")
                    
                    # Draw orientation at intervals
                    interval = max(1, len(path) // 10)
                    for i in range(0, len(path), interval):
                        x, y, theta = path[i]
                        dx = math.cos(theta) * 3
                        dy = math.sin(theta) * 3
                        ax.arrow(x, y, dx, dy, head_width=2, head_length=2, fc=color, ec=color, alpha=0.7)
            
            # Set axis properties
            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_title("Hybrid Path Planning")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                plt.savefig("hybrid_path_planning.png")
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
    
    return env, planning_time, planner

def test_path_planning_with_varying_radii(vis_mode="save", use_enhanced_vis=True):
    """
    Test path planning with different turning radii
    
    Args:
        vis_mode: 'save', 'show', or 'both'
        use_enhanced_vis: Whether to use enhanced visualization
    """
    print("\n=== Testing Path Planning with Varying Turning Radii ===")
    
    # Create environment
    env = create_test_environment()
    
    # Create hybrid path planner
    map_service = MapService()
    
    # Set obstacle grid
    obstacle_grid = set([(x, y) for x in range(env.width) for y in range(env.height) if env.grid[x, y] == 1])
    
    # Define different vehicle configurations
    vehicle_configs = [
        {"id": 1, "length": 6.0, "width": 3.0, "turning_radius": 5.0},   # Small turning radius
        {"id": 2, "length": 6.0, "width": 3.0, "turning_radius": 10.0},  # Medium turning radius
        {"id": 3, "length": 6.0, "width": 3.0, "turning_radius": 15.0}   # Large turning radius
    ]
    
    # Create vehicles with different start/end points
    start_point = (50, 50, 0)
    end_point = (150, 150, math.pi/2)  # End facing up
    
    # Set up environment
    for config in vehicle_configs:
        env.add_vehicle(config["id"], start_point, end_point)
        env.vehicles[config["id"]]['color'] = [
            0.2 + 0.6 * (config["id"]-1) / (len(vehicle_configs)-1),
            0.2,
            0.8 - 0.6 * (config["id"]-1) / (len(vehicle_configs)-1)
        ]
    
    # Create base planner
    planner = HybridPathPlanner(env)
    planner.obstacle_grids = obstacle_grid
    
    # Store results for each configuration
    results = []
    
    # Plan paths for different vehicle configurations
    for config in vehicle_configs:
        vid = config["id"]
        vehicle = env.vehicles[vid]
        
        # Update planner with vehicle config
        planner.vehicle_length = config["length"]
        planner.vehicle_width = config["width"]
        planner.turning_radius = config["turning_radius"]
        
        # Plan path
        start_time = time.time()
        path = planner.plan_path(start_point, end_point)
        planning_time = time.time() - start_time
        
        if path and len(path) > 1:
            vehicle['path'] = path
            
            # Calculate path metrics
            total_length = 0
            for i in range(1, len(path)):
                x1, y1 = path[i-1][0], path[i-1][1]
                x2, y2 = path[i][0], path[i][1]
                segment_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                total_length += segment_length
            
            # Calculate path smoothness
            angle_changes = []
            for i in range(1, len(path) - 1):
                p1, p2, p3 = path[i-1], path[i], path[i+1]
                
                # Calculate vectors
                v1 = (p2[0] - p1[0], p2[1] - p1[1])
                v2 = (p3[0] - p2[0], p3[1] - p2[1])
                
                # Calculate magnitudes
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                
                # Skip if too short
                if mag1 < 1e-6 or mag2 < 1e-6:
                    continue
                    
                # Calculate dot product
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                
                # Calculate angle
                cos_angle = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
                angle = math.acos(cos_angle)
                angle_changes.append(angle)
            
            avg_angle_change = sum(angle_changes) / len(angle_changes) if angle_changes else 0
            smoothness = 1.0 - avg_angle_change / math.pi
            
            # Store results
            results.append({
                "id": vid,
                "turning_radius": config["turning_radius"],
                "planning_time": planning_time,
                "path_length": len(path),
                "total_distance": total_length,
                "smoothness": smoothness
            })
            
            print(f"Vehicle {vid} (radius={config['turning_radius']}) path: {len(path)} points, " +
                 f"length={total_length:.1f}, smoothness={smoothness:.3f}, time={planning_time:.2f}s")
        else:
            print(f"Could not plan path for vehicle {vid}")
    
    # Visualize environment
    if HAS_MATPLOTLIB:
        if use_enhanced_vis:
            # Use enhanced visualization
            visualizer = EnhancedVisualization(figsize=(12, 10))
            fig, ax, info_panel = visualizer.setup_figure("Path Planning with Different Turning Radii")
            
            # Draw environment
            visualizer.draw_environment(env)
            
            # Draw vehicles and paths
            for vid, vehicle in env.vehicles.items():
                config = next((c for c in vehicle_configs if c["id"] == vid), None)
                if not config:
                    continue
                    
                position = vehicle['position']
                color = vehicle['color']
                
                # Draw vehicle
                vehicle_length = config["length"]
                vehicle_width = config["width"]
                
                visualizer.draw_vehicle(
                    position, 
                    vehicle_length, 
                    vehicle_width,
                    color=color, 
                    label=f"V{vid} (R={config['turning_radius']})"
                )
                
                # Draw path
                if 'path' in vehicle and vehicle['path']:
                    path = vehicle['path']
                    
                    visualizer.draw_path(
                        path, 
                        color=color, 
                        label=f"Vehicle {vid} (R={config['turning_radius']})",
                        linewidth=3
                    )
                    
                    # Draw start and end
                    visualizer.ax.scatter(
                        path[0][0], path[0][1], 
                        c=color, marker='o', s=100, 
                        edgecolors='black', linewidths=1,
                        zorder=20, alpha=0.8
                    )
                    
                    visualizer.ax.scatter(
                        path[-1][0], path[-1][1], 
                        c=color, marker='x', s=100, 
                        linewidths=2, zorder=20
                    )
            
            # Show legend
            visualizer.show_legend(loc='upper left')
            
            # Add results as text
            if results:
                result_text = "Results:\n"
                for r in results:
                    result_text += f"R={r['turning_radius']}: "
                    result_text += f"Length={r['total_distance']:.1f}, "
                    result_text += f"Smoothness={r['smoothness']:.3f}\n"
                
                visualizer.info_panel.text(
                    0.05, 0.5, result_text,
                    fontsize=9, ha='left', va='center'
                )
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                visualizer.save_figure("varying_turning_radii.png", dpi=150)
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
        else:
            # Original visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Draw obstacles
            obstacle_mask = env.grid.T == 1
            ax.imshow(obstacle_mask, cmap='gray', alpha=0.5, extent=(0, env.width, 0, env.height), origin='lower')
            
            # Draw vehicles and paths
            for vid, vehicle in env.vehicles.items():
                config = next((c for c in vehicle_configs if c["id"] == vid), None)
                if not config:
                    continue
                    
                position = vehicle['position']
                color = vehicle['color']
                
                # Draw vehicle
                vehicle_length = config["length"]
                vehicle_width = config["width"]
                
                x, y, theta = position
                dx = vehicle_length / 2
                dy = vehicle_width / 2
                corners = [
                    (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy), (-dx, -dy)
                ]
                
                # Rotate and translate corners
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                corners_rotated = []
                
                for cx, cy in corners:
                    rx = x + cx * cos_t - cy * sin_t
                    ry = y + cx * sin_t + cy * cos_t
                    corners_rotated.append((rx, ry))
                
                # Draw polygon
                xs, ys = zip(*corners_rotated)
                ax.fill(xs, ys, color=color, alpha=0.7)
                
                # Draw path
                if 'path' in vehicle and vehicle['path']:
                    path = vehicle['path']
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    
                    # Draw path line
                    ax.plot(path_x, path_y, c=color, linestyle='-', linewidth=2, 
                          label=f"Vehicle {vid} (R={config['turning_radius']})")
                    
                    # Draw orientation at intervals
                    interval = max(1, len(path) // 10)
                    for i in range(0, len(path), interval):
                        x, y, theta = path[i]
                        dx = math.cos(theta) * 3
                        dy = math.sin(theta) * 3
                        ax.arrow(x, y, dx, dy, head_width=2, head_length=2, fc=color, ec=color, alpha=0.7)
            
            # Set axis properties
            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_title("Path Planning with Different Turning Radii")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper left')
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                plt.savefig("varying_turning_radii.png")
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
    
    return env, planner

def test_multiple_vehicles_with_cbs(vis_mode="save", use_enhanced_vis=True, animate=False):
    """
    Test multiple vehicles with CBS conflict resolution
    
    Args:
        vis_mode: 'save', 'show', or 'both'
        use_enhanced_vis: Whether to use enhanced visualization
        animate: Whether to create an animation of conflict resolution
    """
    print("\n=== Testing Multiple Vehicles with CBS Conflict Resolution ===")
    
    # Create environment
    env = create_test_environment()
    
    # Create hybrid path planner
    map_service = MapService()
    
    # Set obstacle grid
    obstacle_grid = set([(x, y) for x in range(env.width) for y in range(env.height) if env.grid[x, y] == 1])
    
    # Create hybrid planner
    planner = HybridPathPlanner(env)
    planner.vehicle_length = 6.0
    planner.vehicle_width = 3.0
    planner.turning_radius = 8.0
    planner.step_size = 0.8
    planner.obstacle_grids = obstacle_grid
    
    # Create CBS algorithm
    cbs = ConflictBasedSearch(planner)
    
    # Create multiple vehicles with crossing paths
    env.add_vehicle(1, (30, 30, 0), (170, 170, 0))      # Bottom-left to top-right
    env.add_vehicle(2, (30, 170, -math.pi/2), (170, 30, 0))  # Top-left to bottom-right
    env.add_vehicle(3, (100, 30, math.pi/2), (100, 170, math.pi/2))  # Bottom-center to top-center
    
    # Set distinct colors for vehicles
    env.vehicles[1]['color'] = [0.8, 0.2, 0.2]  # Red
    env.vehicles[2]['color'] = [0.2, 0.6, 0.8]  # Blue
    env.vehicles[3]['color'] = [0.2, 0.8, 0.2]  # Green
    
    # Initialize visualizer for animation
    if HAS_MATPLOTLIB and animate and use_enhanced_vis:
        visualizer = EnhancedVisualization(figsize=(12, 10))
        fig, ax, info_panel = visualizer.setup_figure(
            "Multiple Vehicles with CBS Conflict Resolution", 
            interactive=True
        )
        
        # Draw environment
        visualizer.draw_environment(env)
        
        # Show initial state
        for vid, vehicle in env.vehicles.items():
            position = vehicle['position']
            visualizer.draw_vehicle(
                position,
                planner.vehicle_length,
                planner.vehicle_width,
                color=vehicle['color'],
                label=f"V{vid}"
            )
        
        visualizer.update_info_panel({
            "Status": "Planning initial paths...",
            "Vehicles": f"{len(env.vehicles)}",
            "Conflicts": "Detecting...",
            "CBS": "Not started"
        })
        
        visualizer.update_and_pause(0.5)
        
        if animate:
            visualizer.add_animation_frame()
    
    # Plan initial paths without conflict resolution
    print("Planning initial paths...")
    vehicle_paths = {}
    for vid, vehicle in env.vehicles.items():
        start = vehicle['position']
        target = vehicle['target']
        
        # Plan path using hybrid planner
        path = planner.plan_path(start, target)
        
        if path and len(path) > 1:
            vehicle['path'] = path
            vehicle_paths[str(vid)] = path
            print(f"Vehicle {vid} path planned with {len(path)} points")
        else:
            print(f"Could not plan path for vehicle {vid}")
    
    # Update animation if enabled
    if HAS_MATPLOTLIB and animate and use_enhanced_vis:
        # Draw paths
        for vid, vehicle in env.vehicles.items():
            if 'path' in vehicle and vehicle['path']:
                visualizer.draw_path(
                    vehicle['path'],
                    color=vehicle['color'],
                    label=f"Vehicle {vid} Path"
                )
        
        visualizer.update_info_panel({
            "Status": "Detecting conflicts...",
            "Vehicles": f"{len(env.vehicles)}",
            "Conflicts": "Detecting...",
            "CBS": "Not started"
        })
        
        visualizer.update_and_pause(0.5)
        
        if animate:
            visualizer.add_animation_frame()
    
    # Check for conflicts
    conflicts = cbs.find_conflicts(vehicle_paths)
    print(f"Detected {len(conflicts)} conflicts between initial paths")
    
    # Update animation if enabled
    if HAS_MATPLOTLIB and animate and use_enhanced_vis and conflicts:
        # Highlight conflicts
        visualizer.draw_conflict_points(conflicts)
        
        # Update info panel
        visualizer.update_info_panel({
            "Status": "Conflicts detected",
            "Vehicles": f"{len(env.vehicles)}",
            "Conflicts": f"{len(conflicts)}",
            "CBS": "Not started"
        })
        
        visualizer.update_and_pause(1.0)
        
        if animate:
            visualizer.add_animation_frame()
    
    # Resolve conflicts using CBS
    if conflicts:
        print("Resolving conflicts with CBS...")
        
        # Measure resolution time
        start_time = time.time()
        resolved_paths = cbs.resolve_conflicts(vehicle_paths)
        resolution_time = time.time() - start_time
        
        # Update vehicle paths with resolved paths
        changed_count = 0
        for vid_str, new_path in resolved_paths.items():
            vid = int(vid_str)
            if vid in env.vehicles:
                old_path = env.vehicles[vid].get('path', [])
                if old_path != new_path:
                    env.vehicles[vid]['path'] = new_path
                    changed_count += 1
        
        print(f"CBS resolved conflicts by changing {changed_count} vehicle paths in {resolution_time:.2f} seconds")
        
        # Check for remaining conflicts
        remaining_conflicts = cbs.find_conflicts(resolved_paths)
        print(f"Remaining conflicts after resolution: {len(remaining_conflicts)}")
        
        # Update animation if enabled
        if HAS_MATPLOTLIB and animate and use_enhanced_vis:
            # Clear previous paths
            ax.clear()
            visualizer.draw_environment(env)
            
            # Draw vehicles
            for vid, vehicle in env.vehicles.items():
                position = vehicle['position']
                visualizer.draw_vehicle(
                    position,
                    planner.vehicle_length,
                    planner.vehicle_width,
                    color=vehicle['color'],
                    label=f"V{vid}"
                )
            
            # Draw resolved paths
            for vid, vehicle in env.vehicles.items():
                if 'path' in vehicle and vehicle['path']:
                    visualizer.draw_path(
                        vehicle['path'],
                        color=vehicle['color'],
                        label=f"Vehicle {vid} Path (Resolved)"
                    )
            
            # Draw any remaining conflicts
            if remaining_conflicts:
                visualizer.draw_conflict_points(remaining_conflicts)
            
            # Update info panel
            visualizer.update_info_panel({
                "Status": "Conflicts resolved",
                "Vehicles": f"{len(env.vehicles)}",
                "Initial Conflicts": f"{len(conflicts)}",
                "Remaining Conflicts": f"{len(remaining_conflicts)}",
                "Paths Changed": f"{changed_count}/{len(env.vehicles)}",
                "Resolution Time": f"{resolution_time:.2f}s"
            })
            
            visualizer.update_and_pause(0.5)
            
            if animate:
                visualizer.add_animation_frame()
    
    # Visualize environment with conflict-free paths
    if HAS_MATPLOTLIB:
        if use_enhanced_vis:
            # Create new visualization if we weren't animating
            if not animate:
                visualizer = EnhancedVisualization(figsize=(12, 10))
                fig, ax, info_panel = visualizer.setup_figure("Multiple Vehicles with CBS Conflict Resolution")
                
                # Draw environment
                visualizer.draw_environment(env)
            
            # Draw vehicles and paths
            for vid, vehicle in env.vehicles.items():
                position = vehicle['position']
                
                # Draw vehicle
                visualizer.draw_vehicle(
                    position,
                    planner.vehicle_length,
                    planner.vehicle_width,
                    color=vehicle['color'],
                    label=f"V{vid}"
                )
                
                # Draw path
                if 'path' in vehicle and vehicle['path']:
                    path = vehicle['path']
                    visualizer.draw_path(
                        path, 
                        color=vehicle['color'], 
                        label=f"Vehicle {vid}"
                    )
                    
                    # Draw target
                    target = vehicle['target']
                    ax.scatter(
                        target[0], target[1], 
                        c=[vehicle['color']], marker='x', s=100,
                        linewidths=2, zorder=20
                    )
            
            # Draw any remaining conflicts after resolution
            if conflicts:
                remaining_conflicts = cbs.find_conflicts({str(vid): vehicle['path'] 
                                                        for vid, vehicle in env.vehicles.items() 
                                                        if 'path' in vehicle})
                
                if remaining_conflicts:
                    visualizer.draw_conflict_points(remaining_conflicts)
            
            # Update info panel with CBS stats
            info_dict = {
                "Algorithm": "Conflict-Based Search (CBS)",
                "Vehicles": f"{len(env.vehicles)}",
                "Initial Conflicts": f"{len(conflicts) if conflicts else 0}",
            }
            
            if conflicts:
                resolution_time = getattr(cbs, 'resolution_time', 'N/A') 
                if isinstance(resolution_time, (int, float)):
                    resolution_time = f"{resolution_time:.2f}s"
                
                info_dict.update({
                    "Resolution Time": resolution_time,
                    "Paths Changed": f"{changed_count}/{len(env.vehicles)}",
                    "Remaining Conflicts": f"{len(remaining_conflicts) if 'remaining_conflicts' in locals() else 0}"
                })
            
            visualizer.update_info_panel(info_dict)
            
            # Show legend
            visualizer.show_legend(loc='upper right')
            
            # Save animation if created
            if animate:
                visualizer.create_animation("cbs_conflict_resolution.gif", fps=5)
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                visualizer.save_figure("cbs_conflict_resolution.png", dpi=150)
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
        else:
            # Original visualization
            fig, ax = plt.subplots(figsize=(12, 12))
            
            # Draw obstacles
            obstacle_mask = env.grid.T == 1
            ax.imshow(obstacle_mask, cmap='gray', alpha=0.5, extent=(0, env.width, 0, env.height), origin='lower')
            
            # Draw vehicles and paths
            for vid, vehicle in env.vehicles.items():
                position = vehicle['position']
                color = vehicle['color']
                
                # Draw vehicle
                x, y, theta = position
                dx = planner.vehicle_length / 2
                dy = planner.vehicle_width / 2
                corners = [
                    (-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy), (-dx, -dy)
                ]
                
                # Rotate and translate corners
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                corners_rotated = []
                
                for cx, cy in corners:
                    rx = x + cx * cos_t - cy * sin_t
                    ry = y + cx * sin_t + cy * cos_t
                    corners_rotated.append((rx, ry))
                
                # Draw polygon
                xs, ys = zip(*corners_rotated)
                ax.fill(xs, ys, color=color, alpha=0.7)
                
                # Add vehicle label
                ax.text(x, y, f"V{vid}", fontsize=10, ha='center', va='center', color='white', weight='bold')
                
                # Draw path
                if 'path' in vehicle and vehicle['path']:
                    path = vehicle['path']
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    
                    # Draw path line
                    ax.plot(path_x, path_y, c=color, linestyle='-', linewidth=2, label=f"Vehicle {vid}")
                    
                    # Draw orientation at intervals
                    interval = max(1, len(path) // 10)
                    for i in range(0, len(path), interval):
                        x, y, theta = path[i]
                        dx = math.cos(theta) * 3
                        dy = math.sin(theta) * 3
                        ax.arrow(x, y, dx, dy, head_width=2, head_length=2, fc=color, ec=color, alpha=0.7)
                
                # Draw target
                target = vehicle['target']
                ax.scatter(target[0], target[1], c=color, marker='x', s=100)
            
            # Highlight conflict points if any
            if conflicts:
                remaining_conflicts = cbs.find_conflicts({str(vid): vehicle['path'] 
                                                        for vid, vehicle in env.vehicles.items() 
                                                        if 'path' in vehicle})
                
                for conflict in remaining_conflicts:
                    # Mark conflict location
                    location = conflict["location"]
                    ax.scatter(location[0], location[1], c='red', marker='*', s=200, alpha=0.7)
                    
                    # Add conflict info
                    time = conflict.get("time", "?")
                    ax.text(location[0], location[1]+5, f"Conflict at t={time}", 
                           ha='center', va='bottom', color='red', fontsize=10)
            
            # Set axis properties
            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_title("Multiple Vehicles with CBS Conflict Resolution")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                plt.savefig("cbs_conflict_resolution.png")
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
    
    return env, planner, cbs

def test_path_smoothing(vis_mode="save", use_enhanced_vis=True):
    """
    Test path smoothing effect
    
    Args:
        vis_mode: 'save', 'show', or 'both'
        use_enhanced_vis: Whether to use enhanced visualization
    """
    print("\n=== Testing Path Smoothing Effect ===")
    
    # Create environment
    env = create_test_environment()
    
    # Create hybrid path planner
    map_service = MapService()
    
    # Set obstacle grid
    obstacle_grid = set([(x, y) for x in range(env.width) for y in range(env.height) if env.grid[x, y] == 1])
    
    # Create hybrid planner
    planner = HybridPathPlanner(env)
    planner.vehicle_length = 6.0
    planner.vehicle_width = 3.0
    planner.turning_radius = 8.0
    planner.obstacle_grids = obstacle_grid
    
    # Create test points that require complex paths
    start_point = (30, 30, 0)
    end_point = (170, 170, 0)
    
    # Create two identical vehicles
    env.add_vehicle(1, start_point, end_point)
    env.add_vehicle(2, start_point, end_point)
    
    # Use different colors
    env.vehicles[1]['color'] = [0.8, 0.2, 0.2]  # Red for no smoothing
    env.vehicles[2]['color'] = [0.2, 0.6, 0.8]  # Blue for smoothing
    
    # Plan path without smoothing
    planner.path_smoothing = False
    path1 = planner.plan_path(start_point, end_point)
    
    if path1 and len(path1) > 1:
        env.vehicles[1]['path'] = path1
        print(f"Path without smoothing: {len(path1)} points")
    
    # Plan path with smoothing
    planner.path_smoothing = True
    planner.smoothing_factor = 0.5
    planner.smoothing_iterations = 10
    path2 = planner.plan_path(start_point, end_point)
    
    if path2 and len(path2) > 1:
        env.vehicles[2]['path'] = path2
        print(f"Path with smoothing: {len(path2)} points")
    
    # Compute path metrics
    def calculate_path_smoothness(path):
        if len(path) < 3:
            return 1.0
        
        angle_changes = []
        for i in range(1, len(path) - 1):
            prev, curr, next_p = path[i-1], path[i], path[i+1]
            
            # Calculate vectors
            v1 = (curr[0] - prev[0], curr[1] - prev[1])
            v2 = (next_p[0] - curr[0], next_p[1] - curr[1])
            
            # Calculate magnitudes
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if mag1 < 1e-3 or mag2 < 1e-3:
                continue
                
            # Calculate dot product and angle
            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
            cos_angle = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
            angle = math.acos(cos_angle)
            angle_changes.append(angle)
        
        if not angle_changes:
            return 1.0
            
        avg_angle_change = sum(angle_changes) / len(angle_changes)
        smoothness = 1.0 - avg_angle_change / math.pi
        
        return max(0.0, min(1.0, smoothness))
    
    # Calculate metrics for both paths
    if path1 and path2:
        # Calculate total path length
        def calc_path_length(path):
            total = 0
            for i in range(1, len(path)):
                dx = path[i][0] - path[i-1][0]
                dy = path[i][1] - path[i-1][1]
                total += math.sqrt(dx*dx + dy*dy)
            return total
        
        length1 = calc_path_length(path1)
        length2 = calc_path_length(path2)
        
        # Calculate smoothness
        smoothness1 = calculate_path_smoothness(path1)
        smoothness2 = calculate_path_smoothness(path2)
        
        print(f"Path metrics without smoothing: Length={length1:.1f}, Smoothness={smoothness1:.4f}")
        print(f"Path metrics with smoothing: Length={length2:.1f}, Smoothness={smoothness2:.4f}")
        print(f"Smoothness improvement: {(smoothness2-smoothness1)/smoothness1*100:.1f}%")
        print(f"Length change: {(length2-length1)/length1*100:.1f}%")
    
    # Visualize environment
    if HAS_MATPLOTLIB:
        if use_enhanced_vis:
            # Create enhanced visualization
            visualizer = EnhancedVisualization(figsize=(14, 10))
            fig, ax, info_panel = visualizer.setup_figure("Effect of Path Smoothing")
            
            # Draw environment
            visualizer.draw_environment(env)
            
            # Draw start and end points with labels
            ax.scatter(start_point[0], start_point[1], c='green', marker='o', s=120, 
                     edgecolors='black', linewidths=1, zorder=20)
            ax.text(start_point[0]+5, start_point[1]+5, "Start", fontsize=12, weight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            ax.scatter(end_point[0], end_point[1], c='red', marker='s', s=120,
                     edgecolors='black', linewidths=1, zorder=20)
            ax.text(end_point[0]+5, end_point[1]+5, "End", fontsize=12, weight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            # Draw paths and data
            if path1 and path2:
                # Draw paths
                visualizer.draw_path(
                    path1, 
                    color=env.vehicles[1]['color'], 
                    label="Without Smoothing",
                    linewidth=3
                )
                
                visualizer.draw_path(
                    path2, 
                    color=env.vehicles[2]['color'], 
                    label="With Smoothing",
                    linewidth=3
                )
                
                # Draw vehicles
                visualizer.draw_vehicle(
                    path1[0], 
                    planner.vehicle_length, 
                    planner.vehicle_width, 
                    color=env.vehicles[1]['color']
                )
                
                visualizer.draw_vehicle(
                    path2[-1], 
                    planner.vehicle_length, 
                    planner.vehicle_width, 
                    color=env.vehicles[2]['color']
                )
                
                # Add metrics to info panel
                visualizer.update_info_panel({
                    "Path Comparison": "",
                    "Without Smoothing": f"Points: {len(path1)}, Length: {length1:.1f}",
                    "With Smoothing": f"Points: {len(path2)}, Length: {length2:.1f}",
                    "Smoothness (No)": f"{smoothness1:.4f}",
                    "Smoothness (Yes)": f"{smoothness2:.4f}",
                    "Improvement": f"{(smoothness2-smoothness1)/smoothness1*100:.1f}%"
                })
            
            # Show legend
            visualizer.show_legend(loc='upper left')
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                visualizer.save_figure("path_smoothing_effect.png", dpi=150)
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
            
            # Create detail visualization - compare path curvature
            if path1 and path2:
                # Create new plot for curvature comparison
                fig2, ax2 = plt.subplots(1, 2, figsize=(14, 6))
                
                # Function to calculate and plot curvature
                def plot_curvature(path, ax, title, color):
                    if len(path) < 3:
                        return
                    
                    # Calculate angle changes at each point
                    angles = []
                    points = []
                    
                    for i in range(1, len(path) - 1):
                        prev, curr, next_p = path[i-1], path[i], path[i+1]
                        
                        # Calculate vectors
                        v1 = (curr[0] - prev[0], curr[1] - prev[1])
                        v2 = (next_p[0] - curr[0], next_p[1] - curr[1])
                        
                        # Calculate magnitudes
                        mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                        mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                        
                        if mag1 < 1e-3 or mag2 < 1e-3:
                            continue
                            
                        # Calculate dot product and angle
                        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                        cos_angle = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))
                        angle = math.acos(cos_angle)
                        
                        # Point index along path
                        point_idx = i / (len(path) - 1)
                        
                        angles.append(angle)
                        points.append(point_idx)
                    
                    # Plot angles
                    ax.plot(points, angles, color=color, linewidth=2)
                    
                    # Plot moving average for smoother visualization
                    window = min(5, len(angles) // 3)
                    if window > 1 and len(angles) > window:
                        moving_avg = []
                        for i in range(len(angles) - window + 1):
                            avg = sum(angles[i:i+window]) / window
                            moving_avg.append(avg)
                        
                        avg_points = [points[i + window//2] for i in range(len(moving_avg))]
                        ax.plot(avg_points, moving_avg, color=color, linewidth=3, alpha=0.7, linestyle='--')
                    
                    # Set labels
                    ax.set_title(title)
                    ax.set_xlabel("Position along path")
                    ax.set_ylabel("Angle change (radians)")
                    ax.grid(True, alpha=0.3)
                
                # Plot curvature for both paths
                plot_curvature(path1, ax2[0], "Without Smoothing", env.vehicles[1]['color'])
                plot_curvature(path2, ax2[1], "With Smoothing", env.vehicles[2]['color'])
                
                # Set common y-axis limits
                y_min = min(ax2[0].get_ylim()[0], ax2[1].get_ylim()[0])
                y_max = max(ax2[0].get_ylim()[1], ax2[1].get_ylim()[1])
                ax2[0].set_ylim(y_min, y_max)
                ax2[1].set_ylim(y_min, y_max)
                
                plt.tight_layout()
                
                # Save curvature plot
                if vis_mode in ["save", "both"]:
                    plt.savefig("path_smoothing_curvature.png", dpi=150)
                
                if vis_mode in ["show", "both"]:
                    plt.show()
                else:
                    plt.close()
        else:
            # Original visualization
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Draw obstacles
            obstacle_mask = env.grid.T == 1
            ax.imshow(obstacle_mask, cmap='gray', alpha=0.5, extent=(0, env.width, 0, env.height), origin='lower')
            
            # Draw start and end points
            ax.scatter(start_point[0], start_point[1], c='green', marker='o', s=100, label='Start')
            ax.scatter(end_point[0], end_point[1], c='red', marker='s', s=100, label='End')
            
            # Draw paths
            if path1 and path2:
                # Path 1 (without smoothing)
                path1_x = [p[0] for p in path1]
                path1_y = [p[1] for p in path1]
                ax.plot(path1_x, path1_y, c=env.vehicles[1]['color'], linestyle='-', 
                       linewidth=2, label="Without Smoothing")
                
                # Draw orientation indicators for path 1
                interval = max(1, len(path1) // 10)
                for i in range(0, len(path1), interval):
                    x, y, theta = path1[i]
                    dx = math.cos(theta) * 3
                    dy = math.sin(theta) * 3
                    ax.arrow(x, y, dx, dy, head_width=2, head_length=2, 
                            fc=env.vehicles[1]['color'], ec=env.vehicles[1]['color'], alpha=0.7)
                
                # Path 2 (with smoothing)
                path2_x = [p[0] for p in path2]
                path2_y = [p[1] for p in path2]
                ax.plot(path2_x, path2_y, c=env.vehicles[2]['color'], linestyle='-', 
                       linewidth=2, label="With Smoothing")
                
                # Draw orientation indicators for path 2
                interval = max(1, len(path2) // 10)
                for i in range(0, len(path2), interval):
                    x, y, theta = path2[i]
                    dx = math.cos(theta) * 3
                    dy = math.sin(theta) * 3
                    ax.arrow(x, y, dx, dy, head_width=2, head_length=2, 
                            fc=env.vehicles[2]['color'], ec=env.vehicles[2]['color'], alpha=0.7)
            
            # Set axis properties
            ax.set_xlim(0, env.width)
            ax.set_ylim(0, env.height)
            ax.set_title("Effect of Path Smoothing")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            # Add legend and metrics
            handles, labels = ax.get_legend_handles_labels()
            
            if path1 and path2:
                from matplotlib.lines import Line2D
                
                # Add metrics to legend
                metrics_label1 = f"No Smoothing: Points={len(path1)}, Smoothness={smoothness1:.4f}"
                metrics_label2 = f"With Smoothing: Points={len(path2)}, Smoothness={smoothness2:.4f}"
                
                handles.append(Line2D([0], [0], color='white', lw=0))
                labels.append(metrics_label1)
                
                handles.append(Line2D([0], [0], color='white', lw=0))
                labels.append(metrics_label2)
                
                handles.append(Line2D([0], [0], color='white', lw=0))
                labels.append(f"Improvement: {(smoothness2-smoothness1)/smoothness1*100:.1f}%")
            
            ax.legend(handles, labels, loc='upper right')
            
            # Save or show figure
            if vis_mode in ["save", "both"]:
                plt.savefig("path_smoothing_effect.png")
            
            if vis_mode in ["show", "both"]:
                plt.show()
            else:
                plt.close()
    
    return env, planner

def test_multi_step_transport_task(vis_mode="save", use_enhanced_vis=True, animate=False):
    """
    Test a complete multi-step mining transport task
    
    Args:
        vis_mode: 'save', 'show', or 'both'
        use_enhanced_vis: Whether to use enhanced visualization
        animate: Whether to create an animation of the task
    """
    print("\n=== Testing Complete Mining Transport Task ===")
    
    # Create environment
    env = create_test_environment()
    
    # Create hybrid path planner
    map_service = MapService()
    
    # Set obstacle grid
    obstacle_grid = set([(x, y) for x in range(env.width) for y in range(env.height) if env.grid[x, y] == 1])
    
    # Create hybrid planner
    planner = HybridPathPlanner(env)
    planner.vehicle_length = 6.0
    planner.vehicle_width = 3.0
    planner.turning_radius = 8.0
    planner.obstacle_grids = obstacle_grid
    
    # Define task phases
    phases = [
        {"name": "Empty vehicle to loading", "status": "moving_to_load", "color": [0.2, 0.2, 0.8]},
        {"name": "Loading material", "status": "loading", "color": [0.8, 0.8, 0.2]},
        {"name": "Loaded vehicle to unloading", "status": "moving_to_unload", "color": [0.8, 0.2, 0.2]},
        {"name": "Unloading material", "status": "unloading", "color": [0.2, 0.8, 0.2]},
        {"name": "Return to parking", "status": "returning", "color": [0.2, 0.2, 0.8]}
    ]
    
    # Define key points
    loading_point = env.loading_points[0]
    unloading_point = env.unloading_points[0]
    parking_point = (100, 15, 0)  # Bottom middle
    
    # Create a vehicle
    env.add_vehicle(1, parking_point, None, status="idle")
    
    # Initialize visualization if matplotlib is available
    if HAS_MATPLOTLIB and use_enhanced_vis:
        visualizer = EnhancedVisualization(figsize=(12, 10))
        if animate:
            fig, ax, info_panel = visualizer.setup_figure("Mining Transport Task", interactive=True)
        else:
            fig, ax, info_panel = visualizer.setup_figure("Mining Transport Task")
        
        # Draw environment
        visualizer.draw_environment(env)
        
        # Draw parking point
        ax.scatter(parking_point[0], parking_point[1], c='blue', marker='^', s=100, zorder=20)
        ax.text(parking_point[0]+5, parking_point[1]+5, "Parking", fontsize=12, weight='bold',
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Execute each phase of the transport task
    for phase_idx, phase in enumerate(phases):
        print(f"\nPhase {phase_idx+1}: {phase['name']}")
        
        # Update vehicle status
        env.vehicles[1]['status'] = phase['status']
        env.vehicles[1]['color'] = phase['color']
        
        # Update vehicle loading status based on phase
        if phase['status'] == 'moving_to_load' or phase['status'] == 'loading':
            env.vehicles[1]['load'] = 0  # Empty
        elif phase['status'] == 'moving_to_unload' or phase['status'] == 'unloading':
            env.vehicles[1]['load'] = env.vehicles[1]['max_load']  # Full
        elif phase['status'] == 'returning':
            env.vehicles[1]['load'] = 0  # Empty again
        
        # Plan path for movement phases
        if 'moving' in phase['status'] or phase['status'] == 'returning':
            # Determine target
            if phase['status'] == 'moving_to_load':
                target = (loading_point[0], loading_point[1], 0)
            elif phase['status'] == 'moving_to_unload':
                target = (unloading_point[0], unloading_point[1], 0)
            elif phase['status'] == 'returning':
                target = parking_point
                
            # Plan path
            start = env.vehicles[1]['position']
            path = planner.plan_path(start, target)
            
            if path and len(path) > 1:
                env.vehicles[1]['path'] = path
                print(f"  Planned path with {len(path)} points from {start[:2]} to {target[:2]}")
                
                # Visualize if enabled
                if HAS_MATPLOTLIB and use_enhanced_vis:
                    # Draw path
                    visualizer.draw_path(
                        path, 
                        color=phase['color'], 
                        label=f"Phase {phase_idx+1}: {phase['name']}"
                    )
                    
                    if animate:
                        # Draw initial vehicle position
                        visualizer.draw_vehicle(
                            start,
                            planner.vehicle_length,
                            planner.vehicle_width,
                            color=phase['color'],
                            load_percent=(env.vehicles[1]['load'] / env.vehicles[1]['max_load']) * 100,
                            label="V1"
                        )
                        
                        # Update info panel
                        visualizer.update_info_panel({
                            "Phase": f"{phase_idx+1}/{len(phases)}: {phase['name']}",
                            "Status": "Starting movement",
                            "Load": f"{env.vehicles[1]['load']}/{env.vehicles[1]['max_load']}",
                            "Position": f"({start[0]:.1f}, {start[1]:.1f})",
                            "Target": f"({target[0]:.1f}, {target[1]:.1f})"
                        })
                        
                        visualizer.update_and_pause(0.5)
                        
                        if animate:
                            visualizer.add_animation_frame()
                
                # Simulate vehicle movement along path
                steps = max(1, len(path) // 10)  # Show 10 animation steps
                
                for i in range(0, len(path), steps):
                    # Update vehicle position
                    point = path[i]
                    env.vehicles[1]['position'] = point
                    
                    # Update visualization
                    if HAS_MATPLOTLIB and use_enhanced_vis and animate:
                        # Clear previous vehicle
                        ax.clear()
                        
                        # Redraw environment
                        visualizer.draw_environment(env)
                        
                        # Draw parking, loading, unloading points again
                        ax.scatter(parking_point[0], parking_point[1], c='blue', marker='^', s=100, zorder=20)
                        ax.text(parking_point[0]+5, parking_point[1]+5, "Parking", fontsize=12, weight='bold',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
                        
                        # Draw path
                        visualizer.draw_path(
                            path, 
                            color=phase['color'], 
                            label=f"Phase {phase_idx+1}: {phase['name']}"
                        )
                        
                        # Draw vehicle at current position
                        visualizer.draw_vehicle(
                            point,
                            planner.vehicle_length,
                            planner.vehicle_width,
                            color=phase['color'],
                            load_percent=(env.vehicles[1]['load'] / env.vehicles[1]['max_load']) * 100,
                            label="V1"
                        )
                        
                        # Update info panel
                        progress = (i+1) / len(path) * 100
                        visualizer.update_info_panel({
                            "Phase": f"{phase_idx+1}/{len(phases)}: {phase['name']}",
                            "Status": "Moving",
                            "Load": f"{env.vehicles[1]['load']}/{env.vehicles[1]['max_load']}",
                            "Position": f"({point[0]:.1f}, {point[1]:.1f})",
                            "Progress": f"{progress:.1f}%"
                        })
                        
                        visualizer.update_and_pause(0.1)
                        
                        if animate and i % (steps * 2) == 0:  # Add fewer frames to keep file size manageable
                            visualizer.add_animation_frame()
                
                # Update vehicle to final position
                env.vehicles[1]['position'] = path[-1]
                
                # Final animation frame for this movement phase
                if HAS_MATPLOTLIB and use_enhanced_vis and animate:
                    # Clear and redraw
                    ax.clear()
                    visualizer.draw_environment(env)
                    
                    # Draw parking point
                    ax.scatter(parking_point[0], parking_point[1], c='blue', marker='^', s=100, zorder=20)
                    ax.text(parking_point[0]+5, parking_point[1]+5, "Parking", fontsize=12, weight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
                    
                    # Draw path
                    visualizer.draw_path(
                        path, 
                        color=phase['color'], 
                        label=f"Phase {phase_idx+1}: {phase['name']}"
                    )
                    
                    # Draw vehicle at final position
                    visualizer.draw_vehicle(
                        path[-1],
                        planner.vehicle_length,
                        planner.vehicle_width,
                        color=phase['color'],
                        load_percent=(env.vehicles[1]['load'] / env.vehicles[1]['max_load']) * 100,
                        label="V1"
                    )
                    
                    # Update info panel
                    visualizer.update_info_panel({
                        "Phase": f"{phase_idx+1}/{len(phases)}: {phase['name']}",
                        "Status": "Arrived",
                        "Load": f"{env.vehicles[1]['load']}/{env.vehicles[1]['max_load']}",
                        "Position": f"({path[-1][0]:.1f}, {path[-1][1]:.1f})",
                        "Progress": "100%"
                    })
                    
                    visualizer.update_and_pause(0.5)
                    
                    if animate:
                        visualizer.add_animation_frame()
            else:
                print(f"  Path planning failed from {start[:2]} to {target[:2]}")
                
        elif phase['status'] == 'loading' or phase['status'] == 'unloading':
            # Simulate loading/unloading process
            load_steps = 5
            if phase['status'] == 'loading':
                start_load = 0
                end_load = env.vehicles[1]['max_load']
            else:  # unloading
                start_load = env.vehicles[1]['max_load']
                end_load = 0
            
            # Simulate gradual loading/unloading
            for step in range(load_steps + 1):
                # Calculate current load
                current_load = start_load + (end_load - start_load) * step / load_steps
                env.vehicles[1]['load'] = current_load
                
                # Visualize loading/unloading
                if HAS_MATPLOTLIB and use_enhanced_vis and animate:
                    # Clear and redraw
                    ax.clear()
                    visualizer.draw_environment(env)
                    
                    # Draw parking point
                    ax.scatter(parking_point[0], parking_point[1], c='blue', marker='^', s=100, zorder=20)
                    ax.text(parking_point[0]+5, parking_point[1]+5, "Parking", fontsize=12, weight='bold',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
                    
                    # Draw loading zone with animation effect
                    position = env.vehicles[1]['position']
                    if phase['status'] == 'loading':
                        # Draw loading point with animation
                        radius = 5 + step * 0.5  # Expanding circle effect
                        alpha = 0.5 + step * 0.1  # Increasing opacity
                        ax.add_patch(plt.Circle((loading_point[0], loading_point[1]), radius, 
                                              color=[0.8, 0.8, 0.2], alpha=alpha * 0.5, zorder=10))
                    else:  # unloading
                        # Draw unloading point with animation
                        radius = 5 + step * 0.5  # Expanding circle effect
                        alpha = 0.5 + step * 0.1  # Increasing opacity
                        ax.add_patch(plt.Circle((unloading_point[0], unloading_point[1]), radius, 
                                              color=[0.2, 0.8, 0.2], alpha=alpha * 0.5, zorder=10))
                    
                    # Draw vehicle with updated load
                    visualizer.draw_vehicle(
                        position,
                        planner.vehicle_length,
                        planner.vehicle_width,
                        color=phase['color'],
                        load_percent=(current_load / env.vehicles[1]['max_load']) * 100,
                        label="V1"
                    )
                    
                    # Update info panel
                    visualizer.update_info_panel({
                        "Phase": f"{phase_idx+1}/{len(phases)}: {phase['name']}",
                        "Status": "In Progress",
                        "Load": f"{int(current_load)}/{env.vehicles[1]['max_load']}",
                        "Position": f"({position[0]:.1f}, {position[1]:.1f})",
                        "Progress": f"{step}/{load_steps} steps"
                    })
                    
                    visualizer.update_and_pause(0.3)
                    
                    if animate:
                        visualizer.add_animation_frame()
            
            print(f"  {phase['name']} completed")
    
    # Create final visualization
    if HAS_MATPLOTLIB and use_enhanced_vis:
        # If we weren't animating, we need to draw final state
        if not animate:
            # Draw environment
            visualizer.draw_environment(env)
            
            # Draw key points
            ax.scatter(parking_point[0], parking_point[1], c='blue', marker='^', s=100, zorder=20)
            ax.text(parking_point[0]+5, parking_point[1]+5, "Parking", fontsize=12, weight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
            
            # Draw final vehicle position
            visualizer.draw_vehicle(
                env.vehicles[1]['position'],
                planner.vehicle_length,
                planner.vehicle_width,
                color=[0.2, 0.2, 0.8],  # Blue for return phase
                load_percent=0,
                label="V1"
            )
        
        # Show legend and update info panel
        visualizer.show_legend()
        visualizer.update_info_panel({
            "Transport Task": "Completed",
            "Total Phases": f"{len(phases)}",
            "Final Status": "Returned to Parking",
            "Position": f"({env.vehicles[1]['position'][0]:.1f}, {env.vehicles[1]['position'][1]:.1f})"
        })
        
        # Save animation if created
        if animate:
            visualizer.create_animation("complete_transport_task.gif", fps=5)
        
        # Save or show figure
        if vis_mode in ["save", "both"]:
            visualizer.save_figure("complete_transport_task.png", dpi=150)
        
        if vis_mode in ["show", "both"]:
            plt.show()
        else:
            plt.close()
    
    print("\nComplete transport task execution finished!")
    return env, planner

def main():
    """Main function to run tests"""
    parser = argparse.ArgumentParser(description="Test Hybrid Path Planner")
    parser.add_argument("--test", type=str, default="all", 
                        help="Test to run (basic, hybrid, radii, smoothing, cbs, transport, all)")
    parser.add_argument("--vis", type=str, default="save", 
                        help="Visualization mode (save, show, both, none)")
    parser.add_argument("--no-vis", action="store_true", default=False,
                        help="Disable enhanced visualization")
    parser.add_argument("--animate", action="store_true", default=False,
                        help="Create animations for applicable tests")
    
    args = parser.parse_args()
    
    # Set up visualization parameters
    use_enhanced_vis = not args.no_vis and HAS_MATPLOTLIB
    vis_mode = args.vis
    animate = args.animate and HAS_MATPLOTLIB
    
    # Determine which tests to run
    if args.test == "basic" or args.test == "all":
        # Test 1: Basic path planning
        test_basic_path_planning(vis_mode=vis_mode, use_enhanced_vis=use_enhanced_vis)
    
    if args.test == "hybrid" or args.test == "all":
        # Test 2: Hybrid path planning
        test_hybrid_path_planning(vis_mode=vis_mode, use_enhanced_vis=use_enhanced_vis)
    
    if args.test == "radii" or args.test == "all":
        # Test 3: Path planning with varying turning radii
        test_path_planning_with_varying_radii(vis_mode=vis_mode, use_enhanced_vis=use_enhanced_vis)
    
    if args.test == "smoothing" or args.test == "all":
        # Test 4: Path smoothing
        test_path_smoothing(vis_mode=vis_mode, use_enhanced_vis=use_enhanced_vis)
    
    if args.test == "cbs" or args.test == "all":
        # Test 5: Multiple vehicles with CBS
        test_multiple_vehicles_with_cbs(vis_mode=vis_mode, use_enhanced_vis=use_enhanced_vis, animate=animate)
    
    if args.test == "transport" or args.test == "all":
        # Test 6: Complete transport task
        test_multi_step_transport_task(vis_mode=vis_mode, use_enhanced_vis=use_enhanced_vis, animate=animate)
    
    print("\nAll tests completed!")
    
    # Keep plots open if matplotlib is available and show mode is enabled
    if HAS_MATPLOTLIB and vis_mode in ["show", "both"]:
        plt.show()

if __name__ == "__main__":
    main()