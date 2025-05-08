import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Arrow, FancyArrowPatch
from matplotlib.collections import PatchCollection
import numpy as np
import math
import time

class EnhancedVisualization:
    """Enhanced visualization module for Hybrid Path Planner tests"""
    
    def __init__(self, figsize=(12, 12), dpi=100):
        """Initialize visualization with figure size and DPI"""
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        self.bg_cache = None
        self.is_interactive = False
        self.animation_frames = []
        
        # Color schemes
        self.color_schemes = {
            'default': {
                'vehicle_empty': [0.2, 0.4, 0.8],      # Blue for empty vehicle
                'vehicle_loaded': [0.8, 0.3, 0.2],     # Red for loaded vehicle
                'loading_point': [0.2, 0.8, 0.2],      # Green for loading points
                'unloading_point': [0.8, 0.2, 0.4],    # Pink for unloading points
                'parking_point': [0.4, 0.4, 0.7],      # Purple for parking
                'obstacle': [0.3, 0.3, 0.3],           # Dark gray for obstacles
                'path': [0.0, 0.6, 0.8],               # Cyan for paths
                'conflict': [0.9, 0.2, 0.2],           # Bright red for conflicts
                'background': [0.95, 0.95, 0.98]       # Light gray-blue for background
            },
            'colorblind_friendly': {
                'vehicle_empty': [0.0, 0.45, 0.7],     # Blue
                'vehicle_loaded': [0.9, 0.6, 0.0],     # Orange
                'loading_point': [0.0, 0.6, 0.5],      # Teal
                'unloading_point': [0.8, 0.4, 0.0],    # Vermillion
                'parking_point': [0.8, 0.6, 0.7],      # Reddish purple
                'obstacle': [0.4, 0.4, 0.4],           # Gray
                'path': [0.35, 0.7, 0.9],              # Sky blue
                'conflict': [0.8, 0.0, 0.0],           # Red
                'background': [0.95, 0.95, 0.95]       # Light gray
            }
        }
        
        # Default color scheme
        self.current_scheme = 'default'
        
    def setup_figure(self, title="Path Planning Visualization", interactive=False):
        """Set up the figure and axes"""
        self.is_interactive = interactive
        
        if interactive:
            plt.ion()  # Turn on interactive mode
        
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_facecolor(self.get_color('background'))
        
        # Add info panel area
        self.info_panel = self.fig.add_axes([0.02, 0.02, 0.25, 0.25], frameon=True, facecolor='white', alpha=0.8)
        self.info_panel.set_xticks([])
        self.info_panel.set_yticks([])
        self.info_panel.set_title("Status", fontsize=10)
        
        if interactive:
            # Cache background for blitting in animations
            self.fig.canvas.draw()
            self.bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        
        return self.fig, self.ax, self.info_panel
    
    def get_color(self, color_type):
        """Get color from current color scheme"""
        return self.color_schemes[self.current_scheme].get(
            color_type, 
            self.color_schemes['default'].get(color_type, [0.5, 0.5, 0.5])
        )
    
    def set_color_scheme(self, scheme_name):
        """Set the active color scheme"""
        if scheme_name in self.color_schemes:
            self.current_scheme = scheme_name
            # Update figure if it exists
            if self.fig:
                self.ax.set_facecolor(self.get_color('background'))
    
    def draw_environment(self, env, clear=True):
        """Draw environment with obstacles, loading, and unloading points"""
        if clear and self.ax:
            self.ax.clear()
            self.ax.set_facecolor(self.get_color('background'))
        
        # Draw grid with obstacles
        obstacle_mask = env.grid.T == 1  # Transpose for correct orientation
        
        # Draw obstacles with improved appearance
        obstacle_color = self.get_color('obstacle')
        self.ax.imshow(
            obstacle_mask, 
            cmap='binary', 
            alpha=0.7, 
            extent=(0, env.width, 0, env.height), 
            origin='lower',
            vmin=0, vmax=1
        )
        
        # Draw grid lines
        grid_step = 20  # Adjust grid line frequency
        for x in range(0, env.width + 1, grid_step):
            self.ax.axvline(x, color='gray', linestyle='-', alpha=0.2)
        for y in range(0, env.height + 1, grid_step):
            self.ax.axhline(y, color='gray', linestyle='-', alpha=0.2)
        
        # Draw loading points
        for i, point in enumerate(env.loading_points):
            self.ax.scatter(
                point[0], point[1], 
                c=[self.get_color('loading_point')], 
                marker='o', s=150, edgecolors='black', linewidths=1, zorder=10
            )
            self.ax.text(
                point[0] + 5, point[1] + 5, 
                f"Loading {i+1}", 
                fontsize=10, weight='bold', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )
        
        # Draw unloading points
        for i, point in enumerate(env.unloading_points):
            self.ax.scatter(
                point[0], point[1], 
                c=[self.get_color('unloading_point')], 
                marker='s', s=150, edgecolors='black', linewidths=1, zorder=10
            )
            self.ax.text(
                point[0] + 5, point[1] + 5, 
                f"Unloading {i+1}", 
                fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
            )
        
        # Set axis properties
        self.ax.set_xlim(0, env.width)
        self.ax.set_ylim(0, env.height)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
    
    def draw_vehicle(self, position, length, width, color=None, alpha=0.8, load_percent=0, label=None):
        """Draw vehicle as a rectangle with orientation and loading indicator"""
        if color is None:
            if load_percent > 50:
                color = self.get_color('vehicle_loaded')
            else:
                color = self.get_color('vehicle_empty')
        
        x, y, theta = position
        
        # Create rectangle centered at origin
        dx = length / 2
        dy = width / 2
        corners = [
            (-dx, -dy),  # Bottom-left
            (dx, -dy),   # Bottom-right
            (dx, dy),    # Top-right
            (-dx, dy),   # Top-left
            (-dx, -dy)   # Back to start
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
        vehicle_patch = self.ax.fill(xs, ys, color=color, alpha=alpha, zorder=20)[0]
        
        # Draw direction indicator (front of vehicle)
        front_x = x + dx * 0.8 * cos_t - 0 * sin_t
        front_y = y + dx * 0.8 * sin_t + 0 * cos_t
        
        # Draw a nicer arrow for direction
        arrow = FancyArrowPatch(
            (x, y), (front_x, front_y),
            arrowstyle='simple', 
            mutation_scale=20,
            color='black', 
            alpha=0.8,
            zorder=25
        )
        self.ax.add_patch(arrow)
        
        # Add load indicator if load_percent > 0
        if load_percent > 0:
            # Draw load as a rectangle inside the vehicle
            load_width = width * 0.6
            load_length = length * 0.6 * (load_percent / 100)
            load_x = x - dx * 0.5  # Position load at the rear half
            load_y = y
            
            # Create load rectangle
            load_corners = [
                (0, -load_width/2),
                (load_length, -load_width/2),
                (load_length, load_width/2),
                (0, load_width/2),
                (0, -load_width/2)
            ]
            
            # Rotate and translate load corners
            load_rotated = []
            for cx, cy in load_corners:
                rx = load_x + cx * cos_t - cy * sin_t
                ry = load_y + cx * sin_t + cy * cos_t
                load_rotated.append((rx, ry))
            
            # Draw load polygon with slightly darker color
            load_xs, load_ys = zip(*load_rotated)
            load_color = [c * 0.8 for c in color]  # Darker shade
            self.ax.fill(load_xs, load_ys, color=load_color, alpha=alpha, zorder=21)
        
        # Add label if provided
        if label:
            self.ax.text(
                x, y, label,
                fontsize=10, weight='bold', color='white',
                ha='center', va='center', zorder=30
            )
        
        return vehicle_patch
    
    def draw_path(self, path, color=None, linewidth=2, alpha=0.7, show_orientations=True, label=None):
        """Draw vehicle path with orientation indicators"""
        if not path or len(path) < 2:
            return None
            
        if color is None:
            color = self.get_color('path')
            
        # Extract path coordinates
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        
        # Draw path line
        path_line = self.ax.plot(
            path_x, path_y, 
            c=color, linestyle='-', linewidth=linewidth, alpha=alpha,
            label=label if label else None,
            zorder=5
        )[0]
        
        # Draw orientation at intervals if requested
        if show_orientations:
            # Calculate reasonable interval based on path length
            interval = max(1, len(path) // 20)  # Show at most 20 orientations
            
            for i in range(0, len(path), interval):
                if len(path[i]) >= 3:  # Has orientation
                    x, y, theta = path[i]
                    dx = math.cos(theta) * 3  # Arrow length
                    dy = math.sin(theta) * 3
                    
                    # Draw orientation arrow
                    arrow = FancyArrowPatch(
                        (x, y), (x + dx, y + dy),
                        arrowstyle='-|>', 
                        mutation_scale=10,
                        color=color, 
                        alpha=alpha,
                        zorder=15
                    )
                    self.ax.add_patch(arrow)
        
        return path_line
    
    def draw_conflict_points(self, conflicts, size=200, alpha=0.7):
        """Draw conflict points on the map"""
        if not conflicts:
            return
            
        for conflict in conflicts:
            # Mark conflict location
            location = conflict["location"]
            time = conflict.get("time", "?")
            
            # Draw star marker for conflict
            self.ax.scatter(
                location[0], location[1], 
                c=[self.get_color('conflict')], 
                marker='*', s=size, alpha=alpha,
                edgecolors='black', linewidths=1,
                zorder=30
            )
            
            # Add conflict info text
            self.ax.text(
                location[0], location[1] + 5, 
                f"Conflict at t={time}", 
                ha='center', va='bottom', 
                color=self.get_color('conflict'), 
                fontsize=10, weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2),
                zorder=31
            )
    
    def update_info_panel(self, info_dict):
        """Update info panel with vehicle and simulation status"""
        if not hasattr(self, 'info_panel') or self.info_panel is None:
            return
            
        self.info_panel.clear()
        self.info_panel.set_xticks([])
        self.info_panel.set_yticks([])
        
        # Title with larger, bold font
        self.info_panel.set_title("Status", fontsize=12, fontweight='bold')
        
        # Format and display information
        y_pos = 0.9
        line_height = 0.1
        
        for key, value in info_dict.items():
            # Format key as bold
            self.info_panel.text(
                0.05, y_pos, f"{key}:", 
                fontsize=9, fontweight='bold', 
                ha='left', va='center'
            )
            
            # Format value
            self.info_panel.text(
                0.35, y_pos, f"{value}", 
                fontsize=9,
                ha='left', va='center'
            )
            
            y_pos -= line_height
    
    def create_animation(self, save_path, fps=10, dpi=100):
        """Create animation from stored frames"""
        if not self.animation_frames:
            print("No animation frames available")
            return False
            
        print(f"Creating animation with {len(self.animation_frames)} frames...")
        
        # Create figure for animation
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Animation function
        def animate(i):
            ax.clear()
            ax.imshow(self.animation_frames[i])
            ax.axis('off')
            return [ax]
        
        # Create animation
        ani = animation.FuncAnimation(
            fig, animate, frames=len(self.animation_frames),
            interval=1000/fps, blit=True
        )
        
        # Save animation
        ani.save(save_path, writer='pillow', fps=fps, dpi=dpi)
        print(f"Animation saved to {save_path}")
        
        plt.close(fig)
        return True
    
    def add_animation_frame(self):
        """Capture current figure as animation frame"""
        if not self.fig:
            return
            
        # Render figure
        self.fig.canvas.draw()
        
        # Get image from canvas
        frame = np.array(self.fig.canvas.renderer.buffer_rgba())
        self.animation_frames.append(frame)
    
    def show_legend(self, loc='upper right'):
        """Add legend to the plot"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=self.get_color('loading_point'), markersize=10, 
                   label='Loading Point'),
            Line2D([0], [0], marker='s', color='w', 
                   markerfacecolor=self.get_color('unloading_point'), markersize=10, 
                   label='Unloading Point'),
            Rectangle((0, 0), 1, 1, facecolor=self.get_color('vehicle_empty'), 
                      label='Empty Vehicle'),
            Rectangle((0, 0), 1, 1, facecolor=self.get_color('vehicle_loaded'), 
                      label='Loaded Vehicle'),
            Line2D([0], [0], color=self.get_color('path'), lw=2, label='Vehicle Path'),
            Line2D([0], [0], marker='*', color='w', 
                   markerfacecolor=self.get_color('conflict'), markersize=10, 
                   label='Conflict Point')
        ]
        
        self.ax.legend(handles=legend_elements, loc=loc, framealpha=0.9)
    
    def setup_comparison_view(self, rows=1, cols=2, titles=None):
        """Set up a multi-panel comparison view"""
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0]*cols, self.figsize[1]*rows))
        
        # Handle single row case
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Set titles if provided
        if titles:
            for i, ax in enumerate(axes.flat):
                if i < len(titles):
                    ax.set_title(titles[i])
        
        return fig, axes
    
    def save_figure(self, filename, dpi=None, bbox_inches='tight'):
        """Save current figure to file"""
        if not self.fig:
            print("No figure to save")
            return False
            
        if dpi is None:
            dpi = self.dpi
            
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Figure saved to {filename}")
        return True
        
    def update_and_pause(self, pause_time=0.01):
        """Update display and pause for interactive mode"""
        if not self.is_interactive:
            return
            
        # Update canvas
        self.fig.canvas.draw()
        
        # Pause to let the GUI catch up
        plt.pause(pause_time)
    
    def close(self):
        """Close figure and clean up"""
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.info_panel = None
        
        if self.is_interactive:
            plt.ioff()  # Turn off interactive mode


# Example usage in test file:
if __name__ == "__main__":
    # Simple test to verify the visualization module
    class DummyEnv:
        def __init__(self):
            self.width = 200
            self.height = 200
            self.grid = np.zeros((200, 200))
            
            # Add some obstacles
            for x in range(50, 70):
                for y in range(50, 150):
                    self.grid[x, y] = 1
                    
            for x in range(130, 150):
                for y in range(50, 150):
                    self.grid[x, y] = 1
            
            self.loading_points = [(30, 30), (30, 170)]
            self.unloading_points = [(170, 30), (170, 170)]
    
    # Create dummy environment
    env = DummyEnv()
    
    # Create visualization
    vis = EnhancedVisualization()
    fig, ax, info = vis.setup_figure("Test Visualization")
    
    # Draw environment
    vis.draw_environment(env)
    
    # Draw vehicle
    vis.draw_vehicle((100, 100, math.pi/4), 6.0, 3.0, label="V1")
    
    # Draw path
    path = [(100, 100, math.pi/4)]
    for i in range(1, 20):
        x = 100 + i * 3
        y = 100 + i * 2
        theta = math.pi/4 + i * 0.05
        path.append((x, y, theta))
    
    vis.draw_path(path)
    
    # Update info panel
    vis.update_info_panel({
        "Vehicle": "V1", 
        "Status": "Moving", 
        "Load": "0/100",
        "Speed": "5.0 m/s"
    })
    
    # Show legend
    vis.show_legend()
    
    # Save figure
    vis.save_figure("test_visualization.png")
    
    plt.show()