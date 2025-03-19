#!/usr/bin/env python3
"""
Real-time Manifold Visualization
--------------------------------
This script creates a real-time 3D visualization of the manifold as it evolves
during training. It monitors the metric tensor files and updates the visualization
when changes are detected.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import json
from matplotlib import cm
import threading
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Set the matplotlib backend to work properly on macOS
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend which works well for interactive plotting on macOS

# Paths for metric tensor files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRIC_PATHS = [
    os.path.join(BASE_DIR, 'metric_tensor.csv'),
    os.path.join(BASE_DIR, 'paths_on_manifold', 'metric_tensor.csv'),
    os.path.join(BASE_DIR, 'paths_on_manifold', 'data', 'metric_tensor.csv'),
    os.path.join(BASE_DIR, 'paths_on_manifold', 'metric_data_current.csv')
]

METADATA_PATHS = [
    os.path.join(BASE_DIR, 'metric_metadata.json'),
    os.path.join(BASE_DIR, 'paths_on_manifold', 'metric_metadata.json'),
    os.path.join(BASE_DIR, 'paths_on_manifold', 'data', 'metric_metadata.json')
]

# Create a custom colormap for better visualization
colors = [(0.2, 0.4, 0.6, 1.0), (0.95, 0.95, 0.95, 1.0), (0.8, 0.2, 0.2, 1.0)]
cmap_name = 'manifold_cmap'
manifold_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

class ManifoldVisualizer:
    def __init__(self, grid_size=20, update_interval=100):
        """
        Initialize the manifold visualizer.
        
        Parameters:
        -----------
        grid_size : int
            Number of points in each dimension of the grid
        update_interval : int
            Interval in milliseconds at which to check for updates
        """
        self.grid_size = grid_size
        self.update_interval = update_interval
        
        # Initialize figure and axes
        plt.style.use('dark_background')  # Use dark background for better visualization
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize metric tensor
        self.metric = None
        self.last_modified_time = 0
        
        # Create coordinate grid
        self.x = np.linspace(-10, 10, grid_size)
        self.y = np.linspace(-10, 10, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.Z = np.zeros((grid_size, grid_size))
        
        # Training progress tracking
        self.epoch = 0
        self.total_epochs = 100
        self.step = 0
        self.total_steps = 1000
        self.loss = 0.0
        
        # Animation object
        self.ani = None
        
        # Initialize the plot
        self.surface = None
        self.setup_plot()
        
    def setup_plot(self):
        """Set up the initial plot elements."""
        self.ax.set_title("Real-time Manifold Visualization", fontsize=16)
        self.ax.set_xlabel("X", fontsize=12)
        self.ax.set_ylabel("Y", fontsize=12)
        self.ax.set_zlabel("Z", fontsize=12)
        
        # Create the initial surface plot
        self.surface = self.ax.plot_surface(
            self.X, self.Y, self.Z, 
            cmap=manifold_cmap, 
            linewidth=0,  # Changed from 0.2 to 0 to hide coordinate lines
            antialiased=True,
            alpha=0.8
        )
        
        # Add a color bar
        cbar = self.fig.colorbar(self.surface, ax=self.ax, shrink=0.5, aspect=5)
        cbar.set_label("Metric Curvature")
        
        # Add training progress text
        self.progress_text = self.ax.text2D(
            0.02, 0.95, "Training Progress: Initializing...", 
            transform=self.ax.transAxes,
            fontsize=12
        )
        
        # Set view angle
        self.ax.view_init(elev=30, azim=-45)
        
        # Set axis limits
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_zlim(-10, 10)
        
    def load_metric(self):
        """Load the metric tensor from file."""
        # Check each possible metric path
        latest_time = self.last_modified_time
        latest_metric = None
        
        for path in METRIC_PATHS:
            if os.path.exists(path):
                mod_time = os.path.getmtime(path)
                if mod_time > latest_time:
                    try:
                        metric = np.loadtxt(path, delimiter=',')
                        if metric.shape[0] > 0:  # Ensure metric is not empty
                            latest_time = mod_time
                            latest_metric = metric
                            print(f"Loaded metric from {path}")
                    except Exception as e:
                        print(f"Error loading metric from {path}: {e}")
        
        # Update if we found a newer metric tensor
        if latest_metric is not None:
            self.metric = latest_metric
            self.last_modified_time = latest_time
            return True
        return False
    
    def load_metadata(self):
        """Load training metadata from file."""
        for path in METADATA_PATHS:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                        self.epoch = metadata.get('epoch', 0)
                        self.total_epochs = metadata.get('total_epochs', 100)
                        self.step = metadata.get('step', 0)
                        self.total_steps = metadata.get('total_steps', 1000)
                        self.loss = metadata.get('loss', 0.0)
                        print(f"Loaded metadata from {path}")
                        return True
                except Exception as e:
                    print(f"Error loading metadata from {path}: {e}")
        return False
    
    def calculate_manifold(self):
        """Calculate the manifold geometry from the metric tensor."""
        if self.metric is None:
            return
        
        # Reset the Z values
        self.Z = np.zeros((self.grid_size, self.grid_size))
        
        # Calculate manifold geometry based on metric tensor
        # For visualization, use various components of the metric tensor to influence the shape
        dim = self.metric.shape[0]
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x, y = self.X[i, j], self.Y[i, j]
                
                # Calculate the curvature/displacement based on the metric tensor
                # Using a combination of metric components for visualization
                
                # Base displacement - use determinant of the metric as a guide for curvature
                try:
                    if dim >= 2:
                        det = np.linalg.det(self.metric[:2, :2])
                    elif dim == 1:
                        det = self.metric[0, 0]
                    else:
                        det = 1.0
                        
                    # Scale factor to make visualization more dramatic
                    scale = 5.0
                    
                    # Calculate curved surface based on metric components
                    # For 2x2 or larger metrics
                    if dim >= 2:
                        # Use metric components for different visual effects
                        g00 = self.metric[0, 0]  # time-time component
                        g11 = self.metric[1, 1]  # x-x component
                        
                        # Off-diagonal terms create twisting effects
                        g01 = self.metric[0, 1] if dim > 1 else 0  # time-x component
                        
                        # Higher dimensions add additional curvature
                        if dim >= 3:
                            g22 = self.metric[2, 2]  # y-y component
                        else:
                            g22 = 1.0
                            
                        # Calculate distance from origin for radial effects
                        r = np.sqrt(x**2 + y**2)
                        
                        # Create visual wave patterns based on metric components
                        base_z = 0.0
                        
                        # Time component (g00) creates valleys
                        base_z -= abs(g00) * np.exp(-0.05 * r**2) * 2.0
                        
                        # Spatial components (g11, g22) create peaks and ripples
                        base_z += g11 * np.sin(r * 0.2) * np.exp(-0.02 * r**2) * 1.0
                        base_z += g22 * np.cos(r * 0.3) * np.exp(-0.03 * r**2) * 1.5
                        
                        # Off-diagonal components create twisting
                        twist = g01 * x * y * 0.05
                        base_z += twist
                        
                        # Apply scaling and set Z value
                        self.Z[i, j] = base_z * scale
                    else:
                        # Simple case for 1x1 metrics
                        self.Z[i, j] = self.metric[0, 0] * np.exp(-0.02 * (x**2 + y**2)) * scale
                except Exception as e:
                    print(f"Error calculating manifold: {e}")
                    self.Z[i, j] = 0
        
    def update_plot(self, frame):
        """Update function for animation."""
        # Check for metric updates
        metric_updated = self.load_metric()
        metadata_updated = self.load_metadata()
        
        if metric_updated:
            # Calculate new manifold geometry
            self.calculate_manifold()
            
            # Remove previous surface
            if self.surface:
                self.surface.remove()
            
            # Create new surface
            self.surface = self.ax.plot_surface(
                self.X, self.Y, self.Z, 
                cmap=manifold_cmap, 
                linewidth=0,  # Changed from 0.2 to 0 to hide coordinate lines
                antialiased=True,
                alpha=0.8
            )
            
            # Update title with metric details
            if self.metric is not None:
                try:
                    det = np.linalg.det(self.metric)
                    title = f"Manifold Visualization (Det: {det:.4f})"
                    self.ax.set_title(title, fontsize=16)
                except:
                    self.ax.set_title("Manifold Visualization", fontsize=16)
        
        # Update progress text if metadata changed or every frame
        if metadata_updated or frame % 10 == 0:
            progress = f"Epoch: {self.epoch}/{self.total_epochs}, Step: {self.step}/{self.total_steps}, Loss: {self.loss:.6f}"
            self.progress_text.set_text(progress)
        
        # Rotate view slightly on each frame for dynamic effect
        if frame % 5 == 0:
            azimuth = self.ax.azim + 0.5
            self.ax.view_init(elev=self.ax.elev, azim=azimuth)
            
        return [self.surface, self.progress_text]
    
    def start_animation(self):
        """Start the animation."""
        self.ani = FuncAnimation(
            self.fig, 
            self.update_plot, 
            frames=range(10000),  # Large number for continuous animation
            interval=self.update_interval,
            blit=False  # Redraw the entire figure
        )
        
        plt.tight_layout()
        plt.show()

def run_visualization():
    """Main function to run the visualization."""
    print("Starting real-time manifold visualization...")
    print("Press Ctrl+C to exit")
    
    visualizer = ManifoldVisualizer(grid_size=40, update_interval=300)
    
    # Initial loading of metric and metadata
    visualizer.load_metric()
    visualizer.load_metadata()
    visualizer.calculate_manifold()
    
    try:
        visualizer.start_animation()
    except KeyboardInterrupt:
        print("\nVisualization terminated by user")

if __name__ == "__main__":
    run_visualization() 