#!/usr/bin/env python3
"""
Test script to generate dramatic changes in the metric tensor over time.
This helps verify that the visualization is correctly responding to changes in the data files.
"""
import numpy as np
import json
import time
import os
import math

def save_metric_to_csv(metric, filename):
    """Save metric tensor to CSV file"""
    np.savetxt(filename, metric, delimiter=',')
    print(f"Saved metric to {filename}")

def save_christoffel_to_csv(christoffel, filename):
    """Save Christoffel symbols to CSV file"""
    with open(filename, 'w') as f:
        dim = christoffel.shape[0]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    value = christoffel[i, j, k]
                    if abs(value) > 1e-10:  # Only save non-zero values
                        f.write(f"{i},{j},{k},{value}\n")
    print(f"Saved Christoffel symbols to {filename}")

def compute_christoffel(metric):
    """Compute Christoffel symbols from metric tensor"""
    dim = metric.shape[0]
    christoffel = np.zeros((dim, dim, dim))
    
    # Compute inverse metric
    try:
        metric_inv = np.linalg.inv(metric)
    except np.linalg.LinAlgError:
        # Add small diagonal term for stability
        metric_reg = metric + np.eye(dim) * 1e-6
        metric_inv = np.linalg.inv(metric_reg)
    
    # Compute Christoffel symbols numerically
    for l in range(dim):
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    # Simplified computation that creates non-zero values
                    if i == j or i == k or j == k:
                        christoffel[l, i, j] += 0.5 * metric_inv[l, k] * 0.3 * metric[i, j]  # Increased factor
    
    return christoffel

def save_metadata(step, total_steps, epoch, total_epochs, loss, filename):
    """Save metadata to JSON file"""
    metadata = {
        'epoch': epoch,
        'total_epochs': total_epochs,
        'step': step,
        'total_steps': total_steps,
        'loss': loss,
        'is_training': True
    }
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {filename}")

def main():
    """Main function to generate test data"""
    # Parameters
    dim = 4  # Dimension of metric tensor (4 for spacetime)
    num_steps = 50  # Reduced number of steps for faster changes
    delay = 0.3  # Reduced delay for faster updates
    
    # File paths - save to all possible locations
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(base_dir, 'paths_on_manifold')
    data_dir = os.path.join(project_dir, 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Define all file paths to update
    metric_paths = [
        os.path.join(base_dir, 'metric_tensor.csv'),
        os.path.join(project_dir, 'metric_tensor.csv'),
        os.path.join(data_dir, 'metric_tensor.csv'),
        os.path.join(project_dir, 'metric_data_current.csv')
    ]
    
    christoffel_paths = [
        os.path.join(base_dir, 'christoffel_symbols.csv'),
        os.path.join(project_dir, 'christoffel_symbols.csv'),
        os.path.join(data_dir, 'christoffel_symbols.csv'),
        os.path.join(project_dir, 'christoffel_data_current.csv')
    ]
    
    metadata_paths = [
        os.path.join(base_dir, 'metric_metadata.json'),
        os.path.join(project_dir, 'metric_metadata.json'),
        os.path.join(data_dir, 'metric_metadata.json')
    ]
    
    print(f"Starting EXTREME metric tensor animation with {num_steps} steps...")
    print(f"Press Ctrl+C to stop the test")
    
    try:
        for step in range(num_steps):
            print(f"Step {step+1}/{num_steps}")
            
            # Create a dramatically changing metric tensor over time
            # Base metric is Minkowski for spacetime (diag(-1,1,1,1))
            metric = np.eye(dim)
            metric[0, 0] = -1.0  # Time component
            
            # Add time-varying components to make the changes visible
            t = step / num_steps
            
            # MORE EXTREME CHANGES: Different oscillation patterns
            # Use multiple frequencies for more complex behavior
            amplitude1 = 1.0 * (math.sin(t * 4 * math.pi) + 1)  # Range 0 to 2, faster oscillation
            amplitude2 = 1.5 * (math.cos(t * 6 * math.pi) + 1)  # Different phase
            amplitude3 = 2.0 * (math.sin(t * 2 * math.pi + math.pi/4) + 1)  # Different phase & amplitude
            
            # Apply extreme deformations
            for i in range(dim):
                for j in range(dim):
                    if i == j:
                        # Make diagonal elements vary dramatically
                        if i == 0:  # Time component
                            metric[i, j] = -1.0 - amplitude1 * 3.0  # Much stronger negative component
                        else:  # Space components
                            # Each spatial dimension behaves differently
                            if i == 1:
                                metric[i, j] = 1.0 + amplitude1 * 1.5  # First dimension
                            elif i == 2:
                                metric[i, j] = 1.0 + amplitude2 * 2.0  # Second dimension
                            else:
                                metric[i, j] = 1.0 + amplitude3 * 2.5  # Third dimension
                    else:
                        # Add strong off-diagonal elements - make them vary differently
                        phase = (i+j) * t * math.pi + (i*j) * math.pi/4  # Different phase for each component
                        # Stronger coupling between dimensions
                        metric[i, j] = math.sin(phase) * amplitude2 * 0.8
            
            # Ensure the metric is symmetric
            for i in range(dim):
                for j in range(i+1, dim):
                    metric[j, i] = metric[i, j]
            
            # Additional oscillating wave pattern to make variations more obvious
            wave_factor = math.sin(t * 8 * math.pi)  # Fast oscillation
            for i in range(dim):
                for j in range(dim):
                    metric[i, j] += wave_factor * 0.5 * (1 if i == j else 0.3)
            
            # Compute Christoffel symbols
            christoffel = compute_christoffel(metric)
            
            # Simulate a decreasing loss
            loss = 1.0 * (1.0 - t) + 0.1
            
            # Update files with the new tensors - update all locations
            for path in metric_paths:
                save_metric_to_csv(metric, path)
            
            for path in christoffel_paths:
                save_christoffel_to_csv(christoffel, path)
            
            for path in metadata_paths:
                save_metadata(step + 1, num_steps, step // 5 + 1, 10, loss, path)
            
            # Force file changes to be detected
            for path in metric_paths + christoffel_paths + metadata_paths:
                if os.path.exists(path):
                    os.system(f"touch {path}")
            
            # Print a visual representation of the metric to see changes
            print("Current metric tensor:")
            for row in metric:
                print("[" + " ".join(f"{val:6.2f}" for val in row) + "]")
            
            print(f"Generated EXTREME metric tensor with amplitudes {amplitude1:.2f}, {amplitude2:.2f}, {amplitude3:.2f}")
            print("-----")
            
            # Sleep to allow visualization time to update
            time.sleep(delay)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    print("Test complete!")

if __name__ == "__main__":
    main() 