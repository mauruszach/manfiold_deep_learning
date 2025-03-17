import numpy as np
import time
import json
import os
import math
import wandb  # Import Weights & Biases
import matplotlib.pyplot as plt

# Simple test script to verify the file-based communication works correctly
# This simulates a training process by generating a series of evolving metrics

def save_metric_to_csv(metric, filename='./metric_data_current.csv'):
    """Save a metric tensor to CSV file"""
    np.savetxt(filename, metric, delimiter=',')
    print(f"Saved metric to {filename}")

def save_christoffel_to_csv(christoffel, filename='./christoffel_data_current.csv'):
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

def save_metadata(epoch, total_epochs, step, total_steps, loss, is_training=True, filename='./metric_metadata.json'):
    """Save metadata to JSON file"""
    metadata = {
        'epoch': epoch,
        'total_epochs': total_epochs,
        'step': step,
        'total_steps': total_steps,
        'loss': loss,
        'is_training': is_training
    }
    
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {filename}")

def calculate_christoffel(metric):
    """Calculate Christoffel symbols from metric tensor (simplified)"""
    dim = metric.shape[0]
    christoffel = np.zeros((dim, dim, dim))
    
    # Calculate inverse metric
    try:
        metric_inv = np.linalg.inv(metric)
    except np.linalg.LinAlgError:
        # Add small regularization if metric is not invertible
        regularized_metric = metric + np.eye(dim) * 1e-6
        metric_inv = np.linalg.inv(regularized_metric)
    
    # Simple approximation of Christoffel symbols
    # In a real implementation, we would calculate derivatives of the metric
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    # Simplified calculation (not physically accurate)
                    christoffel[i, j, k] += 0.5 * metric_inv[i, l] * (
                        0.1 * metric[j, l] * (j == k) +  # Approximate derivative
                        0.1 * metric[k, l] * (k == j) -
                        0.1 * metric[j, k] * (j == l or k == l)
                    )
    
    return christoffel

def generate_evolving_metric(epoch, total_epochs, dim=4):
    """Generate an evolving metric tensor that transitions from flat to curved with a smooth field-like appearance"""
    # Start with Minkowski metric
    metric = np.eye(dim)
    if dim > 0:
        metric[0, 0] = -1.0  # Time component
    
    # Calculate progress (0 to 1) - use a smoother sigmoid function for transition
    raw_progress = epoch / total_epochs
    # Apply sigmoid to make transitions smoother at the beginning and end
    progress = 1.0 / (1.0 + np.exp(-10 * (raw_progress - 0.5)))
    
    # Create a smooth field-like metric that varies continuously in space
    # Use a combination of smooth functions rather than abrupt changes
    
    # Base curvature strength with smooth progression
    curvature_strength = progress * 3.0
    
    # Create a smooth field-like metric
    if dim >= 4:
        # Create a smooth gravitational well
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    # Diagonal elements vary smoothly
                    if i == 0:  # Time component
                        # Smooth time dilation effect
                        metric[i, j] = -1.0 - curvature_strength * 0.5 * np.sin(np.pi * progress)
                    else:  # Space components
                        # Smooth space contraction/expansion
                        phase_shift = i * np.pi / dim  # Different phase for each dimension
                        metric[i, j] = 1.0 + curvature_strength * 0.3 * np.sin(np.pi * progress + phase_shift)
                else:
                    # Off-diagonal elements represent smooth mixing of dimensions
                    # Use smooth sinusoidal functions with phase differences
                    angle = np.pi * progress + (i * j) * np.pi / (dim * 2)
                    # Gradually increase the coupling between dimensions
                    coupling = curvature_strength * 0.2 * np.sin(angle)
                    metric[i, j] = coupling
    
    # Ensure the metric remains valid by enforcing symmetry
    metric = 0.5 * (metric + metric.T)
    
    # Apply a smoothing filter to the entire metric
    # This creates a more field-like appearance by removing sharp transitions
    smoothed_metric = np.zeros_like(metric)
    for i in range(dim):
        for j in range(dim):
            # Simple smoothing using neighboring elements
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < dim and 0 <= nj < dim:
                        neighbors.append(metric[ni, nj])
            smoothed_metric[i, j] = sum(neighbors) / len(neighbors)
    
    # Blend between original and smoothed metric for even smoother transition
    blend_factor = 0.7  # 70% smoothed, 30% original
    metric = (1 - blend_factor) * metric + blend_factor * smoothed_metric
    
    # Ensure the metric maintains proper signature
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    
    # Modify eigenvalues to maintain signature with smooth scaling
    if dim > 0:
        eigenvalues[0] = -abs(eigenvalues[0])  # Keep time component negative
    for i in range(1, dim):
        eigenvalues[i] = abs(eigenvalues[i])  # Keep space components positive
    
    # Reconstruct metric
    metric = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return metric

def main():
    # Simulation parameters
    dim = 4  # 4D spacetime
    total_epochs = 30  # Increased for smoother evolution
    steps_per_epoch = 8  # Increased for more granular updates
    total_steps = total_epochs * steps_per_epoch
    update_interval = 0.2  # seconds between updates - faster updates for smoother animation
    
    # Initialize Weights & Biases
    wandb.init(
        project="manifold-deep-learning",  # Project name
        config={
            "dimensions": dim,
            "total_epochs": total_epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "update_interval": update_interval,
        },
        name="gravitational-field-simulation",  # Run name
        tags=["spacetime", "metric-evolution", "geodesics"]  # Tags for filtering
    )
    
    # Create custom W&B plots for the metric tensor
    wandb.define_metric("epoch")
    wandb.define_metric("step")
    wandb.define_metric("loss", step_metric="step")
    wandb.define_metric("curvature_strength", step_metric="step")
    wandb.define_metric("metric_determinant", step_metric="step")
    
    print(f"Starting visualization test with {total_epochs} epochs, {steps_per_epoch} steps per epoch")
    print(f"Updates will be sent every {update_interval} seconds")
    print("Press Ctrl+C to stop the test")
    
    try:
        for epoch in range(total_epochs):
            for step in range(steps_per_epoch):
                # Calculate global step
                global_step = epoch * steps_per_epoch + step
                
                # Generate evolving metric
                metric = generate_evolving_metric(epoch, total_epochs, dim)
                
                # Calculate Christoffel symbols
                christoffel = calculate_christoffel(metric)
                
                # Calculate fake loss that decreases over time
                loss = 1.0 - 0.8 * (global_step / total_steps)
                loss = max(0.1, loss + 0.05 * np.random.randn())  # Add some noise
                
                # Calculate some additional metrics for monitoring
                metric_det = np.linalg.det(metric)
                eigenvalues = np.linalg.eigvalsh(metric)
                curvature_str = (epoch / total_epochs) * 3.0  # Same calculation as in generate_evolving_metric
                
                # Save data for visualization in files
                save_metric_to_csv(metric)
                save_christoffel_to_csv(christoffel)
                save_metadata(epoch, total_epochs, global_step, total_steps, loss)
                
                # Log metrics to W&B
                wandb_log_data = {
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss,
                    "curvature_strength": curvature_str,
                    "metric_determinant": metric_det,
                    "min_eigenvalue": min(eigenvalues),
                    "max_eigenvalue": max(eigenvalues),
                    "metric_trace": np.trace(metric),
                }
                
                # Add individual metric components to the log
                for i in range(dim):
                    for j in range(dim):
                        wandb_log_data[f"metric_{i}_{j}"] = metric[i, j]
                
                # Create a matplotlib figure for the heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(metric, cmap='viridis')
                ax.set_title(f'Metric Tensor (Epoch {epoch}, Step {step})')
                plt.colorbar(im)
                plt.tight_layout()
                
                # Log directly to wandb without saving temporary file
                wandb_log_data["metric_heatmap"] = wandb.Image(plt)
                
                # Clean up
                plt.close(fig)
                
                # Log data to W&B
                wandb.log(wandb_log_data)
                
                print(f"Epoch {epoch}/{total_epochs}, Step {global_step}/{total_steps}, Loss: {loss:.4f}")
                
                # Wait before next update
                time.sleep(update_interval)
        
        # Final update with is_training=False
        metric = generate_evolving_metric(total_epochs, total_epochs, dim)
        christoffel = calculate_christoffel(metric)
        save_metric_to_csv(metric)
        save_christoffel_to_csv(christoffel)
        save_metadata(total_epochs, total_epochs, total_steps, total_steps, 0.1, is_training=False)
        
        # Log final summary metrics
        wandb.run.summary["final_loss"] = 0.1
        wandb.run.summary["final_metric_determinant"] = np.linalg.det(metric)
        wandb.run.summary["training_completed"] = True
        
        print("Test completed successfully!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        wandb.run.summary["training_completed"] = False
    except Exception as e:
        print(f"Error during test: {e}")
        wandb.run.summary["training_completed"] = False
    finally:
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main() 