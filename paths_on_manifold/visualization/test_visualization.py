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

def save_riemann_to_csv(riemann, filename='./riemann_tensor.csv'):
    """Save Riemann tensor to CSV file"""
    with open(filename, 'w') as f:
        dim = riemann.shape[0]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        value = riemann[i, j, k, l]
                        if abs(value) > 1e-10:  # Only save non-zero values
                            f.write(f"{i},{j},{k},{l},{value}\n")
    print(f"Saved Riemann tensor to {filename}")

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
    """Main test function to simulate a training process"""
    # Specify data directories - make data directory path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, 'paths_on_manifold', 'data')
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Set up file paths
    metric_file = os.path.join(data_dir, 'metric_tensor.csv')
    christoffel_file = os.path.join(data_dir, 'christoffel_symbols.csv')
    riemann_file = os.path.join(data_dir, 'riemann_tensor.csv')
    metadata_file = os.path.join(data_dir, 'metric_metadata.json')
    
    # Also save to current directory for backward compatibility
    metric_file_current = 'metric_data_current.csv'
    christoffel_file_current = 'christoffel_data_current.csv'
    riemann_file_current = 'riemann_tensor.csv'
    metadata_file_current = 'metric_metadata.json'
    
    # Clean up old wandb runs if any
    wandb_dir = os.path.join(data_dir, 'wandb')
    if os.path.exists(wandb_dir):
        import shutil
        for subdir in os.listdir(wandb_dir):
            if subdir.startswith('run-'):
                shutil.rmtree(os.path.join(wandb_dir, subdir))
    
    # Initialize Weights & Biases
    run = wandb.init(project="manifold-deep-learning", name="test-visualization")
    
    # Simulate a training process
    total_epochs = 100
    total_steps = 1000
    
    # Create a 5x5 metric tensor that will evolve over time
    dim = 5
    metric = np.eye(dim)
    
    for epoch in range(total_epochs):
        time.sleep(0.1)  # Slow down to avoid CPU overuse
        
        # Create a mock loss value that decreases over time
        loss = 1.0 * (1.0 - epoch / total_epochs) + 0.1 * np.random.random()
        
        # Periodically save metrics data
        if epoch % 5 == 0:
            step = int(epoch * (total_steps / total_epochs))
            
            # Evolve the metric tensor over time (add a perturbation)
            perturbation = np.random.normal(0, 0.05, (dim, dim))
            perturbation = (perturbation + perturbation.T) / 2  # Make it symmetric
            metric = metric + perturbation
            
            # Ensure metric is positive definite by adding to diagonal if needed
            eigenvalues = np.linalg.eigvalsh(metric)
            if np.any(eigenvalues <= 0):
                min_eig = np.min(eigenvalues)
                if min_eig <= 0:
                    metric = metric + np.eye(dim) * (abs(min_eig) + 0.1)
                    
            # Calculate Christoffel symbols
            christoffel = calculate_christoffel(metric)
            
            # Calculate Riemann tensor 
            riemann = calculate_riemann_tensor(metric, christoffel, dim)
            
            # Save data to files
            save_metric_to_csv(metric, metric_file)
            save_christoffel_to_csv(christoffel, christoffel_file)
            save_riemann_to_csv(riemann, riemann_file)
            save_metadata(epoch, total_epochs, step, total_steps, loss, True, metadata_file)
            
            # Also save to current directory for backward compatibility
            save_metric_to_csv(metric, metric_file_current)
            save_christoffel_to_csv(christoffel, christoffel_file_current)
            save_riemann_to_csv(riemann, riemann_file_current)
            save_metadata(epoch, total_epochs, step, total_steps, loss, True, metadata_file_current)
            
            # Log to W&B
            wandb.log({
                "epoch": epoch,
                "step": step,
                "loss": loss,
                "metric_determinant": np.linalg.det(metric),
                "metric_condition_number": np.linalg.cond(metric)
            })
            
            # Visualize the metric tensor
            plt.figure(figsize=(8, 6))
            plt.imshow(metric, cmap='viridis')
            plt.colorbar(label='Value')
            plt.title(f'Metric Tensor (Epoch {epoch})')
            plt.xlabel('j')
            plt.ylabel('i')
            plt.tight_layout()
            
            # Save the figure and log to W&B
            plt.savefig(f'metric_epoch_{epoch}.png')
            wandb.log({"metric_visualization": wandb.Image(f'metric_epoch_{epoch}.png')})
            plt.close()
            
            print(f"Epoch {epoch}/{total_epochs}, Step {step}/{total_steps}, Loss: {loss:.4f}")
            
    wandb.finish()
    print("Test visualization completed.")

def calculate_riemann_tensor(metric, christoffel, dim):
    """Calculate Riemann tensor from metric and Christoffel symbols"""
    riemann = np.zeros((dim, dim, dim, dim))
    
    # For each component of the Riemann tensor
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    # R^a_bcd = ∂_c Γ^a_bd - ∂_d Γ^a_bc + Γ^a_ce Γ^e_bd - Γ^a_de Γ^e_bc
                    
                    # We'll approximate the derivatives (∂_c Γ^a_bd - ∂_d Γ^a_bc) 
                    # with small values to make it interesting
                    derivative_term = np.random.normal(0, 0.01)
                    
                    # Calculate the Christoffel product terms
                    product_term = 0
                    for e in range(dim):
                        product_term += christoffel[a, c, e] * christoffel[e, b, d]
                        product_term -= christoffel[a, d, e] * christoffel[e, b, c]
                    
                    # Combine terms
                    riemann[a, b, c, d] = derivative_term + product_term
    
    return riemann

if __name__ == "__main__":
    main() 