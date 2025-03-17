import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import json
from path_analysis import ManifoldDataset, Model, CovariantDescent, GeodesicLoss, RicciFlow, visualize_geodesics, RicciCurvature
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_synthetic_manifold_dataset(num_samples=100, noise_level=0.1):
    """
    Create a synthetic dataset with a known manifold structure
    
    This creates data on a 2D manifold embedded in higher dimensions
    """
    # Create a 2D grid of points (the manifold)
    u = np.linspace(-1, 1, int(np.sqrt(num_samples)))
    v = np.linspace(-1, 1, int(np.sqrt(num_samples)))
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()
    
    # Take only the first num_samples points
    u = u[:num_samples]
    v = v[:num_samples]
    
    # Create a mapping from the 2D manifold to a higher-dimensional space
    # Here we'll use a simple quadratic embedding
    x = u
    y = v
    z = u**2 + v**2
    
    # Stack to create 3D points
    points_3d = np.stack([x, y, z], axis=1)
    
    # Convert to torch tensors
    points_3d_tensor = torch.tensor(points_3d, dtype=torch.float32)
    
    # Create input data by adding noise and increasing dimensionality
    input_dim = 10  # Higher dimensional input
    embedding_dim = 3  # 3D embedding
    
    # Add noise to the 3D points
    noisy_points = points_3d_tensor + torch.randn_like(points_3d_tensor) * noise_level
    
    # Pad with random features to create higher-dimensional input
    random_features = torch.randn(num_samples, input_dim - 3) * 0.1
    input_data = torch.cat([noisy_points, random_features], dim=1)
    
    # The target is the clean 3D points
    target_data = points_3d_tensor
    
    return input_data, target_data, input_dim, embedding_dim

def visualize_spacetime_curvature(model, save_dir):
    """
    Visualize the learned metric tensor and its curvature properties
    to demonstrate the curved spacetime characteristics
    """
    logger.info("Visualizing spacetime curvature...")
    
    # Get the learned metric tensor
    metric_tensor = model.embedding.get_metric().detach().cpu().numpy()
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Visualize the metric tensor as a heatmap
    ax1 = fig.add_subplot(221)
    im = ax1.imshow(metric_tensor, cmap='coolwarm')
    ax1.set_title('Learned Metric Tensor')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax1)
    
    # 2. Compute and visualize the Ricci curvature
    ricci_curvature = RicciCurvature(model.embedding.get_metric())
    ricci_tensor = ricci_curvature.ricci_tensor.detach().cpu().numpy()
    
    ax2 = fig.add_subplot(222)
    im = ax2.imshow(ricci_tensor, cmap='coolwarm')
    ax2.set_title('Ricci Curvature Tensor')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Dimension')
    plt.colorbar(im, ax=ax2)
    
    # 3. Visualize scalar curvature
    scalar_curvature = ricci_curvature.scalar_curvature.item()
    ax3 = fig.add_subplot(223)
    ax3.text(0.5, 0.5, f'Scalar Curvature: {scalar_curvature:.6f}', 
             horizontalalignment='center', verticalalignment='center',
             fontsize=14)
    ax3.axis('off')
    
    # 4. Create a 2D grid and visualize the curvature effect
    # We'll create a 2D grid and warp it based on the metric
    ax4 = fig.add_subplot(224, projection='3d')
    
    # Create a 2D grid
    u = np.linspace(-1, 1, 20)
    v = np.linspace(-1, 1, 20)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Compute the warping based on the metric (simplified)
    # In a real spacetime, this would be based on solving geodesic equations
    # Here we'll use a simplified approach to visualize the effect
    
    # Convert to flat coordinates
    points = np.stack([u_grid.flatten(), v_grid.flatten()], axis=1)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Use the model to embed these points
    with torch.no_grad():
        # Create a batch of inputs (pad with zeros to match input_dim)
        input_dim = model.embedding.input_dim
        padded_points = torch.zeros((points_tensor.shape[0], input_dim))
        padded_points[:, :2] = points_tensor
        
        # Embed the points
        embedded_points = model(padded_points)
        
        # Extract the first 3 dimensions for visualization
        embedded_points = embedded_points.cpu().numpy()
    
    # Reshape back to grid
    x = embedded_points[:, 0].reshape(u_grid.shape)
    y = embedded_points[:, 1].reshape(v_grid.shape)
    z = embedded_points[:, 2].reshape(v_grid.shape) if embedded_points.shape[1] > 2 else np.zeros_like(u_grid)
    
    # Plot the warped grid
    surf = ax4.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax4.set_title('Warped Grid (Spacetime Curvature)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/visualizations/spacetime_curvature.png")
    plt.close()
    
    logger.info(f"Spacetime curvature visualization saved to {save_dir}/visualizations/spacetime_curvature.png")

def visualize_particle_in_spacetime(model, save_dir):
    """
    Create an animation of a particle moving along a geodesic in the curved spacetime
    """
    logger.info("Creating particle motion animation in curved spacetime...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Create a figure for the animation
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get the metric tensor and compute Christoffel symbols
    metric = model.embedding.get_metric().cpu()
    ricci_curvature = RicciCurvature(metric)
    christoffel = ricci_curvature.christoffel_symbols
    
    # Create a 2D grid to represent the spacetime
    u = np.linspace(-1, 1, 20)
    v = np.linspace(-1, 1, 20)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Convert to flat coordinates
    points = np.stack([u_grid.flatten(), v_grid.flatten()], axis=1)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Use the model to embed these points
    with torch.no_grad():
        # Create a batch of inputs (pad with zeros to match input_dim)
        input_dim = model.embedding.input_dim
        padded_points = torch.zeros((points_tensor.shape[0], input_dim))
        padded_points[:, :2] = points_tensor
        
        # Embed the points
        embedded_points = model(padded_points)
        
        # Extract the first 3 dimensions for visualization
        embedded_points = embedded_points.cpu().numpy()
    
    # Reshape back to grid
    x = embedded_points[:, 0].reshape(u_grid.shape)
    y = embedded_points[:, 1].reshape(v_grid.shape)
    z = embedded_points[:, 2].reshape(v_grid.shape) if embedded_points.shape[1] > 2 else np.zeros_like(u_grid)
    
    # Plot the warped grid (spacetime)
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.7)
    
    # Define start and end points for the particle
    start_point = torch.tensor([[-0.8, -0.8, 0.0]], dtype=torch.float32)
    end_point = torch.tensor([[0.8, 0.8, 0.0]], dtype=torch.float32)
    
    # Pad with zeros to match input dimension
    padded_start = torch.zeros((1, input_dim))
    padded_start[:, :start_point.shape[1]] = start_point
    
    padded_end = torch.zeros((1, input_dim))
    padded_end[:, :end_point.shape[1]] = end_point
    
    # Embed start and end points
    with torch.no_grad():
        start_embedded = model(padded_start).cpu()
        end_embedded = model(padded_end).cpu()
    
    # Compute geodesic path with more steps for smoother animation
    num_steps = 50
    
    # Initial straight line path
    current_point = start_embedded.clone()
    velocity = (end_embedded - start_embedded) / num_steps
    
    # Normalize velocity
    velocity_norm = torch.norm(velocity)
    if velocity_norm > 0:
        velocity = velocity / velocity_norm
    
    # Simulate geodesic
    geodesic_path = [current_point.clone()]
    
    for step in range(num_steps-1):
        # Compute acceleration using Christoffel symbols
        acceleration = torch.zeros_like(velocity)
        for i in range(velocity.shape[1]):
            for j in range(velocity.shape[1]):
                for k in range(velocity.shape[1]):
                    # Ensure all indices are within bounds
                    if (i < christoffel.shape[0] and j < christoffel.shape[1] and 
                        k < christoffel.shape[2] and j < velocity.shape[1] and 
                        k < velocity.shape[1]):
                        acceleration[0, i] = acceleration[0, i] - christoffel[i, j, k] * velocity[0, j] * velocity[0, k]
        
        # Update velocity and position (avoid in-place operations)
        velocity = velocity + acceleration * (1.0 / num_steps)
        current_point = current_point + velocity * (1.0 / num_steps)
        geodesic_path.append(current_point.clone())
    
    # Convert path to numpy for plotting
    path = torch.cat(geodesic_path, dim=0).numpy()
    
    # Plot the full geodesic path
    ax.plot(path[:, 0], path[:, 1], path[:, 2] if path.shape[1] > 2 else np.zeros_like(path[:, 0]), 
           color='blue', linewidth=2, alpha=0.5)
    
    # Plot start and end points
    ax.scatter(start_embedded[0, 0], start_embedded[0, 1], 
              start_embedded[0, 2] if start_embedded.shape[1] > 2 else 0, 
              color='green', marker='o', s=100, label='Start')
    ax.scatter(end_embedded[0, 0], end_embedded[0, 1], 
              end_embedded[0, 2] if end_embedded.shape[1] > 2 else 0, 
              color='red', marker='x', s=100, label='End')
    
    # Create a point object for the particle
    particle, = ax.plot([], [], [], 'ko', markersize=10)
    
    # Add a time indicator
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Particle Moving Along a Geodesic in Curved Spacetime')
    ax.legend()
    
    # Set consistent view limits
    ax.set_xlim([np.min(path[:, 0])-0.1, np.max(path[:, 0])+0.1])
    ax.set_ylim([np.min(path[:, 1])-0.1, np.max(path[:, 1])+0.1])
    if path.shape[1] > 2:
        ax.set_zlim([np.min(path[:, 2])-0.1, np.max(path[:, 2])+0.1])
    else:
        ax.set_zlim([-0.1, 0.1])
    
    # Animation update function
    def update(frame):
        # Update particle position
        if frame < len(path):
            # Fix the deprecation warning by passing sequences to set_data
            particle.set_data([path[frame, 0]], [path[frame, 1]])
            particle.set_3d_properties([path[frame, 2]] if path.shape[1] > 2 else [0])
            time_text.set_text(f'Time: {frame/(len(path)-1):.2f}')
        return particle, time_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(path), interval=100, blit=True)
    
    # Save animation
    animation_path = f"{save_dir}/visualizations/particle_motion.mp4"
    ani.save(animation_path, writer='ffmpeg', fps=10)
    
    # Also save a static image of the final frame
    static_path = f"{save_dir}/visualizations/particle_trajectory.png"
    plt.savefig(static_path)
    plt.close()
    
    logger.info(f"Particle motion animation saved to {animation_path}")
    logger.info(f"Static trajectory image saved to {static_path}")

def export_metric_for_cpp_simulation(model, save_dir):
    """
    Export the learned metric tensor and other relevant data to a format
    that can be read by the C++ simulation.
    
    This creates a JSON file with the metric tensor, Christoffel symbols,
    and other data needed for the C++ simulation to visualize the curved spacetime.
    """
    logger.info("Exporting metric tensor for C++ simulation...")
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get the metric tensor
    metric_tensor = model.embedding.get_metric().detach().cpu().numpy().tolist()
    
    # Compute Ricci curvature and related quantities
    ricci_curvature = RicciCurvature(model.embedding.get_metric())
    christoffel_symbols = ricci_curvature.christoffel_symbols.detach().cpu().numpy().tolist()
    ricci_tensor = ricci_curvature.ricci_tensor.detach().cpu().numpy().tolist()
    scalar_curvature = float(ricci_curvature.scalar_curvature.item())
    
    # Create a grid of points to visualize the embedding
    grid_size = 20
    u = np.linspace(-1, 1, grid_size)
    v = np.linspace(-1, 1, grid_size)
    u_grid, v_grid = np.meshgrid(u, v)
    
    # Convert to flat coordinates
    points = np.stack([u_grid.flatten(), v_grid.flatten()], axis=1)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    
    # Use the model to embed these points
    embedded_points = []
    with torch.no_grad():
        # Create a batch of inputs (pad with zeros to match input_dim)
        input_dim = model.embedding.input_dim
        padded_points = torch.zeros((points_tensor.shape[0], input_dim))
        padded_points[:, :2] = points_tensor
        
        # Embed the points
        embedded = model(padded_points)
        
        # Extract the first 3 dimensions for visualization
        embedded_np = embedded.cpu().numpy()
        
        # Store as list of [x, y, z] points
        for i in range(embedded_np.shape[0]):
            point = embedded_np[i].tolist()
            # Ensure we have at least 3 dimensions for visualization
            while len(point) < 3:
                point.append(0.0)
            embedded_points.append(point[:3])  # Keep only first 3 dimensions
    
    # Create a dictionary with all the data
    export_data = {
        "metric_tensor": metric_tensor,
        "christoffel_symbols": christoffel_symbols,
        "ricci_tensor": ricci_tensor,
        "scalar_curvature": scalar_curvature,
        "grid_size": grid_size,
        "embedded_points": embedded_points,
        "grid_points": points.tolist(),
        "embedding_dim": model.embedding.embedding_dim,
        "is_spacetime": True  # Indicate this is a spacetime metric
    }
    
    # Save to JSON file
    export_path = f"{save_dir}/metric_for_cpp.json"
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Metric tensor and related data exported to {export_path}")
    
    return export_path

def test_run(save_dir='./test_results'):
    """Run the model for one epoch on a synthetic dataset"""
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/visualizations", exist_ok=True)
    
    # Create synthetic dataset
    logger.info("Creating synthetic dataset...")
    input_data, target_data, input_dim, embedding_dim = create_synthetic_manifold_dataset(num_samples=100)
    
    # Create a simple dataset class
    class SyntheticDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets
            
        def __len__(self):
            return len(self.inputs)
            
        def __getitem__(self, idx):
            return self.inputs[idx], self.targets[idx]
    
    dataset = SyntheticDataset(input_data, target_data)
    
    # Visualize the dataset
    logger.info("Visualizing dataset...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    x = target_data[:, 0].numpy()
    y = target_data[:, 1].numpy()
    z = target_data[:, 2].numpy()
    
    ax.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=20, alpha=0.8)
    
    ax.set_title('Synthetic Manifold Dataset')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/visualizations/dataset.png")
    plt.close()
    
    # Create dataloader
    batch_size = 16
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model and optimizer
    logger.info("Creating model...")
    model = Model(input_dim, embedding_dim, device=device)
    
    # Important: Set the model's metric tensor to have spacetime signature
    # This ensures we're learning a spacetime metric
    with torch.no_grad():
        metric = model.embedding.get_metric()
        # Set the first component (time) to be negative
        metric[0, 0] = -1.0
        # Set the spatial components to be positive
        for i in range(1, embedding_dim):
            metric[i, i] = 1.0
    
    optimizer = CovariantDescent(
        model.parameters(), 
        lr=0.01,
        weight_decay=1e-5,
        curvature_func=lambda p: 0.01 * torch.eye(p.shape[0] if len(p.shape) > 0 else 1, device=device)
    )
    
    criterion = GeodesicLoss(num_steps=5)  # Use fewer steps for testing
    
    # Create Ricci flow optimizer with spacetime flag set to True
    ricci_flow = RicciFlow(
        model, 
        optimizer, 
        criterion,
        num_iterations=2,  # Use fewer iterations for testing
        checkpoint_dir=f"{save_dir}/checkpoints",
        device=device,
        is_spacetime=True  # Explicitly set to learn a spacetime metric
    )
    
    # Run for one epoch
    logger.info("Running for one epoch...")
    model.train()
    total_loss = 0
    global_step = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move data to device
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Perform Ricci flow step
        try:
            loss = ricci_flow.flow_step(inputs, targets, global_step)
            total_loss += loss
            
            logger.info(f"Batch {batch_idx+1}/{len(dataloader)}, Loss: {loss:.6f}")
            
            global_step += 1
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Compute average loss
    avg_loss = total_loss / len(dataloader)
    logger.info(f"Epoch completed, Average Loss: {avg_loss:.6f}")
    
    # Visualize metric evolution
    logger.info("Visualizing metric evolution...")
    ricci_flow.visualize_metric_evolution(save_path=f"{save_dir}/visualizations/metric_evolution.png")
    
    # Visualize geodesics
    logger.info("Visualizing geodesics...")
    try:
        visualize_geodesics(
            model, 
            dataset, 
            num_geodesics=3,
            num_steps=10,
            save_path=f"{save_dir}/visualizations/geodesics.png"
        )
    except Exception as e:
        logger.error(f"Error visualizing geodesics: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Visualize spacetime curvature
    visualize_spacetime_curvature(model, save_dir)
    
    # Add visualization of particle motion
    try:
        visualize_particle_in_spacetime(model, save_dir)
    except Exception as e:
        logger.error(f"Error visualizing particle motion: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Export metric tensor for C++ simulation
    try:
        export_path = export_metric_for_cpp_simulation(model, save_dir)
        logger.info(f"Metric data exported for C++ simulation to {export_path}")
    except Exception as e:
        logger.error(f"Error exporting metric for C++ simulation: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Save model
    logger.info("Saving model...")
    torch.save(model.state_dict(), f"{save_dir}/model.pt")
    
    logger.info(f"Test run completed. Results saved to {save_dir}")
    
    return model, ricci_flow

if __name__ == "__main__":
    test_run() 