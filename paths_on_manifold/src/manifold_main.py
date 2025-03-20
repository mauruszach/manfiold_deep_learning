#!/usr/bin/env python
"""
Manifold Learning Demo Script

This script demonstrates the usage of the manifold_learning package
for learning and visualizing manifold structures in data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import logging
import sys
from typing import Tuple, Dict, List, Optional

# Add parent directory to path so we can import the package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifold_learning import (
    ManifoldModel,
    ManifoldEmbedding,
    RicciCurvature,
    GeodesicPath,
    GeodesicLoss,
    CovariantDescent,
    ManifoldVisualizer
)
from manifold_learning.train import Trainer, create_dataloaders
from manifold_learning.ricci_flow import RicciFlow

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("manifold_demo.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_synthetic_manifold_data(
    n_samples: int = 1000,
    n_features: int = 20,
    manifold_dim: int = 2,
    noise: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data that lies on a lower-dimensional manifold.
    
    Args:
        n_samples: Number of data points to generate
        n_features: Dimensionality of the ambient space
        manifold_dim: Intrinsic dimensionality of the manifold
        noise: Amount of Gaussian noise to add
        
    Returns:
        Tuple of (features, targets)
    """
    # Generate points on a lower-dimensional manifold (e.g., a 2D surface)
    t = torch.linspace(0, 1, n_samples)
    
    if manifold_dim == 1:
        # 1D manifold: a curve in high-dimensional space
        X_manifold = torch.sin(2 * np.pi * t.unsqueeze(1) * torch.linspace(1, 3, n_features))
        
    elif manifold_dim == 2:
        # 2D manifold: a Swiss roll or torus-like structure
        t1 = torch.linspace(0, 4*np.pi, int(np.sqrt(n_samples)))
        t2 = torch.linspace(0, 2*np.pi, int(np.sqrt(n_samples)))
        t1, t2 = torch.meshgrid(t1, t2)
        t1, t2 = t1.flatten(), t2.flatten()
        
        # Keep only n_samples points
        indices = torch.randperm(t1.shape[0])[:n_samples]
        t1, t2 = t1[indices], t2[indices]
        
        # Create manifold in high-dimensional space
        X_manifold = torch.zeros((n_samples, n_features))
        for i in range(n_features):
            # Each feature combines t1 and t2 differently
            X_manifold[:, i] = torch.sin(t1 * (i % 5 + 1) / 5) * torch.cos(t2 * (i % 7 + 1) / 7)
    else:
        # Higher-dimensional manifolds
        raise NotImplementedError("Manifolds with dimension > 2 not yet implemented in this demo")
    
    # Add Gaussian noise to the data
    X = X_manifold + noise * torch.randn_like(X_manifold)
    
    # Normalize the data
    X = (X - X.mean(0)) / X.std(0)
    
    # For this demo, we'll create an autoencoder-like task
    y = X_manifold  # Try to recover the true manifold points
    
    return X, y

def manifold_embedding_demo(args):
    """
    Demonstrate manifold embedding with the package.
    """
    logger.info("Starting manifold embedding demonstration")
    
    # Generate synthetic data
    X, y = generate_synthetic_manifold_data(
        n_samples=args.n_samples,
        n_features=args.input_dim,
        manifold_dim=args.manifold_dim,
        noise=args.noise
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y, batch_size=args.batch_size,
        train_ratio=0.8, val_ratio=0.1
    )
    
    # Create model
    model = ManifoldModel(
        input_dim=args.input_dim,
        output_dim=args.input_dim,  # Autoencoder-style
        embedding_dim=args.embedding_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.2
    )
    
    # Initialize output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.learning_rate,
        output_dir=args.output_dir,
        visualize=True
    )
    
    # Train the model
    history = trainer.train(
        num_epochs=args.epochs,
        patience=10,
        visualize_every=5
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Visualize geodesic paths between points
    visualize_geodesics(model, X, args.output_dir)
    
    # Visualize learned metric and curvature
    visualize_metric_curvature(model, args.output_dir)
    
def visualize_geodesics(model, data, output_dir):
    """
    Visualize geodesic paths between random points in the dataset.
    """
    logger.info("Visualizing geodesic paths")
    
    # Select random pairs of points
    n_paths = 5
    indices = torch.randperm(len(data))[:n_paths*2].reshape(-1, 2)
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "geodesics")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = ManifoldVisualizer()
    
    # Compute and visualize geodesic paths
    paths = []
    for i in range(n_paths):
        x_start = data[indices[i, 0]].unsqueeze(0)
        x_end = data[indices[i, 1]].unsqueeze(0)
        
        # Compute geodesic path
        path = model.compute_geodesic_path(x_start, x_end)
        paths.append(path)
    
    # If embedding dimension is 2 or 3, visualize the paths
    if model.embedding_dim <= 3:
        visualizer.plot_geodesic_paths(
            paths,
            labels=[f"Path {i+1}" for i in range(n_paths)],
            title="Geodesic Paths on Learned Manifold"
        )
        visualizer.save_figure(os.path.join(vis_dir, "geodesic_paths.png"))
        visualizer.close_figure()
    
def visualize_metric_curvature(model, output_dir):
    """
    Visualize the learned metric tensor and Ricci curvature.
    """
    logger.info("Visualizing metric tensor and curvature")
    
    # Create visualization directory
    vis_dir = os.path.join(output_dir, "geometry")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get the metric tensor
    metric = model.get_metric_tensor()
    
    # Compute Ricci curvature
    curvature = model.compute_curvature()
    ricci_tensor = curvature.ricci_tensor
    scalar_curvature = curvature.scalar_curvature
    
    # Initialize visualizer
    visualizer = ManifoldVisualizer()
    
    # Visualize metric tensor
    visualizer.plot_metric_tensor_heatmap(
        metric, title="Learned Metric Tensor"
    )
    visualizer.save_figure(os.path.join(vis_dir, "metric_tensor.png"))
    visualizer.close_figure()
    
    # Visualize Ricci curvature tensor
    visualizer.plot_curvature_heatmap(
        ricci_tensor, title="Ricci Curvature Tensor"
    )
    visualizer.save_figure(os.path.join(vis_dir, "ricci_tensor.png"))
    visualizer.close_figure()
    
    # Log scalar curvature
    logger.info(f"Scalar curvature: {scalar_curvature.item():.4f}")
    
def ricci_flow_demo(args):
    """
    Demonstrate Ricci flow for evolving a manifold's metric.
    """
    logger.info("Starting Ricci flow demonstration")
    
    # Create an initial metric tensor
    # For demonstration, we'll start with a perturbed version of a flat metric
    dim = args.embedding_dim
    initial_metric = torch.eye(dim)
    
    # Add some perturbation
    perturbation = 0.2 * torch.randn(dim, dim)
    perturbation = 0.5 * (perturbation + perturbation.T)  # Make symmetric
    initial_metric = initial_metric + perturbation
    
    # Ensure the metric is positive definite
    min_eigenvalue = 0.1
    eigenvalues, eigenvectors = torch.linalg.eigh(initial_metric)
    adjusted_eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
    initial_metric = eigenvectors @ torch.diag(adjusted_eigenvalues) @ eigenvectors.T
    
    # Create Ricci flow instance
    flow = RicciFlow(
        initial_metric=initial_metric,
        flow_rate=args.flow_rate,
        normalization='volume'
    )
    
    # Create output directory
    flow_dir = os.path.join(args.output_dir, "ricci_flow")
    os.makedirs(flow_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = ManifoldVisualizer()
    
    # Define callback for visualization
    def flow_callback(step, info):
        if (step + 1) % 10 == 0:
            # Visualize current metric
            visualizer.plot_metric_tensor_heatmap(
                info['metric'], 
                title=f"Metric Tensor (Step {step+1})"
            )
            visualizer.save_figure(os.path.join(flow_dir, f"metric_step{step+1}.png"))
            visualizer.close_figure()
            
            # Visualize Ricci tensor
            visualizer.plot_curvature_heatmap(
                info['ricci_tensor'], 
                title=f"Ricci Tensor (Step {step+1})"
            )
            visualizer.save_figure(os.path.join(flow_dir, f"ricci_step{step+1}.png"))
            visualizer.close_figure()
    
    # Evolve the metric
    results = flow.evolve(
        steps=args.flow_steps,
        callback=flow_callback,
        save_interval=10,
        save_path=os.path.join(flow_dir, "flow")
    )
    
    # Visualize evolution of scalar curvature
    curvature_values = [float(c) for c in results['curvatures']]
    plt.figure(figsize=(10, 6))
    plt.plot(curvature_values)
    plt.xlabel('Step')
    plt.ylabel('Scalar Curvature')
    plt.title('Evolution of Scalar Curvature Under Ricci Flow')
    plt.grid(True)
    plt.savefig(os.path.join(flow_dir, "scalar_curvature_evolution.png"))
    plt.close()
    
    logger.info("Ricci flow demonstration completed")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Manifold Learning Demonstration")
    
    # General arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./output',
                      help='Directory to save outputs')
    
    # Demo selection
    parser.add_argument('--demo', type=str, choices=['embedding', 'ricci-flow', 'all'],
                      default='all', help='Which demo to run')
    
    # Data generation arguments
    parser.add_argument('--n-samples', type=int, default=1000,
                      help='Number of data points to generate')
    parser.add_argument('--input-dim', type=int, default=20,
                      help='Ambient space dimensionality')
    parser.add_argument('--manifold-dim', type=int, default=2,
                      help='Intrinsic manifold dimensionality')
    parser.add_argument('--noise', type=float, default=0.1,
                      help='Noise level for synthetic data')
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=3,
                      help='Dimensionality of the manifold embedding')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    
    # Ricci flow arguments
    parser.add_argument('--flow-steps', type=int, default=100,
                      help='Number of Ricci flow steps')
    parser.add_argument('--flow-rate', type=float, default=0.01,
                      help='Flow rate for Ricci flow')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Run demos based on user selection
    if args.demo == 'embedding' or args.demo == 'all':
        manifold_embedding_demo(args)
        
    if args.demo == 'ricci-flow' or args.demo == 'all':
        ricci_flow_demo(args)
        
    logger.info("Manifold learning demonstrations completed") 