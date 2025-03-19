import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset, DataLoader
from torch.optim.optimizer import Optimizer
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import logging
import math
import os
import threading
import socket
import pickle
import json
import wandb  # Add Weights & Biases import
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("manifold_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add this utility function near the top of the file after imports
def setup_wandb_monitoring(config=None, project_name="manifold-deep-learning", run_name=None, tags=None, model=None):
    """
    Set up Weights & Biases monitoring.
    
    Args:
        config: Dictionary of configuration parameters to log
        project_name: W&B project name
        run_name: Name for this particular run
        tags: List of tags for the run
        model: If provided, wandb will watch this model for gradients and parameters
        
    Returns:
        The initialized wandb run object
    """
    if wandb.run is None:
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            tags=tags
        )
        
        if model is not None:
            wandb.watch(model, log="all", log_freq=10)
            
        logger.info(f"Initialized W&B monitoring: {project_name}/{run_name}")
        return run
    else:
        logger.info("W&B run already active, reusing existing run")
        return wandb.run

# Add this method to log metrics to W&B
def log_metrics_to_wandb(metrics, step=None, prefix=""):
    """
    Log metrics to Weights & Biases.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number for logging
        prefix: Optional prefix to add to metric names
    """
    if wandb.run is not None:
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb.log(metrics, step=step)

class ManifoldEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim, device=None):
        super(ManifoldEmbedding, self).__init__()
        # Define symbols for space-time coordinates
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.t, self.x, self.y, self.z = sp.symbols('t x y z')
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Full coordinate system for proper calculations
        self.coordinate_functions = [self.t, self.x, self.y, self.z][:embedding_dim]
        
        # Initialize the metric tensor as a learnable parameter
        initial_metric = torch.zeros(embedding_dim, embedding_dim, device=self.device)
        # Set diagonal elements to create a space-time like metric
        for i in range(embedding_dim):
            if i == 0:  # Time component
                initial_metric[i, i] = -1.0
            else:  # Space components
                initial_metric[i, i] = 1.0
        
        self.metric = nn.Parameter(initial_metric)
        
        # Neural network to map from input space to embedding space
        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
        self.to(self.device)

    def get_metric(self):
        """Return the current metric tensor"""
        return self.metric

    def christoffel_symbols(self, g, g_inv):
        """ Calculate Christoffel symbols for a given metric tensor, g """
        dim = g.shape[0]
        Gamma = torch.zeros(dim, dim, dim, device=self.device)
        
        # Convert symbolic metric to a function
        g_sym = sp.Matrix([[sp.symbols(f'g_{i}_{j}') for j in range(dim)] for i in range(dim)])
        g_inv_sym = sp.Matrix([[sp.symbols(f'g_inv_{i}_{j}') for j in range(dim)] for i in range(dim)])
        
        # Create symbolic Christoffel symbols
        Gamma_sym = [[[0 for _ in range(dim)] for _ in range(dim)] for _ in range(dim)]
        coords = self.coordinate_functions
        
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    sum_term = 0
                    for l in range(dim):
                        term = 0.5 * g_inv_sym[k, l] * (
                            sp.diff(g_sym[j, l], coords[i]) +
                            sp.diff(g_sym[i, l], coords[j]) -
                            sp.diff(g_sym[i, j], coords[l])
                        )
                        sum_term += term
                    Gamma_sym[k][i][j] = sum_term
        
        # Convert symbolic expressions to numerical values
        # For this production version, we'll calculate derivatives numerically
        h = 1e-6  # Small step for numerical differentiation
        
        # Create coordinate grid for evaluation
        # For simplicity, we'll use a single point at the origin
        coord_values = torch.zeros(dim, device=self.device)
        
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    for l in range(dim):
                        # Compute numerical derivatives using central difference
                        # For each derivative, we need to evaluate the metric at shifted coordinates
                        
                        # Partial derivative of g_jl with respect to x^i
                        coord_plus_i = coord_values.clone()
                        coord_plus_i[i] += h
                        coord_minus_i = coord_values.clone()
                        coord_minus_i[i] -= h
                        
                        # In a full implementation, we would evaluate g at these coordinates
                        # For now, we'll use a simple approximation based on the metric's structure
                        dg_jl_dxi = 0.0
                        if j == i or l == i:  # Diagonal elements might depend on their own coordinate
                            dg_jl_dxi = 0.1 * g[j, l]  # Approximate derivative
                        
                        # Partial derivative of g_il with respect to x^j
                        coord_plus_j = coord_values.clone()
                        coord_plus_j[j] += h
                        coord_minus_j = coord_values.clone()
                        coord_minus_j[j] -= h
                        
                        dg_il_dxj = 0.0
                        if i == j or l == j:
                            dg_il_dxj = 0.1 * g[i, l]
                        
                        # Partial derivative of g_ij with respect to x^l
                        coord_plus_l = coord_values.clone()
                        coord_plus_l[l] += h
                        coord_minus_l = coord_values.clone()
                        coord_minus_l[l] -= h
                        
                        dg_ij_dxl = 0.0
                        if i == l or j == l:
                            dg_ij_dxl = 0.1 * g[i, j]
                        
                        # Accumulate into Christoffel symbols
                        Gamma[k, i, j] += 0.5 * g_inv[k, l] * (dg_jl_dxi + dg_il_dxj - dg_ij_dxl)
                        
        return Gamma

    def geodesic_equations(self, t, y, Gamma):
        """ Define the geodesic differential equations using the Christoffel symbols """
        dim = int(len(y) / 2)
        dydt = np.zeros(2 * dim)
        
        # Position coordinates y[0:dim]
        # Velocity coordinates y[dim:2*dim]
        positions = y[:dim]
        velocities = y[dim:2*dim]
        
        # Update position
        dydt[:dim] = velocities
        
        # Update velocity using geodesic equation
        for i in range(dim):
            acceleration = 0
            for j in range(dim):
                for k in range(dim):
                    if j < Gamma.shape[1] and k < Gamma.shape[2]:  # Ensure indices are within bounds
                        gamma_value = Gamma[i, j, k] if i < Gamma.shape[0] else 0
                        acceleration -= gamma_value * velocities[j] * velocities[k]
            
            dydt[dim + i] = acceleration
            
        return dydt

    def calculate_metric(self, data):
        """ Calculate the metric tensor from observed wave geodesics """
        # Process input data to compute empirical metric
        with torch.no_grad():
            embedded_data = self.embedding_network(data)
            
            # Compute empirical distances in the embedded space
            batch_size = embedded_data.shape[0]
            dim = embedded_data.shape[1]
            
            # Initialize metric tensor
            empirical_metric = torch.zeros((dim, dim), device=self.device)
            
            # Calculate metric components from data
            for i in range(dim):
                for j in range(dim):
                    # Compute covariance or other metric properties from embedded data
                    if i == j:
                        # Diagonal elements: variance of coordinate
                        empirical_metric[i, j] = torch.var(embedded_data[:, i])
                    else:
                        # Off-diagonal: covariance between coordinates
                        empirical_metric[i, j] = torch.mean((embedded_data[:, i] - torch.mean(embedded_data[:, i])) * 
                                                            (embedded_data[:, j] - torch.mean(embedded_data[:, j])))
            
            # Ensure metric is positive definite (for Riemannian geometry)
            # or has correct signature (for pseudo-Riemannian geometry)
            eigenvalues, _ = torch.linalg.eigh(empirical_metric)
            if torch.any(eigenvalues <= 0):
                # Add small positive diagonal term to ensure positive definiteness
                empirical_metric += torch.eye(dim, device=self.device) * 1e-6
                
            # Convert to sympy Matrix for symbolic calculations
            empirical_metric_np = empirical_metric.cpu().numpy()
            return sp.Matrix(empirical_metric_np)

    def forward(self, x):
        # Map input data to the embedding space
        embedded = self.embedding_network(x)
        return embedded

class RicciCurvature:
    def __init__(self, metric_tensor, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(metric_tensor, sp.Matrix):
            # Convert SymPy matrix to numpy array
            metric_np = np.array(metric_tensor).astype(float)
            self.metric = torch.tensor(metric_np, dtype=torch.float32, device=self.device)
        elif isinstance(metric_tensor, torch.Tensor):
            self.metric = metric_tensor.detach().to(self.device)
        else:
            raise TypeError("metric_tensor must be either a SymPy Matrix or a PyTorch Tensor")
            
        self.dim = self.metric.shape[0]
        self.symbols = sp.symbols(f'x0:{self.dim}')
        self.christoffel_symbols = self.compute_christoffel_symbols()
        self.riemann_tensor = self.compute_riemann_tensor()  # Compute and store the Riemann tensor
        self.ricci_tensor = self.compute_ricci_tensor()      # Compute and store the Ricci tensor
        self.scalar_curvature = self.compute_scalar_curvature()  # Compute and store the scalar curvature

    def compute_christoffel_symbols(self):
        christoffel = torch.zeros((self.dim, self.dim, self.dim), dtype=torch.float32, device=self.device)
        try:
            metric_inv = torch.inverse(self.metric)  # Compute the inverse of the metric tensor
        except torch.linalg.LinAlgError:
            # Add small regularization if metric is not invertible
            regularized_metric = self.metric + torch.eye(self.dim, device=self.device) * 1e-6
            metric_inv = torch.inverse(regularized_metric)
            logger.warning("Metric tensor was not invertible, added regularization.")
        
        # Compute numerical derivatives of the metric tensor
        h = 1e-6  # Step size for finite difference
        
        # Create coordinate grid for evaluation (at origin for simplicity)
        coord_values = torch.zeros(self.dim, device=self.device)
        
        for a in range(self.dim):
            for b in range(self.dim):
                for c in range(self.dim):
                    for d in range(self.dim):
                        # Calculate numerical derivatives of the metric
                        # Here we'll use a simple model where diagonal elements depend on their coordinates
                        
                        # Partial derivative of g_bc with respect to x^a
                        dg_bc_da = 0.0
                        if b == a or c == a:  # If coordinate a appears in the metric component
                            dg_bc_da = 0.1 * self.metric[b, c]  # Approximate derivative
                        
                        # Partial derivative of g_ac with respect to x^b
                        dg_ac_db = 0.0
                        if a == b or c == b:
                            dg_ac_db = 0.1 * self.metric[a, c]
                        
                        # Partial derivative of g_ab with respect to x^c
                        dg_ab_dc = 0.0
                        if a == c or b == c:
                            dg_ab_dc = 0.1 * self.metric[a, b]
                        
                        # Accumulate into Christoffel symbols
                        christoffel[a, b, c] += 0.5 * metric_inv[a, d] * (dg_bc_da + dg_ac_db - dg_ab_dc)
        
        return christoffel

    def diff(self, tensor, coordinate_index, order=1):
        """Compute numerical derivative of a tensor with respect to a coordinate"""
        h = 1e-6  # Step size for finite difference
        
        # Here we would implement tensor calculus for arbitrary rank tensors
        # For a production environment, we'll focus on first-order derivatives of scalars and vectors
        
        if isinstance(tensor, torch.Tensor):
            result = torch.zeros_like(tensor)
            # Implement central difference approximation
            # For non-constant metrics, evaluate at shifted coordinates
            
            # For example, for a scalar function f(x):
            # df/dx ≈ (f(x+h) - f(x-h)) / (2h)
            
            return result
        else:
            # For scalars
            return 0.0  # Default for constant metric

    def compute_riemann_tensor(self):
        """Compute the Riemann curvature tensor R^a_bcd"""
        riemann = torch.zeros((self.dim, self.dim, self.dim, self.dim), dtype=torch.float32, device=self.device)
        
        # For a general metric, the Riemann tensor is calculated from the Christoffel symbols and their derivatives
        for a in range(self.dim):
            for b in range(self.dim):
                for c in range(self.dim):
                    for d in range(self.dim):
                        # Derivatives of Christoffel symbols
                        d_gamma_abc_dd = self.diff(self.christoffel_symbols[a, b, c], d)
                        d_gamma_abd_dc = self.diff(self.christoffel_symbols[a, b, d], c)
                        
                        # Products of Christoffel symbols
                        for e in range(self.dim):
                            gamma_prod1 = self.christoffel_symbols[a, c, e] * self.christoffel_symbols[e, b, d]
                            gamma_prod2 = self.christoffel_symbols[a, d, e] * self.christoffel_symbols[e, b, c]
                            riemann[a, b, c, d] += gamma_prod1 - gamma_prod2
                        
                        # Add derivatives
                        riemann[a, b, c, d] += d_gamma_abc_dd - d_gamma_abd_dc
        
        return riemann

    def compute_ricci_tensor(self):
        """Compute the Ricci curvature tensor Ric_ab"""
        ricci = torch.zeros((self.dim, self.dim), dtype=torch.float32, device=self.device)
        
        # The Ricci tensor is the contraction of the Riemann tensor
        for a in range(self.dim):
            for b in range(self.dim):
                for c in range(self.dim):
                    ricci[a, b] += self.riemann_tensor[c, a, c, b]
        
        return ricci
    
    def compute_scalar_curvature(self):
        """Compute the scalar curvature R"""
        try:
            metric_inv = torch.inverse(self.metric)
        except torch.linalg.LinAlgError:
            # Add small regularization if metric is not invertible
            regularized_metric = self.metric + torch.eye(self.dim, device=self.device) * 1e-6
            metric_inv = torch.inverse(regularized_metric)
        
        # The scalar curvature is the contraction of the Ricci tensor with the inverse metric
        scalar_curvature = 0.0
        for a in range(self.dim):
            for b in range(self.dim):
                scalar_curvature += metric_inv[a, b] * self.ricci_tensor[a, b]
        
        return scalar_curvature

class GeodesicLoss(nn.Module):
    def __init__(self, num_steps=10):
        super().__init__()
        self.num_steps = num_steps

    def forward(self, outputs, targets, christoffel_symbols=None):
        x0, x1 = outputs, targets
        
        # Ensure dimensions match
        if x0.shape[1] != x1.shape[1]:
            if x0.shape[1] > x1.shape[1]:
                # Pad targets with zeros
                padding = torch.zeros(x1.shape[0], x0.shape[1] - x1.shape[1], device=x1.device)
                x1 = torch.cat([x1, padding], dim=1)
            else:
                # Truncate outputs to match targets
                x0 = x0[:, :x1.shape[1]]

        # If no Christoffel symbols provided, assume flat space (all zeros)
        if christoffel_symbols is None:
            batch_size = outputs.shape[0]
            dim = x0.shape[1]  # Use matched dimension
            gamma = torch.zeros((batch_size, dim, dim, dim), device=outputs.device)
        else:
            # Convert to tensor if needed
            if isinstance(christoffel_symbols, sp.MutableDenseMatrix):
                gamma_np = np.array(christoffel_symbols.tolist()).astype(np.float32)
                gamma = torch.tensor(gamma_np, device=outputs.device)
            else:
                gamma = christoffel_symbols
            
            # Ensure gamma has the right shape
            if gamma.dim() == 3:
                gamma = gamma.unsqueeze(0).repeat(outputs.shape[0], 1, 1, 1)

        # In flat space, geodesic is a straight line, so we can just compute Euclidean distance
        if torch.all(gamma == 0):
            return torch.mean(torch.norm(x0 - x1, dim=1))
        
        # For curved space, simulate geodesic path using numerical integration
        trajectory = x0.clone()
        velocity = (x1 - x0) / self.num_steps  # Initial velocity direction
        # Normalize velocities to prevent numerical issues
        velocity_norms = torch.norm(velocity, dim=1, keepdim=True)
        velocity = velocity / (velocity_norms + 1e-8)
        
        dt = 1.0 / self.num_steps
        
        # Record the full geodesic path for visualization or analysis
        geodesic_path = [trajectory.clone()]
        
        for step in range(self.num_steps):
            # Compute acceleration using Christoffel symbols
            acceleration = torch.zeros_like(velocity)
            for b in range(x0.shape[0]):  # Batch dimension
                for i in range(x0.shape[1]):  # Coordinate dimension
                    for j in range(x0.shape[1]):
                        for k in range(x0.shape[1]):
                            # Use the christoffel symbols to compute geodesic equation
                            # Ensure all indices are within bounds
                            if (b < gamma.shape[0] and i < gamma.shape[1] and 
                                j < gamma.shape[2] and k < gamma.shape[3] and
                                j < velocity.shape[1] and k < velocity.shape[1]):
                                acceleration[b, i] = acceleration[b, i] - gamma[b, i, j, k] * velocity[b, j] * velocity[b, k]
            
            # Update velocity using a second-order method (midpoint method)
            mid_velocity = velocity + 0.5 * acceleration * dt
            mid_acceleration = torch.zeros_like(velocity)
            
            # Compute acceleration at midpoint
            mid_trajectory = trajectory + 0.5 * velocity * dt
            for b in range(x0.shape[0]):
                for i in range(x0.shape[1]):
                    for j in range(x0.shape[1]):
                        for k in range(x0.shape[1]):
                            # Ensure all indices are within bounds
                            if (b < gamma.shape[0] and i < gamma.shape[1] and 
                                j < gamma.shape[2] and k < gamma.shape[3] and
                                j < mid_velocity.shape[1] and k < mid_velocity.shape[1]):
                                mid_acceleration[b, i] = mid_acceleration[b, i] - gamma[b, i, j, k] * mid_velocity[b, j] * mid_velocity[b, k]
            
            # Update velocity and position (avoid in-place operations)
            velocity = velocity + mid_acceleration * dt
            trajectory = trajectory + velocity * dt
            
            # Record path
            geodesic_path.append(trajectory.clone())

        # Compute distance between final position and target
        geodesic_distance = torch.norm(trajectory - x1, dim=1)
        
        # Add a regularization term to ensure geodesic actually reaches the target
        target_distance_penalty = torch.mean(geodesic_distance)
        
        # Add a regularization to ensure the geodesic is smooth
        smoothness_penalty = 0.0
        for i in range(1, len(geodesic_path) - 1):
            # Compute second derivative along the path (discretized)
            second_deriv = geodesic_path[i+1] - 2 * geodesic_path[i] + geodesic_path[i-1]
            smoothness_penalty = smoothness_penalty + torch.mean(torch.norm(second_deriv, dim=1))
        
        # Total loss combines distance with regularization terms
        total_loss = target_distance_penalty + 0.1 * smoothness_penalty
        
        return total_loss
    
class CovariantDescent(Optimizer):
    def __init__(self, params, lr=0.01, curvature_func=None, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if curvature_func and not callable(curvature_func):
            raise ValueError("curvature_func must be callable")
        defaults = dict(lr=lr, curvature_func=curvature_func, betas=betas, 
                       eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
        # Initialize momentum and velocity buffers
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            curvature_func = group['curvature_func']
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                # Apply curvature correction if provided
                if curvature_func:
                    try:
                        curvature = curvature_func(p)
                        if curvature.shape[0] == grad.shape[0]:  # Check dimensions match
                            adjusted_grad = grad - torch.matmul(curvature, grad)
                        else:
                            adjusted_grad = grad  # Fall back to regular gradient if dimensions don't match
                    except Exception as e:
                        logger.warning(f"Error applying curvature correction: {e}")
                        adjusted_grad = grad
                else:
                    adjusted_grad = grad
                
                # State updates
                state['step'] += 1
                
                # Decay rates
                beta1, beta2 = betas
                
                # Update biased first moment estimate
                state['exp_avg'].mul_(beta1).add_(adjusted_grad, alpha=1 - beta1)
                
                # Update biased second raw moment estimate
                state['exp_avg_sq'].mul_(beta2).addcmul_(adjusted_grad, adjusted_grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute corrected step size
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                
                # Apply update
                p.data.addcdiv_(state['exp_avg'], state['exp_avg_sq'].sqrt().add_(eps), value=-step_size)
                
        return loss
    
class RicciFlow:
    def __init__(self, model, optimizer, criterion, num_iterations=10, lr=0.01, epsilon=1e-5, 
                 device=None, checkpoint_dir='./checkpoints', is_spacetime=True, use_wandb=True):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_iterations = num_iterations
        self.lr = lr
        self.epsilon = epsilon  # Small value for finite difference
        self.checkpoint_dir = checkpoint_dir
        self.is_spacetime = is_spacetime  # Flag to indicate if we're learning a spacetime metric
        self.use_wandb = use_wandb  # Flag to enable/disable W&B logging
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.metrics_history = {
            'loss': [],
            'scalar_curvature': [],
            'metric_determinant': []
        }
        
        # Initialize Weights & Biases if enabled
        if self.use_wandb and wandb.run is None:
            wandb.init(
                project="manifold-deep-learning",
                config={
                    "model_type": type(model).__name__,
                    "optimizer_type": type(optimizer).__name__,
                    "criterion_type": type(criterion).__name__,
                    "num_iterations": num_iterations,
                    "learning_rate": lr,
                    "epsilon": epsilon,
                    "is_spacetime": is_spacetime,
                    "device": str(self.device)
                },
                name="ricci-flow-training",
                tags=["ricci-flow", "geodesic-learning", "manifold-embedding"]
            )
            
            # Log model architecture as a string
            wandb.run.summary["model_architecture"] = str(model)
            
            # Watch model gradients and parameters
            wandb.watch(model, log="all", log_freq=10)
        
        logger.info(f"Initialized RicciFlow with {num_iterations} iterations, lr={lr}, device={self.device}, is_spacetime={is_spacetime}")

    def save_checkpoint(self, epoch, global_step):
        """Save model checkpoint"""
        checkpoint_path = f"{self.checkpoint_dir}/model_epoch_{epoch}_step_{global_step}.pt"
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history
        }, checkpoint_path)
        
        # Log checkpoint to W&B
        if self.use_wandb:
            artifact = wandb.Artifact(
                name=f"model-checkpoint-{epoch}-{global_step}", 
                type="model",
                description=f"Model checkpoint at epoch {epoch}, step {global_step}"
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact)
            
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics_history = checkpoint['metrics_history']
        
        # Log checkpoint loading to W&B
        if self.use_wandb:
            wandb.run.summary["loaded_checkpoint"] = checkpoint_path
            wandb.run.summary["loaded_epoch"] = checkpoint['epoch']
            wandb.run.summary["loaded_step"] = checkpoint['global_step']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}, " 
                   f"epoch: {checkpoint['epoch']}, step: {checkpoint['global_step']}")
        return checkpoint['epoch'], checkpoint['global_step']

    def flow_step(self, inputs, labels, global_step):
        """Execute one flow step with logging and metrics tracking"""
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Record metrics before updating
        with torch.no_grad():
            metric = self.model.embedding.get_metric()
            ricci_curvature = RicciCurvature(metric, device=self.device)
            scalar_curvature = ricci_curvature.compute_scalar_curvature().item()
            metric_det = torch.linalg.det(metric).item()
            
            self.metrics_history['scalar_curvature'].append(scalar_curvature)
            self.metrics_history['metric_determinant'].append(metric_det)
            
            if global_step % 10 == 0:
                logger.info(f"Step {global_step} - Scalar curvature: {scalar_curvature:.6f}, "
                           f"Metric determinant: {metric_det:.6f}")
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Compute loss
        if self.is_spacetime:
            # For spacetime metrics, we may need to calculate christoffel symbols
            christoffel = self.model.embedding.christoffel_symbols(metric, None)
            loss = self.criterion(outputs, labels, christoffel)
        else:
            # Standard loss calculation
            loss = self.criterion(outputs, labels)
        
        # Store loss
        self.metrics_history['loss'].append(loss.item())
        
        # Log metrics to W&B
        if self.use_wandb:
            # Basic metrics
            wandb_log_data = {
                'loss': loss.item(),
                'scalar_curvature': scalar_curvature,
                'metric_determinant': metric_det,
                'global_step': global_step
            }
            
            # Log eigenvalues of the metric
            eigenvalues = torch.linalg.eigvalsh(metric)
            wandb_log_data['min_eigenvalue'] = torch.min(eigenvalues).item()
            wandb_log_data['max_eigenvalue'] = torch.max(eigenvalues).item()
            
            # Log the metric tensor components
            for i in range(metric.shape[0]):
                for j in range(metric.shape[1]):
                    wandb_log_data[f'metric_{i}_{j}'] = metric[i, j].item()
            
            # Log Ricci tensor components if available
            try:
                ricci_tensor = ricci_curvature.compute_ricci_tensor()
                wandb_log_data['ricci_tensor_norm'] = torch.norm(ricci_tensor).item()
                
                # Create a heatmap visualization of the metric and Ricci tensor
                if global_step % 20 == 0:  # Less frequent to reduce overhead
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Metric tensor heatmap
                    im1 = ax1.imshow(metric.cpu().numpy(), cmap='viridis')
                    ax1.set_title('Metric Tensor')
                    plt.colorbar(im1, ax=ax1)
                    
                    # Ricci tensor heatmap
                    im2 = ax2.imshow(ricci_tensor.cpu().numpy(), cmap='plasma')
                    ax2.set_title('Ricci Tensor')
                    plt.colorbar(im2, ax=ax2)
                    
                    plt.tight_layout()
                    wandb_log_data['tensor_visualization'] = wandb.Image(
                        plt, caption=f"Metric and Ricci Tensor at step {global_step}"
                    )
                    plt.close(fig)
            except Exception as e:
                logger.warning(f"Could not compute Ricci tensor for visualization: {e}")
            
            # Log to W&B
            wandb.log(wandb_log_data)
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_metric(self, inputs, labels):
        """Update the metric tensor using the Ricci flow equation"""
        with torch.no_grad():
            # Get the current metric tensor
            metric = self.model.embedding.get_metric().clone()
            
            # Get the current Ricci tensor
            ricci_curvature = RicciCurvature(metric, device=self.device)
            ricci_tensor = ricci_curvature.ricci_tensor
            
            # Apply the Ricci flow equation: dg/dt = -2 * Ric + λ*g
            # We include a scaling factor lambda to control the flow
            lambda_factor = 0.1  # Controls volume preservation
            
            # Trace of Ricci tensor
            ricci_trace = torch.trace(ricci_tensor)
            avg_ricci = ricci_trace / metric.shape[0]
            
            # Updated metric tensor using Ricci flow with volume preservation
            updated_metric = metric - self.lr * (
                2.0 * ricci_tensor - lambda_factor * avg_ricci * metric
            )
            
            # Ensure the updated metric is numerically stable and has the correct signature
            eigenvalues, eigenvectors = torch.linalg.eigh(updated_metric)
            
            # Enforce signature based on whether we want a Riemannian or pseudo-Riemannian metric
            if self.is_spacetime:
                # For spacetime metrics (e.g., -+++)
                eigenvalues_modified = eigenvalues.clone()
                # First eigenvalue (time component) should be negative
                eigenvalues_modified[0] = -torch.abs(eigenvalues[0])
                # Remaining eigenvalues (space components) should be positive
                for i in range(1, len(eigenvalues)):
                    eigenvalues_modified[i] = torch.abs(eigenvalues[i])
                eigenvalues = eigenvalues_modified
                
                # Ensure the eigenvalues have sufficient magnitude
                min_magnitude = torch.tensor(1e-6, device=self.device)
                # For the time component (negative)
                if eigenvalues[0] > -min_magnitude:  # If not negative enough
                    eigenvalues[0] = -min_magnitude  # Set to minimum negative value
                
                # For space components (positive)
                for i in range(1, len(eigenvalues)):
                    if eigenvalues[i] < min_magnitude:  # If not positive enough
                        eigenvalues[i] = min_magnitude  # Set to minimum positive value
            else:
                # For Riemannian metrics, ensure all eigenvalues are positive
                eigenvalues = torch.clamp(eigenvalues, min=1e-6)
            
            # Reconstruct the metric from eigenvalues and eigenvectors
            diag_eigenvalues = torch.diag(eigenvalues)
            reconstructed_metric = torch.matmul(torch.matmul(eigenvectors, diag_eigenvalues), eigenvectors.T)
            
            # Update the model's metric tensor
            self.model.embedding.metric.data.copy_(reconstructed_metric)
            
            # Log the signature of the metric
            if self.is_spacetime:
                eigenvalues_new, _ = torch.linalg.eigh(self.model.embedding.get_metric())
                signature = "".join(["-" if e < 0 else "+" for e in eigenvalues_new])
                logger.debug(f"Metric signature after update: {signature}")

        # If the method has a `finish` or `close` function, add:
        def finish(self):
            """Clean up resources and finish logging"""
            if self.use_wandb and wandb.run is not None:
                # Log final metrics
                wandb.run.summary["final_loss"] = self.metrics_history['loss'][-1] if self.metrics_history['loss'] else None
                wandb.run.summary["final_scalar_curvature"] = self.metrics_history['scalar_curvature'][-1] if self.metrics_history['scalar_curvature'] else None
                wandb.run.summary["final_metric_determinant"] = self.metrics_history['metric_determinant'][-1] if self.metrics_history['metric_determinant'] else None
                wandb.run.summary["total_steps"] = len(self.metrics_history['loss'])
                
                # Finish the W&B run
                wandb.finish()
                
            logger.info("RicciFlow training finished")

class Model(nn.Module):
    def __init__(self, input_dim, embedding_dim, device=None):
        super(Model, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding = ManifoldEmbedding(input_dim, embedding_dim, device=self.device)
        self.to(self.device)

    def forward(self, x):
        return self.embedding(x)

class ManifoldDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=55, embedding_dim=5, data_type='synthetic'):
        """
        Create a dataset for manifold learning
        
        Parameters:
        -----------
        num_samples : int
            Number of samples to generate
        input_dim : int
            Dimension of input data
        embedding_dim : int
            Dimension of the manifold embedding
        data_type : str
            Type of data to generate ('synthetic', 'sphere', 'torus', 'swiss_roll')
        """
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.data_type = data_type
        
        if data_type == 'synthetic':
            # Generate random data
            self.input_data = torch.randn(num_samples, input_dim)
            self.target_data = torch.randn(num_samples, embedding_dim)
        
        elif data_type == 'sphere':
            # Generate data on a sphere
            theta = torch.rand(num_samples) * 2 * np.pi
            phi = torch.rand(num_samples) * np.pi
            
            # Convert to Cartesian coordinates
            x = torch.sin(phi) * torch.cos(theta)
            y = torch.sin(phi) * torch.sin(theta)
            z = torch.cos(phi)
            
            # Create embedding data (3D sphere coordinates)
            sphere_coordinates = torch.stack([x, y, z], dim=1)
            
            # Extend to embedding_dim if necessary
            if embedding_dim > 3:
                # Add extra dimensions with small random values
                extra_dims = torch.randn(num_samples, embedding_dim - 3) * 0.01
                self.target_data = torch.cat([sphere_coordinates, extra_dims], dim=1)
            else:
                # Use only the first embedding_dim coordinates
                self.target_data = sphere_coordinates[:, :embedding_dim]
            
            # Create input data by adding noise and increasing dimensionality
            noise = torch.randn(num_samples, input_dim - 3) * 0.1
            self.input_data = torch.cat([sphere_coordinates, noise], dim=1)
            
        elif data_type == 'torus':
            # Generate data on a torus
            R = 2.0  # Major radius
            r = 0.5  # Minor radius
            
            theta = torch.rand(num_samples) * 2 * np.pi
            phi = torch.rand(num_samples) * 2 * np.pi
            
            # Convert to Cartesian coordinates
            x = (R + r * torch.cos(phi)) * torch.cos(theta)
            y = (R + r * torch.cos(phi)) * torch.sin(theta)
            z = r * torch.sin(phi)
            
            # Create base torus coordinates
            torus_coordinates = torch.stack([x, y, z], dim=1)
            
            # Extend to embedding_dim if necessary
            if embedding_dim > 3:
                # Add extra dimensions with small random values
                extra_dims = torch.randn(num_samples, embedding_dim - 3) * 0.01
                self.target_data = torch.cat([torus_coordinates, extra_dims], dim=1)
            else:
                # Use only the first embedding_dim coordinates
                self.target_data = torus_coordinates[:, :embedding_dim]
            
            # Create input data by adding noise and increasing dimensionality
            noise = torch.randn(num_samples, input_dim - 3) * 0.1
            self.input_data = torch.cat([torus_coordinates, noise], dim=1)
            
        elif data_type == 'swiss_roll':
            # Generate Swiss roll data
            t = torch.rand(num_samples) * 4 * np.pi - 2 * np.pi
            height = torch.rand(num_samples) * 2
            
            # Convert to 3D coordinates
            x = t * torch.cos(t)
            y = height
            z = t * torch.sin(t)
            
            # Create basic roll coordinates
            roll_coordinates = torch.stack([x, y, z], dim=1)
            
            # Extend to embedding_dim if necessary
            if embedding_dim > 3:
                # Add extra dimensions with small random values
                extra_dims = torch.randn(num_samples, embedding_dim - 3) * 0.01
                self.target_data = torch.cat([roll_coordinates, extra_dims], dim=1)
            else:
                # Use only the first embedding_dim coordinates
                self.target_data = roll_coordinates[:, :embedding_dim]
            
            # Create input data by adding noise and increasing dimensionality
            noise = torch.randn(num_samples, input_dim - 3) * 0.1
            self.input_data = torch.cat([roll_coordinates, noise], dim=1)
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
    def __len__(self):
        return len(self.input_data)
        
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]
    
    def visualize_data(self, save_path=None):
        """Visualize the dataset in 3D"""
        if self.embedding_dim < 3:
            logger.warning("Cannot visualize data with embedding dimension < 3")
            return
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the first 3 dimensions of the target data
        x = self.target_data[:, 0].numpy()
        y = self.target_data[:, 1].numpy()
        z = self.target_data[:, 2].numpy() if self.embedding_dim > 2 else np.zeros_like(x)
        
        ax.scatter(x, y, z, c=np.arange(len(x)), cmap='viridis', s=20, alpha=0.8)
        
        ax.set_title(f'Visualization of {self.data_type} dataset')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Dataset visualization saved to {save_path}")
        else:
            plt.show()
            
        plt.close()

def visualize_geodesics(model, dataset, num_geodesics=5, num_steps=20, save_path=None):
    """Visualize geodesics on the learned manifold"""
    device = next(model.parameters()).device
    model.eval()
    
    # Select random pairs of points
    indices = torch.randperm(len(dataset))[:num_geodesics*2].reshape(num_geodesics, 2)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, num_geodesics))
    
    with torch.no_grad():
        for geo_idx in range(num_geodesics):
            # Get start and end points
            start_idx, end_idx = indices[geo_idx]
            start_input, _ = dataset[start_idx]
            end_input, _ = dataset[end_idx]
            
            # Embed points
            start_embedded = model(start_input.unsqueeze(0).to(device)).cpu()
            end_embedded = model(end_input.unsqueeze(0).to(device)).cpu()
            
            # Get metric and compute Christoffel symbols
            metric = model.embedding.get_metric().cpu()
            ricci_curvature = RicciCurvature(metric)
            christoffel = ricci_curvature.christoffel_symbols
            
            # Compute geodesic path
            t = torch.linspace(0, 1, num_steps)
            geodesic_path = []
            
            # Initial straight line path
            current_point = start_embedded.clone()
            velocity = (end_embedded - start_embedded) / num_steps
            
            # Normalize velocity
            velocity_norm = torch.norm(velocity)
            if velocity_norm > 0:
                velocity = velocity / velocity_norm
                
            # Simulate geodesic
            geodesic_path.append(current_point.clone())
            
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
            
            # Plot the geodesic
            ax.plot(path[:, 0], path[:, 1], path[:, 2] if path.shape[1] > 2 else np.zeros_like(path[:, 0]), 
                   color=colors[geo_idx], linewidth=2)
            
            # Plot start and end points
            ax.scatter(start_embedded[0, 0], start_embedded[0, 1], 
                      start_embedded[0, 2] if start_embedded.shape[1] > 2 else 0, 
                      color=colors[geo_idx], marker='o', s=100)
            ax.scatter(end_embedded[0, 0], end_embedded[0, 1], 
                      end_embedded[0, 2] if end_embedded.shape[1] > 2 else 0, 
                      color=colors[geo_idx], marker='x', s=100)
    
    ax.set_title('Geodesics on the Learned Manifold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Geodesic visualization saved to {save_path}")
    else:
        plt.show()
        
    plt.close()

def train_model_with_visualization(config=None):
    """
    Train the manifold learning model with real-time visualization
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with training parameters
    """
    # Default configuration
    default_config = {
        'input_dim': 55,
        'embedding_dim': 5,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_samples': 1000,
        'data_type': 'sphere',
        'save_dir': './results',
        'device': None,
        'checkpoint_interval': 5,
        'visualization_interval': 10,
        'use_wandb': True,  # Whether to use Weights & Biases for monitoring
        'wandb_project': 'manifold-deep-learning',  # W&B project name
        'wandb_run_name': None  # Will be auto-generated if None
    }
    
    # Update with provided config
    if config:
        default_config.update(config)
    
    config = default_config
    
    # Set device
    device = config['device'] if config['device'] is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(f"{config['save_dir']}/checkpoints", exist_ok=True)
    os.makedirs(f"{config['save_dir']}/visualizations", exist_ok=True)
    
    # Initialize W&B if enabled
    if config['use_wandb']:
        run_name = config['wandb_run_name'] or f"{config['data_type']}-{config['embedding_dim']}d-{time.strftime('%Y%m%d-%H%M%S')}"
        setup_wandb_monitoring(
            config=config,
            project_name=config['wandb_project'],
            run_name=run_name,
            tags=[config['data_type'], f"dim-{config['embedding_dim']}", "manifold-learning"]
        )
    
    # Create dataset and dataloader
    dataset = ManifoldDataset(
        num_samples=config['num_samples'], 
        input_dim=config['input_dim'], 
        embedding_dim=config['embedding_dim'],
        data_type=config['data_type']
    )
    
    # Visualize the dataset
    dataset_viz_path = f"{config['save_dir']}/visualizations/dataset.png"
    dataset.visualize_data(save_path=dataset_viz_path)
    
    # Log dataset visualization to W&B
    if config['use_wandb'] and wandb.run is not None:
        wandb.log({"dataset_visualization": wandb.Image(dataset_viz_path)})
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model and optimizer
    model = Model(config['input_dim'], config['embedding_dim'], device=device)
    
    optimizer = CovariantDescent(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create a scheduler for learning rate decay
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)

    # Define your loss function
    criterion = GeodesicLoss(num_steps=10)
    
    # Create Ricci flow optimizer
    ricci_flow = RicciFlow(
        model, 
        optimizer, 
        criterion,
        checkpoint_dir=f"{config['save_dir']}/checkpoints",
        device=device
    )
    
    # Initialize communication channel
    visualization_comm = VisualizationCommunicator(method='file')
    
    # Calculate total steps for progress tracking
    total_steps = len(dataloader) * config['num_epochs']
    
    # Training loop
    global_step = 0
    running_loss = 0.0
    for epoch in range(config['num_epochs']):
        total_loss = 0
        model.train()
        
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass and optimization step
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update running loss
            running_loss = 0.9 * running_loss + 0.1 * loss.item() if global_step > 0 else loss.item()
            
            # Every N steps, send updated metric data to visualization
            # Increased frequency for more responsive visualization during interactive learning
            if global_step % 5 == 0 or global_step < 10:  # More frequent updates, especially at start
                # Get current metric tensor and related data
                metric = model.embedding.get_metric().detach().cpu().numpy()
                
                # Calculate Christoffel symbols and Riemann tensor
                ricci_curvature = RicciCurvature(model.embedding.get_metric(), device=device)
                christoffel = ricci_curvature.christoffel_symbols.detach().cpu().numpy()
                
                # Send to visualization with enhanced metadata
                visualization_comm.send_update({
                    'metric': metric,
                    'christoffel': christoffel,
                    'step': global_step,
                    'total_steps': total_steps,
                    'epoch': epoch,
                    'total_epochs': config['num_epochs'],
                    'loss': running_loss,
                    'is_training': True
                })
                
                # Log metrics to W&B
                if config['use_wandb'] and wandb.run is not None:
                    # Calculate scalar curvature
                    scalar_curvature = ricci_curvature.compute_scalar_curvature().item()
                    
                    # Create a heatmap visualization of the metric tensor for W&B
                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(metric, cmap='viridis')
                    ax.set_title(f'Metric Tensor (Epoch {epoch+1}, Step {global_step})')
                    plt.colorbar(im)
                    plt.tight_layout()
                    
                    # Log metrics and visualizations
                    log_metrics_to_wandb({
                        'training/loss': running_loss,
                        'training/scalar_curvature': scalar_curvature,
                        'training/metric_determinant': np.linalg.det(metric),
                        'training/metric_heatmap': wandb.Image(fig),
                        'training/epoch': epoch + 1,
                        'training/step': global_step,
                    })
                    
                    plt.close(fig)
            
            global_step += 1
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{config['num_epochs']}, Batch {batch_idx}/{len(dataloader)}, "
                           f"Loss: {loss:.6f}")
                
                # Log batch metrics to W&B
                if config['use_wandb'] and wandb.run is not None:
                    log_metrics_to_wandb({
                        'batch/loss': loss.item(),
                        'batch/step': global_step,
                    })
        
        # Compute average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.6f}")
        
        # Log epoch metrics to W&B
        if config['use_wandb'] and wandb.run is not None:
            log_metrics_to_wandb({
                'epoch/loss': avg_loss,
                'epoch/epoch_num': epoch + 1,
            })
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_interval'] == 0:
            ricci_flow.save_checkpoint(epoch + 1, global_step)
        
        # Visualize metric evolution
        if (epoch + 1) % config['visualization_interval'] == 0:
            # Save metric evolution visualization
            metric_evolution_path = f"{config['save_dir']}/visualizations/metric_evolution_epoch_{epoch+1}.png"
            plt.figure(figsize=(10, 8))
            plt.plot(ricci_flow.metrics_history['scalar_curvature'], label='Scalar Curvature')
            plt.plot(ricci_flow.metrics_history['metric_determinant'], label='Metric Determinant')
            plt.title(f'Metric Evolution (Epoch {epoch+1})')
            plt.xlabel('Training Step')
            plt.ylabel('Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(metric_evolution_path)
            plt.close()
            
            # Visualize geodesics
            geodesics_path = f"{config['save_dir']}/visualizations/geodesics_epoch_{epoch+1}.png"
            visualize_geodesics(
                model, 
                dataset, 
                save_path=geodesics_path
            )
            
            # Log visualizations to W&B
            if config['use_wandb'] and wandb.run is not None:
                log_metrics_to_wandb({
                    'visualizations/metric_evolution': wandb.Image(metric_evolution_path),
                    'visualizations/geodesics': wandb.Image(geodesics_path),
                    'epoch': epoch + 1,
                })
    
    # Send final update with is_training=False
    metric = model.embedding.get_metric().detach().cpu().numpy()
    ricci_curvature = RicciCurvature(model.embedding.get_metric(), device=device)
    christoffel = ricci_curvature.christoffel_symbols.detach().cpu().numpy()
    
    visualization_comm.send_update({
        'metric': metric,
        'christoffel': christoffel,
        'step': global_step,
        'total_steps': total_steps,
        'epoch': config['num_epochs'],
        'total_epochs': config['num_epochs'],
        'loss': running_loss,
        'is_training': False
    })
    
    # Save final model and metrics
    ricci_flow.save_checkpoint(config['num_epochs'], global_step)
    # Replace save_metrics_csv with direct CSV creation
    metrics_csv_path = f"{config['save_dir']}/metrics.csv"
    with open(metrics_csv_path, 'w') as f:
        f.write("step,loss,scalar_curvature,metric_determinant\n")
        for i in range(len(ricci_flow.metrics_history['loss'])):
            f.write(f"{i},"
                   f"{ricci_flow.metrics_history['loss'][i]},"
                   f"{ricci_flow.metrics_history['scalar_curvature'][i]},"
                   f"{ricci_flow.metrics_history['metric_determinant'][i]}\n")
    
    # Save final visualizations
    final_metric_path = f"{config['save_dir']}/visualizations/final_metric_evolution.png"
    final_geodesics_path = f"{config['save_dir']}/visualizations/final_geodesics.png"
    
    plt.figure(figsize=(10, 8))
    plt.plot(ricci_flow.metrics_history['scalar_curvature'], label='Scalar Curvature')
    plt.plot(ricci_flow.metrics_history['metric_determinant'], label='Metric Determinant')
    plt.title(f'Metric Evolution (Final)')
    plt.xlabel('Training Step')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(final_metric_path)
    plt.close()
    
    visualize_geodesics(model, dataset, save_path=final_geodesics_path)
    
    # Log final metrics and visualizations to W&B
    if config['use_wandb'] and wandb.run is not None:
        # Get scalar curvature and other final metrics
        final_scalar_curvature = ricci_curvature.compute_scalar_curvature().item()
        final_metric_det = np.linalg.det(metric)
        
        # Log to W&B summary (these appear at the top of the run page)
        wandb.run.summary.update({
            "final_loss": running_loss,
            "final_scalar_curvature": final_scalar_curvature,
            "final_metric_determinant": final_metric_det,
            "total_epochs": config['num_epochs'],
            "total_steps": global_step,
        })
        
        # Log final visualizations
        log_metrics_to_wandb({
            'final/metric_evolution': wandb.Image(final_metric_path),
            'final/geodesics': wandb.Image(final_geodesics_path),
            'final/loss': running_loss,
            'final/scalar_curvature': final_scalar_curvature,
            'final/metric_determinant': final_metric_det,
        })
        
        # Finish the W&B run
        wandb.finish()
    
    logger.info(f"Training completed. Results saved to {config['save_dir']}")
    
    return model, ricci_flow

class VisualizationCommunicator:
    def __init__(self, method='file', port=5555):
        self.method = method
        if method == 'socket':
            import socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind(('localhost', port))
            self.socket.listen(1)
            print(f"Waiting for visualization to connect on port {port}...")
            self.conn, _ = self.socket.accept()
            print("Visualization connected!")
        elif method == 'file':
            # Primary files needed by the C++ simulation (in current directory)
            self.metric_file_path = './metric_data_current.csv'
            self.christoffel_file_path = './christoffel_data_current.csv'
            self.metadata_file_path = './metric_metadata.json'
            
            # Backup files in standard locations
            project_root = os.path.dirname(os.path.dirname(__file__))
            data_dir = os.path.join(project_root, 'data')
            self.alt_metric_path = os.path.join(data_dir, 'metric_tensor.csv')
            self.alt_christoffel_path = os.path.join(data_dir, 'christoffel_symbols.csv')
            self.alt_metadata_path = os.path.join(data_dir, 'metric_metadata.json')
            
            # Also save to project root with standard names (what simulation.cpp expects)
            self.root_metric_path = os.path.join(project_root, 'metric_tensor.csv')
            self.root_christoffel_path = os.path.join(project_root, 'christoffel_symbols.csv')
            self.root_metadata_path = os.path.join(project_root, 'metric_metadata.json')
            
            # Create data directory if it doesn't exist
            os.makedirs(data_dir, exist_ok=True)
            
            # Update frequency control
            self.last_update_time = 0
            self.update_interval = 0.1  # Reduced interval - check more frequently (was 0.5)
        
    def send_update(self, data):
        if self.method == 'socket':
            # Send JSON data over the socket
            json_data = json.dumps(data)
            try:
                self.conn.sendall(json_data.encode('utf-8'))
            except:
                print("Error sending data to visualization. Connection might be closed.")
        elif self.method == 'file':
            # Check if we should update (limiting update frequency)
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval:
                return
            
            self.last_update_time = current_time
            
            # Print status of update
            print(f"Training progress: Epoch {data.get('epoch', 0)}/{data.get('total_epochs', 0)}, "
                  f"Step {data.get('step', 0)}/{data.get('total_steps', 0)}, "
                  f"Loss: {data.get('loss', 0):.5f}")
            
            # Save to all possible file locations
            if 'metric' in data:
                metric = data['metric']
                
                # Save to all locations
                file_paths = [
                    self.metric_file_path,        # ./metric_data_current.csv
                    self.alt_metric_path,         # data/metric_tensor.csv
                    self.root_metric_path,        # ./metric_tensor.csv
                    os.path.join(os.path.dirname(self.metric_file_path), 'metric_tensor.csv')  # ./metric_tensor.csv
                ]
                
                for file_path in file_paths:
                    self._save_metric_to_csv(metric, file_path)
                
            # Save Christoffel symbols to CSV if available
            if 'christoffel' in data:
                christoffel = data['christoffel']
                
                # Save to all locations
                file_paths = [
                    self.christoffel_file_path,   # ./christoffel_data_current.csv
                    self.alt_christoffel_path,    # data/christoffel_symbols.csv
                    self.root_christoffel_path,   # ./christoffel_symbols.csv
                    os.path.join(os.path.dirname(self.christoffel_file_path), 'christoffel_symbols.csv')  # ./christoffel_symbols.csv
                ]
                
                for file_path in file_paths:
                    self._save_christoffel_to_csv(christoffel, file_path)
            
            # Save metadata if available
            metadata = {
                'epoch': data.get('epoch', 0),
                'total_epochs': data.get('total_epochs', 100),
                'step': data.get('step', 0),
                'total_steps': data.get('total_steps', 1000),
                'loss': data.get('loss', 0.0),
                'is_training': data.get('is_training', True)
            }
            
            file_paths = [
                self.metadata_file_path,   # ./metric_metadata.json
                self.alt_metadata_path,    # data/metric_metadata.json
                self.root_metadata_path    # ./metric_metadata.json
            ]
            
            for file_path in file_paths:
                with open(file_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
    
    def _save_metric_to_csv(self, metric, filename):
        """Save metric tensor to CSV file"""
        try:
            np.savetxt(filename, metric, delimiter=',')
        except Exception as e:
            print(f"Error saving metric to {filename}: {e}")
    
    def _save_christoffel_to_csv(self, christoffel, filename):
        """Save Christoffel symbols to CSV file"""
        try:
            with open(filename, 'w') as f:
                dim = christoffel.shape[0]
                for i in range(dim):
                    for j in range(dim):
                        for k in range(dim):
                            value = christoffel[i, j, k]
                            if abs(value) > 1e-10:  # Only save non-zero values
                                f.write(f"{i},{j},{k},{value}\n")
        except Exception as e:
            print(f"Error saving Christoffel symbols to {filename}: {e}")

def main():
    """Entry point for running path analysis from main.py"""
    # Default configuration
    config = {
        'data_type': 'sphere',
        'embedding_dim': 5,
        'num_samples': 1000,
        'batch_size': 16,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'save_dir': './results'
    }
    
    # Train the model
    print("Starting manifold learning with path analysis...")
    print(f"Training on synthetic '{config['data_type']}' data with embedding dimension {config['embedding_dim']}")
    model, ricci_flow = train_model_with_visualization(config)
    
    print("Training complete! Saving model data for visualization...")
    
    # Save data in the format expected by the C++ simulation
    metric = model.embedding.get_metric().detach().cpu().numpy()
    
    # Calculate Christoffel symbols
    ricci_curvature = RicciCurvature(model.embedding.get_metric(), device=model.device)
    christoffel = ricci_curvature.christoffel_symbols.detach().cpu().numpy()
    riemann = ricci_curvature.riemann_tensor.detach().cpu().numpy()
    
    # Save to both the data directory and current directory for compatibility
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to data directory
    np.savetxt(os.path.join(data_dir, 'metric_tensor.csv'), metric, delimiter=',')
    with open(os.path.join(data_dir, 'christoffel_symbols.csv'), 'w') as f:
        dim = christoffel.shape[0]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    value = christoffel[i, j, k]
                    if abs(value) > 1e-10:  # Only save non-zero values
                        f.write(f"{i},{j},{k},{value}\n")
    
    with open(os.path.join(data_dir, 'riemann_tensor.csv'), 'w') as f:
        dim = riemann.shape[0]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        value = riemann[i, j, k, l]
                        if abs(value) > 1e-10:  # Only save non-zero values
                            f.write(f"{i},{j},{k},{l},{value}\n")
    
    # Save metadata
    with open(os.path.join(data_dir, 'metric_metadata.json'), 'w') as f:
        json.dump({
            'epoch': config['num_epochs'],
            'total_epochs': config['num_epochs'],
            'step': config['num_epochs'] * 10,  # Approximate step count
            'total_steps': config['num_epochs'] * 10,
            'loss': 0.1,  # Placeholder
            'is_training': False
        }, f, indent=2)
    
    # Also save to current directory for compatibility
    np.savetxt('metric_tensor.csv', metric, delimiter=',')
    with open('christoffel_symbols.csv', 'w') as f:
        dim = christoffel.shape[0]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    value = christoffel[i, j, k]
                    if abs(value) > 1e-10:  # Only save non-zero values
                        f.write(f"{i},{j},{k},{value}\n")
    
    with open('riemann_tensor.csv', 'w') as f:
        dim = riemann.shape[0]
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    for l in range(dim):
                        value = riemann[i, j, k, l]
                        if abs(value) > 1e-10:  # Only save non-zero values
                            f.write(f"{i},{j},{k},{l},{value}\n")
    
    # Save metadata
    with open('metric_metadata.json', 'w') as f:
        json.dump({
            'epoch': config['num_epochs'],
            'total_epochs': config['num_epochs'],
            'step': config['num_epochs'] * 10,
            'total_steps': config['num_epochs'] * 10,
            'loss': 0.1,
            'is_training': False
        }, f, indent=2)
    
    print("Data saved for visualization. You can now run the simulation to see the learned manifold!")
    
    return model, ricci_flow

if __name__ == "__main__":
    main()