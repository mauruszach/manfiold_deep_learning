#!/usr/bin/env python3
"""
Weights & Biases Monitoring for Manifold Deep Learning

This script demonstrates how to use Weights & Biases to monitor the training
of the manifold deep learning model and the spacetime simulation.
"""

import argparse
import os
import time
import wandb
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import io
from path_analysis import train_model_with_visualization, setup_wandb_monitoring, log_metrics_to_wandb
from test_visualization import generate_evolving_metric, calculate_christoffel, save_metric_to_csv, save_christoffel_to_csv, save_metadata

def run_simulation():
    """Run the C++ simulation in a separate process"""
    try:
        # Check if the simulation executable exists
        if not os.path.exists('./simulation'):
            print("Simulation executable not found. Compiling...")
            compile_cmd = "g++ -std=c++17 -DGL_SILENCE_DEPRECATION -o simulation simulation.cpp -I/opt/homebrew/opt/sfml/include -L/opt/homebrew/opt/sfml/lib -lsfml-graphics -lsfml-window -lsfml-system -framework OpenGL"
            subprocess.run(compile_cmd, shell=True, check=True)
        
        # Start simulation in background
        print("Starting simulation...")
        sim_process = subprocess.Popen('./simulation', shell=True)
        return sim_process
    except Exception as e:
        print(f"Error starting simulation: {e}")
        return None

def run_visualization_test(args):
    """Generate evolving metrics and monitor with W&B"""
    # Initialize W&B
    run = setup_wandb_monitoring(
        config={
            "dimensions": 4,
            "total_epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "update_interval": args.update_interval,
        },
        project_name=args.project,
        run_name=args.run_name or f"metric-evolution-{time.strftime('%Y%m%d-%H%M%S')}",
        tags=["spacetime", "metric-evolution", "geodesics"]
    )
    
    # Create a custom panel for the metric heatmap in W&B
    wandb.define_metric("epoch")
    wandb.define_metric("step")
    wandb.define_metric("loss", step_metric="step")
    wandb.define_metric("curvature_strength", step_metric="step")
    wandb.define_metric("metric_determinant", step_metric="step")
    
    # Simulation parameters
    dim = 4  # 4D spacetime
    total_epochs = args.epochs
    steps_per_epoch = args.steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    update_interval = args.update_interval  # seconds between updates
    
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
                
                # Calculate additional metrics for monitoring
                metric_det = np.linalg.det(metric)
                eigenvalues = np.linalg.eigvalsh(metric)
                curvature_str = (epoch / total_epochs) * 3.0
                
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
                
                # Create and log a heatmap of the metric tensor
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(metric, cmap='viridis')
                ax.set_title(f'Metric Tensor (Epoch {epoch}, Step {step})')
                plt.colorbar(im)
                plt.tight_layout()
                
                # Save figure to a temporary file and use that for wandb
                temp_img_path = f"./temp_metric_{epoch}_{step}.png"
                plt.savefig(temp_img_path)
                wandb_log_data["metric_heatmap"] = wandb.Image(temp_img_path, caption=f"Metric Tensor (Epoch {epoch}, Step {step})")
                
                plt.close(fig)  # Close figure to avoid memory leaks
                
                # Remove temporary file after logging
                if os.path.exists(temp_img_path):
                    try:
                        os.remove(temp_img_path)
                    except:
                        pass
                
                # Log data to W&B
                wandb.log(wandb_log_data)
                
                print(f"Epoch {epoch}/{total_epochs}, Step {global_step}/{total_steps}, Loss: {loss:.4f}")
                
                # Wait before next update
                time.sleep(update_interval)
        
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

def run_deep_learning_model(args):
    """Run the deep learning model with W&B monitoring"""
    # Configure the training
    config = {
        'input_dim': 55,
        'embedding_dim': 5,
        'batch_size': 16,
        'num_epochs': args.epochs,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_samples': 1000,
        'data_type': 'sphere',
        'save_dir': './results',
        'device': None,
        'checkpoint_interval': 5,
        'visualization_interval': 10,
        'use_wandb': True,
        'wandb_project': args.project,
        'wandb_run_name': args.run_name or f"manifold-learning-{time.strftime('%Y%m%d-%H%M%S')}"
    }
    
    # Train the model with W&B monitoring
    print(f"Starting deep learning model training with W&B monitoring...")
    model, ricci_flow = train_model_with_visualization(config)
    
    return model, ricci_flow

def main():
    parser = argparse.ArgumentParser(description='W&B Monitoring for Manifold Deep Learning')
    parser.add_argument('--run-type', choices=['simulation', 'deep-learning', 'both'], default='both',
                      help='Which component to run (simulation, deep-learning, or both)')
    parser.add_argument('--project', default='manifold-deep-learning',
                      help='W&B project name')
    parser.add_argument('--run-name', default=None,
                      help='W&B run name (will be auto-generated if not provided)')
    parser.add_argument('--epochs', type=int, default=30,
                      help='Number of epochs for training')
    parser.add_argument('--steps-per-epoch', type=int, default=8,
                      help='Steps per epoch')
    parser.add_argument('--update-interval', type=float, default=0.2,
                      help='Update interval in seconds')
    
    args = parser.parse_args()
    
    sim_process = None
    
    try:
        if args.run_type in ['simulation', 'both']:
            # Run the C++ simulation
            sim_process = run_simulation()
        
        if args.run_type in ['simulation', 'both']:
            # Run the visualization test with W&B monitoring
            run_visualization_test(args)
        
        if args.run_type in ['deep-learning', 'both']:
            # Run the deep learning model with W&B monitoring
            run_deep_learning_model(args)
            
    finally:
        # Cleanup
        if sim_process is not None:
            print("Terminating simulation process...")
            sim_process.terminate()
            sim_process.wait()

if __name__ == '__main__':
    main() 