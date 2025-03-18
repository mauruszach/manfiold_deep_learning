import numpy as np
import torch
import os

class ManifoldExporter:
    """
    Class to export manifold data from Python model to CSV files for C++ visualization.
    """
    @staticmethod
    def export_metric_tensor(filename, model, grid_points=None):
        """
        Export the metric tensor to a CSV file for visualization in C++.
        
        Parameters:
        -----------
        filename : str
            Path to save the CSV file
        model : Model
            Trained manifold model
        grid_points : torch.Tensor or None
            Grid points to evaluate the metric at (if None, uses a default grid)
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Get the metric tensor
        with torch.no_grad():
            metric = model.embedding.get_metric().cpu().numpy()
            
            # Save to CSV
            np.savetxt(filename, metric, delimiter=',')
            
        print(f"Exported metric tensor to {filename}")
        return True
    
    @staticmethod
    def export_christoffel_symbols(filename, model, grid_points=None):
        """
        Export Christoffel symbols to a CSV file for C++ visualization.
        Format: i,j,k,value
        
        Parameters:
        -----------
        filename : str
            Path to save the CSV file
        model : Model
            Trained manifold model
        grid_points : torch.Tensor or None
            Grid points to evaluate the Christoffel symbols at
        """
        device = next(model.parameters()).device
        model.eval()
        
        with torch.no_grad():
            # Get the metric tensor
            metric = model.embedding.get_metric()
            
            # Calculate the inverse metric tensor
            try:
                metric_inv = torch.inverse(metric)
            except:
                # Add small regularization
                metric_reg = metric + torch.eye(metric.shape[0], device=metric.device) * 1e-6
                metric_inv = torch.inverse(metric_reg)
            
            # Calculate Christoffel symbols
            christoffel = model.embedding.christoffel_symbols(metric, metric_inv)
            christoffel_np = christoffel.cpu().numpy()
            
            # Export in the format i,j,k,value
            with open(filename, 'w') as f:
                for i in range(christoffel_np.shape[0]):
                    for j in range(christoffel_np.shape[1]):
                        for k in range(christoffel_np.shape[2]):
                            # Only write non-zero values to save space
                            if abs(christoffel_np[i, j, k]) > 1e-6:
                                f.write(f"{i},{j},{k},{christoffel_np[i, j, k]}\n")
            
        print(f"Exported Christoffel symbols to {filename}")
        return True
    
    @staticmethod
    def export_riemann_tensor(filename, model, grid_points=None):
        """
        Export Riemann tensor to a CSV file for C++ visualization.
        Format: i,j,k,l,value
        
        Parameters:
        -----------
        filename : str
            Path to save the CSV file
        model : Model
            Trained manifold model
        grid_points : torch.Tensor or None
            Grid points to evaluate the Riemann tensor at
        """
        device = next(model.parameters()).device
        model.eval()
        
        with torch.no_grad():
            # Get the metric tensor
            metric = model.embedding.get_metric()
            
            # Create a RicciCurvature calculator
            from_model = RicciCurvature(metric, device=device)
            
            # Get the Riemann tensor
            riemann = from_model.riemann_tensor
            riemann_np = riemann.cpu().numpy()
            
            # Export in the format i,j,k,l,value
            with open(filename, 'w') as f:
                for i in range(riemann_np.shape[0]):
                    for j in range(riemann_np.shape[1]):
                        for k in range(riemann_np.shape[2]):
                            for l in range(riemann_np.shape[3]):
                                # Only write non-zero values to save space
                                if abs(riemann_np[i, j, k, l]) > 1e-6:
                                    f.write(f"{i},{j},{k},{l},{riemann_np[i, j, k, l]}\n")
            
        print(f"Exported Riemann tensor to {filename}")
        return True
    
    @staticmethod
    def export_all(output_dir, model, grid_points=None):
        """
        Export all tensor data for C++ visualization.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the CSV files
        model : Model
            Trained manifold model
        grid_points : torch.Tensor or None
            Grid points to evaluate the tensors at
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export metric tensor
        metric_file = os.path.join(output_dir, "metric_tensor.csv")
        ManifoldExporter.export_metric_tensor(metric_file, model, grid_points)
        
        # Export Christoffel symbols
        christoffel_file = os.path.join(output_dir, "christoffel_symbols.csv")
        ManifoldExporter.export_christoffel_symbols(christoffel_file, model, grid_points)
        
        # Export Riemann tensor
        riemann_file = os.path.join(output_dir, "riemann_tensor.csv")
        ManifoldExporter.export_riemann_tensor(riemann_file, model, grid_points)
        
        print(f"Exported all tensor data to {output_dir}")
        return True


# Add this to the train_model function at the end
def export_model_for_visualization(model, config):
    """
    Export model data for C++ visualization.
    
    Parameters:
    -----------
    model : Model
        Trained manifold model
    config : dict
        Configuration dictionary
    """
    export_dir = os.path.join(config['save_dir'], 'cpp_export')
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"Exporting model data for C++ visualization to {export_dir}...")
    
    # Export tensor data
    ManifoldExporter.export_all(export_dir, model)
    
    print(f"Exported model data to {export_dir}")
    
    # Instruction for the user
    print("\nTo use with C++ visualization:")
    print(f"1. Copy files from {export_dir} to the directory where the C++ executable is located")
    print("2. Run the C++ visualization")
    print("3. Press 'M' to toggle between Python model data and traditional simulation")
    print("4. Press 'L' to reload the model data if needed")
    

# Modify the train_model function to include export at the end:
def train_model_with_export(config=None):
    """Modified version that includes export for C++ visualization"""
    model, ricci_flow = train_model(config)
    
    # Add export step
    export_model_for_visualization(model, config)
    
    return model, ricci_flow


# For command line execution, modify the main block to use the new function
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Train a manifold learning model')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration JSON file')
    parser.add_argument('--data_type', type=str, default='sphere', 
                        choices=['synthetic', 'sphere', 'torus', 'swiss_roll'],
                        help='Type of data to generate')
    parser.add_argument('--embedding_dim', type=int, default=5, help='Dimension of the manifold embedding')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--export_cpp', action='store_true', help='Export data for C++ visualization')
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Create config from command line arguments
        config = {
            'data_type': args.data_type,
            'embedding_dim': args.embedding_dim,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'save_dir': args.save_dir,
            'export_cpp': args.export_cpp
        }
    
    # Train the model with export option
    if config.get('export_cpp', False):
        model, ricci_flow = train_model_with_export(config)
    else:
        model, ricci_flow = train_model(config)