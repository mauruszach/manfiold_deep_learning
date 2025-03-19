#!/usr/bin/env python3
"""
Main entry point for Manifold Learning project.
This script provides a command-line interface to run various aspects of the project.
"""
# Set matplotlib backend to non-interactive to avoid GUI issues in threads
import matplotlib
matplotlib.use('Agg')

import os
import sys
import argparse
import subprocess
import multiprocessing
import threading
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def run_visualization():
    """Run the test visualization."""
    from visualization.test_visualization import main as visualization_main
    visualization_main()

def run_simulation(background=False, count=1):
    """
    Run the C++ simulation using the executable.
    
    Args:
        background (bool): If True, run in background (non-blocking)
        count (int): Number of parallel simulations to run
    """
    bin_path = project_root / "bin" / "simulation"
    if not bin_path.exists():
        print(f"Error: Simulation executable not found at {bin_path}")
        return
    
    if background:
        # Run in background using subprocess
        print(f"Starting simulation in background...")
        proc = subprocess.Popen([str(bin_path)], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE,
                         start_new_session=True)
        print("Simulation is running in the background. You can continue working.")
        return proc
        
    if count > 1:
        # Run multiple simulations in parallel
        print(f"Starting {count} parallel simulations...")
        processes = []
        for i in range(count):
            # Create a unique output file for each simulation
            out_file = f"simulation_output_{i}.txt"
            with open(out_file, 'w') as f:
                p = subprocess.Popen([str(bin_path)], 
                                    stdout=f,
                                    stderr=f)
                processes.append(p)
                print(f"  - Started simulation {i+1}, output: {out_file}")
        
        print(f"All {count} simulations are running concurrently.")
        print("Press Ctrl+C to terminate all simulations.")
        
        try:
            # Wait for all processes to complete
            for p in processes:
                p.wait()
            print("All simulations completed successfully.")
        except KeyboardInterrupt:
            print("Terminating simulations...")
            for p in processes:
                p.terminate()
            print("All simulations terminated.")
    else:
        # Run single simulation in foreground (blocking)
        print("Running simulation (this will block until completed)...")
        proc = subprocess.Popen([str(bin_path)])
        return proc

def run_tests():
    """Run the test suite."""
    from tests.run_test import main as test_main
    test_main()

def run_blender_visualization():
    """Run the Blender visualization script."""
    from visualization.blender_live_monitor import main as blender_main
    blender_main()

def run_path_analysis():
    """Run the path analysis module."""
    print("Running path analysis to learn a manifold from synthetic data...")
    from src.path_analysis import main as path_analysis_main
    model, ricci_flow = path_analysis_main()
    print("Path analysis complete. You can now run the simulation to visualize the learned manifold.")
    return model, ricci_flow

def run_interactive_learning():
    """
    Run path analysis and simulation concurrently, allowing real-time 
    visualization of the manifold training process.
    """
    print("Starting interactive manifold learning with real-time visualization...")

    # First, check if the simulation executable exists
    bin_path = project_root / "bin" / "simulation"
    if not bin_path.exists():
        print(f"Simulation executable not found at {bin_path}")
        print("Attempting to compile the simulation...")
        
        # Check if build script exists and is executable
        build_script = project_root / "build_simulation.sh"
        if build_script.exists():
            # Try to run the build script
            try:
                print("Running build script...")
                subprocess.run([str(build_script)], check=True)
                print("Build completed successfully!")
            except subprocess.CalledProcessError:
                print("Build failed. Please check that SFML and OpenGL are installed.")
                print("On macOS: brew install sfml")
                print("On Ubuntu: sudo apt-get install libsfml-dev libgl1-mesa-dev")
                return
            except Exception as e:
                print(f"Error running build script: {e}")
                return
        else:
            print("Build script not found. Please compile the simulation manually.")
            print("You can create the build script by running:")
            print("echo '#!/bin/bash' > build_simulation.sh")
            print("echo 'c++ -std=c++17 src/simulation.cpp -o bin/simulation -framework SFML -framework OpenGL -lsfml-graphics -lsfml-window -lsfml-system' >> build_simulation.sh")
            print("chmod +x build_simulation.sh")
            print("./build_simulation.sh")
            return
    
    # Ensure data directories exist
    print("Setting up data directories...")
    os.makedirs(project_root / "data", exist_ok=True)
    
    # Create placeholder files initially to make sure simulation loads correctly
    print("Creating initial placeholder data files...")
    import numpy as np
    # Create a 4x4 identity matrix for initial metric tensor
    placeholder_metric = np.eye(4)
    placeholder_metric[0, 0] = -1.0  # Minkowski metric for spacetime
    np.savetxt(project_root / "metric_tensor.csv", placeholder_metric, delimiter=',')
    np.savetxt(project_root / "data" / "metric_tensor.csv", placeholder_metric, delimiter=',')
    
    # Create empty Christoffel symbols file
    with open(project_root / "christoffel_symbols.csv", 'w') as f:
        f.write("0,0,0,0.0\n")
    with open(project_root / "data" / "christoffel_symbols.csv", 'w') as f:
        f.write("0,0,0,0.0\n")
    
    # Create initial metadata
    import json
    metadata = {
        'epoch': 0,
        'total_epochs': 50,
        'step': 0,
        'total_steps': 500,
        'loss': 0.0,
        'is_training': True
    }
    with open(project_root / "metric_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    with open(project_root / "data" / "metric_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Define a thread function to run the path analysis
    def run_training():
        print("Starting manifold learning thread...")
        # Set matplotlib to use non-interactive backend to avoid GUI issues in the thread
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        from src.path_analysis import main as path_analysis_main
        try:
            model, ricci_flow = path_analysis_main()
            print("Training complete! The simulation will continue to run.")
            print("You can close the simulation window when you're done.")
        except Exception as e:
            print(f"Error in training process: {e}")

    # Start the training in a separate thread
    training_thread = threading.Thread(target=run_training)
    training_thread.daemon = True  # Allow the program to exit even if thread is running
    
    # Start the C++ simulation as a subprocess
    print("Starting C++ simulation for real-time visualization...")
    try:
        # Start the simulation process
        simulation_process = subprocess.Popen([str(bin_path)])
        
        # Wait for the simulation window to initialize
        time.sleep(2)
        
        # Now start the training thread
        training_thread.start()
        
        print("\nInteractive learning mode active:")
        print("- The manifold is being trained in the background")
        print("- The simulation window shows real-time visualization")
        print("- Training data will be automatically loaded by the simulation")
        print("- If needed, press 'M' to toggle between model-based and traditional simulation")
        print("- Press 'L' to manually reload the model data if visualization is not updating")
        print("- Press Ctrl+C in this terminal to stop both processes")
        
        # Wait for the simulation to finish or user to interrupt
        simulation_process.wait()
        
    except KeyboardInterrupt:
        print("\nInterrupting interactive learning mode...")
    finally:
        # Clean up processes
        if 'simulation_process' in locals() and simulation_process.poll() is None:
            print("Terminating simulation...")
            simulation_process.terminate()
        
        print("Interactive learning session ended.")

def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Manifold Learning CLI")
    
    parser.add_argument(
        "--visualize", 
        action="store_true", 
        help="Run visualization test"
    )
    parser.add_argument(
        "--simulate", 
        action="store_true", 
        help="Run C++ simulation"
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run simulation in background (non-blocking)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel simulation instances to run (default: 1)"
    )
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run tests"
    )
    parser.add_argument(
        "--blender", 
        action="store_true", 
        help="Run Blender visualization"
    )
    parser.add_argument(
        "--analyze", 
        action="store_true", 
        help="Run path analysis"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run path analysis and simulation concurrently for real-time visualization"
    )
    
    return parser

def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.interactive:
        run_interactive_learning()
    elif args.visualize:
        run_visualization()
    elif args.simulate:
        run_simulation(background=args.background, count=args.parallel)
    elif args.test:
        run_tests()
    elif args.blender:
        run_blender_visualization()
    elif args.analyze:
        run_path_analysis()
    else:
        # If no arguments provided, show help
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --visualize                # Run test visualization")
        print("  python main.py --simulate                 # Run C++ simulation (blocking)")
        print("  python main.py --simulate --background    # Run simulation in background")
        print("  python main.py --simulate --parallel 4    # Run 4 simulation instances in parallel")
        print("  python main.py --interactive              # Run training with real-time visualization")
        print("  python main.py --blender                  # Run Blender visualization")

if __name__ == "__main__":
    main() 