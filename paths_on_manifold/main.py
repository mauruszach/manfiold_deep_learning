#!/usr/bin/env python3
"""
Main entry point for Manifold Learning project.
This script provides a command-line interface to run various aspects of the project.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

def run_visualization():
    """Run the test visualization."""
    from visualization.test_visualization import main as visualization_main
    visualization_main()

def run_simulation():
    """Run the C++ simulation using the executable."""
    bin_path = project_root / "bin" / "simulation"
    if not bin_path.exists():
        print(f"Error: Simulation executable not found at {bin_path}")
        return
    
    os.system(str(bin_path))

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
    from src.path_analysis import main as path_analysis_main
    path_analysis_main()

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
    
    return parser

def main():
    """Main entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.visualize:
        run_visualization()
    elif args.simulate:
        run_simulation()
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
        print("  python main.py --visualize  # Run test visualization")
        print("  python main.py --simulate   # Run C++ simulation")
        print("  python main.py --blender    # Run Blender visualization")

if __name__ == "__main__":
    main() 