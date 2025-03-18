#!/usr/bin/env python3
import os
import sys
import subprocess
import argparse
import time
import signal
import platform

"""
This script runs both the training model (test_visualization.py) and the Blender
visualization monitor (blender_live_monitor.py) simultaneously.

Usage:
python run_live_visualization.py --blender_path "/path/to/blender"
"""

def find_script_path(script_name):
    """Find the path to a script in the current directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, script_name)
    
    if os.path.exists(script_path):
        return script_path
    else:
        return None

def main():
    parser = argparse.ArgumentParser(description="Run manifold learning with live Blender visualization")
    parser.add_argument("--blender_path", help="Path to Blender executable")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval for visualization updates (seconds)")
    parser.add_argument("--auto_launch", action="store_true", help="Automatically launch Blender GUI when changes detected")
    args = parser.parse_args()
    
    # Find script paths
    training_script = find_script_path("test_visualization.py")
    monitor_script = find_script_path("blender_live_monitor.py")
    
    if not training_script:
        print("Error: Could not find test_visualization.py")
        sys.exit(1)
    
    if not monitor_script:
        print("Error: Could not find blender_live_monitor.py")
        sys.exit(1)
    
    print("Starting manifold learning with live Blender visualization...")
    
    # Start the training model in a separate process
    training_cmd = [sys.executable, training_script]
    training_process = subprocess.Popen(training_cmd)
    
    # Start the Blender visualization monitor in a separate process
    monitor_cmd = [sys.executable, monitor_script]
    if args.blender_path:
        monitor_cmd.extend(["--blender_path", args.blender_path])
    monitor_cmd.extend(["--interval", str(args.interval)])
    if args.auto_launch:
        monitor_cmd.append("--auto_launch")
    
    monitor_process = subprocess.Popen(monitor_cmd)
    
    print("Both processes started. Press Ctrl+C to stop.")
    
    # Handle graceful termination of both processes
    try:
        # Wait for both processes to complete
        training_process.wait()
        monitor_process.wait()
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        
        # Terminate processes gracefully
        if platform.system() == "Windows":
            # Windows requires different termination approach
            if training_process.poll() is None:
                training_process.terminate()
            if monitor_process.poll() is None:
                monitor_process.terminate()
        else:
            # Unix-like systems can use signals
            if training_process.poll() is None:
                os.kill(training_process.pid, signal.SIGINT)
            if monitor_process.poll() is None:
                os.kill(monitor_process.pid, signal.SIGINT)
        
        # Wait for processes to terminate
        try:
            training_process.wait(timeout=5)
            monitor_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing processes that didn't terminate gracefully...")
            if training_process.poll() is None:
                training_process.kill()
            if monitor_process.poll() is None:
                monitor_process.kill()
    
    print("All processes terminated. Visualization complete.")

if __name__ == "__main__":
    main() 