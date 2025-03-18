#!/usr/bin/env python3
import os
import time
import subprocess
import sys
import argparse
import json
import platform
import shutil
from pathlib import Path

"""
This script monitors for changes in metric data files and updates the Blender
visualization in real-time, creating a live visualization of your manifold as it learns.

Usage:
python blender_live_monitor.py --blender_path "/path/to/blender" [--interval 1.0]
"""

# File paths
METRIC_FILE = "metric_data_current.csv"
CHRISTOFFEL_FILE = "christoffel_data_current.csv"
METADATA_FILE = "metric_metadata.json"
BLENDER_SCRIPT = "blender_script.py"

# Default paths for Blender by platform
DEFAULT_BLENDER_PATHS = {
    "darwin": ["/Applications/Blender.app/Contents/MacOS/Blender"],  # macOS
    "linux": ["/usr/bin/blender", "/snap/bin/blender"],             # Linux
    "win32": ["C:\\Program Files\\Blender Foundation\\Blender 3.5\\blender.exe", 
              "C:\\Program Files\\Blender Foundation\\Blender 3.4\\blender.exe",
              "C:\\Program Files\\Blender Foundation\\Blender 3.3\\blender.exe"]  # Windows
}

def find_blender_executable():
    """Find the Blender executable on the system."""
    system = platform.system().lower()
    
    if system == "darwin":
        return find_executable_in_paths(DEFAULT_BLENDER_PATHS.get("darwin", []))
    elif system == "linux":
        return find_executable_in_paths(DEFAULT_BLENDER_PATHS.get("linux", []))
    elif system == "windows" or system == "win32":
        return find_executable_in_paths(DEFAULT_BLENDER_PATHS.get("win32", []))
    else:
        print(f"Unsupported platform: {system}")
        return None

def find_executable_in_paths(paths):
    """Check if the executable exists in any of the provided paths."""
    for path in paths:
        if os.path.exists(path):
            return path
    return None

def get_file_modification_time(file_path):
    """Get the last modification time of a file."""
    if os.path.exists(file_path):
        return os.path.getmtime(file_path)
    return 0

def check_files_changed(last_check_times):
    """Check if any of the metric data files have changed."""
    files_to_check = {
        "metric": METRIC_FILE,
        "christoffel": CHRISTOFFEL_FILE,
        "metadata": METADATA_FILE
    }
    
    changed = False
    current_times = {}
    
    for key, file_path in files_to_check.items():
        current_time = get_file_modification_time(file_path)
        current_times[key] = current_time
        
        if current_time > last_check_times.get(key, 0):
            changed = True
    
    return changed, current_times

def get_metadata_info():
    """Get current epoch and loss information from metadata file."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            
            epoch = metadata.get('epoch', 'unknown')
            total_epochs = metadata.get('total_epochs', 'unknown')
            loss = metadata.get('loss', 'unknown')
            
            return f"Epoch {epoch}/{total_epochs}, Loss: {loss}"
        except Exception as e:
            return f"Error reading metadata: {e}"
    
    return "No metadata available"

def launch_blender(blender_path, script_path, args=None):
    """Launch Blender with the specified script."""
    if args is None:
        args = []
    
    # Ensure absolute paths
    script_path = os.path.abspath(script_path)
    
    # Build command line arguments
    cmd = [blender_path, "--background", "--python", script_path]
    cmd.extend(args)
    
    try:
        print(f"Launching Blender with command: {' '.join(cmd)}")
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error launching Blender: {e}")
        return None

def launch_blender_gui(blender_path, blend_file_path):
    """Launch Blender with GUI showing a specific .blend file."""
    # Ensure absolute path
    blend_file_path = os.path.abspath(blend_file_path)
    
    # Build command line arguments
    cmd = [blender_path, blend_file_path]
    
    try:
        print(f"Launching Blender GUI with file: {blend_file_path}")
        process = subprocess.Popen(cmd)
        return process
    except Exception as e:
        print(f"Error launching Blender GUI: {e}")
        return None

def make_blender_update_script(input_script_path, output_script_path):
    """Create a modified version of the Blender script that loads the latest data."""
    with open(input_script_path, 'r') as f:
        script_content = f.read()
    
    # Modify the script to disable FORCE_VISUALIZATION
    modified_content = script_content.replace("FORCE_VISUALIZATION = True", "FORCE_VISUALIZATION = False")
    
    with open(output_script_path, 'w') as f:
        f.write(modified_content)
    
    print(f"Created modified Blender script at {output_script_path}")

def save_visualization(blender_path, script_path, output_blend_file):
    """Run Blender with the script to update visualization and save as a .blend file."""
    temp_script_path = "temp_save_script.py"
    
    # Create a temporary script that loads the script and saves the result
    with open(temp_script_path, 'w') as f:
        f.write(f"""
import bpy
import sys
import os

# Execute the main visualization script
exec(open(r"{script_path}").read())

# Save the resulting scene to a blend file
bpy.ops.wm.save_as_mainfile(filepath=r"{output_blend_file}")
print(f"Saved visualization to {{os.path.abspath(r'{output_blend_file}')}}")
""")
    
    # Launch Blender to execute the temporary script
    cmd = [blender_path, "--background", "--python", temp_script_path]
    
    try:
        print("Creating and saving visualization to .blend file...")
        subprocess.run(cmd, check=True)
        
        # Clean up the temporary script
        os.remove(temp_script_path)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error saving visualization: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Monitor metric data files and update Blender visualization")
    parser.add_argument("--blender_path", help="Path to Blender executable")
    parser.add_argument("--interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--auto_launch", action="store_true", help="Automatically launch Blender GUI when changes detected")
    args = parser.parse_args()
    
    # Find or use provided Blender path
    blender_path = args.blender_path
    if not blender_path:
        blender_path = find_blender_executable()
        if not blender_path:
            print("Error: Blender executable not found. Please specify with --blender_path.")
            sys.exit(1)
    
    print(f"Using Blender at: {blender_path}")
    
    # Make sure the Blender script exists
    script_path = BLENDER_SCRIPT
    if not os.path.exists(script_path):
        print(f"Error: Blender script not found at {script_path}")
        sys.exit(1)
    
    # Create a modified version of the script that loads from files
    update_script_path = "blender_update_script.py"
    make_blender_update_script(script_path, update_script_path)
    
    # Initialize file monitoring
    last_check_times = {}
    output_blend_file = "manifold_visualization_latest.blend"
    blender_process = None
    
    print(f"Monitoring for changes in metric data files (interval: {args.interval}s)")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            # Check if files have changed
            changed, current_times = check_files_changed(last_check_times)
            
            if changed:
                metadata_info = get_metadata_info()
                print(f"\nDetected changes in metric data files - {metadata_info}")
                
                # Update the last check times
                last_check_times = current_times
                
                # Save the visualization as a .blend file
                if save_visualization(blender_path, update_script_path, output_blend_file):
                    if args.auto_launch:
                        # Close previous Blender instance if it exists
                        if blender_process and blender_process.poll() is None:
                            try:
                                blender_process.terminate()
                                print("Closed previous Blender instance")
                            except:
                                pass
                        
                        # Launch Blender with the updated visualization
                        blender_process = launch_blender_gui(blender_path, output_blend_file)
                
                print(f"Visualization updated - {metadata_info}")
                print(f"Open {output_blend_file} in Blender to see the latest visualization")
            
            # Wait before checking again
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    # Clean up
    if os.path.exists(update_script_path):
        os.remove(update_script_path)
    
    print("Monitoring script terminated")

if __name__ == "__main__":
    main() 