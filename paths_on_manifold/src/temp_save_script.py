
import bpy
import sys
import os

# Execute the main visualization script
exec(open(r"blender_update_script.py").read())

# Save the resulting scene to a blend file
bpy.ops.wm.save_as_mainfile(filepath=r"manifold_visualization_latest.blend")
print(f"Saved visualization to {os.path.abspath(r'manifold_visualization_latest.blend')}")
