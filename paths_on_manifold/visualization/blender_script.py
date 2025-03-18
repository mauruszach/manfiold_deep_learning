import bpy
import numpy as np
import math
import os
import json
import mathutils
import traceback

# Configuration options - adjust these if Blender is crashing
COMPATIBILITY_MODE = True  # Set to True to disable features that might cause crashes
SKIP_PARTICLES = True      # Set to True to skip particle creation (most crash-prone part)
MAX_GRID_SIZE = 80         # Increased for better granularity
DISABLE_MODIFIERS = True   # Set to True to disable subdivision modifiers
SHOW_COORDINATE_LINES = True  # Set to True to display coordinate grid lines on the manifold
COORDINATE_LINE_DENSITY = 24  # Increased density of coordinate lines
CURVATURE_INTENSITY = 25.0    # Dramatically increased curvature strength for extreme effect
FORCE_VISUALIZATION = True    # Force direct visualization even if data is loaded - set to False for real-time updates
PURE_BLACK_BACKGROUND = True  # Force pure black background with no ambient lighting

def clear_scene():
    """Clear existing objects in Blender scene."""
    # Deselect all first
    bpy.ops.object.select_all(action='DESELECT')
    
    # Select and delete all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clear all materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    
    # Clear all textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture)
    
    # Clear all images
    for image in bpy.data.images:
        bpy.data.images.remove(image)
    
    # Set up a completely pure black background with no ambient lighting
    world = bpy.data.worlds.new('PureBlackWorld')
    bpy.context.scene.world = world
    world.use_nodes = True
    
    # Clear any existing nodes in the world tree
    for node in world.node_tree.nodes:
        world.node_tree.nodes.remove(node)
    
    # Create a simple background and output node
    background = world.node_tree.nodes.new('ShaderNodeBackground')
    output = world.node_tree.nodes.new('ShaderNodeOutputWorld')
    
    # Set background to absolute pure black with no emission
    background.inputs[0].default_value = (0, 0, 0, 1)  # Pure black
    background.inputs[1].default_value = 0.0  # Zero strength for complete darkness
    
    # Connect nodes
    world.node_tree.links.new(background.outputs[0], output.inputs[0])
    
    print("Scene cleared and absolute pure black background set")

def load_metric_data(file_path):
    """Load metric tensor data from CSV file with diagnostics."""
    try:
        data = np.loadtxt(file_path, delimiter=',')
        print(f"Loaded metric data shape: {data.shape}")
        
        # Reshape if needed - 4x4 metric tensor
        if len(data.shape) == 1 and data.shape[0] == 16:
            data = data.reshape(4, 4)
            print("Reshaped 1D array to 4x4 metric tensor")
        
        # Print diagnostic information
        print(f"Metric tensor summary:")
        print(f"  Shape: {data.shape}")
        print(f"  Determinant: {np.linalg.det(data)}")
        print(f"  Trace: {np.trace(data)}")
        
        try:
            eigenvalues = np.linalg.eigvalsh(data)
            print(f"  Eigenvalues: {eigenvalues}")
        except:
            print("  Could not compute eigenvalues")
        
        print(f"  Sample values: \n{data}")
        
        return data
    except Exception as e:
        print(f"Error loading metric data: {str(e)}")
        # Create a default metric tensor if loading fails
        print("Creating default metric tensor")
        return np.array([
            [-1.0, -0.24, -0.28, -0.28],
            [-0.24, 0.06, -0.14, -0.26],
            [-0.28, -0.14, 0.01, -0.21],
            [-0.28, -0.26, -0.21, 0.07]
        ])

def load_christoffel_data(file_path):
    """Load Christoffel symbols from CSV file with diagnostics."""
    try:
        data = np.loadtxt(file_path, delimiter=',')
        print(f"Loaded Christoffel data shape: {data.shape}")
        
        # Reshape if needed to 4x4x4 tensor
        if len(data.shape) == 1:
            data = data.reshape(4, 4, 4)
            print("Reshaped 1D array to 4x4x4 Christoffel tensor")
        
        print(f"Christoffel symbols summary:")
        print(f"  Shape: {data.shape}")
        print(f"  Max value: {np.max(data)}")
        print(f"  Min value: {np.min(data)}")
        
        return data
    except Exception as e:
        print(f"Error loading Christoffel data: {str(e)}")
        # Create default data if loading fails
        print("Creating default Christoffel symbols")
        return np.zeros((4, 4, 4))

def load_metadata(file_path):
    """Load metadata from JSON file."""
    try:
        with open(file_path, 'r') as f:
            metadata = json.load(f)
            print(f"Loaded metadata: {metadata}")
            return metadata
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return {"epoch": "unknown", "loss": "unknown"}

def extract_ricci_curvature(metric):
    """Calculate Ricci scalar curvature from metric tensor."""
    try:
        # Calculate determinant
        det = np.linalg.det(metric)
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(metric)
        
        # Calculate trace
        trace = np.trace(metric)
        
        # Calculate curvature (simplified formula)
        # In real GR, this would involve derivatives of the metric
        curvature = -det * (1.0 + abs(trace)) * (max(eigenvalues) - min(eigenvalues))
        
        # Ensure we get a reasonable value
        if abs(curvature) < 0.01:
            curvature = -0.1  # Minimum effect for visualization
        
        print(f"Calculated curvature: {curvature}")
        return curvature
    except Exception as e:
        print(f"Error in curvature calculation: {str(e)}")
        return -0.1

def create_spacetime_grid(metric_data, size=10, divisions=40):
    """Create a grid representing spacetime with the given metric tensor."""
    print("Creating spacetime grid...")
    
    # Adjust divisions based on compatibility mode
    if COMPATIBILITY_MODE:
        divisions = min(divisions, MAX_GRID_SIZE)
    else:
        divisions = min(divisions, 100)  # Higher resolution when not in compatibility mode
    
    print(f"Creating grid with {divisions} subdivisions")
    
    # Create grid mesh
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=divisions,
        y_subdivisions=divisions,
        size=size
    )
    
    grid = bpy.context.active_object
    grid.name = "SpacetimeGrid"
    
    # Create material
    mat = bpy.data.materials.new(name="GridMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Get the principled BSDF node
    principled = nodes.get('Principled BSDF')
    output = nodes.get('Material Output')
    
    # Clear all links and create a simple color ramp setup
    mat.node_tree.links.clear()
    
    # Add nodes for enhanced material
    geometry = nodes.new(type='ShaderNodeNewGeometry')
    separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    
    # Set up a red-focused color ramp for better curvature visualization
    color_ramp.color_ramp.elements.clear()  # Remove default elements
    
    # Add elements for red-focused gradient
    e1 = color_ramp.color_ramp.elements.new(0.0)
    e1.color = (0.8, 0.0, 0.0, 1.0)  # Deep red for high curvature
    
    e2 = color_ramp.color_ramp.elements.new(0.3)
    e2.color = (0.9, 0.3, 0.0, 1.0)  # Orange-red
    
    e3 = color_ramp.color_ramp.elements.new(0.6)
    e3.color = (1.0, 0.5, 0.5, 1.0)  # Light red
    
    e4 = color_ramp.color_ramp.elements.new(0.8)
    e4.color = (1.0, 0.7, 0.7, 1.0)  # Pink
    
    e5 = color_ramp.color_ramp.elements.new(1.0)
    e5.color = (1.0, 0.9, 0.9, 1.0)  # Near white - low curvature
    
    # Connect nodes
    links.new(geometry.outputs['Position'], separate_xyz.inputs[0])
    links.new(separate_xyz.outputs['Z'], color_ramp.inputs[0])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs[0], output.inputs[0])
    
    # Add nice material properties
    if 'Specular' in principled.inputs:
        principled.inputs['Specular'].default_value = 0.7
    elif 'Specular IOR' in principled.inputs:
        principled.inputs['Specular IOR'].default_value = 0.7
    
    principled.inputs['Roughness'].default_value = 0.2
    principled.inputs['Metallic'].default_value = 0.2
    
    grid.data.materials.append(mat)
    
    return grid

def deform_grid_with_metric(grid, metric_data):
    """Apply metric-based deformation to the grid."""
    if grid is None:
        print("No grid to deform")
        return 0, 0

    print("Applying metric-based deformation...")
    
    try:
        # Convert metric data to numpy for easier handling
        metric_np = np.array(metric_data)
        
        # Extract curvature scalar
        curvature = extract_ricci_curvature(metric_data)
        print(f"Extracted curvature: {curvature}")
        
        # DRAMATICALLY increased scale factor for deformation - extreme curvature
        scale_factor = -CURVATURE_INTENSITY  # Very pronounced curvature
        
        # Keep track of min and max z values for color mapping
        min_z = float('inf')
        max_z = float('-inf')
        
        print(f"Applying deformation with scale factor: {scale_factor}")
        
        # First pass: calculate z values
        for i, vertex in enumerate(grid.data.vertices):
            x, y = vertex.co.x, vertex.co.y
            
            # Calculate radial distance from center
            r = math.sqrt(x*x + y*y)
            if r < 0.001:  # Avoid division by zero
                r = 0.001  # Small value instead of continuing
                
            # Calculate z-offset based on curvature - using a much stronger effect formula
            try:
                # Use a more dramatic gravity well model for visualization
                # This function creates a very deep well in the center
                # z_offset = scale_factor * math.exp(-r*r/8)  # Gaussian curve
                
                # Alternative formula for stronger visual effect at the edges
                z_offset = scale_factor * (1.0 / (0.5 + r*r*0.2))  # Stronger hyperbolic effect
                
                # For a more accurate model with actual metric data:
                # Use determinant as a measure of volume distortion
                det = metric_np[0][0] * metric_np[1][1] - metric_np[0][1] * metric_np[1][0]
                if abs(det) > 0.01:  # avoid division by very small determinants
                    z_offset *= (1.0 - det)
                
                # Apply the deformation
                vertex.co.z = z_offset
                
                # Update min and max for color mapping
                min_z = min(min_z, z_offset)
                max_z = max(max_z, z_offset)
                
            except Exception as e:
                print(f"Error in deformation calculation: {e}")
                vertex.co.z = 0
                
        # Update mesh
        grid.data.update()
        
        print(f"Deformation applied. Z range: {min_z} to {max_z}")
        return min_z, max_z
        
    except Exception as e:
        print(f"Error deforming grid: {e}")
        traceback.print_exc()
        return 0, 0

def apply_curvature_color_gradient(grid, min_z, max_z):
    """Apply an enhanced color gradient based on curvature values."""
    # Create a vertex group to store curvature values
    if "CurvatureValues" not in grid.vertex_groups:
        curvature_group = grid.vertex_groups.new(name="CurvatureValues")
    else:
        curvature_group = grid.vertex_groups["CurvatureValues"]
    
    # Normalize z values to 0-1 range for proper color mapping
    z_range = max_z - min_z
    if z_range == 0:
        z_range = 1.0  # Prevent division by zero
    
    # Store normalized curvature values in vertex group
    for i, vertex in enumerate(grid.data.vertices):
        normalized_value = (vertex.co.z - min_z) / z_range
        # Invert the value for color gradient (deeper curvature = more intense color)
        normalized_value = 1.0 - normalized_value
        curvature_group.add([i], normalized_value, 'REPLACE')
    
    # Create a new material for the grid with advanced curvature visualization
    mat = bpy.data.materials.new(name="CurvatureGradientMaterial")
    grid.data.materials.clear()  # Remove any existing materials
    grid.data.materials.append(mat)
    mat.use_nodes = True
    
    # Get material nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes for advanced shader network
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    attribute = nodes.new('ShaderNodeAttribute')
    color_ramp = nodes.new('ShaderNodeValToRGB')
    normal = nodes.new('ShaderNodeNormal')
    bump = nodes.new('ShaderNodeBump')
    noise_texture = nodes.new('ShaderNodeTexNoise')
    mix_rgb = nodes.new('ShaderNodeMixRGB')
    
    # Set up attribute to read vertex group weights
    attribute.attribute_name = "CurvatureValues"
    
    # Set up colorful gradient for curvature - blue to purple to red
    color_ramp.color_ramp.elements.clear()  # Remove default elements
    
    # Lowest curvature - blue
    e1 = color_ramp.color_ramp.elements.new(0.0)
    e1.color = (0.0, 0.2, 0.8, 1.0)  # Deep blue for lowest curvature
    
    # Low-medium curvature - cyan
    e2 = color_ramp.color_ramp.elements.new(0.2)
    e2.color = (0.0, 0.6, 0.8, 1.0)  # Cyan-blue
    
    # Medium curvature - magenta
    e3 = color_ramp.color_ramp.elements.new(0.4)
    e3.color = (0.6, 0.0, 0.8, 1.0)  # Purple
    
    # Medium-high curvature - orange
    e4 = color_ramp.color_ramp.elements.new(0.7)
    e4.color = (0.9, 0.4, 0.0, 1.0)  # Orange
    
    # Highest curvature - deep red
    e5 = color_ramp.color_ramp.elements.new(1.0)
    e5.color = (0.8, 0.0, 0.0, 1.0)  # Deep red for highest curvature
    
    # Set up noise for surface detail
    noise_texture.inputs['Scale'].default_value = 30.0
    noise_texture.inputs['Detail'].default_value = 10.0
    noise_texture.inputs['Roughness'].default_value = 0.7
    
    # Mix noise with color gradient
    mix_rgb.blend_type = 'OVERLAY'
    mix_rgb.inputs['Fac'].default_value = 0.10  # Subtle effect
    
    # Set up bump mapping for surface detail
    bump.inputs['Strength'].default_value = 0.2
    bump.inputs['Distance'].default_value = 0.02
    
    # Set material properties for a scientific visualization look
    if 'Specular' in principled.inputs:
        principled.inputs['Specular'].default_value = 0.9
    elif 'Specular IOR' in principled.inputs:
        principled.inputs['Specular IOR'].default_value = 0.9
    
    principled.inputs['Roughness'].default_value = 0.1
    principled.inputs['Metallic'].default_value = 0.8
    
    # Add emission for better visibility against black
    if 'Emission' in principled.inputs:
        principled.inputs['Emission'].default_value = (0.05, 0.05, 0.05, 1.0)  # Slight emission
        principled.inputs['Emission Strength'].default_value = 0.2  # Subtle glow
    
    # Properly handle properties that might not exist in all Blender versions
    if 'Clearcoat' in principled.inputs:
        principled.inputs['Clearcoat'].default_value = 0.5
    if 'Clearcoat Roughness' in principled.inputs:
        principled.inputs['Clearcoat Roughness'].default_value = 0.1
    if 'Sheen' in principled.inputs:
        principled.inputs['Sheen'].default_value = 0.1
    
    # Connect nodes
    links.new(attribute.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(noise_texture.outputs['Fac'], mix_rgb.inputs[2])
    links.new(color_ramp.outputs['Color'], mix_rgb.inputs[1])
    links.new(mix_rgb.outputs['Color'], principled.inputs['Base Color'])
    links.new(noise_texture.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], principled.inputs['Normal'])
    links.new(principled.outputs[0], output.inputs[0])
    
    print("Applied enhanced color gradient for curvature visualization")

def create_particles(num_particles=20, grid_object=None, metric_data=None):
    """Create particles to visualize geodesic paths."""
    print(f"Creating {num_particles} particles...")
    particles = []
    
    # Create particle collection
    if "Particles" not in bpy.data.collections:
        particle_collection = bpy.data.collections.new("Particles")
        bpy.context.scene.collection.children.link(particle_collection)
    else:
        particle_collection = bpy.data.collections["Particles"]
    
    # Create particles
    for i in range(num_particles):
        # Create a sphere for the particle
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.15, 
            segments=16, 
            rings=16
        )
        particle = bpy.context.active_object
        particle.name = f"Particle_{i}"
        
        # Move to particle collection
        bpy.ops.object.move_to_collection(collection_index=bpy.data.collections.find("Particles") + 1)
        
        # Create glossy material for the particle
        mat = bpy.data.materials.new(name=f"ParticleMaterial_{i}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Clear existing nodes
        for node in nodes:
            nodes.remove(node)
        
        # Create nodes for a shiny material
        output = nodes.new('ShaderNodeOutputMaterial')
        principled = nodes.new('ShaderNodeBsdfPrincipled')
        
        # Set random color based on position
        h = i / num_particles  # Hue
        s = 0.8  # Saturation
        v = 1.0  # Value
        
        # HSV to RGB conversion
        h_i = int(h * 6)
        f = h * 6 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        # Set material properties
        principled.inputs['Base Color'].default_value = (r, g, b, 1.0)
        if 'Specular' in principled.inputs:
            principled.inputs['Specular'].default_value = 1.0
        elif 'Specular IOR' in principled.inputs:
            principled.inputs['Specular IOR'].default_value = 1.0
        principled.inputs['Roughness'].default_value = 0.1
        principled.inputs['Metallic'].default_value = 0.8
        
        # Connect nodes
        mat.node_tree.links.new(principled.outputs[0], output.inputs[0])
        
        # Assign material
        particle.data.materials.append(mat)
        
        # Position randomly with higher starting position
        angle = 2 * math.pi * (i / num_particles)
        radius = 4.0 * (0.5 + 0.5 * np.random.random())  # Random radius between 2-4
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        z = 4.0 + 2.0 * np.random.random()  # Start above the grid
        
        particle.location = (x, y, z)
        
        # Add a trail renderer
        trail = create_particle_trail(particle, (r, g, b))
        
        particles.append(particle)
    
    print("Particle creation complete")
    return particles

def create_particle_trail(particle, color):
    """Create a trail effect for a particle."""
    # Create a curve for the trail
    curve_data = bpy.data.curves.new('TrailCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2
    curve_data.bevel_depth = 0.03
    
    # Create the spline for the curve
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(count=9)  # 10 points including the first one
    
    # Position points in a line behind the particle
    for i, point in enumerate(spline.bezier_points):
        # Set position slightly behind particle
        point.co = (
            particle.location.x - i * 0.05, 
            particle.location.y - i * 0.05, 
            particle.location.z - i * 0.05
        )
        point.handle_left_type = 'AUTO'
        point.handle_right_type = 'AUTO'
    
    # Create curve object
    trail_obj = bpy.data.objects.new("Trail", curve_data)
    
    # Create material for trail
    mat = bpy.data.materials.new(name="TrailMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes for trail material
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    
    # Set color with emission
    r, g, b = color
    emission.inputs['Color'].default_value = (r, g, b, 1.0)
    emission.inputs['Strength'].default_value = 2.0
    
    # Connect nodes
    mat.node_tree.links.new(emission.outputs[0], output.inputs[0])
    
    # Assign material
    trail_obj.data.materials.append(mat)
    
    # Link to same collection as particle
    bpy.data.collections["Particles"].objects.link(trail_obj)
    
    return trail_obj

def setup_animation(particles, christoffel_data, metric_data, frames=250):
    """Set up improved animation for particles following geodesics."""
    print("Setting up animation...")
    
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames
    
    # Create curve guides for better visualization
    curve_guides = []
    
    for particle in particles:
        # Create animation data
        particle.animation_data_create()
        particle.animation_data.action = bpy.data.actions.new(name=f"{particle.name}_Action")
        
        # Create keyframes for location
        fcurves = []
        for i in range(3):  # x, y, z
            fcurve = particle.animation_data.action.fcurves.new(data_path=f"location", index=i)
            fcurves.append(fcurve)
        
        # Initial conditions
        x = particle.location.x
        y = particle.location.y
        z = particle.location.z
        
        # More dynamic initial velocities
        if "orbit" in particle.name.lower():
            # Create orbital motion
            dist = math.sqrt(x**2 + y**2)
            speed = 0.15
            vx = -y / dist * speed
            vy = x / dist * speed
            vz = 0
        else:
            # Random initial velocity with some gravitational bias
            vx = np.random.random() * 0.1 - 0.05
            vy = np.random.random() * 0.1 - 0.05
            vz = np.random.random() * 0.05 - 0.10  # Slight downward bias
        
        # Numerical integration of geodesic equations
        dt = 0.05  # Time step
        positions = []
        position = [x, y, z]
        velocity = [vx, vy, vz]
        
        # Extract scalar curvature for force calculation
        curvature = extract_ricci_curvature(metric_data)
        
        # Get metric components for calculations
        if isinstance(metric_data, np.ndarray) and metric_data.shape[0] >= 4:
            g00 = metric_data[0, 0] if metric_data[0, 0] != 0 else -1.0
            g11 = metric_data[1, 1] if metric_data[1, 1] != 0 else 0.05
            g22 = metric_data[2, 2] if metric_data[2, 2] != 0 else 0.01
            g33 = metric_data[3, 3] if metric_data[3, 3] != 0 else 0.07
        else:
            g00, g11, g22, g33 = -1.0, 0.05, 0.01, 0.07
        
        for frame in range(frames):
            positions.append(position.copy())
            
            # Enhanced geodesic path simulation
            # Calculate distance from center
            r_squared = position[0]**2 + position[1]**2
            r = math.sqrt(r_squared) if r_squared > 0 else 0.01
            
            # Calculate grid height at current position
            x_norm = position[0] / 5.0
            y_norm = position[1] / 5.0
            grid_z = curvature * -1.0 * (x_norm**2 + y_norm**2)
            
            # Calculate gravitational force (proportional to gradient of metric)
            # This is a simplified version of the geodesic equation
            ax = -curvature * position[0] * 0.3 * abs(g00)
            ay = -curvature * position[1] * 0.3 * abs(g11)
            
            # Vertical acceleration based on height difference and metric
            height_diff = position[2] - grid_z
            az = -0.4 * height_diff * (abs(g22) + abs(g33))
            
            # Apply asymmetric effects from off-diagonal terms
            if isinstance(metric_data, np.ndarray) and metric_data.shape[0] >= 4:
                ax += 0.1 * (metric_data[0, 1] * position[1] + 
                             metric_data[0, 2] * position[2])
                ay += 0.1 * (metric_data[1, 0] * position[0] + 
                             metric_data[1, 2] * position[2])
            
            # Update velocity (verlet integration)
            velocity[0] += ax * dt
            velocity[1] += ay * dt
            velocity[2] += az * dt
            
            # Apply damping
            damping = 0.997
            velocity[0] *= damping
            velocity[1] *= damping
            velocity[2] *= damping
            
            # Update position
            position[0] += velocity[0] * dt
            position[1] += velocity[1] * dt
            position[2] += velocity[2] * dt
            
            # Ensure particles don't go through the grid
            if position[2] < grid_z + 0.2:
                position[2] = grid_z + 0.2
                velocity[2] = -velocity[2] * 0.6  # Bounce with energy loss
        
        # Create keyframes from positions
        for i, frame_pos in enumerate(positions):
            frame = i + 1
            for axis in range(3):
                fcurves[axis].keyframe_points.insert(frame, frame_pos[axis])
                
                # Improve interpolation
                keyframe = fcurves[axis].keyframe_points[-1]
                keyframe.interpolation = 'BEZIER'
                keyframe.easing = 'EASE_IN_OUT'
        
        # Also animate the trail
        trail_obj = None
        for obj in bpy.data.collections["Particles"].objects:
            if obj.name.startswith("Trail") and obj.data.type == 'CURVE':
                trail_obj = obj
                break
        
        if trail_obj:
            animate_trail(trail_obj, positions, frames)
    
    print("Animation setup complete")

def animate_trail(trail_obj, positions, frames):
    """Animate the trail to follow the particle."""
    # Create animation data
    trail_obj.animation_data_create()
    trail_obj.animation_data.action = bpy.data.actions.new(name=f"{trail_obj.name}_Action")
    
    spline = trail_obj.data.splines[0]
    num_points = len(spline.bezier_points)
    
    # For each frame, update trail points
    for frame in range(1, frames+1):
        # Skip some frames for efficiency
        if frame % 5 != 0 and frame != 1 and frame != frames:
            continue
            
        # Set current frame
        bpy.context.scene.frame_set(frame)
        
        # Get past positions (if available)
        past_positions = []
        for i in range(num_points):
            if frame - i > 0 and frame - i <= len(positions):
                past_positions.append(positions[frame - i - 1])
            else:
                # If not enough history, use the earliest available
                if len(positions) > 0:
                    past_positions.append(positions[0])
                else:
                    past_positions.append((0, 0, 0))
        
        # Update spline points
        for i, point in enumerate(spline.bezier_points):
            if i < len(past_positions):
                point.co = past_positions[i]
                # Create keyframe
                point.keyframe_insert(data_path="co")

def setup_scene_lighting():
    """Setup lighting for better visualization with minimal glare."""
    print("Setting up lighting...")
    
    # Create main directional light (sun) with much lower energy to eliminate glare
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.data.energy = 0.7  # Significantly reduced brightness to eliminate glare
    sun.rotation_euler = (math.radians(45), 0, math.radians(45))
    
    # Create fill light from opposite direction
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    fill = bpy.context.active_object
    fill.data.energy = 0.5  # Very soft fill light
    fill.rotation_euler = (math.radians(45), 0, math.radians(225))
    
    # Add a third light to ensure grid visibility
    bpy.ops.object.light_add(type='SUN', location=(0, 0, -10))
    bottom_light = bpy.context.active_object
    bottom_light.data.energy = 0.3  # Very subtle light from below
    bottom_light.rotation_euler = (math.radians(-30), 0, math.radians(0))
    
    # Add a fourth light to improve surface visibility
    bpy.ops.object.light_add(type='SUN', location=(0, 10, 5))
    side_light = bpy.context.active_object
    side_light.data.energy = 0.4  # Gentle side light
    side_light.rotation_euler = (math.radians(15), math.radians(-90), 0)
    
    # Force absolute pure black world background
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("PureBlackWorld")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    
    # Clear any existing nodes
    for node in world.node_tree.nodes:
        world.node_tree.nodes.remove(node)
    
    # Add background node for black background - explicitly zero emission
    output = world.node_tree.nodes.new(type='ShaderNodeOutputWorld')
    background = world.node_tree.nodes.new(type='ShaderNodeBackground')
    
    # Set background to absolute pure black with zero ambient lighting
    background.inputs[0].default_value = (0, 0, 0, 1)  # Pure black
    background.inputs[1].default_value = 0.0  # No strength (pure black)
    
    # Connect nodes
    world.node_tree.links.new(background.outputs[0], output.inputs[0])
    
    print("Lighting setup complete with minimal glare")

def create_info_text(metadata):
    """Create text object showing metadata."""
    print("Creating info text...")
    
    # Create text object
    bpy.ops.object.text_add(location=(-4, 4, 0.5))
    text_obj = bpy.context.active_object
    text_obj.name = "InfoText"
    
    # Set up text content
    epoch = metadata.get('epoch', 'unknown')
    loss = metadata.get('loss', 'unknown')
    determinant = metadata.get('metric_determinant', 'unknown')
    
    text_obj.data.body = f"Manifold Visualization\n"
    text_obj.data.body += f"Epoch: {epoch}\n"
    text_obj.data.body += f"Loss: {loss}\n"
    text_obj.data.body += f"Det(g): {determinant}"
    
    # Format text
    text_obj.data.size = 0.4
    text_obj.data.extrude = 0.02
    
    # Create material for the text
    mat = bpy.data.materials.new(name="TextMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Set color - use safer property access
    principled = nodes.get('Principled BSDF')
    if principled:
        principled.inputs['Base Color'].default_value = (1.0, 0.8, 0.2, 1.0)
        if 'Specular' in principled.inputs:
            principled.inputs['Specular'].default_value = 0.5
        elif 'Specular IOR' in principled.inputs:
            principled.inputs['Specular IOR'].default_value = 0.5
        if 'Metallic' in principled.inputs:
            principled.inputs['Metallic'].default_value = 1.0
    
    text_obj.data.materials.append(mat)
    
    print("Info text created")
    return text_obj

def create_test_visualization():
    """Create a test visualization with known curvature."""
    print("Creating test visualization with EXTREME curvature...")
    
    # Create a grid with appropriate size and resolution
    divisions = min(80, MAX_GRID_SIZE) if COMPATIBILITY_MODE else 100
    size = 10.0
    
    # We're going to create a mesh by hand for complete control
    print(f"Creating custom grid with {divisions} subdivisions")
    
    verts = []
    faces = []
    
    # Create vertices in a grid pattern
    scale_factor = -CURVATURE_INTENSITY
    
    # Create vertices with a controlled z-coordinate (curvature)
    for i in range(divisions + 1):
        for j in range(divisions + 1):
            # Calculate normalized position
            x = -size/2.0 + i * size / divisions
            y = -size/2.0 + j * size / divisions
            
            # Calculate radial distance from center
            r = math.sqrt(x*x + y*y)
            if r < 0.001:
                r = 0.001
            
            # Apply strong curve formula - smoother transition
            z = scale_factor * (1.0 / (0.5 + r*r*0.15))
            
            verts.append((x, y, z))
    
    # Create faces as quads
    for i in range(divisions):
        for j in range(divisions):
            # Calculate vertex indices for this quad
            v1 = i * (divisions + 1) + j
            v2 = i * (divisions + 1) + (j + 1)
            v3 = (i + 1) * (divisions + 1) + (j + 1)
            v4 = (i + 1) * (divisions + 1) + j
            
            # Create the face
            faces.append((v1, v2, v3, v4))
    
    # Create the mesh
    mesh = bpy.data.meshes.new("ManifoldMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    # Create the object
    grid = bpy.data.objects.new("SpacetimeGrid", mesh)
    bpy.context.collection.objects.link(grid)
    
    # Calculate min/max z values for normalization
    z_values = [v[2] for v in verts]
    min_z = min(z_values)
    max_z = max(z_values)
    print(f"Z range: {min_z} to {max_z}")
    
    # Calculate normalization factor
    z_range = max_z - min_z
    if z_range == 0:
        z_range = 1.0
    
    # Create NEW APPROACH: DIRECT VERTEX COLORING
    # This method bypasses material systems and directly assigns colors
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    
    color_layer = mesh.vertex_colors.active
    
    # Assign colors directly to vertices for faces
    i = 0
    for poly in mesh.polygons:
        for idx in poly.loop_indices:
            # Get the vertex for this face corner
            v_idx = mesh.loops[idx].vertex_index
            vertex = mesh.vertices[v_idx]
            
            # Normalize height (0 = deepest/center, 1 = highest/edge)
            norm_height = 1.0 - ((vertex.co.z - min_z) / z_range)
            
            # PURE RED GRADIENT: from darkest red to light red/pink
            # Apply a gamma correction for better visual gradient
            gamma_corrected = pow(norm_height, 0.7)  # Gamma correction
            
            if gamma_corrected < 0.2:
                # Deepest center - dark rich red
                r, g, b = 0.7, 0.0, 0.0
            elif gamma_corrected < 0.4:
                # Deep red
                r, g, b = 0.85, 0.05, 0.05
            elif gamma_corrected < 0.6:
                # Medium red
                r, g, b = 1.0, 0.1, 0.1
            elif gamma_corrected < 0.8:
                # Lighter red
                r, g, b = 1.0, 0.3, 0.3
            else:
                # Edges - pale red/pink
                r, g, b = 1.0, 0.6, 0.6
            
            # Assign the color
            color_layer.data[i].color = (r, g, b, 1.0)
            i += 1
    
    # Create a simple emission material to make colors glow without needing strong lights
    mat = bpy.data.materials.new(name="PureRedEmissionMaterial")
    mat.use_nodes = True
    
    # Get material nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes for very simple emission material
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    vertex_color = nodes.new('ShaderNodeVertexColor')
    vertex_color.layer_name = color_layer.name
    
    # Connect nodes
    links.new(vertex_color.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs[0], output.inputs[0])
    
    # Increase emission strength to compensate for reduced lighting
    emission.inputs['Strength'].default_value = 2.5
    
    # Assign material to the grid
    grid.data.materials.append(mat)
    
    # Create a mathematically correct Schwarzschild-like metric for reference
    r_s = 2.0  # Schwarzschild radius
    test_metric = [
        [-(1 - r_s/10), 0.0, 0.0, 0.0],
        [0.0, 1/(1 - r_s/10), 0.0, 0.0],
        [0.0, 0.0, 10**2, 0.0],
        [0.0, 0.0, 0.0, 10**2 * (math.sin(math.pi/4))**2]
    ]
    
    # Create coordinate lines with higher brightness
    lines = create_direct_coordinate_lines(grid, divisions=COORDINATE_LINE_DENSITY)
    
    # Create test particles to visualize geodesics
    particles = []
    if not SKIP_PARTICLES:
        particles = create_particles(num_particles=25, grid_object=grid, metric_data=test_metric)
    
    # Create info text
    test_metadata = {
        'epoch': 'Test',
        'step': '0',
        'loss': '0.0',
        'curvature_strength': str(scale_factor),
        'metric': 'Schwarzschild-like (exaggerated)'
    }
    create_info_text(test_metadata)
    
    # Setup camera
    bpy.ops.object.camera_add(location=(12, -12, 8))
    camera = bpy.context.active_object
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))
    bpy.context.scene.camera = camera
    
    print("Test visualization created successfully with direct vertex coloring")
    return grid, particles, test_metric

def create_direct_coordinate_lines(grid_object, divisions=10):
    """Create coordinate lines directly, not relying on the grid mesh."""
    if not SHOW_COORDINATE_LINES:
        return None
    
    print(f"Creating direct coordinate lines with {divisions} divisions")
    
    # Get grid dimensions
    size = 10.0  # Fixed size for consistency
    
    # Create line objects for the coordinate grid
    lines_collection = []
    
    # Function to calculate z value at a given (x,y) position
    def get_z_at_position(x, y):
        r = math.sqrt(x*x + y*y)
        if r < 0.001:
            r = 0.001
        return -CURVATURE_INTENSITY * (1.0 / (0.5 + r*r*0.15))
    
    # Create x-fixed lines
    for i in range(divisions + 1):
        # Create a curve
        curve_data = bpy.data.curves.new(f'XLine_{i}', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 16  # Increased for smoother curves
        
        # Set bevel depth for thickness
        curve_data.bevel_depth = 0.02
        
        # Create spline
        spline = curve_data.splines.new('NURBS')
        
        # Calculate x position
        x = -size/2.0 + i * size / divisions
        
        # Set number of points
        samples = 64  # More samples for smoother curves
        spline.points.add(samples - 1)  # Already has one point
        
        # Set points
        for j in range(samples):
            y = -size/2.0 + j * size / (samples - 1)
            z = get_z_at_position(x, y)
            
            # Set point coordinates (NURBS points use 4D coordinates w=1.0)
            spline.points[j].co = (x, y, z + 0.02, 1.0)  # Smaller offset above surface
        
        # Create object
        curve_obj = bpy.data.objects.new(f'XLine_{i}', curve_data)
        bpy.context.collection.objects.link(curve_obj)
        
        # Create material - simplified pure emission for visibility
        mat = bpy.data.materials.new(name=f"LineXMaterial_{i}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Clear default nodes
        for node in nodes:
            nodes.remove(node)
        
        # Create emission shader with much higher brightness to compensate for dimmer lighting
        output = nodes.new('ShaderNodeOutputMaterial')
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
        emission.inputs[1].default_value = 20.0  # Significantly increased emission for visibility
        
        # Connect nodes
        mat.node_tree.links.new(emission.outputs[0], output.inputs[0])
        
        # Assign material
        curve_obj.data.materials.append(mat)
        
        lines_collection.append(curve_obj)
    
    # Create y-fixed lines
    for i in range(divisions + 1):
        # Create a curve
        curve_data = bpy.data.curves.new(f'YLine_{i}', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 16  # Increased for smoother curves
        
        # Set bevel depth for thickness
        curve_data.bevel_depth = 0.02
        
        # Create spline
        spline = curve_data.splines.new('NURBS')
        
        # Calculate y position
        y = -size/2.0 + i * size / divisions
        
        # Set number of points
        samples = 64  # More samples for smoother curves
        spline.points.add(samples - 1)  # Already has one point
        
        # Set points
        for j in range(samples):
            x = -size/2.0 + j * size / (samples - 1)
            z = get_z_at_position(x, y)
            
            # Set point coordinates (NURBS points use 4D coordinates w=1.0)
            spline.points[j].co = (x, y, z + 0.02, 1.0)  # Smaller offset above surface
        
        # Create object
        curve_obj = bpy.data.objects.new(f'YLine_{i}', curve_data)
        bpy.context.collection.objects.link(curve_obj)
        
        # Create material - simplified pure emission for visibility
        mat = bpy.data.materials.new(name=f"LineYMaterial_{i}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        
        # Clear default nodes
        for node in nodes:
            nodes.remove(node)
        
        # Create emission shader with much higher brightness to compensate for dimmer lighting
        output = nodes.new('ShaderNodeOutputMaterial')
        emission = nodes.new('ShaderNodeEmission')
        emission.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
        emission.inputs[1].default_value = 20.0  # Significantly increased emission for visibility
        
        # Connect nodes
        mat.node_tree.links.new(emission.outputs[0], output.inputs[0])
        
        # Assign material
        curve_obj.data.materials.append(mat)
        
        lines_collection.append(curve_obj)
    
    print(f"Created {len(lines_collection)} coordinate lines")
    return lines_collection

def main():
    """Main function to run the visualization."""
    print("Starting manifold visualization...")
    
    try:
        # Setup compatibility mode for older Blender versions
        global COMPATIBILITY_MODE, SKIP_PARTICLES, MAX_GRID_SIZE, DISABLE_MODIFIERS
        global COORDINATE_LINE_DENSITY, CURVATURE_INTENSITY, FORCE_VISUALIZATION
        global PURE_BLACK_BACKGROUND
        
        # Clear existing scene
        clear_scene()
        
        # Setup scene lighting first
        try:
            setup_scene_lighting()
        except Exception as e:
            print(f"Warning: Could not set up lighting: {e}")
            # Create a basic light if advanced lighting fails
            bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
            light = bpy.context.active_object
            light.data.energy = 5.0  # High energy
        
        # Check if we should load from files or use test visualization
        if not FORCE_VISUALIZATION:
            # Try to load data from files
            metric_file = './metric_data_current.csv'
            christoffel_file = './christoffel_data_current.csv'
            metadata_file = './metric_metadata.json'
            
            if os.path.exists(metric_file):
                print(f"Found metric data file: {metric_file}")
                metric_data = load_metric_data(metric_file)
                
                if os.path.exists(christoffel_file):
                    christoffel_data = load_christoffel_data(christoffel_file)
                else:
                    christoffel_data = None
                    
                if os.path.exists(metadata_file):
                    metadata = load_metadata(metadata_file)
                else:
                    metadata = {"epoch": "unknown", "loss": "unknown"}
                
                # Create a grid with the loaded metric
                grid = create_spacetime_grid(metric_data)
                min_z, max_z = deform_grid_with_metric(grid, metric_data)
                
                # Apply direct vertex coloring for the manifold
                create_vertex_color_visualization(grid, min_z, max_z)
                
                # Create coordinate lines
                if SHOW_COORDINATE_LINES:
                    lines = create_direct_coordinate_lines(grid, divisions=COORDINATE_LINE_DENSITY)
                
                # Create particles
                particles = []
                if not SKIP_PARTICLES:
                    particles = create_particles(num_particles=15, grid_object=grid, metric_data=metric_data)
                
                # Create info text
                create_info_text(metadata)
                
                # Setup camera
                bpy.ops.object.camera_add(location=(12, -12, 8))
                camera = bpy.context.active_object
                camera.rotation_euler = (math.radians(60), 0, math.radians(45))
                bpy.context.scene.camera = camera
                
                print("Created visualization from loaded metric data")
            else:
                print("No metric data file found, using test visualization")
                # Force the test visualization as fallback
                grid, particles, test_metric = create_test_visualization()
        else:
            # Force the test visualization to ensure we have color
            print("Forcing direct visualization with color gradient...")
            grid, particles, test_metric = create_test_visualization()
        
        # Set up safer viewport settings
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        # Force material preview mode for better display
                        space.shading.type = 'MATERIAL'
                        space.shading.use_scene_lights = True
                        space.shading.use_scene_world = True
                        
                        # Force pure black background in viewport too
                        space.shading.background_type = 'WORLD'
                        space.shading.background_color = (0, 0, 0)
                        space.shading.studio_light = 'DEFAULT'
                        
                        # Remove all overlays
                        space.overlay.show_floor = False
                        space.overlay.show_axis_x = False
                        space.overlay.show_axis_y = False
                        space.overlay.show_axis_z = False
                        
                        # Try to enable bloom if available
                        if hasattr(space.shading, 'use_bloom'):
                            space.shading.use_bloom = True
                            space.shading.bloom_intensity = 0.5  # higher bloom intensity
        
        # Force viewport redraw
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        
        print("Visualization setup complete with color gradients on black background.")
        
    except Exception as e:
        print(f"Critical error in visualization: {e}")
        traceback.print_exc()

def create_vertex_color_visualization(grid, min_z, max_z):
    """Create vertex colors for the manifold based on height values."""
    mesh = grid.data
    
    # Create a vertex color layer if it doesn't exist
    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    
    color_layer = mesh.vertex_colors.active
    
    # Calculate normalization factor
    z_range = max_z - min_z
    if z_range == 0:
        z_range = 1.0
    
    # Assign colors directly to vertices for faces
    i = 0
    for poly in mesh.polygons:
        for idx in poly.loop_indices:
            # Get the vertex for this face corner
            v_idx = mesh.loops[idx].vertex_index
            vertex = mesh.vertices[v_idx]
            
            # Normalize height (0 = deepest/center, 1 = highest/edge)
            norm_height = 1.0 - ((vertex.co.z - min_z) / z_range)
            
            # Apply a gamma correction for better visual gradient
            gamma_corrected = pow(norm_height, 0.7)  # Gamma correction
            
            if gamma_corrected < 0.2:
                # Deepest center - dark rich red
                r, g, b = 0.7, 0.0, 0.0
            elif gamma_corrected < 0.4:
                # Deep red
                r, g, b = 0.85, 0.05, 0.05
            elif gamma_corrected < 0.6:
                # Medium red
                r, g, b = 1.0, 0.1, 0.1
            elif gamma_corrected < 0.8:
                # Lighter red
                r, g, b = 1.0, 0.3, 0.3
            else:
                # Edges - pale red/pink
                r, g, b = 1.0, 0.6, 0.6
            
            # Assign the color
            color_layer.data[i].color = (r, g, b, 1.0)
            i += 1
    
    # Create a simple emission material to make colors glow
    mat = bpy.data.materials.new(name="PureRedEmissionMaterial")
    mat.use_nodes = True
    
    # Get material nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes for very simple emission material
    output = nodes.new('ShaderNodeOutputMaterial')
    emission = nodes.new('ShaderNodeEmission')
    vertex_color = nodes.new('ShaderNodeVertexColor')
    vertex_color.layer_name = color_layer.name
    
    # Connect nodes
    links.new(vertex_color.outputs['Color'], emission.inputs['Color'])
    links.new(emission.outputs[0], output.inputs[0])
    
    # Set emission strength
    emission.inputs['Strength'].default_value = 2.5
    
    # Assign material to the grid
    grid.data.materials.clear()  # Remove any existing materials
    grid.data.materials.append(mat)
    
    print("Applied vertex color visualization to manifold")
    
    return color_layer

# Execute main function when run directly
if __name__ == "__main__":
    main()