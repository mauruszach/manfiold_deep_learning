import bpy
import numpy as np
import math
import os
import json
import mathutils

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
    
    print("Scene cleared successfully")

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
    """Create a more detailed grid with stronger deformation."""
    print("Creating spacetime grid...")
    
    # Create a grid mesh
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=divisions, 
        y_subdivisions=divisions, 
        size=size,
        location=(0, 0, 0)
    )
    grid = bpy.context.active_object
    grid.name = "SpacetimeGrid"
    
    # Create a better material for the grid
    mat = bpy.data.materials.new(name="GridMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)
    
    # Create nodes for advanced shading
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    geometry = nodes.new('ShaderNodeNewGeometry')
    separate_xyz = nodes.new('ShaderNodeSeparateXYZ')
    mapping = nodes.new('ShaderNodeMapping')
    color_ramp = nodes.new('ShaderNodeValToRGB')
    
    # Set up an enhanced color gradient
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.6, 1.0)  # Deep blue
    
    # Add middle elements for better gradient
    color_ramp.color_ramp.elements.new(0.3)
    color_ramp.color_ramp.elements[1].color = (0.0, 0.5, 0.8, 1.0)  # Light blue
    
    color_ramp.color_ramp.elements.new(0.6)
    color_ramp.color_ramp.elements[2].color = (0.7, 0.7, 0.0, 1.0)  # Yellow
    
    color_ramp.color_ramp.elements.new(0.8)
    color_ramp.color_ramp.elements[3].color = (0.9, 0.4, 0.0, 1.0)  # Orange
    
    color_ramp.color_ramp.elements[4].position = 1.0
    color_ramp.color_ramp.elements[4].color = (0.8, 0.0, 0.0, 1.0)  # Red
    
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
    
    # Apply metric deformation to the grid
    deform_grid_with_metric(grid, metric_data)
    
    # Add subdivision surface modifier for smoothness
    subdiv = grid.modifiers.new(name="Subdivision", type='SUBSURF')
    subdiv.levels = 2
    subdiv.render_levels = 3
    
    # Add wireframe for better visual understanding
    wireframe = grid.modifiers.new(name="Wireframe", type='WIREFRAME')
    wireframe.thickness = 0.01
    wireframe.use_relative_offset = True
    
    print("Grid creation complete")
    return grid

def deform_grid_with_metric(grid, metric_data):
    """Apply stronger deformation based on the metric tensor."""
    print("Deforming grid with metric tensor...")
    
    # Ensure metric is in correct format
    if len(metric_data.shape) == 1:
        # Reshape if it's a flattened array
        dim = int(math.sqrt(metric_data.shape[0]))
        metric = metric_data.reshape(dim, dim)
    else:
        metric = metric_data
    
    # Get mesh data
    mesh = grid.data
    vertices = mesh.vertices
    
    # Calculate scalar curvature for visualization
    curvature = extract_ricci_curvature(metric)
    
    # Determine magnification factor - scale based on curvature magnitude
    scale_factor = -4.0  # Base scale, stronger than before
    if abs(curvature) < 0.05:
        scale_factor *= 5.0  # Boost small curvatures
    
    # Get additional factors from metric components for asymmetric effects
    asymmetry_factor = abs(np.sum([
        metric[0,1], metric[0,2], metric[0,3],
        metric[1,2], metric[1,3], metric[2,3]
    ])) * 0.5
    
    print(f"Applied scale factor: {scale_factor}, Asymmetry: {asymmetry_factor}")
    
    # Store curvature values in vertex groups for color mapping
    if "CurvatureValues" not in grid.vertex_groups:
        grid.vertex_groups.new(name="CurvatureValues")
    curvature_group = grid.vertex_groups["CurvatureValues"]
    
    # Apply deformation to vertices
    min_z = float('inf')
    max_z = float('-inf')
    
    # First pass: calculate z values and find min/max
    for i, vertex in enumerate(vertices):
        # Get normalized position in the grid
        x_norm = vertex.co.x / 5.0
        y_norm = vertex.co.y / 5.0
        
        # Calculate distance from center
        r_squared = x_norm**2 + y_norm**2
        r = math.sqrt(r_squared)
        
        # Enhanced deformation formula with more terms from metric components
        # More complex deformation using multiple metric components
        z_deformation = scale_factor * (
            curvature * r_squared +  # Basic quadratic well
            0.5 * asymmetry_factor * x_norm * y_norm +  # Cross-term for asymmetry
            0.2 * (metric[0,0] * x_norm**3 + metric[1,1] * y_norm**3) +  # Cubic terms
            0.1 * (metric[2,2] + metric[3,3]) * math.sin(r * 3.0)  # Oscillation term
        )
        
        # Apply the deformation
        vertex.co.z = z_deformation
        
        # Track min/max for normalization
        min_z = min(min_z, z_deformation)
        max_z = max(max_z, z_deformation)
    
    # Second pass: store normalized curvature values in vertex group
    z_range = max_z - min_z
    if z_range == 0:
        z_range = 1.0  # Prevent division by zero
        
    for i, vertex in enumerate(vertices):
        # Normalize z value to 0-1 range for color mapping
        normalized_curvature = (vertex.co.z - min_z) / z_range
        # Store in vertex group with weight for shader access
        curvature_group.add([i], normalized_curvature, 'REPLACE')
    
    print(f"Grid deformation complete. Z range: {min_z} to {max_z}")
    
    # Apply enhanced color based on curvature
    apply_curvature_color_gradient(grid, min_z, max_z)

def apply_curvature_color_gradient(grid, min_z, max_z):
    """Apply an enhanced color gradient based on curvature values."""
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
    
    # Set up an enhanced physics-inspired color gradient
    color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])
    
    # Deep valleys (high negative curvature) - deep blue
    pos0 = color_ramp.color_ramp.elements.new(0.0)
    pos0.color = (0.0, 0.05, 0.6, 1.0)
    
    # Medium negative curvature - cyan/teal
    pos1 = color_ramp.color_ramp.elements.new(0.2)
    pos1.color = (0.0, 0.5, 0.8, 1.0)
    
    # Flat regions - green
    pos2 = color_ramp.color_ramp.elements.new(0.4)
    pos2.color = (0.0, 0.8, 0.2, 1.0)
    
    # Slight positive curvature - yellow
    pos3 = color_ramp.color_ramp.elements.new(0.6)
    pos3.color = (0.8, 0.8, 0.0, 1.0)
    
    # Medium positive curvature - orange
    pos4 = color_ramp.color_ramp.elements.new(0.8)
    pos4.color = (1.0, 0.4, 0.0, 1.0)
    
    # High positive curvature - red
    pos5 = color_ramp.color_ramp.elements.new(1.0)
    pos5.color = (0.8, 0.0, 0.0, 1.0)
    
    # Set up noise for surface detail
    noise_texture.inputs['Scale'].default_value = 30.0
    noise_texture.inputs['Detail'].default_value = 10.0
    noise_texture.inputs['Roughness'].default_value = 0.7
    
    # Mix noise with color gradient
    mix_rgb.blend_type = 'OVERLAY'
    mix_rgb.inputs['Fac'].default_value = 0.15  # Subtle effect
    
    # Set up bump mapping for surface detail
    bump.inputs['Strength'].default_value = 0.2
    bump.inputs['Distance'].default_value = 0.02
    
    # Set material properties for a scientific visualization look
    if 'Specular' in principled.inputs:
        principled.inputs['Specular'].default_value = 0.8
    elif 'Specular IOR' in principled.inputs:
        principled.inputs['Specular IOR'].default_value = 0.8
    
    principled.inputs['Roughness'].default_value = 0.2
    principled.inputs['Metallic'].default_value = 0.7
    
    # Properly handle properties that might not exist in all Blender versions
    if 'Clearcoat' in principled.inputs:
        principled.inputs['Clearcoat'].default_value = 0.3
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
    
    print("Applied enhanced curvature color gradient")

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
    """Set up enhanced lighting in the scene."""
    print("Setting up lighting...")
    
    # Create a sun light
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.name = "MainSun"
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(60), 0, math.radians(45))
    
    # Create a rim light
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 5))
    rim = bpy.context.active_object
    rim.name = "RimLight"
    rim.data.energy = 1.0
    rim.rotation_euler = (math.radians(-60), 0, math.radians(135))
    
    # Create an ambient light
    bpy.ops.object.light_add(type='AREA', radius=1, location=(0, 0, 8))
    ambient = bpy.context.active_object
    ambient.name = "AmbientLight"
    ambient.data.energy = 3.0
    ambient.data.size = 15.0
    
    # Set up camera for better angle
    bpy.ops.object.camera_add(location=(12, -12, 10))
    camera = bpy.context.active_object
    camera.name = "MainCamera"
    camera.rotation_euler = (math.radians(55), 0, math.radians(45))
    bpy.context.scene.camera = camera
    
    # Create a custom world background
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    
    world.use_nodes = True
    world_nodes = world.node_tree.nodes
    
    # Clear existing nodes
    for node in world_nodes:
        world_nodes.remove(node)
    
    # Create a nice gradient background
    output = world_nodes.new('ShaderNodeOutputWorld')
    bg = world_nodes.new('ShaderNodeBackground')
    gradient = world_nodes.new('ShaderNodeTexGradient')
    mapping = world_nodes.new('ShaderNodeMapping')
    texcoord = world_nodes.new('ShaderNodeTexCoord')
    colorramp = world_nodes.new('ShaderNodeValToRGB')
    
    # Set up gradient from dark blue to black
    colorramp.color_ramp.elements[0].position = 0.0
    colorramp.color_ramp.elements[0].color = (0.0, 0.02, 0.05, 1.0)
    colorramp.color_ramp.elements[1].position = 1.0
    colorramp.color_ramp.elements[1].color = (0.0, 0.0, 0.0, 1.0)
    
    # Connect nodes
    world.node_tree.links.new(texcoord.outputs['Generated'], mapping.inputs[0])
    world.node_tree.links.new(mapping.outputs[0], gradient.inputs[0])
    world.node_tree.links.new(gradient.outputs[0], colorramp.inputs[0])
    world.node_tree.links.new(colorramp.outputs[0], bg.inputs[0])
    world.node_tree.links.new(bg.outputs[0], output.inputs[0])
    
    print("Lighting setup complete")

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
    print("Creating test visualization...")
    
    # Clean scene
    clear_scene()
    setup_scene_lighting()
    
    # Create a grid with higher density for smoother curvature
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=50, 
        y_subdivisions=50, 
        size=10,
        location=(0, 0, 0))
    
    grid = bpy.context.active_object
    grid.name = "SpacetimeGrid"
    
    # Create predefined metric tensor (Schwarzschild-like)
    # Using metric that should show visible curvature
    test_metric = np.array([
        [-1.0, -0.24, -0.28, -0.28],
        [-0.24, 0.06, -0.14, -0.26],
        [-0.28, -0.14, 0.01, -0.21],
        [-0.28, -0.26, -0.21, 0.07]
    ])
    
    # Apply subdivision modifier for smoother surface
    bpy.ops.object.modifier_add(type='SUBSURF')
    bpy.context.object.modifiers["Subdivision"].levels = 2
    bpy.context.object.modifiers["Subdivision"].render_levels = 3
    bpy.ops.object.modifier_apply(modifier="Subdivision")
    
    # Create vertex group for curvature values
    curvature_group = grid.vertex_groups.new(name="CurvatureValues")
    
    # Deform the grid
    deform_vertices = grid.data.vertices
    
    # Track min/max curvature values
    min_z = float('inf')
    max_z = float('-inf')
    
    # First pass: apply curvature and find min/max
    for v in deform_vertices:
        # Get normalized position
        pos = v.co.copy()
        
        # Calculate distance from center
        r = math.sqrt(pos.x**2 + pos.y**2)
        
        # Skip center point to avoid division by zero
        if r < 0.1:
            continue
            
        # Enhanced Schwarzschild-like curvature with more pronounced effect
        z_offset = -3.5 / (r + 0.3) * (1 + 0.3 * math.sin(r * 2))
        
        # Apply deformation
        v.co.z = z_offset
        
        # Track min/max for normalization
        min_z = min(min_z, z_offset)
        max_z = max(max_z, z_offset)
    
    # Second pass: store normalized curvature values
    z_range = max_z - min_z
    if z_range == 0:
        z_range = 1.0  # Prevent division by zero
        
    for i, v in enumerate(deform_vertices):
        # Normalize z value to 0-1 range for color mapping
        normalized_curvature = (v.co.z - min_z) / z_range
        # Store in vertex group with weight for shader access
        curvature_group.add([i], normalized_curvature, 'REPLACE')
    
    # Apply mesh update
    grid.data.update()
    
    # Apply enhanced color gradient
    apply_curvature_color_gradient(grid, min_z, max_z)
    
    # Create particles to visualize geodesic paths
    particles = create_particles(num_particles=25, grid_object=grid, metric_data=test_metric)
    
    # Create test metadata for info display
    test_metadata = {
        'epoch': 30,
        'step': 240,
        'loss': 0.1,
        'curvature_strength': 2.9,
        'max_eigenvalue': 0.347,
        'min_eigenvalue': -0.693,
        'metric_determinant': -0.012
    }
    
    # Create info text
    text_obj = create_info_text(test_metadata)
    
    # Add continuous animation to the grid to simulate metric evolution
    anim_frames = 120
    grid.animation_data_create()
    grid.animation_data.action = bpy.data.actions.new(name="GridAnimation")
    
    # Create shape keys for animation
    sk_basis = grid.shape_key_add(name='Basis')
    sk_basis.interpolation = 'KEY_LINEAR'
    sk_target = grid.shape_key_add(name='Deformed')
    sk_target.interpolation = 'KEY_LINEAR'
    
    # Set up target shape key with stronger deformation
    for i, v in enumerate(grid.data.vertices):
        pos = v.co.copy()
        r = math.sqrt(pos.x**2 + pos.y**2)
        if r < 0.1:
            continue
        # Create a slightly different deformation for animation
        z_offset = -4.0 / (r + 0.2) * (1 + 0.2 * math.cos(r * 3))
        sk_target.data[i].co.z = z_offset
    
    # Animate the shape key influence
    sk_target.value = 0.0
    sk_target.keyframe_insert(data_path="value", frame=1)
    sk_target.value = 1.0
    sk_target.keyframe_insert(data_path="value", frame=anim_frames // 2)
    sk_target.value = 0.0
    sk_target.keyframe_insert(data_path="value", frame=anim_frames)
    
    # Set the animation to cycle
    sk_target.animation_data_create()
    sk_target.animation_data.action = bpy.data.actions.new(name="ShapeKeyAnimation")
    for fc in sk_target.animation_data.action.fcurves:
        fc.extrapolation = 'CYCLIC'
    
    # Set up continuous rotating camera
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    
    if camera:
        camera.animation_data_create()
        camera.animation_data.action = bpy.data.actions.new(name="CameraAnimation")
        
        # Create orbital camera motion
        for i in range(0, anim_frames + 1, 20):
            angle = 2 * math.pi * i / anim_frames
            camera.location.x = 15 * math.sin(angle)
            camera.location.y = 15 * math.cos(angle)
            camera.location.z = 8 + 2 * math.sin(angle * 2)
            camera.keyframe_insert(data_path="location", frame=i)
            
            # Always look at the center
            direction = mathutils.Vector((0, 0, 0)) - camera.location
            rot_quat = direction.to_track_quat('Z', 'Y')
            camera.rotation_euler = rot_quat.to_euler()
            camera.keyframe_insert(data_path="rotation_euler", frame=i)
    
    # Set Blender to the animation frame
    bpy.context.scene.frame_set(1)
    bpy.context.scene.frame_end = anim_frames
    
    # Enable post-processing effects for better visualization
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.use_ssao = True  # Ambient occlusion
                    space.shading.ssao_factor = 2.0
                    space.shading.use_bloom = True  # Bloom effect
                    space.shading.bloom_intensity = 0.5
    
    print("Test visualization created successfully with enhanced color gradient")
    return grid, particles, test_metric

def main():
    """Main entry point for the Blender script."""
    print("Starting Manifold Visualization...")
    
    # Clear existing scene
    clear_scene()
    
    # Setup lighting and camera
    setup_scene_lighting()
    
    # Define file paths
    metric_file = "/Users/zachmaurus/manifold_deep_learning/paths_on_manifold/metric_data_current.csv"
    christoffel_file = "/Users/zachmaurus/manifold_deep_learning/paths_on_manifold/christoffel_data_current.csv"
    metadata_file = "/Users/zachmaurus/manifold_deep_learning/paths_on_manifold/metric_metadata.json"
    
    # Check if we should run a test visualization first
    run_test = False
    
    if run_test:
        print("Running test visualization...")
        grid, particles, test_metric = create_test_visualization()
    else:
        try:
            # Load the data
            print("Loading metric data...")
            metric_data = load_metric_data(metric_file)
            
            print("Loading Christoffel symbols...")
            christoffel_data = load_christoffel_data(christoffel_file)
            
            try:
                print("Loading metadata...")
                metadata = load_metadata(metadata_file)
            except:
                print("Could not load metadata, creating default")
                metadata = {
                    'epoch': 30,
                    'step': 240,
                    'loss': 0.1,
                    'curvature_strength': 2.9
                }
            
            # Extract Ricci curvature for visualization
            curvature = extract_ricci_curvature(metric_data)
            print(f"Extracted curvature: {curvature}")
            
            # Create spacetime grid with higher density
            print("Creating spacetime grid...")
            grid = create_spacetime_grid(metric_data, size=10, divisions=50)
            
            # Add subdivision for smoother appearance
            bpy.ops.object.select_all(action='DESELECT')
            grid.select_set(True)
            bpy.context.view_layer.objects.active = grid
            bpy.ops.object.modifier_add(type='SUBSURF')
            bpy.context.object.modifiers["Subdivision"].levels = 2
            bpy.context.object.modifiers["Subdivision"].render_levels = 3
            bpy.ops.object.modifier_apply(modifier="Subdivision")
            
            # Apply metric deformation with enhanced curvature visualization
            print("Applying metric deformation...")
            deform_grid_with_metric(grid, metric_data)
            
            # Create particles with trails
            print("Creating particles...")
            particles = create_particles(num_particles=25, grid_object=grid, metric_data=metric_data)
            
            # Setup particle animation
            print("Setting up animation...")
            setup_animation(particles, christoffel_data, metric_data, frames=240)
            
            # Create info text
            print("Creating info text...")
            create_info_text(metadata)
            
            # Add continuous animation
            print("Setting up continuous animation...")
            anim_frames = 240
            
            # Create shape keys for grid evolution
            grid.shape_key_add(name='Basis')
            deformed = grid.shape_key_add(name='Deformed')
            
            # Get vertices of current deformation
            original_positions = np.array([v.co.copy() for v in grid.data.vertices])
            
            # Apply a slightly different deformation for animation target
            for i, v in enumerate(grid.data.vertices):
                pos = original_positions[i]
                r = math.sqrt(pos.x**2 + pos.y**2)
                if r < 0.1:
                    continue
                z_offset = original_positions[i].z * (1 + 0.3 * math.sin(r * 3))
                deformed.data[i].co.z = z_offset
            
            # Animate the shape key influence
            deformed.value = 0.0
            deformed.keyframe_insert(data_path="value", frame=1)
            deformed.value = 1.0
            deformed.keyframe_insert(data_path="value", frame=anim_frames // 2)
            deformed.value = 0.0
            deformed.keyframe_insert(data_path="value", frame=anim_frames)
            
            # Set the animation to cycle
            for fc in grid.animation_data.action.fcurves:
                fc.extrapolation = 'CYCLIC'
            
            # Set up rotating camera
            camera = None
            for obj in bpy.data.objects:
                if obj.type == 'CAMERA':
                    camera = obj
                    break
            
            if camera:
                camera.animation_data_create()
                camera.animation_data.action = bpy.data.actions.new(name="CameraAnimation")
                
                # Create orbital camera motion
                for i in range(0, anim_frames + 1, 20):
                    angle = 2 * math.pi * i / anim_frames
                    camera.location.x = 15 * math.sin(angle)
                    camera.location.y = 15 * math.cos(angle)
                    camera.location.z = 10 + 2 * math.sin(angle * 2)
                    camera.keyframe_insert(data_path="location", frame=i)
                    
                    # Always look at the center
                    direction = mathutils.Vector((0, 0, 0)) - camera.location
                    rot_quat = direction.to_track_quat('-Z', 'Y')
                    camera.rotation_euler = rot_quat.to_euler()
                    camera.keyframe_insert(data_path="rotation_euler", frame=i)
            
            # Set render settings for higher quality
            bpy.context.scene.render.resolution_x = 1920
            bpy.context.scene.render.resolution_y = 1080
            bpy.context.scene.render.film_transparent = True
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            bpy.context.scene.frame_set(1)
            bpy.context.scene.frame_end = anim_frames
            
            print("Visualization completed successfully.")
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            print("Falling back to test visualization...")
            grid, particles, test_metric = create_test_visualization()
    
    # Set up a nice viewpoint with enhanced rendering settings
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    space.shading.use_scene_lights = True
                    space.shading.use_scene_world = True
                    space.shading.light_direction = (0.5, 0.5, 0.9)
                    space.overlay.show_floor = False
                    
                    # Enable screen space ambient occlusion for better depth perception
                    space.shading.use_ssao = True
                    space.shading.ssao_factor = 1.5
                    
                    # Enable bloom effect
                    space.shading.use_bloom = True
                    space.shading.bloom_intensity = 0.3
    
    print("High quality manifold visualization with enhanced color gradients setup complete.")
    
# Execute main function when run directly
if __name__ == "__main__":
    main()