#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include <optional>
#include <fstream>
#include <sstream>
#include <string>

// Include OpenGL headers
#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

// Silence OpenGL deprecation warnings on macOS
#define GL_SILENCE_DEPRECATION

// Constants
const float G = 6.67430e-11;  // Gravitational constant
const float SCALE_FACTOR = 1e-4;  // Adjusted scale factor for smoother visualization
const int GRID_SIZE = 60;  // Increased grid size for smoother curvature
const float GRID_SPACING = 0.5f;  // Decreased spacing for denser grid
const int WINDOW_WIDTH = 800;
const int WINDOW_HEIGHT = 600;
const float TIME_STEP = 0.02f;
const float PARTICLE_SPEED = 3.0f;  // Base speed for particles
// Physical constants for simulation
const float LIGHT_SPEED = 299792458.0f;  // Speed of light in m/s
const float SPACE_SCALE = 1e8;  // Scale for visualization
const float GRAVITY_SCALE = 5.0f;  // Scale factor for gravitational effects
const float TIME_DILATION_FACTOR = 0.2f;  // Factor to visualize time dilation effects

// 3D Vector class
class Vector3 {
public:
    float x, y, z;
    
    Vector3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    
    Vector3 operator+(const Vector3& v) const {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }
    
    Vector3 operator-(const Vector3& v) const {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }
    
    Vector3 operator*(float s) const {
        return Vector3(x * s, y * s, z * s);
    }
    
    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Vector3 normalized() const {
        float len = length();
        if (len < 1e-10) return Vector3();
        return Vector3(x / len, y / len, z / len);
    }
};

// Tensor class to store metric and Christoffel symbols imported from Python model
class Tensor {
public:
    // For metric tensor (rank 2)
    std::vector<std::vector<float>> metric;
    // For Christoffel symbols (rank 3)
    std::vector<std::vector<std::vector<float>>> christoffel;
    // For Riemann curvature tensor (rank 4)
    std::vector<std::vector<std::vector<std::vector<float>>>> riemann;
    int dim;
    
    Tensor(int dimension) : dim(dimension) {
        // Initialize metric tensor with identity matrix
        metric.resize(dim, std::vector<float>(dim, 0.0f));
        for (int i = 0; i < dim; i++) {
            metric[i][i] = (i == 0) ? -1.0f : 1.0f; // Minkowski metric by default
        }
        
        // Initialize Christoffel symbols (all zeros initially)
        christoffel.resize(dim, std::vector<std::vector<float>>(
            dim, std::vector<float>(dim, 0.0f)));
            
        // Initialize Riemann tensor (all zeros initially)
        riemann.resize(dim, std::vector<std::vector<std::vector<float>>>(
            dim, std::vector<std::vector<float>>(
                dim, std::vector<float>(dim, 0.0f))));
    }
    
    // Load metric tensor from CSV file exported by Python model
    bool loadMetricFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open metric file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        int row = 0;
        
        while (std::getline(file, line) && row < dim) {
            std::stringstream ss(line);
            std::string cell;
            int col = 0;
            
            while (std::getline(ss, cell, ',') && col < dim) {
                try {
                    metric[row][col] = std::stof(cell);
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing metric value at row " << row 
                              << ", col " << col << ": " << cell << std::endl;
                    metric[row][col] = 0.0f;
                }
                col++;
            }
            row++;
        }
        
        file.close();
        return true;
    }
    
    // Load Christoffel symbols from CSV file exported by Python model
    bool loadChristoffelFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open Christoffel file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        
        // Format is expected to be: i,j,k,value
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<int> indices;
            float value;
            
            // Parse indices
            for (int i = 0; i < 3; i++) {
                if (!std::getline(ss, cell, ',')) {
                    std::cerr << "Error parsing Christoffel indices: " << line << std::endl;
                    continue;
                }
                try {
                    indices.push_back(std::stoi(cell));
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing index: " << cell << std::endl;
                    continue;
                }
            }
            
            // Parse value
            if (std::getline(ss, cell)) {
                try {
                    value = std::stof(cell);
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing Christoffel value: " << cell << std::endl;
                    continue;
                }
                
                // Check indices are within bounds
                if (indices[0] < dim && indices[1] < dim && indices[2] < dim) {
                    christoffel[indices[0]][indices[1]][indices[2]] = value;
                }
            }
        }
        
        file.close();
        return true;
    }
    
    // Load Riemann tensor from CSV file
    bool loadRiemannFromCSV(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open Riemann file: " << filename << std::endl;
            return false;
        }
        
        std::string line;
        
        // Format is expected to be: i,j,k,l,value
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string cell;
            std::vector<int> indices;
            float value;
            
            // Parse indices
            for (int i = 0; i < 4; i++) {
                if (!std::getline(ss, cell, ',')) {
                    std::cerr << "Error parsing Riemann indices: " << line << std::endl;
                    continue;
                }
                try {
                    indices.push_back(std::stoi(cell));
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing index: " << cell << std::endl;
                    continue;
                }
            }
            
            // Parse value
            if (std::getline(ss, cell)) {
                try {
                    value = std::stof(cell);
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing Riemann value: " << cell << std::endl;
                    continue;
                }
                
                // Check indices are within bounds
                if (indices[0] < dim && indices[1] < dim && 
                    indices[2] < dim && indices[3] < dim) {
                    riemann[indices[0]][indices[1]][indices[2]][indices[3]] = value;
                }
            }
        }
        
        file.close();
        return true;
    }
    
    // Calculate metric at a given point (for non-constant metrics)
    std::vector<std::vector<float>> getMetricAt(const Vector3& position) const {
        // For position-dependent metrics, this would interpolate
        // For now, just return the stored metric
        return metric;
    }
    
    // Calculate Christoffel symbols at a given point
    std::vector<std::vector<std::vector<float>>> getChristoffelAt(const Vector3& position) const {
        // For position-dependent Christoffel symbols, this would interpolate
        // For now, just return the stored symbols
        return christoffel;
    }
};

// Mass class representing objects with gravitational influence
class Mass {
public:
    Vector3 position;
    float mass;
    float radius;
    
    Mass(const Vector3& pos, float m) : position(pos), mass(m) {
        radius = 0.2f + 0.05f * std::cbrt(mass);  // Visualized radius based on mass
    }
};

// Class to represent a point in the spacetime grid
class GridPoint {
public:
    Vector3 position;  // Original position
    Vector3 displaced;  // Position after displacement due to gravity
    Vector3 basisX;    // X basis vector
    Vector3 basisY;    // Y basis vector
    Vector3 basisZ;    // Z basis vector
    float curvatureDepth; // How much the point is curved downward
    
    GridPoint(const Vector3& pos) : position(pos), displaced(pos), curvatureDepth(0.0f) {
        // Initialize basis vectors
        basisX = Vector3(0.3f, 0, 0);
        basisY = Vector3(0, 0.3f, 0);
        basisZ = Vector3(0, 0, 0.3f);
    }
    
    // Calculate displacement based on masses (traditional method)
    void calculateDisplacement(const std::vector<Mass>& masses) {
        // Reset to original position
        displaced = position;
        curvatureDepth = 0.0f;
        
        // Original basis vectors
        Vector3 origBasisX(0.3f, 0, 0);
        Vector3 origBasisY(0, 0.3f, 0);
        Vector3 origBasisZ(0, 0, 0.3f);
        
        // Reset basis vectors
        basisX = origBasisX;
        basisY = origBasisY;
        basisZ = origBasisZ;
        
        // For each mass, calculate a smooth gravitational well
        for (const auto& mass : masses) {
            Vector3 direction = position - mass.position;
            float distance = direction.length();
            
            // Avoid division by zero with a minimum distance
            if (distance < 0.1f) {
                distance = 0.1f;
            }
            
            // Calculate a smooth well function based on distance
            // This uses a modified Gaussian function for a smooth, continuous well
            float wellDepth = mass.mass * 0.01f;
            float wellWidth = 10.0f + mass.mass * 0.02f;  // Wider wells for larger masses
            float curvature = wellDepth * std::exp(-(distance * distance) / (2.0f * wellWidth));
            
            // Apply a distance-based cutoff to prevent long-range effects
            float maxDistance = 20.0f;
            if (distance > maxDistance) {
                float fadeOut = 1.0f - ((distance - maxDistance) / 5.0f);
                fadeOut = std::max(0.0f, fadeOut);
                curvature *= fadeOut;
            }
            
            // Direction normalized
            Vector3 dir = direction.normalized();
            
            // Apply vertical displacement (z-axis) for the well
            float zDisplacement = curvature;
            curvatureDepth += zDisplacement;
            
            // Apply a very subtle horizontal displacement toward the mass
            // This helps create a more natural-looking well
            float horizontalFactor = 0.05f;
            displaced.x -= dir.x * curvature * horizontalFactor;
            displaced.y -= dir.y * curvature * horizontalFactor;
            
            // Warp the basis vectors based on curvature (representing geodesic deviation)
            float basisWarp = curvature * 0.8f;
            
            // Calculate how much each basis vector should be affected
            Vector3 xEffect = dir * basisWarp * std::abs(dir.x);
            Vector3 yEffect = dir * basisWarp * std::abs(dir.y);
            Vector3 zEffect = dir * basisWarp * std::abs(dir.z);
            
            // Warp the basis vectors (representing tidal forces)
            basisX = (origBasisX - xEffect).normalized() * origBasisX.length();
            basisY = (origBasisY - yEffect).normalized() * origBasisY.length();
            basisZ = (origBasisZ - zEffect).normalized() * origBasisZ.length();
        }
        
        // Apply the total z-displacement
        displaced.z -= curvatureDepth;
    }
    
    // Calculate displacement based on the metric tensor from Python model
    void calculateDisplacementFromMetric(const Tensor& tensor) {
        // Reset to original position
        displaced = position;
        
        // Get the metric at this point
        auto metric = tensor.getMetricAt(position);
        int dim = tensor.dim;
        
        // Calculate the Ricci scalar (trace of Ricci tensor) to determine curvature depth
        float ricciScalar = 0.0f;
        
        // Simplified computation of Ricci scalar from the metric
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if (i == j) {
                    // Contribution to Ricci scalar from diagonal elements
                    ricciScalar += metric[i][j] * (i == 0 ? -1.0f : 1.0f);
                }
            }
        }
        
        // Scale for visualization
        curvatureDepth = std::abs(ricciScalar) * 0.5f;
        
        // Apply the curvature to the z-coordinate
        displaced.z -= curvatureDepth;
        
        // Compute basis vectors from the metric tensor
        // These represent how local coordinate frames are distorted by curvature
        // For visualization, we'll just scale the original basis vectors based on metric components
        Vector3 origBasisX(0.3f, 0, 0);
        Vector3 origBasisY(0, 0.3f, 0);
        Vector3 origBasisZ(0, 0, 0.3f);
        
        if (dim >= 2) {
            // X basis affected by g_xx component
            float xScale = std::sqrt(std::abs(metric[1][1]));
            basisX = origBasisX * xScale;
            
            // Y basis affected by g_yy component
            float yScale = std::sqrt(std::abs(metric[2][2]));
            basisY = origBasisY * yScale;
            
            // Z basis affected by g_zz component
            float zScale = (dim > 3) ? std::sqrt(std::abs(metric[3][3])) : 1.0f;
            basisZ = origBasisZ * zScale;
            
            // For off-diagonal components, we add a shear effect
            if (dim > 2) {
                if (std::abs(metric[1][2]) > 0.01f) {
                    basisX = basisX + origBasisY * metric[1][2] * 0.2f;
                    basisY = basisY + origBasisX * metric[1][2] * 0.2f;
                }
            }
        }
    }
    
    void draw() const {
        // Draw the grid point
        glColor3f(0.7f, 0.7f, 0.7f);
        glPointSize(3.0f);
        glBegin(GL_POINTS);
        glVertex3f(displaced.x, displaced.y, displaced.z);
        glEnd();
        
        // Draw basis vectors
        glBegin(GL_LINES);
        
        // X basis vector (red)
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(displaced.x, displaced.y, displaced.z);
        glVertex3f(displaced.x + basisX.x, displaced.y + basisX.y, displaced.z + basisX.z);
        
        // Y basis vector (green)
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(displaced.x, displaced.y, displaced.z);
        glVertex3f(displaced.x + basisY.x, displaced.y + basisY.y, displaced.z + basisY.z);
        
        // Z basis vector (blue)
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(displaced.x, displaced.y, displaced.z);
        glVertex3f(displaced.x + basisZ.x, displaced.y + basisZ.y, displaced.z + basisZ.z);
        
        glEnd();
    }
};

// Class to represent a particle moving in curved spacetime
class Particle {
public:
    Vector3 position;    // Current position
    Vector3 velocity;    // Current velocity
    Vector3 acceleration; // Current acceleration
    std::vector<Vector3> trail; // Trail of previous positions
    float size;          // Visual size
    sf::Color color;     // Particle color
    Vector3 prevPosition; // Previous position for Verlet integration
    float properTime;    // Proper time for the particle (time dilation)
    
    Particle(const Vector3& pos, const Vector3& vel, float s, const sf::Color& col) 
        : position(pos), velocity(vel), acceleration(Vector3()), size(s), color(col), 
          prevPosition(pos - vel * TIME_STEP), properTime(0.0f) {
        // Initialize with zero acceleration
        trail.reserve(100); // Reserve space for trail
        trail.push_back(position); // Add initial position to trail
    }
    
    // Update particle position based on spacetime curvature
    void update(const std::vector<Mass>& masses, const std::vector<GridPoint>& grid, float deltaTime) {
        // Store current position for next iteration
        Vector3 temp = position;
        
        // Calculate acceleration due to spacetime curvature
        calculateAcceleration(masses, grid);
        
        // Update position using Verlet integration (more accurate for orbital motion)
        // x(t+dt) = 2*x(t) - x(t-dt) + a(t)*dt^2
        position = position * 2.0f - prevPosition + acceleration * (deltaTime * deltaTime);
        
        // Update velocity (for visualization and other calculations)
        velocity = (position - prevPosition) * (1.0f / (2.0f * deltaTime));
        
        // Store current position as previous for next iteration
        prevPosition = temp;
        
        // Calculate proper time (time dilation effect)
        // In general relativity, proper time passes more slowly in stronger gravitational fields
        float gravitationalPotential = 0.0f;
        for (const auto& mass : masses) {
            float distance = (position - mass.position).length();
            if (distance > 0.1f) {
                gravitationalPotential += G * mass.mass / distance;
            }
        }
        
        // Time dilation factor (simplified from full GR equation)
        // dτ/dt = sqrt(1 - 2GM/rc²)
        float timeDilationFactor = 1.0f - (2.0f * gravitationalPotential * TIME_DILATION_FACTOR);
        timeDilationFactor = std::max(0.1f, timeDilationFactor); // Prevent negative or zero values
        
        // Update proper time
        properTime += deltaTime * timeDilationFactor;
        
        // Add current position to trail
        if (trail.size() > 100) {
            trail.erase(trail.begin());
        }
        trail.push_back(position);
    }
    
    // Update particle using the metric and Christoffel symbols from Python model
    void updateWithTensor(const Tensor& tensor, float deltaTime) {
        Vector3 temp = position;
        
        // Calculate acceleration using Christoffel symbols (geodesic equation)
        calculateAccelerationFromTensor(tensor);
        
        // Update position using Verlet integration
        position = position * 2.0f - prevPosition + acceleration * (deltaTime * deltaTime);
        
        // Update velocity
        velocity = (position - prevPosition) * (1.0f / (2.0f * deltaTime));
        
        // Store current position as previous for next iteration
        prevPosition = temp;
        
        // Calculate proper time using the metric tensor
        auto metric = tensor.getMetricAt(position);
        
        // ds² = g_μν dx^μ dx^ν
        float ds_squared = 0.0f;
        
        // Assuming velocity components match metric indices
        for (int i = 0; i < tensor.dim && i < 3; i++) {
            for (int j = 0; j < tensor.dim && j < 3; j++) {
                // Get velocity components
                float v_i = 0.0f, v_j = 0.0f;
                
                if (i == 0) v_i = 1.0f;  // dt component (assuming dt=1)
                else if (i == 1) v_i = velocity.x;
                else if (i == 2) v_j = velocity.y;
                
                if (j == 0) v_j = 1.0f;  // dt component
                else if (j == 1) v_j = velocity.x;
                else if (j == 2) v_j = velocity.y;
                
                ds_squared += metric[i][j] * v_i * v_j;
            }
        }
        
        // Proper time increment: dτ = sqrt(-ds²)
        // Note: In a timelike path, ds² < 0
        float dtau = deltaTime;
        if (ds_squared < 0) {
            dtau = deltaTime * std::sqrt(-ds_squared);
        }
        
        properTime += dtau;
        
        // Add current position to trail
        if (trail.size() > 100) {
            trail.erase(trail.begin());
        }
        trail.push_back(position);
    }
    
    // Calculate acceleration based on spacetime curvature (geodesic equation)
    void calculateAcceleration(const std::vector<Mass>& masses, const std::vector<GridPoint>& grid) {
        // Reset acceleration
        acceleration = Vector3(0, 0, 0);
        
        // In general relativity, the geodesic equation describes how objects move in curved spacetime
        // The full equation is complex, but we can approximate it for visualization
        
        // First, calculate the Newtonian gravitational acceleration as a base
        for (const auto& mass : masses) {
            Vector3 direction = mass.position - position;
            float distance = direction.length();
            
            if (distance < 0.5f) {
                distance = 0.5f; // Avoid extreme forces at very close distances
            }
            
            // Calculate gravitational acceleration (a = GM/r²)
            float forceMagnitude = G * mass.mass * GRAVITY_SCALE / (distance * distance);
            acceleration = acceleration + direction.normalized() * forceMagnitude;
        }
        
        // Now add general relativistic corrections
        
        // 1. Perihelion precession effect
        // This causes orbits to rotate slightly with each revolution
        Vector3 velocityDirection = velocity.normalized();
        float speed = velocity.length();
        
        for (const auto& mass : masses) {
            Vector3 direction = mass.position - position;
            float distance = direction.length();
            
            if (distance < 1.0f || speed < 0.1f) continue;
            
            // Calculate the cross product of velocity and direction to mass
            Vector3 perpDirection;
            perpDirection.x = velocityDirection.y * direction.z - velocityDirection.z * direction.y;
            perpDirection.y = velocityDirection.z * direction.x - velocityDirection.x * direction.z;
            perpDirection.z = velocityDirection.x * direction.y - velocityDirection.y * direction.x;
            
            if (perpDirection.length() > 0.001f) {
                perpDirection = perpDirection.normalized();
                
                // The precession effect is stronger for:
                // - Higher speeds
                // - Stronger gravitational fields (larger masses)
                // - Closer distances
                float precessionFactor = 0.2f * mass.mass * speed / (distance * distance);
                
                // Add a small perpendicular acceleration to create the precession
                acceleration = acceleration + perpDirection * precessionFactor;
            }
        }
        
        // 2. Frame-dragging effect (Lense-Thirring effect)
        // This causes particles to be dragged in the direction of a rotating mass
        // For simplicity, we'll assume masses rotate around the z-axis
        for (const auto& mass : masses) {
            Vector3 direction = mass.position - position;
            float distance = direction.length();
            
            if (distance < 1.0f) continue;
            
            // Direction of frame dragging (perpendicular to both the z-axis and the direction to mass)
            Vector3 dragDirection;
            dragDirection.x = -direction.y;
            dragDirection.y = direction.x;
            dragDirection.z = 0;
            
            if (dragDirection.length() > 0.001f) {
                dragDirection = dragDirection.normalized();
                
                // Frame dragging is stronger for larger masses and closer distances
                float dragFactor = 0.1f * mass.mass / (distance * distance * distance);
                
                // Add the frame-dragging acceleration
                acceleration = acceleration + dragDirection * dragFactor;
            }
        }
        
        // 3. Use the grid curvature information to refine the acceleration
        std::vector<std::pair<GridPoint, float>> nearestPoints;
        findNearestGridPoints(grid, nearestPoints);
        
        if (!nearestPoints.empty()) {
            // Calculate weighted average of curvature at particle position
            float totalCurvature = 0.0f;
            float totalWeight = 0.0f;
            
            for (const auto& pointPair : nearestPoints) {
                const GridPoint& point = pointPair.first;
                float weight = pointPair.second;
                
                totalCurvature += point.curvatureDepth * weight;
                totalWeight += weight;
            }
            
            if (totalWeight > 0.0f) {
                float avgCurvature = totalCurvature / totalWeight;
                
                // In areas of higher curvature, the geodesic deviation is stronger
                // This affects how the particle's path bends
                for (const auto& mass : masses) {
                    Vector3 direction = mass.position - position;
                    float distance = direction.length();
                    
                    if (distance < 0.5f) continue;
                    
                    // Calculate a curvature-based correction factor
                    float curvatureFactor = avgCurvature * 0.3f;
                    
                    // Apply additional acceleration based on curvature
                    acceleration = acceleration + direction.normalized() * curvatureFactor;
                }
            }
        }
    }
    
    // Calculate acceleration using the geodesic equation with Christoffel symbols
    void calculateAccelerationFromTensor(const Tensor& tensor) {
        // Reset acceleration
        acceleration = Vector3(0, 0, 0);
        
        // Get Christoffel symbols at the current position
        auto christoffel = tensor.getChristoffelAt(position);
        
        // Apply the geodesic equation:
        // d²x^μ/dt² + Γ^μ_νρ (dx^ν/dt)(dx^ρ/dt) = 0
        
        // Map our 3D velocity to the appropriate tensor components
        std::vector<float> velocity_components;
        velocity_components.push_back(1.0f);  // dt/dt = 1
        velocity_components.push_back(velocity.x);
        velocity_components.push_back(velocity.y);
        if (tensor.dim > 3) {
            velocity_components.push_back(velocity.z);
        }
        
        // Fill remaining components with zeros if needed
        while (velocity_components.size() < tensor.dim) {
            velocity_components.push_back(0.0f);
        }
        
        // Calculate acceleration components
        std::vector<float> accel_components(tensor.dim, 0.0f);
        
        for (int mu = 0; mu < tensor.dim; mu++) {
            for (int nu = 0; nu < tensor.dim; nu++) {
                for (int rho = 0; rho < tensor.dim; rho++) {
                    accel_components[mu] -= christoffel[mu][nu][rho] * 
                                           velocity_components[nu] * 
                                           velocity_components[rho];
                }
            }
        }
        
        // Convert back to 3D acceleration
        if (tensor.dim > 1) acceleration.x = accel_components[1];
        if (tensor.dim > 2) acceleration.y = accel_components[2];
        if (tensor.dim > 3) acceleration.z = accel_components[3];
        
        // Scale for visualization
        float accelMagnitude = acceleration.length();
        if (accelMagnitude > 10.0f) {
            acceleration = acceleration * (10.0f / accelMagnitude);
        }
    }
    
    // Find the nearest grid points for interpolation
    void findNearestGridPoints(const std::vector<GridPoint>& grid, 
                              std::vector<std::pair<GridPoint, float>>& nearestPoints) {
        const float MAX_DISTANCE = 5.0f;
        
        for (const auto& point : grid) {
            float distance = (point.position - position).length();
            
            if (distance < MAX_DISTANCE) {
                // Weight is inversely proportional to distance
                float weight = 1.0f / (1.0f + distance);
                nearestPoints.push_back(std::make_pair(point, weight));
            }
        }
        
        // Sort by weight (highest first)
        std::sort(nearestPoints.begin(), nearestPoints.end(), 
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Limit to the closest few points by removing excess elements
        if (nearestPoints.size() > 8) {
            nearestPoints.erase(nearestPoints.begin() + 8, nearestPoints.end());
        }
    }
    
    // Draw the particle and its trail
    void draw() const {
        // Draw trail with color based on proper time (time dilation visualization)
        if (trail.size() > 1) {
            glLineWidth(2.0f);
            glBegin(GL_LINE_STRIP);
            
            for (size_t i = 0; i < trail.size(); ++i) {
                // Fade trail from full color to transparent
                float alpha = static_cast<float>(i) / trail.size();
                
                // Use the particle's color but adjust based on proper time
                // This visualizes time dilation - redder means time is passing more slowly
                float timeFactor = std::min(1.0f, properTime * 0.01f);
                float r = color.r / 255.0f;
                float g = (color.g / 255.0f) * timeFactor;
                float b = (color.b / 255.0f) * timeFactor;
                
                glColor4f(r, g, b, alpha);
                glVertex3f(trail[i].x, trail[i].y, trail[i].z);
            }
            
            glEnd();
        }
        
        // Draw particle
        glPushMatrix();
        glTranslatef(position.x, position.y, position.z);
        
        // Color particle based on its velocity (relativistic doppler effect)
        float speed = velocity.length();
        float dopplerFactor = std::min(1.0f, speed / 5.0f);
        
        // Approaching = blueshift, receding = redshift (simplified)
        float r = color.r / 255.0f;
        float g = color.g / 255.0f;
        float b = color.b / 255.0f;
        
        // Adjust color based on velocity direction relative to viewer
        Vector3 viewerDirection(0, 0, 1); // Assuming viewer is above the plane
        float dotProduct = velocity.x * viewerDirection.x + 
                          velocity.y * viewerDirection.y + 
                          velocity.z * viewerDirection.z;
        
        if (dotProduct > 0) {
            // Receding - redshift
            g *= (1.0f - dopplerFactor * 0.5f);
            b *= (1.0f - dopplerFactor * 0.5f);
        } else {
            // Approaching - blueshift
            r *= (1.0f - dopplerFactor * 0.5f);
            g *= (1.0f - dopplerFactor * 0.3f);
        }
        
        glColor3f(r, g, b);
        
        // Create a sphere for the particle
        GLUquadricObj* sphere = gluNewQuadric();
        gluQuadricDrawStyle(sphere, GLU_FILL);
        gluSphere(sphere, size, 16, 16);
        gluDeleteQuadric(sphere);
        
        glPopMatrix();
    }
};

class SpacetimeSimulator {
private:
    sf::RenderWindow window;
    std::vector<Mass> masses;
    std::vector<GridPoint> grid;
    std::vector<Particle> particles;  // Add particles vector
    
    // Tensor for metric and Christoffel symbols from Python model
    Tensor* tensor;
    bool useModelData;   // Flag to toggle between traditional simulation and model-based
    
    // Camera parameters
    float cameraAngleX = 35.0f;  // Adjusted for better initial view
    float cameraAngleY = -45.0f;  // Adjusted for better initial view
    float cameraDistance = 25.0f;  // Increased for better overall view
    Vector3 cameraTarget;
    
    // Mouse control parameters
    bool isRotating = false;
    sf::Vector2i lastMousePos;
    
    // Simulation clock
    sf::Clock simulationClock;
    float elapsedTime = 0.0f;
    
public:
    SpacetimeSimulator() : 
        window(sf::VideoMode(sf::Vector2u(WINDOW_WIDTH, WINDOW_HEIGHT)), 
               "3D Spacetime Curvature Simulator with Manifold Integration", 
               sf::Style::Default),
        useModelData(false) {
        
        // Set OpenGL context settings
        sf::ContextSettings settings;
        settings.depthBits = 24;
        settings.stencilBits = 8;
        settings.antiAliasingLevel = 4;
        settings.majorVersion = 3;
        settings.minorVersion = 3;
        settings.attributeFlags = sf::ContextSettings::Core;
        
        // Note: In SFML 3.0, we can't change settings after window creation
        // We would need to recreate the window with these settings
        
        window.setFramerateLimit(60);
        
        // Initialize OpenGL
        initOpenGL();
        
        // Create the spacetime grid
        createGrid();
        
        // Initialize tensor with appropriate dimension
        // Typically 4D for spacetime (t, x, y, z)
        tensor = new Tensor(4);
        
        // Add initial masses with better spacing for visualization
        // Sun-like mass
        addMass(Vector3(-7, -7, 0), 300.0f);
        
        // Supermassive black hole
        addMass(Vector3(7, 7, 0), 600.0f);
        
        // Neutron star
        addMass(Vector3(10, -8, 0), 150.0f);
        
        // Add initial particles with orbital trajectories
        // Particle in elliptical orbit around the first mass
        addParticle(Vector3(-10, -7, 0), Vector3(0, 2.5f, 0), 0.2f, sf::Color(0, 255, 255));  // Cyan
        
        // Particle in tight orbit around the second mass (black hole)
        addParticle(Vector3(7, 10, 0), Vector3(-3.0f, 0, 0), 0.2f, sf::Color(255, 255, 0));  // Yellow
        
        // Particle in figure-8 trajectory between two masses
        addParticle(Vector3(0, 0, 0), Vector3(1.8f, 1.8f, 0), 0.2f, sf::Color(255, 0, 255));  // Magenta
        
        // Try to load tensor data from Python model output
        loadModelData();
    }
    
    ~SpacetimeSimulator() {
        if (tensor) {
            delete tensor;
        }
    }
    
    void initOpenGL() {
        // Set up the OpenGL state
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // Black background
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        
        // Set up the projection matrix
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0f, static_cast<float>(WINDOW_WIDTH) / WINDOW_HEIGHT, 0.1f, 100.0f);
        
        // Set up initial modelview matrix
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
    }
    
    void createGrid() {
        float halfSize = GRID_SIZE * GRID_SPACING / 2.0f;
        for (int x = 0; x < GRID_SIZE; ++x) {
            float xPos = x * GRID_SPACING - halfSize;
            for (int y = 0; y < GRID_SIZE; ++y) {
                float yPos = y * GRID_SPACING - halfSize;
                // Create a 2D grid at z=0 (representing flat spacetime)
                grid.emplace_back(Vector3(xPos, yPos, 0));
            }
        }
        
        // Pre-calculate initial grid displacements to avoid initial animation
        for (auto& point : grid) {
            point.calculateDisplacement(masses);
        }
    }
    
    // Load data exported from Python model
    bool loadModelData() {
        bool success = true;
        
        // Try to load the metric tensor
        if (!tensor->loadMetricFromCSV("metric_tensor.csv")) {
            std::cout << "Warning: Could not load metric tensor, falling back to default simulation" << std::endl;
            success = false;
        }
        
        // Try to load Christoffel symbols
        if (!tensor->loadChristoffelFromCSV("christoffel_symbols.csv")) {
            std::cout << "Warning: Could not load Christoffel symbols, falling back to default simulation" << std::endl;
            success = false;
        }
        
        // Try to load Riemann tensor (optional)
        tensor->loadRiemannFromCSV("riemann_tensor.csv");
        
        // Set flag based on load success
        useModelData = success;
        
        // If successful, update grid with the loaded metric data
        if (useModelData) {
            updateGridFromTensor();
        }
        
        return success;
    }
    
    // Update grid points based on the loaded tensor data
    void updateGridFromTensor() {
        for (auto& point : grid) {
            point.calculateDisplacementFromMetric(*tensor);
        }
    }
    
    void addMass(const Vector3& position, float mass) {
        // Convert mass to solar masses for more realistic scale
        // Sun's mass is about 2e30 kg
        float solarMasses = mass;
        masses.emplace_back(position, solarMasses);
    }
    
    void addRandomMasses(int count) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> posDist(-12.0f, 12.0f);
        std::uniform_real_distribution<float> massDist(80.0f, 300.0f);
        
        for (int i = 0; i < count; ++i) {
            addMass(Vector3(posDist(gen), posDist(gen), 0), massDist(gen));
        }
    }
    
    void addParticle(const Vector3& position, const Vector3& velocity, float size, const sf::Color& color) {
        particles.emplace_back(position, velocity, size, color);
    }
    
    void handleEvents() {
        // Using the new SFML 3.0 event handling API
        while (std::optional<sf::Event> eventOpt = window.pollEvent()) {
            const sf::Event& event = *eventOpt;
            
            if (event.is<sf::Event::Closed>()) {
                window.close();
            }
            else if (const auto* mouseButtonPressed = event.getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressed->button == sf::Mouse::Button::Left) {
                    isRotating = true;
                    lastMousePos = sf::Mouse::getPosition(window);
                }
                else if (mouseButtonPressed->button == sf::Mouse::Button::Right) {
                    // Cast a ray and add a mass where it intersects the grid plane
                    sf::Vector3f worldPos = getWorldPosition(mouseButtonPressed->position.x, mouseButtonPressed->position.y);
                    addMass(Vector3(worldPos.x, worldPos.y, worldPos.z), 50.0f);
                }
            }
            else if (const auto* mouseButtonReleased = event.getIf<sf::Event::MouseButtonReleased>()) {
                if (mouseButtonReleased->button == sf::Mouse::Button::Left) {
                    isRotating = false;
                }
            }
            else if (const auto* mouseMoved = event.getIf<sf::Event::MouseMoved>()) {
                if (isRotating) {
                    sf::Vector2i currentPos = sf::Vector2i(mouseMoved->position.x, mouseMoved->position.y);
                    sf::Vector2i delta = currentPos - lastMousePos;
                    
                    cameraAngleY += delta.x * 0.5f;
                    cameraAngleX += delta.y * 0.5f;
                    
                    // Clamp vertical angle to avoid flipping
                    cameraAngleX = std::max(-89.0f, std::min(89.0f, cameraAngleX));
                    
                    lastMousePos = currentPos;
                }
            }
            else if (const auto* mouseWheelScrolled = event.getIf<sf::Event::MouseWheelScrolled>()) {
                cameraDistance -= mouseWheelScrolled->delta;
                cameraDistance = std::max(5.0f, std::min(50.0f, cameraDistance));
            }
            else if (const auto* keyPressed = event.getIf<sf::Event::KeyPressed>()) {
                if (keyPressed->scancode == sf::Keyboard::Scancode::C) {
                    // Clear all masses except the central one
                    if (!masses.empty()) {
                        Mass centralMass = masses[0];
                        masses.clear();
                        masses.push_back(centralMass);
                    }
                }
                else if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
                    // Add random masses
                    addRandomMasses(3);
                }
                else if (keyPressed->scancode == sf::Keyboard::Scancode::P) {
                    // Add a new particle with orbital velocity around the nearest mass
                    std::random_device rd;
                    std::mt19937 gen(rd());
                    std::uniform_real_distribution<float> posDist(-12.0f, 12.0f);
                    std::uniform_real_distribution<float> speedDist(1.5f, 3.5f);
                    std::uniform_real_distribution<float> colorDist(0.0f, 1.0f);
                    std::uniform_real_distribution<float> eccentricityDist(0.0f, 0.5f);  // For elliptical orbits
                    
                    // Random position
                    Vector3 pos(posDist(gen), posDist(gen), 0);
                    
                    // Find the nearest mass to orbit
                    const Mass* nearestMass = nullptr;
                    float minDistance = std::numeric_limits<float>::max();
                    
                    for (const auto& mass : masses) {
                        float distance = (mass.position - pos).length();
                        if (distance < minDistance) {
                            minDistance = distance;
                            nearestMass = &mass;
                        }
                    }
                    
                    // Calculate orbital velocity if we found a mass
                    Vector3 vel;
                    if (nearestMass && minDistance > 0.5f) {
                        // Direction perpendicular to the line connecting particle and mass
                        Vector3 direction = pos - nearestMass->position;
                        Vector3 perpDirection;
                        
                        // Cross product with up vector to get perpendicular direction
                        perpDirection.x = direction.y;
                        perpDirection.y = -direction.x;
                        perpDirection.z = 0;
                        
                        if (perpDirection.length() > 0.001f) {
                            perpDirection = perpDirection.normalized();
                            
                            // Calculate orbital speed based on mass and distance
                            // v = sqrt(GM/r) for circular orbit
                            float orbitalSpeed = std::sqrt(G * nearestMass->mass * GRAVITY_SCALE / minDistance);
                            
                            // Scale for better visualization
                            orbitalSpeed = std::min(orbitalSpeed, 5.0f);
                            orbitalSpeed = std::max(orbitalSpeed, 1.0f);
                            
                            // Add some eccentricity to create elliptical orbits
                            float eccentricity = eccentricityDist(gen);
                            
                            // Adjust velocity for elliptical orbit
                            // For an elliptical orbit, we adjust the perpendicular component
                            // and add a small radial component
                            Vector3 radialDirection = direction.normalized();
                            
                            // Perpendicular component (determines angular momentum)
                            vel = perpDirection * orbitalSpeed;
                            
                            // Add radial component (creates eccentricity)
                            vel = vel + radialDirection * (orbitalSpeed * eccentricity * 0.5f);
                        } else {
                            vel = Vector3(speedDist(gen), speedDist(gen), 0).normalized() * speedDist(gen);
                        }
                    } else {
                        // Random velocity if no mass is nearby
                        vel = Vector3(speedDist(gen), speedDist(gen), 0).normalized() * speedDist(gen);
                    }
                    
                    // Random color
                    sf::Color color(
                        static_cast<uint8_t>(colorDist(gen) * 255),
                        static_cast<uint8_t>(colorDist(gen) * 255),
                        static_cast<uint8_t>(colorDist(gen) * 255)
                    );
                    
                    addParticle(pos, vel, 0.2f, color);
                }
                else if (keyPressed->scancode == sf::Keyboard::Scancode::M) {
                    // Toggle between model-based and traditional simulation
                    useModelData = !useModelData;
                    std::cout << "Using " << (useModelData ? "model data" : "traditional simulation") << std::endl;
                    
                    // Update grid based on the selected mode
                    if (useModelData) {
                        updateGridFromTensor();
                    } else {
                        for (auto& point : grid) {
                            point.calculateDisplacement(masses);
                        }
                    }
                }
                else if (keyPressed->scancode == sf::Keyboard::Scancode::L) {
                    // Reload model data
                    loadModelData();
                }
            }
        }
    }
    
    sf::Vector3f getWorldPosition(int screenX, int screenY) {
        // Simple ray-plane intersection to get world position from screen position
        // This is a simplified version that intersects with the z=0 plane
        
        // Convert screen coordinates to normalized device coordinates
        float x = 2.0f * screenX / WINDOW_WIDTH - 1.0f;
        float y = 1.0f - 2.0f * screenY / WINDOW_HEIGHT;
        
        // Get camera position
        float camX = cameraDistance * std::sin(cameraAngleY * M_PI / 180.0f) * std::cos(cameraAngleX * M_PI / 180.0f);
        float camY = cameraDistance * std::sin(cameraAngleX * M_PI / 180.0f);
        float camZ = cameraDistance * std::cos(cameraAngleY * M_PI / 180.0f) * std::cos(cameraAngleX * M_PI / 180.0f);
        
        // Calculate ray direction in world space (simplified)
        float dirX = -camX + x * 10.0f;
        float dirY = -camY + y * 10.0f;
        float dirZ = -camZ;
        
        // Normalize direction
        float length = std::sqrt(dirX * dirX + dirY * dirY + dirZ * dirZ);
        dirX /= length;
        dirY /= length;
        dirZ /= length;
        
        // Ray-plane intersection (z=0 plane)
        if (std::abs(dirZ) < 1e-10) return sf::Vector3f(0, 0, 0);  // Parallel to plane
        
        float t = -camZ / dirZ;
        float worldX = camX + dirX * t;
        float worldY = camY + dirY * t;
        
        return sf::Vector3f(worldX, worldY, 0);
    }
    
    void run() {
        while (window.isOpen()) {
            handleEvents();
            update();
            render();
        }
    }
    
    void update() {
        // Get elapsed time since last update
        float deltaTime = simulationClock.restart().asSeconds();
        elapsedTime += deltaTime;
        
        // Limit delta time to avoid instability
        if (deltaTime > 0.1f) {
            deltaTime = 0.1f;
        }
        
        // Calculate grid displacements based on the selected mode
        if (useModelData) {
            updateGridFromTensor();
        } else {
            for (auto& point : grid) {
                point.calculateDisplacement(masses);
            }
        }
        
        // Update particles based on the selected mode
        for (auto& particle : particles) {
            if (useModelData) {
                particle.updateWithTensor(*tensor, deltaTime);
            } else {
                particle.update(masses, grid, deltaTime);
            }
        }
        
        // Check if particles are too far away and remove them
        particles.erase(std::remove_if(particles.begin(), particles.end(),
            [](const Particle& p) { return p.position.length() > 30.0f; }),
            particles.end());
    }
    
    void render() {
        // Clear the window
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Set up the camera
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        
        // Calculate camera position based on angles and distance
        float camX = cameraDistance * std::sin(cameraAngleY * M_PI / 180.0f) * std::cos(cameraAngleX * M_PI / 180.0f);
        float camY = cameraDistance * std::sin(cameraAngleX * M_PI / 180.0f);
        float camZ = cameraDistance * std::cos(cameraAngleY * M_PI / 180.0f) * std::cos(cameraAngleX * M_PI / 180.0f);
        
        // Set up the view matrix
        gluLookAt(
            camX, camY, camZ,  // Camera position
            0, 0, 0,           // Look at the origin
            0, 1, 0            // Up vector
        );
        
        // Draw the spacetime mesh
        drawSpacetimeMesh();
        
        // Draw masses
        for (const auto& mass : masses) {
            // Draw a sphere for each mass
            glPushMatrix();
            glTranslatef(mass.position.x, mass.position.y, mass.position.z);
            
            // Choose color based on mass (larger masses are more red - representing hotter/more energetic objects)
            float colorIntensity = std::min(1.0f, mass.mass / 500.0f);
            glColor3f(1.0f, 1.0f - colorIntensity * 0.8f, 1.0f - colorIntensity * 0.9f);  // White to red gradient
            
            // Create a sphere
            GLUquadricObj* sphere = gluNewQuadric();
            gluQuadricDrawStyle(sphere, GLU_FILL);
            gluSphere(sphere, mass.radius, 32, 32);
            gluDeleteQuadric(sphere);
            
            glPopMatrix();
        }
        
        // Draw particles
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        for (const auto& particle : particles) {
            particle.draw();
        }
        
        glDisable(GL_BLEND);
        
        // Draw instructions text using SFML
        window.pushGLStates();
        
        sf::Font font;
        if (font.openFromFile("/System/Library/Fonts/Helvetica.ttc")) {
            std::string modeStr = useModelData ? "MODEL-BASED (Python Manifold)" : "TRADITIONAL (GR approximation)";
            sf::Text text(font, 
                "Controls:\n"
                "Left-click + drag: Rotate view\n"
                "Right-click: Add mass at location\n"
                "Mouse wheel: Zoom in/out\n"
                "C: Clear additional masses\n"
                "R: Add random masses\n"
                "P: Add random particle\n"
                "M: Toggle simulation mode\n"
                "L: Reload Python model data\n\n"
                "Current mode: " + modeStr,
                14);
            text.setFillColor(sf::Color::White);
            text.setPosition(sf::Vector2f(10, 10));
            window.draw(text);
        }
        
        window.popGLStates();
        
        // Display the window
        window.display();
    }
    
    void drawSpacetimeMesh() {
        // Draw the mesh surface with transparency
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        // Enable depth sorting for transparent objects
        glDepthMask(GL_FALSE);
        
        // Draw the surface as quads
        for (int y = 0; y < GRID_SIZE - 1; ++y) {
            for (int x = 0; x < GRID_SIZE - 1; ++x) {
                int idx00 = y * GRID_SIZE + x;
                int idx10 = y * GRID_SIZE + (x + 1);
                int idx01 = (y + 1) * GRID_SIZE + x;
                int idx11 = (y + 1) * GRID_SIZE + (x + 1);
                
                const auto& p00 = grid[idx00].displaced;
                const auto& p10 = grid[idx10].displaced;
                const auto& p01 = grid[idx01].displaced;
                const auto& p11 = grid[idx11].displaced;
                
                // Average curvature for this quad
                float avgCurvature = (grid[idx00].curvatureDepth + 
                                     grid[idx10].curvatureDepth + 
                                     grid[idx01].curvatureDepth + 
                                     grid[idx11].curvatureDepth) / 4.0f;
                
                // Color based on curvature with transparency
                float intensity = std::min(1.0f, avgCurvature / 4.0f);
                float alpha = 0.25f + intensity * 0.4f;  // More curved = more visible
                
                // Smoother color transition
                glBegin(GL_QUADS);
                
                // Use per-vertex coloring for smoother gradients
                float i00 = std::min(1.0f, grid[idx00].curvatureDepth / 4.0f);
                float i10 = std::min(1.0f, grid[idx10].curvatureDepth / 4.0f);
                float i01 = std::min(1.0f, grid[idx01].curvatureDepth / 4.0f);
                float i11 = std::min(1.0f, grid[idx11].curvatureDepth / 4.0f);
                
                // Use a more physically accurate color scheme (gravitational redshift effect)
                // Deeper curvature = more redshifted
                glColor4f(1.0f, 1.0f - i00 * 0.7f, 1.0f - i00 * 0.8f, alpha);
                glVertex3f(p00.x, p00.y, p00.z);
                
                glColor4f(1.0f, 1.0f - i10 * 0.7f, 1.0f - i10 * 0.8f, alpha);
                glVertex3f(p10.x, p10.y, p10.z);
                
                glColor4f(1.0f, 1.0f - i11 * 0.7f, 1.0f - i11 * 0.8f, alpha);
                glVertex3f(p11.x, p11.y, p11.z);
                
                glColor4f(1.0f, 1.0f - i01 * 0.7f, 1.0f - i01 * 0.8f, alpha);
                glVertex3f(p01.x, p01.y, p01.z);
                
                glEnd();
            }
        }
        
        // Restore depth mask for opaque objects
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        
        // Draw grid lines with gradient color based on curvature
        glLineWidth(1.2f);  // Slightly thinner lines for cleaner look with denser grid
        
        // Draw X grid lines
        for (int y = 0; y < GRID_SIZE; y += 1) {
            glBegin(GL_LINE_STRIP);
            for (int x = 0; x < GRID_SIZE; ++x) {
                int index = y * GRID_SIZE + x;
                const auto& point = grid[index];
                
                // Color gradient based on curvature depth (gravitational redshift effect)
                float intensity = std::min(1.0f, point.curvatureDepth / 4.0f);
                glColor3f(1.0f, 1.0f - intensity * 0.6f, 1.0f - intensity * 0.8f);
                
                glVertex3f(point.displaced.x, point.displaced.y, point.displaced.z);
            }
            glEnd();
        }
        
        // Draw Y grid lines
        for (int x = 0; x < GRID_SIZE; x += 1) {
            glBegin(GL_LINE_STRIP);
            for (int y = 0; y < GRID_SIZE; ++y) {
                int index = y * GRID_SIZE + x;
                const auto& point = grid[index];
                
                // Color gradient based on curvature depth (gravitational redshift effect)
                float intensity = std::min(1.0f, point.curvatureDepth / 4.0f);
                glColor3f(1.0f, 1.0f - intensity * 0.6f, 1.0f - intensity * 0.8f);
                
                glVertex3f(point.displaced.x, point.displaced.y, point.displaced.z);
            }
            glEnd();
        }
    }
};

int main() {
    try {
        std::cout << "Starting 3D Spacetime Curvature Simulator with Manifold Integration..." << std::endl;
        SpacetimeSimulator sim;
        sim.run();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}