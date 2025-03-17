# Real-Time Manifold Learning Visualization

This project provides a real-time visualization of manifold learning, showing how a neural network learns the metric tensor of a manifold and how particles move in the resulting curved spacetime.

## Components

The system consists of two main components:

1. **Python Training Component** (`path_analysis.py`): Trains a neural network to learn the metric tensor of a manifold from data.
2. **C++ Visualization Component** (`simulation.cpp`): Visualizes the learned metric tensor and simulates particle motion in the resulting curved spacetime.

## Setup and Installation

### Prerequisites

- Python 3.7+ with NumPy, PyTorch, SciPy, Matplotlib, and SymPy
- C++ compiler with C++17 support
- SFML library for graphics
- OpenGL

### Installation

1. Install Python dependencies:
   ```bash
   pip install numpy torch scipy matplotlib sympy pandas
   ```

2. Install SFML:
   - macOS: `brew install sfml`
   - Linux: `sudo apt-get install libsfml-dev`
   - Windows: Download from [SFML website](https://www.sfml-dev.org/download.php)

3. Compile the C++ visualization:
   ```bash
   g++ -std=c++17 -DGL_SILENCE_DEPRECATION -o simulation simulation.cpp -I/opt/homebrew/opt/sfml/include -L/opt/homebrew/opt/sfml/lib -lsfml-graphics -lsfml-window -lsfml-system -framework OpenGL
   ```
   
   Note: Adjust the include and library paths based on your SFML installation.

## Usage

### Running the Visualization Test

To test the visualization without running the full training:

1. Run the test script:
   ```bash
   python test_visualization.py
   ```

2. In a separate terminal, run the visualization:
   ```bash
   ./simulation
   ```

The test script will generate a series of evolving metrics that transition from flat to curved spacetime, and the visualization will display the resulting curvature and particle motion.

### Running the Full Training with Visualization

1. Start the visualization:
   ```bash
   ./simulation
   ```

2. In a separate terminal, run the training:
   ```bash
   python path_analysis.py
   ```

The training script will periodically send updated metric tensors to the visualization, allowing you to see how the learned manifold evolves during training.

## Visualization Controls

- **Left-click + drag**: Rotate view
- **Right-click**: Add mass at location
- **Mouse wheel**: Zoom in/out
- **C**: Clear additional masses
- **R**: Add random masses
- **P**: Add random particle
- **M**: Toggle between model-based and traditional simulation
- **L**: Reload Python model data
- **T**: Toggle timeline view
- **Left/Right arrows**: Navigate timeline

## File-Based Communication

The Python and C++ components communicate through files:

- `metric_data_current.csv`: Contains the current metric tensor
- `christoffel_data_current.csv`: Contains the Christoffel symbols
- `metric_metadata.json`: Contains metadata about the training progress

## Timeline Feature

The timeline feature allows you to scrub through the evolution of the metric tensor during training. This is useful for visualizing how the manifold changes over time.

## Customization

You can customize the visualization by modifying the following parameters in `simulation.cpp`:

- `GRID_SIZE`: Number of grid points in each dimension
- `GRID_SPACING`: Spacing between grid points
- `PARTICLE_SPEED`: Base speed for particles
- `GRAVITY_SCALE`: Scale factor for gravitational effects

You can customize the training by modifying the parameters in `path_analysis.py`:

- `input_dim`: Dimension of input data
- `embedding_dim`: Dimension of the manifold embedding
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate for optimization
- `data_type`: Type of synthetic data to generate ('sphere', 'torus', 'swiss_roll', etc.)

## Troubleshooting

- If the visualization doesn't update, check that the file paths in both components match.
- If you see OpenGL deprecation warnings, you can silence them with the `-DGL_SILENCE_DEPRECATION` flag.
- If the visualization crashes, try reducing the grid size or particle count.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 