# Paths on Manifold

A deep learning system for learning manifold structures and visualizing paths in curved spacetime.

## Features

- Train neural networks to learn manifold structures from geodesic paths
- Visualize learned manifolds in real-time using C++ OpenGL simulation
- Interactive learning mode with concurrent visualization
- Export learned metric tensors and Christoffel symbols
- Compatible with Python and C++ components

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.10+
- SFML and OpenGL (for C++ simulation)
- Blender (optional, for 3D visualization)

### Setting up the environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install as development package
pip install -e .
```

### Building the C++ Simulation

The C++ simulation visualizes the learned manifold in real-time. To build it:

```bash
# Compile using the provided build script
./build_simulation.sh
```

On macOS, you may need to install SFML first:

```bash
brew install sfml
```

On Linux:

```bash
# Ubuntu/Debian
sudo apt-get install libsfml-dev libgl1-mesa-dev

# Fedora/RHEL
sudo dnf install SFML-devel mesa-libGL-devel
```

## Usage

### Interactive Learning with Real-Time Visualization

The most exciting way to use this project is with interactive learning mode, where you can see the manifold evolve in real-time as the model trains:

```bash
python main.py --interactive
```

This will:
1. Start the C++ simulation window
2. Begin training the manifold model in the background
3. Update the visualization in real-time as the metric tensor evolves

### Other Usage Options

```bash
# Run path analysis (train the manifold model) without visualization
python main.py --analyze

# Run only the simulation to visualize previously trained models
python main.py --simulate

# Run the simulation in the background
python main.py --simulate --background

# Run multiple instances of the simulation
python main.py --simulate --parallel 4

# Run Blender visualization
python main.py --blender
```

## Controls for the C++ Simulation

- **Left-click + drag**: Rotate the view
- **Right-click**: Add a mass at the cursor location
- **Mouse wheel**: Zoom in/out
- **C key**: Clear additional masses
- **R key**: Add random masses
- **P key**: Add a random particle
- **M key**: Toggle between model-based and traditional simulation
- **L key**: Reload the Python model data
- **T key**: Toggle timeline view
- **Left/Right arrows**: Navigate timeline

## Project Structure

- `src/`: Core Python code for manifold learning
- `bin/`: Compiled executables
- `data/`: Data files and model outputs
- `visualization/`: Visualization tools and scripts
- `tests/`: Test scripts and examples

## Development

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

MIT 