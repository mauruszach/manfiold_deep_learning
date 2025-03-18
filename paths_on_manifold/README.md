# Manifold Deep Learning

A research project for learning and visualizing curved spacetime manifolds using deep learning techniques.

## Project Structure

```
paths_on_manifold/
├── bin/                  # Executables and binaries
│   └── simulation        # C++ simulation executable
├── data/                 # Data files and outputs
│   ├── christoffel_data_current.csv
│   ├── metric_data_current.csv
│   ├── metric_metadata.json
│   └── ...
├── docs/                 # Documentation
│   ├── README.md         # Detailed documentation
│   ├── manifold_learning.pdf
│   └── ...
├── models/               # Trained models and model definitions
├── src/                  # Source code
│   ├── path_analysis.py  # Core logic for path analysis on manifolds
│   ├── python_model_exporter.py
│   ├── run_wandb_monitor.py
│   ├── simulation.cpp    # C++ simulation code
│   └── ...
├── tests/                # Test files and results
│   ├── run_test.py
│   ├── test_results/
│   └── ...
├── visualization/        # Visualization scripts and tools
│   ├── blender_script.py
│   ├── blender_live_monitor.py
│   ├── run_live_visualization.py
│   └── test_visualization.py
├── main.py               # Main entry point
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.8+
- C++ compiler (for simulation)
- Blender 3.0+ (for 3D visualization)
- PyTorch 1.8+

### Setup

Clone the repository:

```bash
git clone <repository-url>
cd paths_on_manifold
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Compile the simulation (if needed):

```bash
cd src
g++ -o ../bin/simulation simulation.cpp -std=c++17 -O2
```

## Usage

The project provides a unified command-line interface through `main.py`:

```bash
# Run test visualization
python main.py --visualize

# Run C++ simulation
python main.py --simulate

# Run Blender visualization
python main.py --blender

# Run tests
python main.py --test

# Run path analysis
python main.py --analyze
```

## Key Components

### Manifold Learning

The core of the project is a deep learning approach to learning the structure of manifolds, particularly those representing curved spacetime in general relativity. The model learns to predict:

- Metric tensors
- Christoffel symbols
- Geodesic paths

### Visualization

Multiple visualization methods are provided:

1. **C++ Real-time Simulation**: Interactive visualization of particles moving on the manifold
2. **Blender Visualization**: High-quality rendering of the manifold and geodesics
3. **Test Visualization**: Simple visualization for testing and debugging

### Data Generation and Analysis

The project includes tools for:

- Generating synthetic manifold data
- Analyzing paths on manifolds
- Computing geometric quantities (Riemann tensor, Ricci curvature, etc.)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 