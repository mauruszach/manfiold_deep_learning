# Manifold Deep Learning

A research framework for implementing and experimenting with deep learning on non-Euclidean spaces, particularly focusing on Riemannian manifolds and their applications in machine learning.

## Project Overview

This project explores the intersection of Riemannian geometry and deep learning, providing tools for:

1. Learning embeddings on manifolds with non-trivial metric structures
2. Computing and visualizing geodesic paths and curvature properties
3. Implementing optimization algorithms that respect manifold structure
4. Evolving metric tensors via Ricci flow and other geometric flows
5. Visualizing manifold structures and their properties

## Repository Structure

- `paths_on_manifold/`: Core implementation of manifold learning algorithms
  - `src/`: Source code
    - `manifold_learning/`: Python package for manifold learning (can be imported)
    - `path_analysis.py`: Original implementation (now refactored into module)
    - `simulation.cpp`: C++ implementation of simulation visualizations
    - `manifold_main.py`: Demonstration script for the package
  - `tests/`: Test cases for the algorithms
  - `examples/`: Example implementations and applications
  - `data/`: Sample datasets and generated manifold structures

## Getting Started

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/manifold_deep_learning.git
cd manifold_deep_learning
pip install -r requirements.txt
```

### Running the Demos

To run the manifold learning demos:

```bash
cd paths_on_manifold/src
python manifold_main.py --demo all
```

To run the simulation visualization:

```bash
./run_simulation.sh
```

To run the interactive mode:

```bash
./run_interactive.sh
```

## Documentation

For detailed documentation on the `manifold_learning` package, see:

```
paths_on_manifold/src/manifold_learning/README.md
```

## Key Features

### Manifold Embeddings

Learn embeddings onto manifolds with learnable metric tensors, allowing the model to adapt the geometry of the space to the data.

### Geodesic Calculations

Compute geodesic paths between points on the manifold, respecting the learned metric structure.

### Ricci Flow

Evolve metric tensors using Ricci flow, which tends to smooth out irregularities in the geometry and reveal underlying structure.

### Visualization Tools

Visualize manifold structures, geodesic paths, and curvature properties in 2D and 3D.

## Citations

If you use this code in your research, please cite:

```
@misc{manifold_deep_learning,
  author = {Maurus, Zach},
  title = {Manifold Deep Learning},
  year = {2023},
  publisher = {GitHub},
  url = {https://github.com/username/manifold_deep_learning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 