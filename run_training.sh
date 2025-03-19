#!/bin/bash
# Run just the path analysis component without the simulation

echo "Starting path analysis training..."
cd "$(dirname "$0")"
python3 paths_on_manifold/main.py --analyze 