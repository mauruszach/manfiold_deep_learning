#!/bin/bash
# Run just the simulation component without the training

echo "Starting simulation..."
cd "$(dirname "$0")"
BIN_PATH="paths_on_manifold/bin/simulation"

if [ -f "$BIN_PATH" ]; then
    "$BIN_PATH"
else
    echo "Error: Simulation executable not found at $BIN_PATH"
    echo "Please compile the simulation first by running paths_on_manifold/build_simulation.sh"
    exit 1
fi 