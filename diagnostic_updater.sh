#!/bin/bash
# This script forcibly updates file timestamps to trigger C++ simulation reload
# Run this script if you notice that your model is training but simulation is not showing updates

cd "$(dirname "$0")"
WORKSPACE_DIR="$(pwd)"

echo "Forcibly updating file timestamps to trigger simulation reload..."

# Function to touch files if they exist
touch_if_exists() {
    if [ -f "$1" ]; then
        echo "Touching file: $1"
        touch "$1"
    fi
}

# Update all possible data files
touch_if_exists "$WORKSPACE_DIR/paths_on_manifold/metric_tensor.csv"
touch_if_exists "$WORKSPACE_DIR/paths_on_manifold/christoffel_symbols.csv"
touch_if_exists "$WORKSPACE_DIR/paths_on_manifold/metric_metadata.json"
touch_if_exists "$WORKSPACE_DIR/paths_on_manifold/data/metric_tensor.csv"
touch_if_exists "$WORKSPACE_DIR/paths_on_manifold/data/christoffel_symbols.csv"
touch_if_exists "$WORKSPACE_DIR/paths_on_manifold/data/metric_metadata.json"
touch_if_exists "$WORKSPACE_DIR/paths_on_manifold/metric_data_current.csv"
touch_if_exists "$WORKSPACE_DIR/paths_on_manifold/christoffel_data_current.csv"
touch_if_exists "$WORKSPACE_DIR/metric_tensor.csv"
touch_if_exists "$WORKSPACE_DIR/christoffel_symbols.csv"
touch_if_exists "$WORKSPACE_DIR/metric_metadata.json"

echo "Done! If the simulation is running, it should now reload the model data."
echo "If this doesn't work, make sure to press 'M' in the simulation to switch to model-based mode,"
echo "then press 'L' to manually trigger a reload." 