#!/bin/bash
# Run simulation and training in separate processes

cd "$(dirname "$0")"
WORKSPACE_DIR="$(pwd)"

# Ensure data directories exist
mkdir -p "$WORKSPACE_DIR/paths_on_manifold/data"

# Create placeholder files initially to make sure simulation loads correctly
echo "Creating initial placeholder data files..."
cat > "$WORKSPACE_DIR/paths_on_manifold/metric_tensor.csv" << EOF
1.0,0.0,0.0,0.0
0.0,1.0,0.0,0.0
0.0,0.0,1.0,0.0
0.0,0.0,0.0,1.0
EOF

cp "$WORKSPACE_DIR/paths_on_manifold/metric_tensor.csv" "$WORKSPACE_DIR/paths_on_manifold/data/metric_tensor.csv"

# Create empty Christoffel symbols file
echo "0,0,0,0.0" > "$WORKSPACE_DIR/paths_on_manifold/christoffel_symbols.csv"
echo "0,0,0,0.0" > "$WORKSPACE_DIR/paths_on_manifold/data/christoffel_symbols.csv"

# Create initial metadata
cat > "$WORKSPACE_DIR/paths_on_manifold/metric_metadata.json" << EOF
{
  "epoch": 0,
  "total_epochs": 50,
  "step": 0,
  "total_steps": 500,
  "loss": 0.0,
  "is_training": true
}
EOF

cp "$WORKSPACE_DIR/paths_on_manifold/metric_metadata.json" "$WORKSPACE_DIR/paths_on_manifold/data/metric_metadata.json"

# Start the C++ simulation in background
echo "Starting simulation in background..."
BIN_PATH="$WORKSPACE_DIR/paths_on_manifold/bin/simulation"

if [ ! -f "$BIN_PATH" ]; then
    echo "Error: Simulation executable not found at $BIN_PATH"
    echo "Attempting to compile the simulation..."
    
    BUILD_SCRIPT="$WORKSPACE_DIR/paths_on_manifold/build_simulation.sh"
    if [ -f "$BUILD_SCRIPT" ]; then
        chmod +x "$BUILD_SCRIPT"
        "$BUILD_SCRIPT"
    else
        echo "Build script not found. Please compile the simulation manually."
        exit 1
    fi
fi

# Run simulation in a separate process
"$BIN_PATH" &
SIMULATION_PID=$!

# Wait a moment for the simulation to initialize
sleep 2

# Start the path analysis in the current terminal
echo "Starting path analysis training..."
echo "The simulation is running in the background. You can interact with it while training progresses."
echo ""
echo "IMPORTANT: Press 'M' in the simulation window to switch to model-based mode if needed."
echo "Press 'L' to manually reload model data if visualization is not updating."
echo ""

python3 "$WORKSPACE_DIR/paths_on_manifold/main.py" --analyze

# When path analysis completes, ask if the user wants to close the simulation
echo "Path analysis complete. Simulation is still running."
read -p "Do you want to close the simulation? (y/n): " choice
if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
    echo "Closing simulation..."
    kill $SIMULATION_PID
else
    echo "Simulation is still running. Press Ctrl+C to close it when done."
    # Wait for the simulation to finish
    wait $SIMULATION_PID
fi

echo "Interactive session ended." 