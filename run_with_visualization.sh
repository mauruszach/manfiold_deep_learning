#!/bin/bash
# Script to run the test simulation and real-time Python visualization together

# Change to the project root directory
cd "$(dirname "$0")"

# Define log files
TEST_LOG="test_simulation.log"
VIZ_LOG="visualization.log"

# Function to run a command in the background and log its output
run_background() {
    local cmd="$1"
    local log="$2"
    
    echo "Starting $cmd (logging to $log)"
    $cmd > "$log" 2>&1 &
    echo "Started process with PID $!"
}

# Clear any existing log files
> "$TEST_LOG"
> "$VIZ_LOG"

echo "Starting manifold learning visualization environment..."
echo

# Start the Python visualization in the background
echo "Starting Python visualization..."
run_background "python3 visualize_manifold.py" "$VIZ_LOG"
VISUALIZATION_PID=$!

# Wait a moment for the visualization to initialize
sleep 2

# Start the test simulation that generates metric changes
echo "Starting test simulation generating metric tensor changes..."
run_background "python3 test_simulation_updates.py" "$TEST_LOG"
TEST_PID=$!

echo
echo "Both processes are now running!"
echo "- Test simulation (PID: $TEST_PID) - Log: $TEST_LOG"
echo "- Visualization (PID: $VISUALIZATION_PID) - Log: $VIZ_LOG"
echo 
echo "Press Ctrl+C to stop both processes"

# Handle interrupt signal to terminate both processes cleanly
trap "echo 'Stopping all processes...'; kill $TEST_PID $VISUALIZATION_PID 2>/dev/null; exit" INT

# Wait for any process to finish or for user interrupt
wait

echo "All processes finished." 