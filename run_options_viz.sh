#!/bin/bash
# Script to collect options data from Interactive Brokers and visualize the option manifold in real-time

# Change to the project root directory
cd "$(dirname "$0")"

# Define log files
COLLECTOR_LOG="options_collector.log"
VIZ_LOG="options_visualization.log"

# Default parameters
SYMBOL="SPY"
DAYS=30
PORT=4001  # Updated to user's socket port
CLIENT_ID=10  # Changed from 1 to 10 to avoid conflicts
INTERVAL=300  # 5 minutes between collections
SAMPLES=20    # Number of snapshots to collect

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --symbol)
      SYMBOL="$2"
      shift 2
      ;;
    --days)
      DAYS="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --client-id)
      CLIENT_ID="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Function to run a command in the background and log its output
run_background() {
    local cmd="$1"
    local log="$2"
    
    echo "Starting $cmd (logging to $log)"
    $cmd > "$log" 2>&1 &
    echo "Started process with PID $!"
    return $!
}

# Check dependencies
if ! pip show ib_insync &> /dev/null; then
    echo "Installing ib_insync package..."
    pip install ib_insync
fi

if ! pip show scikit-learn &> /dev/null; then
    echo "Installing scikit-learn package..."
    pip install scikit-learn
fi

# Clear any existing log files
> "$COLLECTOR_LOG"
> "$VIZ_LOG"

echo "Starting options chain manifold visualization environment..."
echo "Symbol: $SYMBOL, Target days to expiration: $DAYS"
echo "Connecting to IB Gateway on port: $PORT"
echo

# Start the Python visualization in the background
echo "Starting Python visualization..."
run_background "python3 visualize_manifold.py" "$VIZ_LOG"
VISUALIZATION_PID=$!

# Wait a moment for the visualization to initialize
sleep 2

# Start the options data collector
echo "Starting options data collection from Interactive Brokers..."
run_background "python3 fetch_options_data.py --symbol $SYMBOL --days $DAYS --port $PORT --client_id $CLIENT_ID --interval $INTERVAL --samples $SAMPLES" "$COLLECTOR_LOG"
COLLECTOR_PID=$!

echo
echo "Both processes are now running!"
echo "- Options collector (PID: $COLLECTOR_PID) - Log: $COLLECTOR_LOG"
echo "- Visualization (PID: $VISUALIZATION_PID) - Log: $VIZ_LOG"
echo 
echo "Press Ctrl+C to stop both processes"

# Handle interrupt signal to terminate both processes cleanly
trap "echo 'Stopping all processes...'; kill $COLLECTOR_PID $VISUALIZATION_PID 2>/dev/null; exit" INT

# Wait for any process to finish or for user interrupt
wait

echo "All processes finished."
echo
echo "Options data has been saved to the 'options_data' directory."
echo "You can further analyze this data or run the visualization again with:"
echo "  python3 visualize_manifold.py" 