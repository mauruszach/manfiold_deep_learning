#!/bin/bash
# Diagnostic script to monitor file changes during training

cd "$(dirname "$0")"
WORKSPACE_DIR="$(pwd)"

# Function to check file info
check_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo "File exists: $file"
        echo "  Last modified: $(stat -f "%Sm" "$file")"
        echo "  Size: $(stat -f "%z" "$file") bytes"
        echo "  First 3 lines:"
        head -n 3 "$file"
        echo ""
    else
        echo "File DOES NOT exist: $file"
        echo ""
    fi
}

# Check initial state
echo "=== INITIAL FILE STATE ==="
check_file "$WORKSPACE_DIR/paths_on_manifold/metric_tensor.csv"
check_file "$WORKSPACE_DIR/paths_on_manifold/christoffel_symbols.csv"
check_file "$WORKSPACE_DIR/paths_on_manifold/metric_metadata.json"
check_file "$WORKSPACE_DIR/paths_on_manifold/data/metric_tensor.csv"
check_file "$WORKSPACE_DIR/paths_on_manifold/data/christoffel_symbols.csv"
check_file "$WORKSPACE_DIR/paths_on_manifold/metric_data_current.csv"
check_file "$WORKSPACE_DIR/paths_on_manifold/christoffel_data_current.csv"

# Start monitoring for changes
echo "=== MONITORING FOR CHANGES (press Ctrl+C to stop) ==="
while true; do
    current_time=$(date "+%H:%M:%S")
    
    # Check if files have changed in the last 5 seconds
    recent_changes=$(find "$WORKSPACE_DIR/paths_on_manifold" -name "*.csv" -o -name "*.json" -mtime -5s | grep -v "__pycache__")
    
    if [ ! -z "$recent_changes" ]; then
        echo "[$current_time] Recent changes detected:"
        echo "$recent_changes"
        echo ""
        
        # Show updated content for any changed metric files
        for file in $recent_changes; do
            if [[ $file == *"metric"* ]] || [[ $file == *"christoffel"* ]]; then
                echo "Updated file: $file"
                echo "  First 3 lines:"
                head -n 3 "$file"
                echo ""
            fi
        done
    fi
    
    sleep 2
done 