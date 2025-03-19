# Troubleshooting Guide

## Simulation not showing model updates

If the model is training but the simulation doesn't seem to be updating with the learned manifold data, try these steps:

### 1. Verify Model-Based Mode

When the simulation starts, it now defaults to model-based mode, but it's worth verifying:

- Press the 'M' key in the simulation window to toggle model-based mode
- The console should show "Using model data" when in model-based mode
- If it shows "Using traditional simulation", press 'M' again to switch

### 2. Manually Reload Model Data

- Press the 'L' key in the simulation window to manually trigger a reload of the model data
- Check the console output to see if it successfully loads the files

### 3. Run the Diagnostic Scripts

Two diagnostic scripts are provided to help troubleshoot data communication issues:

```bash
# Check if files are being updated during training
./diagnostic.sh

# Force update timestamps to trigger simulation reload
./diagnostic_updater.sh
```

### 4. Check File Locations

The simulation looks for model data in several locations:

- `./metric_tensor.csv` and `./christoffel_symbols.csv` (in the project root)
- `./metric_data_current.csv` and `./christoffel_data_current.csv` (in the project root)
- `./data/metric_tensor.csv` and `./data/christoffel_symbols.csv` (in the data subdirectory)

Make sure at least one of these locations contains valid data files.

### 5. Use Separate Terminal Windows

Instead of using the integrated mode, try running the simulation and training in separate terminal windows:

```bash
# Terminal 1: Run just the simulation
./run_simulation.sh

# Terminal 2: Run just the training
./run_training.sh
```

### 6. Check Model Training Progress

Make sure the model is actually training and updating its parameters:

- Look for progress messages in the training output
- Check for changes in the loss value

## If All Else Fails

If the above steps don't resolve the issue, consider:

1. Restarting both the simulation and training processes
2. Checking for error messages in the simulation console
3. Verifying that the model is actually learning and not stuck in a local minimum 