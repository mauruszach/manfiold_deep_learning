#!/bin/bash
# Script to compile the C++ simulation for manifold visualization

echo "Compiling C++ simulation..."

# Ensure bin directory exists
mkdir -p bin

# Check if on macOS
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS, using appropriate frameworks"
    
    # Use clang++ to compile the simulation
    clang++ -std=c++17 src/simulation.cpp \
        -o bin/simulation \
        -framework SFML \
        -framework OpenGL \
        -lsfml-graphics -lsfml-window -lsfml-system \
        -Wno-deprecated-declarations \
        -O2
    
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "Compilation successful!"
        echo "Executable created at: bin/simulation"
    else
        echo "Compilation failed with error code $RESULT"
        echo "Make sure you have SFML and OpenGL installed."
        echo "On macOS you can install them with: brew install sfml"
    fi
    
else
    # Linux/other compilation
    echo "Detected Linux/other, using appropriate libraries"
    
    g++ -std=c++17 src/simulation.cpp \
        -o bin/simulation \
        -lGL -lGLU -lsfml-graphics -lsfml-window -lsfml-system \
        -O2
    
    RESULT=$?
    if [ $RESULT -eq 0 ]; then
        echo "Compilation successful!"
        echo "Executable created at: bin/simulation"
    else
        echo "Compilation failed with error code $RESULT"
        echo "Make sure you have SFML and OpenGL installed."
        echo "On Ubuntu/Debian: sudo apt-get install libsfml-dev libgl1-mesa-dev"
        echo "On Fedora/RHEL: sudo dnf install SFML-devel mesa-libGL-devel"
    fi
fi

# Make the binary executable
chmod +x bin/simulation

echo "Build process complete!" 