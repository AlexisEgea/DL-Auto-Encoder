#!/bin/bash

echo "Auto-Encoder Execution"
echo "_________________________________________________________________________________________________________________________________________________________"

# shellcheck disable=SC2164

echo "***Requirement: Have python3 & pip installed***"
cd src/
echo "Import library with Pip"
echo

pip install numpy
pip install torch
pip install matplotlib

echo "_________________________________________________________________________________________________________________________________________________________"
echo "Execution"

python3 main.py

read -s -p "Press Enter to finish the script..."
