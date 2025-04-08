#!/bin/bash
#SBATCH --job-name=endure_privacy
#SBATCH --output=logs/endure_privacy_%j.out
#SBATCH --error=logs/endure_privacy_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=shared

# Load required modules
module load python/3.9.7
module load rocksdb/7.7.3

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs
mkdir -p results
mkdir -p workload_traces
mkdir -p privacy_results
mkdir -p performance_results
mkdir -p sensitivity_results

# Run experiments
echo "Running privacy analysis..."
python -m endure.privacy_analysis

echo "Running performance analysis..."
python -m endure.performance_analysis

echo "Running sensitivity analysis..."
python -m endure.sensitivity_analysis

# Deactivate virtual environment
deactivate 