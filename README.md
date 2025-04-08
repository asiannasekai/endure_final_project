# Endure Privacy Analysis

This project implements a differential privacy framework for database workload tuning, focusing on the privacy/utility tradeoff in database configuration optimization.

## Project Overview

The project implements a scenario where:
1. Party A owns a database and wants to optimize its configuration
2. Party B provides database tuning services
3. Due to privacy concerns, Party A sends perturbed workload characteristics instead of the original data
4. The project analyzes how this privacy-preserving approach affects tuning quality

## Key Components

1. **Workload Generator**
   - Generates original and differentially private workloads
   - Applies Laplace noise to workload characteristics
   - Maintains valid ratios after perturbation

2. **Endure Integration**
   - Converts workloads to Endure's format
   - Runs tuning on both original and private workloads
   - Compares resulting configurations

3. **Analysis Tools**
   - Privacy Analysis: Tests different epsilon values
   - Performance Analysis: Measures impact on throughput/latency
   - Sensitivity Analysis: Tests different workload characteristics

## Usage

1. **Local Development**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run experiments
   python -m endure.privacy_analysis
   python -m endure.performance_analysis
   python -m endure.sensitivity_analysis
   ```

2. **SCC Cluster**
   ```bash
   # Submit job
   sbatch run_experiments.slurm
   ```

## Results

Results are stored in the following directories:
- `privacy_results/`: Privacy analysis results
- `performance_results/`: Performance analysis results
- `sensitivity_results/`: Sensitivity analysis results
- `workload_traces/`: Generated workload traces
- `logs/`: Experiment logs

## License

MIT License
