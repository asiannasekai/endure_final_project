# Endure Project with Differential Privacy

This project implements a differential privacy framework for database workload tuning using RocksDB and Endure. The system allows for private database tuning by perturbing workload characteristics while maintaining utility.

## Project Structure

```
endure_final_project/
├── endure/
│   ├── workload_generator.py    # Generates original and private workloads
│   ├── endure_integration.py    # Integrates with Endure tuning framework
│   ├── privacy_analysis.py      # Analyzes privacy-utility tradeoffs
│   ├── performance_analysis.py  # Measures performance impacts
│   └── rocksdb_config.py       # RocksDB configuration templates
└── experiments/
    └── run_experiments.slurm    # Slurm job script for SCC
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/asiannasekai/endure_final_project.git
cd endure_final_project
```

2. Set up Python environment:
```bash
module load python3
python3 -m venv venv
source venv/bin/activate
pip install numpy
```

## Workload Characteristics

The workload generator supports the following characteristics:

1. **Read/Write Ratios**
   - `read_ratio`: Proportion of read operations (0.0-1.0)
   - `write_ratio`: Proportion of write operations (0.0-1.0)

2. **Key Access Patterns**
   - `hot_key_ratio`: Proportion of operations on hot keys (0.0-1.0)
   - `hot_key_count`: Number of frequently accessed keys
   - `key_size`: Size of keys in bytes
   - `value_size`: Size of values in bytes

3. **Operation Counts**
   - `operation_count`: Total number of operations in workload

## Privacy Implementation

The system implements ε-differential privacy using the Laplace mechanism:

1. **Original Workload Generation**
   - Creates workload based on specified characteristics
   - Maintains operation ratios and access patterns

2. **Privacy Mechanism**
   - Adds Laplace noise to workload characteristics
   - Scale parameter based on privacy budget (ε)
   - Maintains valid ratios after perturbation

3. **Configuration Generation**
   - Generates RocksDB configurations from workload characteristics
   - Supports both original and private workloads

## Running Experiments

1. **Basic Experiment**
```bash
sbatch run_experiments.slurm
```

2. **Custom Privacy Settings**
```python
# In endure_experiment.py
epsilon_values = [0.1, 0.5, 1.0, 2.0]  # Privacy parameters
workload_chars = WorkloadCharacteristics(
    read_ratio=0.7,
    write_ratio=0.3,
    hot_key_ratio=0.2,
    operation_count=10000,
    key_size=16,
    value_size=100,
    hot_key_count=100
)
```

3. **Monitoring Results**
```bash
# Check job status
squeue -u $USER

# View results
cat endure_*.out
```

## Result Analysis

The experiments generate three types of results:

1. **Privacy Analysis**
   - Configuration differences between original and private workloads
   - Impact of ε on utility
   - Stored in `privacy_results/`

2. **Performance Analysis**
   - Throughput comparisons
   - Latency measurements
   - Space amplification
   - Stored in `performance_results/`

3. **Workload Metrics**
   - Actual vs. intended operation ratios
   - Key access pattern statistics
   - Stored with experiment results

## Next Steps

Potential areas for extension:

1. **Additional Privacy Mechanisms**
   - Implement different noise distributions
   - Add composition theorems
   - Support for multiple workload perturbations

2. **Enhanced Workload Characteristics**
   - Add temporal patterns
   - Support for range queries
   - Complex value distributions

3. **Integration Improvements**
   - Real-time workload adaptation
   - Automated parameter tuning
   - Extended metrics collection

## Contact

For questions or issues, please contact:
- Project maintainer: asiannah@bu.edu
- Original author: asiannah

## License

MIT License

