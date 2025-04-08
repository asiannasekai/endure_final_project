import os
import time
from typing import Dict, Tuple
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .rocksdb_config import RocksDBConfig, DEFAULT_CONFIG
import rocksdb

class EndureExperiment:
    def __init__(self, epsilon: float = 1.0):
        """Initialize experiment with privacy parameter epsilon."""
        self.workload_generator = WorkloadGenerator(epsilon)
        self.db = None

    def setup_db(self, config: RocksDBConfig) -> None:
        """Set up RocksDB with given configuration."""
        os.makedirs(config.db_path, exist_ok=True)
        opts = rocksdb.Options(**config.to_dict())
        self.db = rocksdb.DB(config.db_path, opts)

    def run_workload(self, workload: list) -> Dict[str, float]:
        """Run a workload and return performance metrics."""
        start_time = time.time()
        
        for operation in workload:
            if operation["type"] == "write":
                self.db.put(operation["key"].encode(), operation["value"].encode())
            else:  # read
                self.db.get(operation["key"].encode())
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "duration_seconds": duration,
            "operations_per_second": len(workload) / duration
        }

    def run_experiment(self, characteristics: WorkloadCharacteristics) -> Tuple[Dict, Dict, Dict]:
        """Run experiment with original and differentially private workloads."""
        # Generate workloads
        original_workload, private_workload = self.workload_generator.generate_workload(characteristics)
        
        # Calculate workload metrics
        original_metrics = self.workload_generator.calculate_workload_metrics(original_workload)
        private_metrics = self.workload_generator.calculate_workload_metrics(private_workload)
        
        # Run workloads with default configuration
        self.setup_db(DEFAULT_CONFIG)
        
        original_performance = self.run_workload(original_workload)
        private_performance = self.run_workload(private_workload)
        
        # Clean up
        if self.db:
            self.db.close()
        
        return {
            "original_workload": original_metrics,
            "private_workload": private_metrics,
            "original_performance": original_performance,
            "private_performance": private_performance
        }

def main():
    # Example workload characteristics
    characteristics = WorkloadCharacteristics(
        read_ratio=0.7,  # 70% reads
        write_ratio=0.3,  # 30% writes
        key_size=16,  # 16 bytes
        value_size=100,  # 100 bytes
        operation_count=100000,  # 100k operations
        hot_key_ratio=0.2,  # 20% operations on hot keys
        hot_key_count=100  # 100 hot keys
    )
    
    # Run experiment with different epsilon values
    for epsilon in [0.1, 0.5, 1.0, 2.0]:
        print(f"\nRunning experiment with epsilon = {epsilon}")
        experiment = EndureExperiment(epsilon)
        results = experiment.run_experiment(characteristics)
        
        print("\nWorkload Characteristics:")
        print("Original:")
        print(f"  Read ratio: {results['original_workload']['read_ratio']:.2f}")
        print(f"  Write ratio: {results['original_workload']['write_ratio']:.2f}")
        print(f"  Hot key ratio: {results['original_workload']['hot_key_ratio']:.2f}")
        
        print("\nPrivate:")
        print(f"  Read ratio: {results['private_workload']['read_ratio']:.2f}")
        print(f"  Write ratio: {results['private_workload']['write_ratio']:.2f}")
        print(f"  Hot key ratio: {results['private_workload']['hot_key_ratio']:.2f}")
        
        print("\nPerformance:")
        print("Original workload:")
        print(f"  Operations/sec: {results['original_performance']['operations_per_second']:.2f}")
        print("Private workload:")
        print(f"  Operations/sec: {results['private_performance']['operations_per_second']:.2f}")

if __name__ == "__main__":
    main() 