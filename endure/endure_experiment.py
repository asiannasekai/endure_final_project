import os
import time
from typing import Dict, Tuple, List
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .rocksdb_config import RocksDBConfig, DEFAULT_CONFIG
from .visualization import EnhancedVisualization
import rocksdb

class EndureExperiment:
    def __init__(self, epsilon: float = 1.0):
        """Initialize experiment with privacy parameter epsilon."""
        self.workload_generator = WorkloadGenerator(epsilon)
        self.db = None
        self.batch_size = 1000
        self.visualization = EnhancedVisualization("results/visualization")

    def setup_db(self, config: RocksDBConfig) -> None:
        """Set up RocksDB with given configuration."""
        os.makedirs(config.db_path, exist_ok=True)
        opts = rocksdb.Options(**config.to_dict())
        self.db = rocksdb.DB(config.db_path, opts)

    def process_batch(self, batch: List[Dict]) -> None:
        """Process a batch of operations."""
        write_batch = rocksdb.WriteBatch()
        
        for operation in batch:
            if operation["type"] == "write":
                write_batch.put(operation["key"].encode(), operation["value"].encode())
            else:  # read
                self.db.get(operation["key"].encode())
        
        if write_batch.count() > 0:
            self.db.write(write_batch)

    def run_workload(self, workload: List[Dict]) -> Dict[str, float]:
        """Run a workload and return performance metrics."""
        start_time = time.time()
        
        # Process workload in batches
        for i in range(0, len(workload), self.batch_size):
            batch = workload[i:i + self.batch_size]
            self.process_batch(batch)
        
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
        
        # Run workloads with optimized configuration
        config = RocksDBConfig(
            db_path="rocksdb_data",
            max_background_jobs=8,
            max_subcompactions=4,
            write_buffer_size=128 * 1024 * 1024,
            max_write_buffer_number=6,
            level0_file_num_compaction_trigger=8,
            level0_slowdown_writes_trigger=32,
            compression_type="lz4",
            block_cache_size=8 * 1024 * 1024 * 1024,
            optimize_filters_for_hits=True,
            bloom_locality=1,
            batch_size=1000
        )
        
        self.setup_db(config)
        
        original_performance = self.run_workload(original_workload)
        private_performance = self.run_workload(private_workload)
        
        # Clean up
        if self.db:
            self.db.close()
        
        # Create visualization
        results = {
            "original_workload": original_metrics,
            "private_workload": private_metrics,
            "original_performance": original_performance,
            "private_performance": private_performance
        }
        
        # Generate visualizations
        self.visualization.plot_workload_characteristics(results)
        self.visualization.plot_privacy_performance_tradeoff(results)
        
        return results

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