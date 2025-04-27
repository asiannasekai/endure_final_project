from typing import Dict, List, Tuple
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .rocksdb_config import RocksDBConfig
import numpy as np
import json
import os
import rocksdb
import time

class EndureIntegration:
    def __init__(self, epsilon: float = 1.0):
        """Initialize integration with privacy parameter epsilon."""
        self.workload_generator = WorkloadGenerator(epsilon)
        self.workload_cache = {}  # Cache for generated workloads

    def generate_workload_trace(self, characteristics: WorkloadCharacteristics) -> Tuple[str, str]:
        """Generate workload traces for both original and private workloads."""
        # Generate workloads
        original_workload, private_workload = self.workload_generator.generate_workload(characteristics)
        
        # Convert to Endure format
        original_trace = self._convert_to_endure_format(original_workload)
        private_trace = self._convert_to_endure_format(private_workload)
        
        # Save traces to files
        original_path = self._save_workload_trace(original_trace, "original")
        private_path = self._save_workload_trace(private_trace, "private")
        
        return original_path, private_path

    def _convert_to_endure_format(self, workload: List[Dict]) -> List[Dict]:
        """Convert workload to Endure's expected format."""
        endure_workload = []
        for op in workload:
            endure_op = {
                "operation": "GET" if op["type"] == "read" else "PUT",
                "key": op["key"],
                "value": op["value"] if op["value"] else "",
                "timestamp": len(endure_workload)  # Sequential timestamps
            }
            endure_workload.append(endure_op)
        return endure_workload

    def _save_workload_trace(self, trace: List[Dict], prefix: str) -> str:
        """Save workload trace to file."""
        trace_dir = "workload_traces"
        os.makedirs(trace_dir, exist_ok=True)
        
        filename = f"{trace_dir}/{prefix}_trace_{len(self.workload_cache)}.json"
        with open(filename, 'w') as f:
            json.dump(trace, f)
        
        self.workload_cache[filename] = trace
        return filename

    def run_endure_tuning(self, trace_path: str) -> Dict:
        """Run Endure tuning on a workload trace."""
        # Load workload trace
        with open(trace_path, 'r') as f:
            workload = json.load(f)
        
        # Create RocksDB configuration
        config = RocksDBConfig(
            db_path=f"rocksdb_data_{os.path.basename(trace_path)}",
            max_background_jobs=16,  # Increased for better parallelism
            max_subcompactions=8,    # Increased for better parallelism
            write_buffer_size=256 * 1024 * 1024,  # Increased buffer size
            max_write_buffer_number=8,  # Increased for better write performance
            level0_file_num_compaction_trigger=8,
            level0_slowdown_writes_trigger=32,
            compression_type="lz4",
            block_cache_size=16 * 1024 * 1024 * 1024,  # Increased cache size
            optimize_filters_for_hits=True,
            bloom_locality=1,
            batch_size=10000  # Increased batch size
        )
        
        # Set up database
        os.makedirs(config.db_path, exist_ok=True)
        opts = rocksdb.Options(**config.to_dict())
        db = rocksdb.DB(config.db_path, opts)
        
        # Run workload
        start_time = time.time()
        total_operations = len(workload)
        batch_size = 10000  # Process in larger batches
        
        # Pre-allocate write batches
        write_batches = []
        current_batch = rocksdb.WriteBatch()
        current_batch_size = 0
        
        for op in workload:
            if op["operation"] == "PUT":
                current_batch.put(op["key"].encode(), op["value"].encode())
                current_batch_size += 1
            else:  # GET
                db.get(op["key"].encode())
            
            if current_batch_size >= batch_size:
                write_batches.append(current_batch)
                current_batch = rocksdb.WriteBatch()
                current_batch_size = 0
        
        if current_batch_size > 0:
            write_batches.append(current_batch)
        
        # Write all batches
        for batch in write_batches:
            db.write(batch)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate performance metrics
        throughput = total_operations / duration
        latency = duration / total_operations * 1000  # Convert to milliseconds
        
        # Get space amplification
        stats = db.get_property(b"rocksdb.stats")
        space_amplification = float(stats.split(b"Space amplification: ")[1].split(b"\n")[0])
        
        # Clean up
        db.close()
        
        return {
            "config": config.to_dict(),
            "performance_metrics": {
                "throughput": throughput,
                "latency": latency,
                "space_amplification": space_amplification
            }
        }

    def compare_configurations(self, original_config: Dict, private_config: Dict) -> Dict:
        """Compare original and private configurations."""
        return {
            "parameter_differences": {
                param: {
                    "original": original_config[param],
                    "private": private_config[param],
                    "difference": abs(original_config[param] - private_config[param]),
                    "difference_percent": abs(original_config[param] - private_config[param]) / original_config[param] * 100
                }
                for param in original_config.keys()
            }
        }

def main():
    # Example usage
    characteristics = WorkloadCharacteristics(
        read_ratio=0.7,
        write_ratio=0.3,
        key_size=16,
        value_size=100,
        operation_count=100000,
        hot_key_ratio=0.2,
        hot_key_count=100
    )
    
    # Test different epsilon values
    for epsilon in [0.1, 0.5, 1.0, 2.0]:
        print(f"\nTesting with epsilon = {epsilon}")
        integration = EndureIntegration(epsilon)
        
        # Generate and save workload traces
        original_path, private_path = integration.generate_workload_trace(characteristics)
        
        # Run Endure tuning
        original_results = integration.run_endure_tuning(original_path)
        private_results = integration.run_endure_tuning(private_path)
        
        # Compare configurations
        comparison = integration.compare_configurations(
            original_results["config"],
            private_results["config"]
        )
        
        print("\nConfiguration Comparison:")
        for param, diff in comparison["parameter_differences"].items():
            print(f"{param}:")
            print(f"  Original: {diff['original']}")
            print(f"  Private: {diff['private']}")
            print(f"  Difference: {diff['difference_percent']:.2f}%")

if __name__ == "__main__":
    main() 