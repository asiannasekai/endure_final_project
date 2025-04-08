import os
import time
import rocksdb
from typing import Dict, Any
from .rocksdb_config import (
    RocksDBConfig, DEFAULT_CONFIG, DEFAULT_PERTURBED_1, DEFAULT_PERTURBED_2,
    WRITE_INTENSIVE_CONFIG, WRITE_INTENSIVE_PERTURBED_1, WRITE_INTENSIVE_PERTURBED_2,
    READ_INTENSIVE_CONFIG, READ_INTENSIVE_PERTURBED_1, READ_INTENSIVE_PERTURBED_2,
    BALANCED_CONFIG, BALANCED_PERTURBED_1, BALANCED_PERTURBED_2
)

def setup_db(config: RocksDBConfig) -> rocksdb.DB:
    """Set up a RocksDB instance with the given configuration."""
    # Create directory if it doesn't exist
    os.makedirs(config.db_path, exist_ok=True)
    
    # Convert config to RocksDB options
    opts = rocksdb.Options(**config.to_dict())
    
    # Open the database
    return rocksdb.DB(config.db_path, opts)

def run_write_experiment(db: rocksdb.DB, num_entries: int, key_size: int = 16, value_size: int = 100) -> Dict[str, Any]:
    """Run a write experiment and return metrics."""
    start_time = time.time()
    
    # Write entries
    for i in range(num_entries):
        key = f"key_{i:016d}".encode()
        value = b"x" * value_size
        db.put(key, value)
    
    end_time = time.time()
    duration = end_time - start_time
    write_rate = num_entries / duration
    
    return {
        "total_entries": num_entries,
        "duration_seconds": duration,
        "write_rate_ops_per_second": write_rate,
        "key_size_bytes": key_size,
        "value_size_bytes": value_size
    }

def run_read_experiment(db: rocksdb.DB, num_entries: int) -> Dict[str, Any]:
    """Run a read experiment and return metrics."""
    start_time = time.time()
    
    # Read entries
    for i in range(num_entries):
        key = f"key_{i:016d}".encode()
        db.get(key)
    
    end_time = time.time()
    duration = end_time - start_time
    read_rate = num_entries / duration
    
    return {
        "total_entries": num_entries,
        "duration_seconds": duration,
        "read_rate_ops_per_second": read_rate
    }

def main():
    # Experiment configurations
    configs = {
        "default": DEFAULT_CONFIG,
        "default_perturbed_1": DEFAULT_PERTURBED_1,
        "default_perturbed_2": DEFAULT_PERTURBED_2,
        "write_intensive": WRITE_INTENSIVE_CONFIG,
        "write_intensive_perturbed_1": WRITE_INTENSIVE_PERTURBED_1,
        "write_intensive_perturbed_2": WRITE_INTENSIVE_PERTURBED_2,
        "read_intensive": READ_INTENSIVE_CONFIG,
        "read_intensive_perturbed_1": READ_INTENSIVE_PERTURBED_1,
        "read_intensive_perturbed_2": READ_INTENSIVE_PERTURBED_2,
        "balanced": BALANCED_CONFIG,
        "balanced_perturbed_1": BALANCED_PERTURBED_1,
        "balanced_perturbed_2": BALANCED_PERTURBED_2
    }
    
    # Experiment parameters
    num_entries = 1_000_000
    results = {}
    
    # Run experiments for each configuration
    for config_name, config in configs.items():
        print(f"\nRunning experiments for {config_name} configuration...")
        print(f"Configuration details:")
        print(f"  write_buffer_size: {config.write_buffer_size / (1024*1024)}MB")
        print(f"  max_write_buffer_number: {config.max_write_buffer_number}")
        print(f"  level0_file_num_compaction_trigger: {config.level0_file_num_compaction_trigger}")
        if hasattr(config, 'optimize_filters_for_hits'):
            print(f"  optimize_filters_for_hits: {config.optimize_filters_for_hits}")
        if hasattr(config, 'bloom_locality'):
            print(f"  bloom_locality: {config.bloom_locality}")
        
        # Set up database
        db = setup_db(config)
        
        # Run write experiment
        write_results = run_write_experiment(db, num_entries)
        print(f"Write results for {config_name}:")
        print(f"  Write rate: {write_results['write_rate_ops_per_second']:.2f} ops/sec")
        
        # Run read experiment
        read_results = run_read_experiment(db, num_entries)
        print(f"Read results for {config_name}:")
        print(f"  Read rate: {read_results['read_rate_ops_per_second']:.2f} ops/sec")
        
        # Store results
        results[config_name] = {
            "write": write_results,
            "read": read_results,
            "config": config.to_dict()
        }
        
        # Clean up
        db.close()
    
    return results

if __name__ == "__main__":
    main() 