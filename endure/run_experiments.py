import os
import time
from typing import Dict, Any, List
from .lsm.cost import Cost
from .lsm.types import System, LSMDesign, Policy, Workload

def setup_system(characteristics: Dict[str, Any]) -> System:
    """Set up system parameters based on workload characteristics."""
    return System(
        entry_size=characteristics.get("key_size", 16) + characteristics.get("value_size", 100),
        selectivity=0.1,  # Default selectivity
        entries_per_page=128,  # Default entries per page
        num_entries=characteristics.get("operation_count", 1000),
        mem_budget=1000,  # Default memory budget
        phi=1.0  # Default phi
    )

def setup_design() -> LSMDesign:
    """Set up default LSM design."""
    return LSMDesign(
        bits_per_elem=10,  # Default bits per element
        size_ratio=10,  # Default size ratio
        policy=Policy.Classic,
        kapacity=()
    )

def run_write_experiment(workload: List[Dict], system: System, design: LSMDesign) -> Dict[str, Any]:
    """Run a write experiment and return metrics."""
    start_time = time.time()
    
    # Convert workload to Endure format
    endure_workload = Workload(
        z0=0.0,  # No empty reads
        z1=0.0,  # No non-empty reads
        q=0.0,  # No range queries
        w=1.0  # All writes
    )
    
    # Calculate cost using the cost model
    cost = Cost(max_levels=10).calc_cost(design, system, endure_workload)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "total_entries": len(workload),
        "duration_seconds": duration,
        "cost": cost,
        "write_rate_ops_per_second": len(workload) / duration
    }

def run_read_experiment(workload: List[Dict], system: System, design: LSMDesign) -> Dict[str, Any]:
    """Run a read experiment and return metrics."""
    start_time = time.time()
    
    # Convert workload to Endure format
    endure_workload = Workload(
        z0=0.5,  # Half empty reads
        z1=0.5,  # Half non-empty reads
        q=0.0,  # No range queries
        w=0.0  # No writes
    )
    
    # Calculate cost using the cost model
    cost = Cost(max_levels=10).calc_cost(design, system, endure_workload)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "total_entries": len(workload),
        "duration_seconds": duration,
        "cost": cost,
        "read_rate_ops_per_second": len(workload) / duration
    }

def run_experiments(characteristics: Dict[str, Any]) -> Dict[str, Any]:
    """Run experiments with different configurations."""
    results = {}
    num_entries = characteristics.get("operation_count", 1000)
    
    # Set up system and design
    system = setup_system(characteristics)
    design = setup_design()
    
    # Run write experiment
    write_results = run_write_experiment([{"type": "write", "key": f"key_{i}", "value": "x" * 100} for i in range(num_entries)], system, design)
    print(f"Write results:")
    print(f"  Write rate: {write_results['write_rate_ops_per_second']:.2f} ops/sec")
    print(f"  Cost: {write_results['cost']:.2f}")
    
    # Run read experiment
    read_results = run_read_experiment([{"type": "read", "key": f"key_{i}"} for i in range(num_entries)], system, design)
    print(f"Read results:")
    print(f"  Read rate: {read_results['read_rate_ops_per_second']:.2f} ops/sec")
    print(f"  Cost: {read_results['cost']:.2f}")
    
    # Store results
    results = {
        "write": write_results,
        "read": read_results,
        "config": {
            "bits_per_elem": design.bits_per_elem,
            "size_ratio": design.size_ratio,
            "policy": design.policy.name,
            "kapacity": design.kapacity
        }
    }
    
    return results

if __name__ == "__main__":
    main() 