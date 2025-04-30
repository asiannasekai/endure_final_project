from typing import Dict, List, Tuple
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .lsm.cost import Cost
from .lsm.types import System, LSMDesign, Policy, Workload
import numpy as np
import json
import os
import time

class EndureIntegration:
    def __init__(self, epsilon: float = 1.0):
        """Initialize integration with privacy parameter epsilon."""
        self.workload_generator = WorkloadGenerator(epsilon)
        self.workload_cache = {}  # Cache for generated workloads
        self.cost_model = Cost(max_levels=10)  # Default max levels

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
        
        # Create system parameters
        system = System(
            entry_size=1024,  # Default entry size
            selectivity=0.1,  # Default selectivity
            entries_per_page=128,  # Default entries per page
            num_entries=len(workload),
            mem_budget=1000,  # Default memory budget
            phi=1.0  # Default phi
        )
        
        # Create LSM design
        design = LSMDesign(
            bits_per_elem=10,  # Default bits per element
            size_ratio=10,  # Default size ratio
            policy=Policy.Classic,
            kapacity=()
        )
        
        # Convert workload to Endure format
        endure_workload = Workload(
            z0=sum(1 for op in workload if op["operation"] == "GET" and not op.get("is_hot", False)) / len(workload),
            z1=sum(1 for op in workload if op["operation"] == "GET" and op.get("is_hot", False)) / len(workload),
            q=0.1,  # Default range query ratio
            w=sum(1 for op in workload if op["operation"] == "PUT") / len(workload)
        )
        
        # Calculate cost using the cost model
        start_time = time.time()
        cost = self.cost_model.calc_cost(design, system, endure_workload)
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate performance metrics
        throughput = len(workload) / duration
        latency = duration / len(workload) * 1000  # Convert to milliseconds
        
        return {
            "config": {
                "bits_per_elem": design.bits_per_elem,
                "size_ratio": design.size_ratio,
                "policy": design.policy.name,
                "kapacity": design.kapacity
            },
            "performance_metrics": {
                "throughput": throughput,
                "latency": latency,
                "cost": cost
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