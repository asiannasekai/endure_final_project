import os
import time
from typing import Dict, Tuple, List
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .lsm.cost import Cost
from .lsm.types import System, LSMDesign, Policy, Workload
from .visualization import EnhancedVisualization

class EndureExperiment:
    def __init__(self, epsilon: float = 1.0):
        """Initialize experiment with privacy parameter epsilon."""
        self.workload_generator = WorkloadGenerator(epsilon)
        self.cost_model = Cost(max_levels=10)  # Default max levels
        self.visualization = EnhancedVisualization("results/visualization")

    def setup_system(self, characteristics: WorkloadCharacteristics) -> System:
        """Set up system parameters based on workload characteristics."""
        return System(
            entry_size=characteristics.key_size + characteristics.value_size,
            selectivity=0.1,  # Default selectivity
            entries_per_page=128,  # Default entries per page
            num_entries=characteristics.operation_count,
            mem_budget=1000,  # Default memory budget
            phi=1.0  # Default phi
        )

    def setup_design(self) -> LSMDesign:
        """Set up default LSM design."""
        return LSMDesign(
            bits_per_elem=10,  # Default bits per element
            size_ratio=10,  # Default size ratio
            policy=Policy.Classic,
            kapacity=()
        )

    def run_workload(self, workload: List[Dict], system: System, design: LSMDesign) -> Dict[str, float]:
        """Run a workload using the cost model and return performance metrics."""
        start_time = time.time()
        
        # Convert workload to Endure format
        endure_workload = Workload(
            z0=sum(1 for op in workload if op["type"] == "read" and not op.get("is_hot", False)) / len(workload),
            z1=sum(1 for op in workload if op["type"] == "read" and op.get("is_hot", False)) / len(workload),
            q=0.1,  # Default range query ratio
            w=sum(1 for op in workload if op["type"] == "write") / len(workload)
        )
        
        # Calculate cost using the cost model
        cost = self.cost_model.calc_cost(design, system, endure_workload)
        
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "duration_seconds": duration,
            "cost": cost,
            "operations_per_second": len(workload) / duration
        }

    def run_experiment(self, characteristics: WorkloadCharacteristics) -> Tuple[Dict, Dict, Dict]:
        """Run experiment with original and differentially private workloads."""
        # Generate workloads
        original_workload, private_workload = self.workload_generator.generate_workload(characteristics)
        
        # Calculate workload metrics
        original_metrics = self.workload_generator.calculate_workload_metrics(original_workload)
        private_metrics = self.workload_generator.calculate_workload_metrics(private_workload)
        
        # Set up system and design
        system = self.setup_system(characteristics)
        design = self.setup_design()
        
        # Run workloads with cost model
        original_performance = self.run_workload(original_workload, system, design)
        private_performance = self.run_workload(private_workload, system, design)
        
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