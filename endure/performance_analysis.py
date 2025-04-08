import numpy as np
from typing import Dict, List
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .endure_integration import EndureIntegration
import matplotlib.pyplot as plt
import json
import os

class PerformanceAnalysis:
    def __init__(self):
        self.results_dir = "performance_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_performance_analysis(self, characteristics: WorkloadCharacteristics,
                               epsilons: List[float] = [0.1, 0.5, 1.0, 2.0],
                               num_trials: int = 5) -> Dict:
        """Run performance analysis with different epsilon values."""
        results = {}
        
        for epsilon in epsilons:
            print(f"\nTesting epsilon = {epsilon}")
            trial_results = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                integration = EndureIntegration(epsilon)
                
                # Generate workloads
                original_path, private_path = integration.generate_workload_trace(characteristics)
                
                # Get configurations
                original_results = integration.run_endure_tuning(original_path)
                private_results = integration.run_endure_tuning(private_path)
                
                # Run performance tests
                performance_metrics = self._run_performance_tests(
                    original_results["config"],
                    private_results["config"],
                    original_path,
                    private_path
                )
                
                trial_results.append(performance_metrics)
            
            results[epsilon] = trial_results
        
        # Save results
        self._save_results(results)
        return results

    def _run_performance_tests(self, original_config: Dict, private_config: Dict,
                             original_trace: str, private_trace: str) -> Dict:
        """Run performance tests for both configurations."""
        # This would run actual performance tests
        # For now, we'll use mock data
        return {
            "original_config": {
                "throughput": np.random.normal(1000, 100),
                "latency": np.random.normal(10, 1),
                "space_amplification": np.random.normal(1.5, 0.1)
            },
            "private_config": {
                "throughput": np.random.normal(950, 100),  # Slightly worse
                "latency": np.random.normal(11, 1),  # Slightly worse
                "space_amplification": np.random.normal(1.6, 0.1)  # Slightly worse
            }
        }

    def _save_results(self, results: Dict) -> None:
        """Save performance results to file."""
        filename = f"{self.results_dir}/performance_analysis_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def plot_results(self, results: Dict) -> None:
        """Plot performance analysis results."""
        epsilons = sorted(results.keys())
        metrics = ["throughput", "latency", "space_amplification"]
        
        # Calculate average metrics
        avg_metrics = {metric: [] for metric in metrics}
        for epsilon in epsilons:
            trials = results[epsilon]
            for metric in metrics:
                original_values = [t["original_config"][metric] for t in trials]
                private_values = [t["private_config"][metric] for t in trials]
                avg_metrics[metric].append({
                    "original": np.mean(original_values),
                    "private": np.mean(private_values)
                })
        
        # Plot results
        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(metrics):
            plt.subplot(1, 3, i + 1)
            original_values = [m["original"] for m in avg_metrics[metric]]
            private_values = [m["private"] for m in avg_metrics[metric]]
            
            plt.plot(epsilons, original_values, 'b-o', label='Original')
            plt.plot(epsilons, private_values, 'r-o', label='Private')
            
            plt.xlabel('Epsilon (ε)')
            plt.ylabel(metric.capitalize())
            plt.title(f'Privacy vs {metric.capitalize()}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_analysis_plot.png")
        plt.close()

def main():
    # Example workload characteristics
    characteristics = WorkloadCharacteristics(
        read_ratio=0.7,
        write_ratio=0.3,
        key_size=16,
        value_size=100,
        operation_count=100000,
        hot_key_ratio=0.2,
        hot_key_count=100
    )
    
    # Run analysis
    analysis = PerformanceAnalysis()
    results = analysis.run_performance_analysis(characteristics)
    
    # Plot results
    analysis.plot_results(results)
    
    print("\nAnalysis complete. Results saved to performance_results/ directory.")

if __name__ == "__main__":
    main() 