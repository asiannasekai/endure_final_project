import numpy as np
from typing import Dict, List
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .endure_integration import EndureIntegration
import matplotlib.pyplot as plt
import json
import os

class PrivacyAnalysis:
    def __init__(self):
        self.results_dir = "privacy_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def _validate_results(self, results: Dict) -> bool:
        """Validate the results of the privacy analysis."""
        try:
            # Check if results is a dictionary
            if not isinstance(results, dict):
                return False
            
            # Check each epsilon entry
            for epsilon, trials in results.items():
                if not isinstance(epsilon, (int, float)):
                    return False
                if not isinstance(trials, list):
                    return False
                
                # Check each trial
                for trial in trials:
                    if not isinstance(trial, dict):
                        return False
                    if "comparison" not in trial or "privacy_metrics" not in trial:
                        return False
                    
                    # Check privacy metrics
                    metrics = trial["privacy_metrics"]
                    if not isinstance(metrics, dict):
                        return False
                    if "configuration_difference" not in metrics or "performance_difference" not in metrics:
                        return False
            
            return True
        except Exception:
            return False

    def run_privacy_sweep(self, characteristics: WorkloadCharacteristics,
                         epsilons: List[float] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0],
                         num_trials: int = 10) -> Dict:
        """Run multiple trials with different epsilon values."""
        results = {}
        
        for epsilon in epsilons:
            print(f"\nTesting epsilon = {epsilon}")
            trial_results = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                integration = EndureIntegration(epsilon)
                
                # Generate and analyze workloads
                original_path, private_path = integration.generate_workload_trace(characteristics)
                original_results = integration.run_endure_tuning(original_path)
                private_results = integration.run_endure_tuning(private_path)
                
                # Compare configurations
                comparison = integration.compare_configurations(
                    original_results["config"],
                    private_results["config"]
                )
                
                # Calculate privacy metrics
                privacy_metrics = self._calculate_privacy_metrics(
                    original_results["config"],
                    private_results["config"],
                    original_results["performance_metrics"],
                    private_results["performance_metrics"]
                )
                
                trial_results.append({
                    "comparison": comparison,
                    "privacy_metrics": privacy_metrics
                })
            
            results[epsilon] = trial_results
        
        # Save results
        self._save_results(results)
        return results

    def _calculate_privacy_metrics(self, original_config: Dict, private_config: Dict,
                                 original_perf: Dict, private_perf: Dict) -> Dict:
        """Calculate privacy and utility metrics."""
        # Configuration difference
        config_diff = sum(
            abs(original_config[param] - private_config[param]) / original_config[param]
            for param in original_config.keys()
        ) / len(original_config)
        
        # Performance difference
        perf_diff = {
            metric: abs(original_perf[metric] - private_perf[metric]) / original_perf[metric]
            for metric in original_perf.keys()
        }
        
        return {
            "configuration_difference": config_diff,
            "performance_difference": perf_diff
        }

    def _save_results(self, results: Dict) -> None:
        """Save analysis results to file."""
        filename = f"{self.results_dir}/privacy_analysis_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def plot_results(self, results: Dict) -> None:
        """Plot privacy/utility tradeoff analysis."""
        epsilons = sorted(results.keys())
        
        # Calculate average metrics
        avg_config_diff = []
        avg_perf_diff = []
        
        for epsilon in epsilons:
            trials = results[epsilon]
            config_diffs = [t["privacy_metrics"]["configuration_difference"] for t in trials]
            perf_diffs = [t["privacy_metrics"]["performance_difference"]["throughput"] for t in trials]
            
            avg_config_diff.append(np.mean(config_diffs))
            avg_perf_diff.append(np.mean(perf_diffs))
        
        # Plot configuration difference
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epsilons, avg_config_diff, 'b-o')
        plt.xlabel('Epsilon (ε)')
        plt.ylabel('Average Configuration Difference')
        plt.title('Privacy vs Configuration Difference')
        
        # Plot performance difference
        plt.subplot(1, 2, 2)
        plt.plot(epsilons, avg_perf_diff, 'r-o')
        plt.xlabel('Epsilon (ε)')
        plt.ylabel('Average Performance Difference')
        plt.title('Privacy vs Performance Difference')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/privacy_analysis_plot.png")
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
    analysis = PrivacyAnalysis()
    results = analysis.run_privacy_sweep(characteristics)
    
    # Plot results
    analysis.plot_results(results)
    
    print("\nAnalysis complete. Results saved to privacy_results/ directory.")

if __name__ == "__main__":
    main() 