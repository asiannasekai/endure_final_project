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
                         epsilons: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
                         num_trials: int = 5) -> Dict:
        """Run multiple trials with different epsilon values."""
        if not characteristics.validate():
            raise ValueError("Invalid workload characteristics")
        
        results = {}
        
        for epsilon in epsilons:
            print(f"\nTesting epsilon = {epsilon}")
            trial_results = []
            
            for trial in range(num_trials):
                print(f"  Trial {trial + 1}/{num_trials}")
                
                # Create workload generator with current epsilon
                workload_generator = WorkloadGenerator(epsilon=epsilon)
                
                # Generate workloads with differential privacy
                original_workload, private_workload = workload_generator.generate_workload(characteristics)
                
                # Convert workloads to traces
                original_trace = self._convert_to_trace(original_workload)
                private_trace = self._convert_to_trace(private_workload)
                
                # Save traces
                original_path = self._save_trace(original_trace, f"original_{epsilon}_{trial}")
                private_path = self._save_trace(private_trace, f"private_{epsilon}_{trial}")
                
                # Run Endure tuning
                integration = EndureIntegration(epsilon)
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
                    "privacy_metrics": privacy_metrics,
                    "workload_metrics": {
                        "original": workload_generator.calculate_workload_metrics(original_workload),
                        "private": workload_generator.calculate_workload_metrics(private_workload)
                    }
                })
            
            results[epsilon] = trial_results
        
        # Save results
        self._save_results(results)
        return results

    def _convert_to_trace(self, workload: List[Dict]) -> List[Dict]:
        """Convert workload to trace format."""
        return [
            {
                "operation": "GET" if op["type"] == "read" else "PUT",
                "key": op["key"],
                "value": op["value"] if op["value"] else "",
                "timestamp": i
            }
            for i, op in enumerate(workload)
        ]

    def _save_trace(self, trace: List[Dict], prefix: str) -> str:
        """Save trace to file."""
        trace_dir = "workload_traces"
        os.makedirs(trace_dir, exist_ok=True)
        
        filename = f"{trace_dir}/{prefix}_trace.json"
        with open(filename, 'w') as f:
            json.dump(trace, f)
        
        return filename

    def _safe_percentage(self, original: float, new: float) -> float:
        """Safely calculate percentage difference."""
        if original == 0:
            return 0.0
        return abs(original - new) / original * 100

    def _calculate_privacy_metrics(self, original_config: Dict, private_config: Dict,
                                 original_perf: Dict, private_perf: Dict) -> Dict:
        """Calculate privacy and utility metrics with statistical analysis."""
        # Configuration difference with statistical significance
        config_diffs = {
            param: {
                'difference': abs(original_config[param] - private_config[param]),
                'difference_percent': self._safe_percentage(original_config[param], private_config[param]),
                'original_value': original_config[param],
                'private_value': private_config[param]
            }
            for param in original_config.keys()
        }
        
        # Performance difference with statistical analysis
        perf_diffs = {
            metric: {
                'difference': abs(original_perf[metric] - private_perf[metric]),
                'difference_percent': self._safe_percentage(original_perf[metric], private_perf[metric]),
                'original_value': original_perf[metric],
                'private_value': private_perf[metric],
                'impact': self._calculate_performance_impact(metric, original_perf[metric], private_perf[metric])
            }
            for metric in original_perf.keys()
        }
        
        # Calculate overall privacy-utility tradeoff score
        privacy_utility_score = self._calculate_privacy_utility_score(config_diffs, perf_diffs)
        
        return {
            "configuration_differences": config_diffs,
            "performance_differences": perf_diffs,
            "privacy_utility_score": privacy_utility_score
        }

    def _calculate_performance_impact(self, metric: str, original: float, private: float) -> str:
        """Calculate the impact level of performance difference."""
        diff_percent = abs(original - private) / original * 100
        
        if metric == 'throughput':
            if diff_percent < 5:
                return 'Negligible'
            elif diff_percent < 15:
                return 'Minor'
            elif diff_percent < 30:
                return 'Moderate'
            else:
                return 'Significant'
        elif metric == 'latency':
            if diff_percent < 10:
                return 'Negligible'
            elif diff_percent < 25:
                return 'Minor'
            elif diff_percent < 50:
                return 'Moderate'
            else:
                return 'Significant'
        else:  # space_amplification
            if diff_percent < 10:
                return 'Negligible'
            elif diff_percent < 20:
                return 'Minor'
            elif diff_percent < 40:
                return 'Moderate'
            else:
                return 'Significant'

    def _calculate_privacy_utility_score(self, config_diffs: Dict, perf_diffs: Dict) -> Dict:
        """Calculate a comprehensive privacy-utility tradeoff score."""
        # Weight factors for different metrics
        weights = {
            'throughput': 0.4,
            'latency': 0.3,
            'space_amplification': 0.3
        }
        
        # Calculate weighted performance impact
        performance_score = sum(
            weights[metric] * (100 - diff['difference_percent'])
            for metric, diff in perf_diffs.items()
        )
        
        # Calculate configuration stability score
        config_score = 100 - np.mean([
            diff['difference_percent']
            for diff in config_diffs.values()
        ])
        
        # Overall score (0-100)
        overall_score = (performance_score * 0.7) + (config_score * 0.3)
        
        return {
            'performance_score': performance_score,
            'configuration_score': config_score,
            'overall_score': overall_score,
            'interpretation': self._interpret_score(overall_score)
        }

    def _interpret_score(self, score: float) -> str:
        """Interpret the privacy-utility tradeoff score."""
        if score >= 90:
            return "Excellent privacy-utility tradeoff"
        elif score >= 75:
            return "Good privacy-utility tradeoff"
        elif score >= 60:
            return "Acceptable privacy-utility tradeoff"
        elif score >= 45:
            return "Suboptimal privacy-utility tradeoff"
        else:
            return "Poor privacy-utility tradeoff"

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
            config_diffs = [t["privacy_metrics"]["configuration_differences"][param]['difference'] for t in trials for param in t["privacy_metrics"]["configuration_differences"]]
            perf_diffs = [t["privacy_metrics"]["performance_differences"][metric]['difference'] for t in trials for metric in t["privacy_metrics"]["performance_differences"]]
            
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