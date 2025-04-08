import numpy as np
from typing import Dict, List
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .endure_integration import EndureIntegration
import matplotlib.pyplot as plt
import json
import os

class SensitivityAnalysis:
    def __init__(self):
        self.results_dir = "sensitivity_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def run_sensitivity_analysis(self, base_characteristics: WorkloadCharacteristics,
                               variations: Dict[str, List[float]],
                               epsilon: float = 1.0,
                               num_trials: int = 5) -> Dict:
        """Run sensitivity analysis for different workload characteristics."""
        results = {}
        
        for characteristic, values in variations.items():
            print(f"\nAnalyzing sensitivity of {characteristic}")
            char_results = []
            
            for value in values:
                print(f"  Testing value = {value}")
                # Create modified characteristics
                modified_char = self._modify_characteristics(base_characteristics, characteristic, value)
                
                trial_results = []
                for trial in range(num_trials):
                    print(f"    Trial {trial + 1}/{num_trials}")
                    integration = EndureIntegration(epsilon)
                    
                    # Generate workloads
                    original_path, private_path = integration.generate_workload_trace(modified_char)
                    
                    # Get configurations
                    original_results = integration.run_endure_tuning(original_path)
                    private_results = integration.run_endure_tuning(private_path)
                    
                    # Calculate sensitivity metrics
                    sensitivity = self._calculate_sensitivity(
                        original_results["config"],
                        private_results["config"],
                        original_results["performance_metrics"],
                        private_results["performance_metrics"]
                    )
                    
                    trial_results.append(sensitivity)
                
                char_results.append({
                    "value": value,
                    "sensitivity": np.mean([r["sensitivity"] for r in trial_results]),
                    "performance_impact": np.mean([r["performance_impact"] for r in trial_results])
                })
            
            results[characteristic] = char_results
        
        # Save results
        self._save_results(results)
        return results

    def _modify_characteristics(self, base: WorkloadCharacteristics,
                              characteristic: str, value: float) -> WorkloadCharacteristics:
        """Create modified characteristics with one changed value."""
        params = {
            "read_ratio": base.read_ratio,
            "write_ratio": base.write_ratio,
            "key_size": base.key_size,
            "value_size": base.value_size,
            "operation_count": base.operation_count,
            "hot_key_ratio": base.hot_key_ratio,
            "hot_key_count": base.hot_key_count
        }
        
        params[characteristic] = value
        return WorkloadCharacteristics(**params)

    def _calculate_sensitivity(self, original_config: Dict, private_config: Dict,
                             original_perf: Dict, private_perf: Dict) -> Dict:
        """Calculate sensitivity and performance impact metrics."""
        # Configuration sensitivity
        config_sensitivity = sum(
            abs(original_config[param] - private_config[param]) / original_config[param]
            for param in original_config.keys()
        ) / len(original_config)
        
        # Performance impact
        perf_impact = {
            metric: abs(original_perf[metric] - private_perf[metric]) / original_perf[metric]
            for metric in original_perf.keys()
        }
        
        return {
            "sensitivity": config_sensitivity,
            "performance_impact": np.mean(list(perf_impact.values()))
        }

    def _save_results(self, results: Dict) -> None:
        """Save sensitivity analysis results to file."""
        filename = f"{self.results_dir}/sensitivity_analysis_results.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)

    def plot_results(self, results: Dict) -> None:
        """Plot sensitivity analysis results."""
        plt.figure(figsize=(15, 5))
        
        # Plot sensitivity
        plt.subplot(1, 2, 1)
        for characteristic, data in results.items():
            values = [d["value"] for d in data]
            sensitivities = [d["sensitivity"] for d in data]
            plt.plot(values, sensitivities, 'o-', label=characteristic)
        
        plt.xlabel('Characteristic Value')
        plt.ylabel('Configuration Sensitivity')
        plt.title('Workload Characteristic Sensitivity')
        plt.legend()
        
        # Plot performance impact
        plt.subplot(1, 2, 2)
        for characteristic, data in results.items():
            values = [d["value"] for d in data]
            impacts = [d["performance_impact"] for d in data]
            plt.plot(values, impacts, 'o-', label=characteristic)
        
        plt.xlabel('Characteristic Value')
        plt.ylabel('Performance Impact')
        plt.title('Workload Characteristic Impact')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/sensitivity_analysis_plot.png")
        plt.close()

def main():
    # Base workload characteristics
    base_characteristics = WorkloadCharacteristics(
        read_ratio=0.7,
        write_ratio=0.3,
        key_size=16,
        value_size=100,
        operation_count=100000,
        hot_key_ratio=0.2,
        hot_key_count=100
    )
    
    # Define variations for each characteristic
    variations = {
        "read_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "hot_key_ratio": [0.1, 0.2, 0.3, 0.4, 0.5],
        "operation_count": [10000, 50000, 100000, 500000, 1000000]
    }
    
    # Run analysis
    analysis = SensitivityAnalysis()
    results = analysis.run_sensitivity_analysis(base_characteristics, variations)
    
    # Plot results
    analysis.plot_results(results)
    
    print("\nAnalysis complete. Results saved to sensitivity_results/ directory.")

if __name__ == "__main__":
    main() 