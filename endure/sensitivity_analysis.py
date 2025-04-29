import numpy as np
from typing import Dict, List, Optional
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .endure_integration import EndureIntegration
import matplotlib.pyplot as plt
import json
import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

class SensitivityAnalysis:
    def __init__(self, results_dir: str = "sensitivity_results", max_workers: int = 4):
        """Initialize sensitivity analysis with resource management."""
        self.results_dir = results_dir
        self.max_workers = max_workers
        self.temp_files = set()  # Track temporary files
        self._setup_logging()
        self._setup_directories()
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = os.path.join(self.results_dir, "sensitivity_analysis.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _setup_directories(self) -> None:
        """Set up required directories with validation."""
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            if not os.access(self.results_dir, os.W_OK):
                raise PermissionError(f"No write permission for {self.results_dir}")
            
            # Check disk space
            disk = shutil.disk_usage(self.results_dir)
            free_gb = disk.free / (1024 * 1024 * 1024)
            if free_gb < 1:  # Need at least 1GB free
                raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB available")
        except Exception as e:
            logging.error(f"Error setting up directories: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logging.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
        self.temp_files.clear()

    def run_sensitivity_analysis(self, base_characteristics: WorkloadCharacteristics,
                               variations: Dict[str, List[float]],
                               epsilon: float = 1.0,
                               num_trials: int = 5) -> Dict:
        """Run sensitivity analysis with enhanced error handling and resource management."""
        try:
            # Validate inputs
            if not isinstance(base_characteristics, WorkloadCharacteristics):
                raise TypeError("base_characteristics must be a WorkloadCharacteristics object")
            if not isinstance(variations, dict):
                raise TypeError("variations must be a dictionary")
            if not isinstance(epsilon, (int, float)) or epsilon <= 0:
                raise ValueError("epsilon must be a positive number")
            if not isinstance(num_trials, int) or num_trials < 1:
                raise ValueError("num_trials must be a positive integer")
            
            # Check system resources
            memory = psutil.virtual_memory()
            total_memory = memory.total
            memory_threshold = min(2 * 1024 * 1024 * 1024, total_memory * 0.2)  # 2GB or 20% of total, whichever is smaller
            
            if memory.available < memory_threshold:
                logging.warning(
                    f"Low memory available ({memory.available / (1024 * 1024 * 1024):.2f}GB), "
                    f"analysis may be slow. Threshold: {memory_threshold / (1024 * 1024 * 1024):.2f}GB"
                )
            
            results = {}
            checkpoint_file = os.path.join(self.results_dir, "sensitivity_checkpoint.json")
            self.temp_files.add(checkpoint_file)
            
            try:
                # Load checkpoint if exists
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r') as f:
                        results = json.load(f)
                    logging.info(f"Loaded {len(results)} results from checkpoint")
                
                # Run analysis for each characteristic in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(
                            self._analyze_characteristic,
                            base_characteristics,
                            char,
                            values,
                            epsilon,
                            num_trials
                        ): char
                        for char, values in variations.items()
                        if char not in results  # Skip already analyzed characteristics
                    }
                    
                    for future in as_completed(futures):
                        char = futures[future]
                        try:
                            char_results = future.result()
                            results[char] = char_results
                            
                            # Save checkpoint after each characteristic
                            with open(checkpoint_file, 'w') as f:
                                json.dump(results, f, indent=2)
                            
                        except Exception as e:
                            logging.error(f"Error analyzing {char}: {str(e)}")
                            raise
                
                # Save final results
                self._save_results(results)
                return results
            
            finally:
                self.cleanup()
        
        except Exception as e:
            logging.error(f"Error in sensitivity analysis: {str(e)}")
            raise
    
    def _analyze_characteristic(self, base: WorkloadCharacteristics,
                              characteristic: str, values: List[float],
                              epsilon: float, num_trials: int) -> List[Dict]:
        """Analyze a single characteristic with error handling."""
        char_results = []
        
        for value in values:
            logging.info(f"Testing {characteristic} = {value}")
            trial_results = []
            
            for trial in range(num_trials):
                try:
                    # Create modified characteristics
                    modified_char = self._modify_characteristics(base, characteristic, value)
                    
                    # Generate and analyze workloads
                    integration = EndureIntegration(epsilon)
                    original_path, private_path = integration.generate_workload_trace(modified_char)
                    self.temp_files.add(original_path)
                    self.temp_files.add(private_path)
                    
                    # Run tuning with retry
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            original_results = integration.run_endure_tuning(original_path)
                            private_results = integration.run_endure_tuning(private_path)
                            break
                        except Exception as e:
                            if attempt == max_retries - 1:
                                raise
                            logging.warning(f"Trial {trial + 1} attempt {attempt + 1} failed: {str(e)}")
                            continue
                    
                    # Calculate sensitivity metrics
                    sensitivity = self._calculate_sensitivity(
                        original_results["config"],
                        private_results["config"],
                        original_results["performance_metrics"],
                        private_results["performance_metrics"]
                    )
                    
                    trial_results.append(sensitivity)
                    
                except Exception as e:
                    logging.error(f"Error in trial {trial + 1}: {str(e)}")
                    raise
            
            # Calculate mean sensitivity and performance impact
            char_results.append({
                "value": value,
                "sensitivity": np.mean([r["sensitivity"] for r in trial_results]),
                "performance_impact": np.mean([r["performance_impact"] for r in trial_results]),
                "std_dev": {
                    "sensitivity": np.std([r["sensitivity"] for r in trial_results]),
                    "performance_impact": np.std([r["performance_impact"] for r in trial_results])
                }
            })
        
        return char_results

    def _modify_characteristics(self, base: WorkloadCharacteristics,
                              characteristic: str, value: float) -> WorkloadCharacteristics:
        """Create modified characteristics with validation."""
        if not hasattr(base, characteristic):
            raise ValueError(f"Invalid characteristic: {characteristic}")
        
        params = {
            "read_ratio": base.read_ratio,
            "write_ratio": base.write_ratio,
            "key_size": base.key_size,
            "value_size": base.value_size,
            "operation_count": base.operation_count,
            "hot_key_ratio": base.hot_key_ratio,
            "hot_key_count": base.hot_key_count
        }
        
        # Validate value ranges
        if characteristic in ["read_ratio", "write_ratio", "hot_key_ratio"]:
            if not 0 <= value <= 1:
                raise ValueError(f"{characteristic} must be between 0 and 1")
        elif characteristic in ["key_size", "value_size", "operation_count", "hot_key_count"]:
            if not value > 0:
                raise ValueError(f"{characteristic} must be positive")
        
        params[characteristic] = value
        return WorkloadCharacteristics(**params)

    def _calculate_sensitivity(self, original_config: Dict, private_config: Dict,
                             original_perf: Dict, private_perf: Dict) -> Dict:
        """Calculate sensitivity and performance impact metrics with validation."""
        try:
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
            
            # Validate results
            if np.isnan(config_sensitivity) or np.isinf(config_sensitivity):
                raise ValueError("Invalid sensitivity value")
            
            for impact in perf_impact.values():
                if np.isnan(impact) or np.isinf(impact):
                    raise ValueError("Invalid performance impact value")
            
            return {
                "sensitivity": config_sensitivity,
                "performance_impact": np.mean(list(perf_impact.values()))
            }
        
        except Exception as e:
            logging.error(f"Error calculating sensitivity: {str(e)}")
            raise

    def _save_results(self, results: Dict) -> None:
        """Save sensitivity analysis results with validation."""
        try:
            # Validate results structure
            if not isinstance(results, dict):
                raise TypeError("Results must be a dictionary")
            
            for char, data in results.items():
                if not isinstance(data, list):
                    raise TypeError(f"Data for {char} must be a list")
                for item in data:
                    if not all(k in item for k in ["value", "sensitivity", "performance_impact"]):
                        raise ValueError(f"Invalid result structure for {char}")
            
            # Save results
            filename = os.path.join(self.results_dir, "sensitivity_analysis_results.json")
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            
            logging.info(f"Results saved to {filename}")
            
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            raise

    def plot_results(self, results: Dict) -> None:
        """Plot sensitivity analysis results with error handling."""
        try:
            plt.figure(figsize=(15, 5))
            
            # Plot sensitivity
            plt.subplot(1, 2, 1)
            for characteristic, data in results.items():
                values = [d["value"] for d in data]
                sensitivities = [d["sensitivity"] for d in data]
                std_devs = [d["std_dev"]["sensitivity"] for d in data]
                plt.errorbar(values, sensitivities, yerr=std_devs, fmt='o-', label=characteristic)
            
            plt.xlabel('Characteristic Value')
            plt.ylabel('Configuration Sensitivity')
            plt.title('Workload Characteristic Sensitivity')
            plt.legend()
            
            # Plot performance impact
            plt.subplot(1, 2, 2)
            for characteristic, data in results.items():
                values = [d["value"] for d in data]
                impacts = [d["performance_impact"] for d in data]
                std_devs = [d["std_dev"]["performance_impact"] for d in data]
                plt.errorbar(values, impacts, yerr=std_devs, fmt='o-', label=characteristic)
            
            plt.xlabel('Characteristic Value')
            plt.ylabel('Performance Impact')
            plt.title('Workload Characteristic Impact')
            plt.legend()
            
            plt.tight_layout()
            plot_file = os.path.join(self.results_dir, "sensitivity_analysis_plot.png")
            plt.savefig(plot_file)
            plt.close()
            
            logging.info(f"Plot saved to {plot_file}")
            
        except Exception as e:
            logging.error(f"Error plotting results: {str(e)}")
            raise

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