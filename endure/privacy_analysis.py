import numpy as np
from typing import Dict, List, Tuple, Any
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .endure_integration import EndureIntegration
from .analysis import BaseAnalysis, AnalysisResult
import matplotlib.pyplot as plt
import json
import os
import logging

logger = logging.getLogger(__name__)

class PrivacyAnalysis(BaseAnalysis):
    """Privacy analysis implementation with enhanced functionality."""
    
    def __init__(self, config=None, results_dir=None):
        """Initialize privacy analysis with configuration and results directory."""
        super().__init__(config)
        if results_dir:
            self.config.analysis.results_dir = results_dir
        self._batch_size = 1000  # Number of operations per batch
        self._progress_file = os.path.join(self.config.analysis.results_dir, "progress.json")
        self.workload_generator = WorkloadGenerator()
        self.integration = EndureIntegration()

    def run_privacy_sweep(self, characteristics: Dict,
                         epsilons: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
                         num_trials: int = 5) -> AnalysisResult:
        """Run multiple trials with different epsilon values."""
        try:
            # Convert characteristics dictionary to WorkloadCharacteristics object
            if not isinstance(characteristics, dict):
                raise ValueError("Invalid workload characteristics format")
            
            try:
                workload_chars = WorkloadCharacteristics(
                    read_ratio=float(characteristics["read_ratio"]),
                    write_ratio=float(characteristics["write_ratio"]),
                    key_size=int(characteristics["key_size"]),
                    value_size=int(characteristics["value_size"]),
                    operation_count=int(characteristics["operation_count"]),
                    hot_key_ratio=float(characteristics["hot_key_ratio"]),
                    hot_key_count=int(characteristics["hot_key_count"])
                )
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid workload characteristics: {str(e)}")
            
            if not isinstance(epsilons, list) or not all(isinstance(e, (int, float)) for e in epsilons):
                raise ValueError("Invalid epsilon values")
            if not isinstance(num_trials, int) or num_trials < 1:
                raise ValueError("Invalid number of trials")
            
            results = {}
            for epsilon in epsilons:
                trial_results = []
                for trial in range(num_trials):
                    try:
                        # Check resources before each trial
                        self._check_resources()
                        
                        # Generate workloads with differential privacy
                        original_workload, private_workload = self.workload_generator.generate_workload(
                            workload_chars, epsilon=epsilon
                        )
                        
                        # Convert workloads to traces
                        original_trace = self._convert_to_trace(original_workload)
                        private_trace = self._convert_to_trace(private_workload)
                        
                        # Save traces
                        original_path = self._save_trace(original_trace, f"original_{epsilon}_{trial}")
                        private_path = self._save_trace(private_trace, f"private_{epsilon}_{trial}")
                        self.temp_files.extend([original_path, private_path])
                        
                        # Run Endure tuning
                        self.integration.epsilon = epsilon
                        original_results = self.integration.run_endure_tuning(original_path)
                        private_results = self.integration.run_endure_tuning(private_path)
                        
                        # Compare configurations
                        comparison = self.integration.compare_configurations(
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
                            "workload_characteristics": {
                                "read_ratio": workload_chars.read_ratio,
                                "write_ratio": workload_chars.write_ratio,
                                "key_size": workload_chars.key_size,
                                "value_size": workload_chars.value_size,
                                "operation_count": workload_chars.operation_count,
                                "hot_key_ratio": workload_chars.hot_key_ratio,
                                "hot_key_count": workload_chars.hot_key_count
                            }
                        })
                        
                        # Monitor resources after each trial
                        self._monitor_resources()
                        
                    except Exception as e:
                        logger.error(f"Error in trial {trial} for epsilon {epsilon}: {str(e)}")
                        continue
                
                results[epsilon] = trial_results
            
            # Process and validate results
            processed_results = self._process_privacy_results(results)
            if not self._validate_privacy_results(processed_results):
                raise ValueError("Invalid privacy analysis results")
            
            return AnalysisResult(
                metrics=processed_results,
                workload_characteristics=characteristics,
                config={'epsilons': epsilons, 'num_trials': num_trials}
            )
            
        except Exception as e:
            logger.error(f"Error in privacy sweep: {str(e)}")
            raise
        finally:
            self.cleanup()

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
        trace_dir = os.path.join(self.config.analysis.results_dir, "workload_traces")
        os.makedirs(trace_dir, exist_ok=True)
        
        filename = os.path.join(trace_dir, f"{prefix}_trace.json")
        with open(filename, 'w') as f:
            json.dump(trace, f)
        
        return filename

    def _calculate_privacy_metrics(self, original_config: Dict, private_config: Dict,
                                 original_perf: Dict, private_perf: Dict) -> Dict:
        """Calculate privacy and utility metrics with statistical analysis."""
        try:
            # Validate input data
            if not all(isinstance(x, dict) for x in [original_config, private_config, original_perf, private_perf]):
                raise ValueError("Invalid input data: all parameters must be dictionaries")
            
            # Configuration difference with statistical significance
            config_diffs = {}
            for param in original_config.keys():
                if param not in private_config:
                    continue
                    
                try:
                    original_val = float(original_config[param])
                    private_val = float(private_config[param])
                    
                    config_diffs[param] = {
                        'difference': abs(original_val - private_val),
                        'difference_percent': self._safe_percentage(original_val, private_val),
                        'original_value': original_val,
                        'private_value': private_val
                    }
                except (ValueError, TypeError):
                    logger.warning(f"Could not process parameter {param}: invalid values")
                    continue
            
            # Performance difference with statistical analysis
            perf_diffs = {}
            for metric in original_perf.keys():
                if metric not in private_perf:
                    continue
                    
                try:
                    original_val = float(original_perf[metric])
                    private_val = float(private_perf[metric])
                    
                    perf_diffs[metric] = {
                        'difference': abs(original_val - private_val),
                        'difference_percent': self._safe_percentage(original_val, private_val),
                        'original_value': original_val,
                        'private_value': private_val,
                        'impact': self._calculate_performance_impact(metric, original_val, private_val)
                    }
                except (ValueError, TypeError):
                    logger.warning(f"Could not process metric {metric}: invalid values")
                    continue
            
            # Calculate overall privacy-utility tradeoff score
            privacy_utility_score = self._calculate_privacy_utility_score(config_diffs, perf_diffs)
            
            return {
                "configuration_differences": config_diffs,
                "performance_differences": perf_diffs,
                "privacy_utility_score": privacy_utility_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating privacy metrics: {str(e)}")
            raise

    def _safe_percentage(self, original: float, new: float) -> float:
        """Safely calculate percentage difference."""
        if original == 0:
            return 0.0
        return abs(original - new) / original * 100

    def _calculate_performance_impact(self, metric: str, original: float, private: float) -> str:
        """Calculate the impact level of performance difference."""
        diff_percent = self._safe_percentage(original, private)
        
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

    def _process_privacy_results(self, results: Dict) -> Dict:
        """Process privacy analysis results."""
        try:
            processed = {
                'privacy_loss': [],
                'utility_loss': [],
                'error_rate': []
            }
            
            for epsilon, trials in results.items():
                for trial in trials:
                    metrics = trial['privacy_metrics']
                    processed['privacy_loss'].append(metrics['privacy_loss'])
                    processed['utility_loss'].append(metrics['utility_loss'])
                    processed['error_rate'].append(metrics['error_rate'])
            
            # Calculate statistics
            return {
                'privacy_loss': {
                    'mean': np.mean(processed['privacy_loss']),
                    'std': np.std(processed['privacy_loss']),
                    'min': np.min(processed['privacy_loss']),
                    'max': np.max(processed['privacy_loss'])
                },
                'utility_loss': {
                    'mean': np.mean(processed['utility_loss']),
                    'std': np.std(processed['utility_loss']),
                    'min': np.min(processed['utility_loss']),
                    'max': np.max(processed['utility_loss'])
                },
                'error_rate': {
                    'mean': np.mean(processed['error_rate']),
                    'std': np.std(processed['error_rate']),
                    'min': np.min(processed['error_rate']),
                    'max': np.max(processed['error_rate'])
                }
            }
        except Exception as e:
            logger.error(f"Error processing privacy results: {str(e)}")
            raise

    def _validate_privacy_results(self, results: Dict) -> bool:
        """Validate privacy analysis results."""
        try:
            # Check required metrics
            required_metrics = ['privacy_loss', 'utility_loss', 'error_rate']
            missing_metrics = [metric for metric in required_metrics if metric not in results]
            if missing_metrics:
                logger.error(f"Missing required metrics in results: {missing_metrics}")
                return False
            
            # Validate metric ranges
            metric_ranges = {
                'privacy_loss': (0, 1),
                'utility_loss': (0, 1),
                'error_rate': (0, 1)
            }
            
            for metric, (min_val, max_val) in metric_ranges.items():
                value = results.get(metric)
                if not isinstance(value, (int, float)):
                    logger.error(f"Invalid type for {metric}: {type(value)}")
                    return False
                if not min_val <= value <= max_val:
                    logger.warning(f"{metric} value {value} is outside valid range [{min_val}, {max_val}]")
                    # Don't fail validation for out-of-range values, just warn
            
            # Validate consistency between metrics
            if results['privacy_loss'] == 0 and results['utility_loss'] > 0.1:
                logger.warning("Non-zero utility loss with zero privacy loss")
            
            if results['error_rate'] > 0.7:
                logger.warning(f"High error rate detected: {results['error_rate']}")
            
            return True
        except Exception as e:
            logger.error(f"Error validating privacy results: {str(e)}")
            return False

def main():
    # Example workload characteristics
    characteristics = {
        "read_ratio": 0.7,
        "write_ratio": 0.3,
        "key_size": 16,
        "value_size": 100,
        "operation_count": 100000,
        "hot_key_ratio": 0.2,
        "hot_key_count": 100
    }
    
    # Run analysis
    analysis = PrivacyAnalysis()
    results = analysis.run_privacy_sweep(characteristics)
    
    # Plot results
    analysis.plot_results(results)
    
    print("\nAnalysis complete. Results saved to privacy_results/ directory.")

if __name__ == "__main__":
    main() 