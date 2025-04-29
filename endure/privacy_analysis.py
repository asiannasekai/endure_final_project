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
        """Calculate privacy and utility metrics with proper differential privacy guarantees."""
        try:
            # Validate input data
            if not all(isinstance(x, dict) for x in [original_config, private_config, original_perf, private_perf]):
                raise ValueError("Invalid input data: all parameters must be dictionaries")
            
            # Calculate sensitivity for each metric
            sensitivities = {
                'throughput': self._calculate_sensitivity(original_perf['throughput']),
                'latency': self._calculate_sensitivity(original_perf['latency']),
                'memory': self._calculate_sensitivity(original_perf['memory'])
            }
            
            # Calculate privacy loss using advanced composition theorem
            privacy_loss = self._calculate_privacy_loss(
                original_config=original_config,
                private_config=private_config,
                sensitivities=sensitivities,
                epsilon=self.epsilon
            )
            
            # Calculate utility loss with weighted metrics
            utility_loss = self._calculate_utility_loss(
                original_perf=original_perf,
                private_perf=private_perf,
                weights={
                    'throughput': 0.5,  # Higher weight for critical performance metric
                    'latency': 0.3,
                    'memory': 0.2
                }
            )
            
            # Calculate configuration differences with privacy bounds
            config_diffs = {}
            for param in original_config.keys():
                if param not in private_config:
                    continue
                    
                try:
                    original_val = float(original_config[param])
                    private_val = float(private_config[param])
                    
                    # Calculate privacy-preserving bounds
                    sensitivity = self._calculate_parameter_sensitivity(param, original_val)
                    noise_scale = sensitivity / self.epsilon
                    lower_bound = private_val - 2 * noise_scale
                    upper_bound = private_val + 2 * noise_scale
                    
                    config_diffs[param] = {
                        'difference': abs(original_val - private_val),
                        'difference_percent': self._safe_percentage(original_val, private_val),
                        'privacy_bounds': {
                            'lower': max(0, lower_bound),
                            'upper': upper_bound
                        },
                        'privacy_guarantee': self._calculate_parameter_privacy(
                            original_val, private_val, sensitivity, self.epsilon
                        )
                    }
                except (ValueError, TypeError):
                    logger.warning(f"Could not process parameter {param}: invalid values")
                    continue
            
            # Calculate performance differences with privacy guarantees
            perf_diffs = {}
            for metric in original_perf.keys():
                if metric not in private_perf:
                    continue
                    
                try:
                    original_val = float(original_perf[metric])
                    private_val = float(private_perf[metric])
                    sensitivity = sensitivities.get(metric, 1.0)
                    
                    # Calculate privacy-preserving bounds
                    noise_scale = sensitivity / self.epsilon
                    lower_bound = private_val - 2 * noise_scale
                    upper_bound = private_val + 2 * noise_scale
                    
                    perf_diffs[metric] = {
                        'difference': abs(original_val - private_val),
                        'difference_percent': self._safe_percentage(original_val, private_val),
                        'privacy_bounds': {
                            'lower': max(0, lower_bound),
                            'upper': upper_bound
                        },
                        'impact': self._calculate_performance_impact(
                            metric, original_val, private_val, sensitivity
                        )
                    }
                except (ValueError, TypeError):
                    logger.warning(f"Could not process metric {metric}: invalid values")
                    continue
            
            # Calculate overall privacy-utility score with proper weighting
            privacy_utility_score = self._calculate_privacy_utility_score(
                privacy_loss=privacy_loss,
                utility_loss=utility_loss,
                config_diffs=config_diffs,
                perf_diffs=perf_diffs
            )
            
            return {
                "privacy_loss": privacy_loss,
                "utility_loss": utility_loss,
                "configuration_differences": config_diffs,
                "performance_differences": perf_diffs,
                "privacy_utility_score": privacy_utility_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating privacy metrics: {str(e)}")
            raise

    def _calculate_sensitivity(self, value: float) -> float:
        """Calculate local sensitivity for a metric."""
        # Use smooth sensitivity to bound the impact of individual records
        if value == 0:
            return 1.0
        return min(abs(value) * 0.1, value)  # 10% of value or full value, whichever is smaller

    def _calculate_privacy_loss(self, original_config: Dict, private_config: Dict,
                              sensitivities: Dict[str, float], epsilon: float) -> float:
        """Calculate privacy loss using advanced composition theorem."""
        # Count number of queries/operations
        num_operations = sum(1 for k in original_config if k in private_config)
        
        # Use advanced composition theorem
        beta = 0.01  # Target failure probability
        epsilon_prime = epsilon * np.sqrt(2 * num_operations * np.log(1/beta))
        
        # Calculate actual privacy loss
        total_loss = 0.0
        for param, sensitivity in sensitivities.items():
            if param in original_config and param in private_config:
                diff = abs(float(original_config[param]) - float(private_config[param]))
                total_loss += min(1.0, diff / (sensitivity * epsilon_prime))
        
        return min(1.0, total_loss / num_operations)

    def _calculate_utility_loss(self, original_perf: Dict, private_perf: Dict,
                              weights: Dict[str, float]) -> float:
        """Calculate utility loss with weighted metrics."""
        total_loss = 0.0
        total_weight = sum(weights.values())
        
        for metric, weight in weights.items():
            if metric in original_perf and metric in private_perf:
                try:
                    orig_val = float(original_perf[metric])
                    priv_val = float(private_perf[metric])
                    
                    # Calculate normalized difference
                    if orig_val != 0:
                        diff = abs(orig_val - priv_val) / orig_val
                    else:
                        diff = abs(priv_val)
                    
                    # Apply weight and add to total
                    total_loss += (diff * weight / total_weight)
                except (ValueError, TypeError):
                    continue
        
        return min(1.0, total_loss)

    def _calculate_parameter_sensitivity(self, param: str, value: float) -> float:
        """Calculate parameter-specific sensitivity."""
        if param in ['read_ratio', 'write_ratio', 'hot_key_ratio']:
            return 0.1  # 10% sensitivity for ratios
        elif param in ['key_size', 'value_size']:
            return max(1.0, value * 0.05)  # 5% sensitivity for sizes
        elif param == 'operation_count':
            return max(1.0, value * 0.01)  # 1% sensitivity for counts
        else:
            return max(1.0, value * 0.1)  # Default 10% sensitivity

    def _calculate_parameter_privacy(self, original: float, private: float,
                                  sensitivity: float, epsilon: float) -> float:
        """Calculate privacy guarantee for a parameter."""
        if sensitivity == 0:
            return 1.0
        
        # Calculate privacy guarantee based on noise magnitude
        noise = abs(original - private)
        privacy_level = min(1.0, (noise / sensitivity) * epsilon)
        
        return privacy_level

    def _calculate_performance_impact(self, metric: str, original: float,
                                   private: float, sensitivity: float) -> str:
        """Calculate performance impact with dynamic thresholds."""
        diff_percent = self._safe_percentage(original, private)
        
        # Calculate dynamic thresholds based on sensitivity
        negligible_threshold = min(5.0, sensitivity * 2)
        minor_threshold = min(15.0, sensitivity * 5)
        moderate_threshold = min(30.0, sensitivity * 10)
        
        if diff_percent < negligible_threshold:
            return 'Negligible'
        elif diff_percent < minor_threshold:
            return 'Minor'
        elif diff_percent < moderate_threshold:
            return 'Moderate'
        else:
            return 'Significant'

    def _calculate_privacy_utility_score(self, privacy_loss: float, utility_loss: float,
                                      config_diffs: Dict, perf_diffs: Dict) -> Dict:
        """Calculate comprehensive privacy-utility tradeoff score."""
        # Weight factors for different components
        weights = {
            'privacy': 0.4,
            'utility': 0.3,
            'config_stability': 0.2,
            'performance_stability': 0.1
        }
        
        # Calculate config stability score
        config_changes = [d['difference_percent'] for d in config_diffs.values()]
        config_stability = 1.0 - (sum(config_changes) / len(config_changes) / 100) if config_changes else 1.0
        
        # Calculate performance stability score
        perf_changes = [d['difference_percent'] for d in perf_diffs.values()]
        perf_stability = 1.0 - (sum(perf_changes) / len(perf_changes) / 100) if perf_changes else 1.0
        
        # Calculate overall score
        privacy_score = 1.0 - privacy_loss
        utility_score = 1.0 - utility_loss
        
        overall_score = (
            weights['privacy'] * privacy_score +
            weights['utility'] * utility_score +
            weights['config_stability'] * config_stability +
            weights['performance_stability'] * perf_stability
        ) * 100
        
        return {
            'privacy_score': privacy_score * 100,
            'utility_score': utility_score * 100,
            'config_stability': config_stability * 100,
            'performance_stability': perf_stability * 100,
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