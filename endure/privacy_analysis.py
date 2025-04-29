import numpy as np
from typing import Dict, List
from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
from .endure_integration import EndureIntegration
import matplotlib.pyplot as plt
import json
import os
import logging

logger = logging.getLogger(__name__)

class PrivacyAnalysis:
    def __init__(self):
        self.results_dir = "privacy_results"
        os.makedirs(self.results_dir, exist_ok=True)

    def _validate_results(self, results: Dict) -> bool:
        """Validate the results of the privacy analysis with more lenient checks."""
        try:
            # Check if results is a dictionary
            if not isinstance(results, dict):
                logger.warning("Results must be a dictionary")
                return False
            
            valid_epsilons = 0
            # Check each epsilon entry
            for epsilon, trials in results.items():
                try:
                    # Convert epsilon to float if it's a string
                    if isinstance(epsilon, str):
                        epsilon = float(epsilon)
                    
                    if not isinstance(epsilon, (int, float)):
                        logger.warning(f"Skipping invalid epsilon value: {epsilon}")
                        continue
                    if not isinstance(trials, list):
                        logger.warning(f"Skipping invalid trials data for epsilon {epsilon}")
                        continue
                    
                    valid_trials = 0
                    # Check each trial
                    for trial in trials:
                        if not isinstance(trial, dict):
                            logger.warning("Skipping invalid trial data structure")
                            continue
                        
                        # Check for required fields with lenient validation
                        has_comparison = "comparison" in trial and isinstance(trial["comparison"], dict)
                        has_privacy_metrics = "privacy_metrics" in trial and isinstance(trial["privacy_metrics"], dict)
                        
                        if not (has_comparison or has_privacy_metrics):
                            logger.warning("Skipping trial with missing required fields")
                            continue
                        
                        # Validate comparison data if present
                        if has_comparison:
                            comparison = trial["comparison"]
                            if not isinstance(comparison, dict):
                                logger.warning("Skipping trial with invalid comparison data")
                                continue
                            if "parameter_differences" not in comparison:
                                logger.warning("Skipping trial with missing parameter differences")
                                continue
                        
                        valid_trials += 1
                    
                    if valid_trials > 0:
                        valid_epsilons += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing epsilon {epsilon}: {str(e)}")
                    continue
            
            # Consider results valid if we have at least one valid epsilon with trials
            return valid_epsilons > 0
            
        except Exception as e:
            logger.warning(f"Error validating results: {str(e)}")
            return False

    def run_privacy_sweep(self, characteristics: Dict,
                         epsilons: List[float] = [0.1, 0.5, 1.0, 2.0, 5.0],
                         num_trials: int = 5) -> Dict:
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
                        # Create workload generator with current epsilon
                        workload_generator = WorkloadGenerator(epsilon=epsilon)
                        
                        # Generate workloads with differential privacy
                        original_workload, private_workload = workload_generator.generate_workload(workload_chars)
                        
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
                    except Exception as e:
                        logger.error(f"Error in trial {trial} for epsilon {epsilon}: {str(e)}")
                        continue
                
                results[epsilon] = trial_results
            
            # Validate results before saving
            if not self._validate_results(results):
                raise ValueError("Invalid results generated")
            
            self._save_results(results)
            return results
            
        except Exception as e:
            logger.error(f"Error in privacy sweep: {str(e)}")
            raise

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
        try:
            if not self._validate_results(results):
                raise ValueError("Invalid results for plotting")
            
            # Create visualization instance
            viz = EnhancedVisualization(self.results_dir)
            
            # Generate plots
            viz.plot_privacy_performance_tradeoff(results)
            viz.plot_configuration_differences(results)
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            raise

    def _calculate_statistical_metrics(self, values: List[float]) -> Dict:
        """Calculate statistical metrics with confidence intervals."""
        try:
            if not values:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'ci_lower': 0.0,
                    'ci_upper': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
                
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample standard deviation
            
            # Calculate 95% confidence interval
            n = len(values)
            if n > 1:
                ci = 1.96 * (std / np.sqrt(n))  # 95% CI
                ci_lower = mean - ci
                ci_upper = mean + ci
            else:
                ci_lower = mean
                ci_upper = mean
            
            return {
                'mean': float(mean),
                'std': float(std),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        except Exception as e:
            logger.error(f"Error calculating statistical metrics: {str(e)}")
            return {
                'mean': 0.0,
                'std': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'min': 0.0,
                'max': 0.0
            }

    def _analyze_tuning_stability(self, results: Dict) -> Dict:
        """Analyze tuning stability across different epsilon values."""
        try:
            stability_metrics = {}
            
            for epsilon, trials in results.items():
                if not trials:
                    continue
                    
                # Collect configuration differences
                config_diffs = []
                perf_diffs = {
                    'throughput': [],
                    'latency': [],
                    'space_amplification': []
                }
                
                for trial in trials:
                    try:
                        # Configuration differences
                        for param_diff in trial['comparison']['parameter_differences'].values():
                            config_diffs.append(param_diff['difference_percent'])
                        
                        # Performance differences
                        for metric, diff in trial['privacy_metrics']['performance_differences'].items():
                            if metric in perf_diffs:
                                perf_diffs[metric].append(diff['difference_percent'])
                    except (KeyError, TypeError):
                        continue
                
                # Calculate statistical metrics
                stability_metrics[epsilon] = {
                    'configuration_stability': self._calculate_statistical_metrics(config_diffs),
                    'performance_stability': {
                        metric: self._calculate_statistical_metrics(values)
                        for metric, values in perf_diffs.items()
                    }
                }
            
            return stability_metrics
        except Exception as e:
            logger.error(f"Error analyzing tuning stability: {str(e)}")
            return {}

    def _analyze_workload_patterns(self, results: Dict) -> Dict:
        """Analyze workload patterns and their impact on tuning."""
        try:
            pattern_metrics = {}
            
            for epsilon, trials in results.items():
                if not trials:
                    continue
                    
                pattern_metrics[epsilon] = {
                    'read_write_ratio': [],
                    'hot_key_access': [],
                    'operation_distribution': []
                }
                
                for trial in trials:
                    try:
                        if 'workload_metrics' in trial:
                            metrics = trial['workload_metrics']
                            pattern_metrics[epsilon]['read_write_ratio'].append(
                                metrics.get('read_ratio', 0.0)
                            )
                            pattern_metrics[epsilon]['hot_key_access'].append(
                                metrics.get('hot_key_ratio', 0.0)
                            )
                            pattern_metrics[epsilon]['operation_distribution'].append(
                                metrics.get('operation_distribution', {})
                            )
                    except (KeyError, TypeError):
                        continue
                
                # Calculate statistical metrics for each pattern
                for pattern, values in pattern_metrics[epsilon].items():
                    if values and isinstance(values[0], (int, float)):
                        pattern_metrics[epsilon][pattern] = self._calculate_statistical_metrics(values)
            
            return pattern_metrics
        except Exception as e:
            logger.error(f"Error analyzing workload patterns: {str(e)}")
            return {}

    def load_results(self) -> Dict[float, List]:
        """Load results from file with proper type conversion for epsilon values."""
        try:
            results_path = os.path.join(self.results_dir, "privacy_analysis_results.json")
            if not os.path.exists(results_path):
                logger.error(f"Results file not found: {results_path}")
                return {}

            with open(results_path, 'r') as f:
                raw_results = json.load(f)

            # Convert string epsilon keys to float and validate
            results = {}
            for eps_str, trials in raw_results.items():
                try:
                    # Convert string to float
                    epsilon = float(eps_str)
                    
                    # Validate epsilon value
                    if epsilon <= 0:
                        logger.warning(f"Invalid epsilon value: {eps_str} (must be positive)")
                        continue
                        
                    # Validate trials data
                    if not isinstance(trials, list):
                        logger.warning(f"Invalid trials data for epsilon {eps_str}: {type(trials)}")
                        continue
                        
                    # Convert any string values in trials to appropriate types
                    processed_trials = []
                    for trial in trials:
                        if not isinstance(trial, dict):
                            logger.warning(f"Invalid trial data format for epsilon {eps_str}")
                            continue
                            
                        processed_trial = {}
                        for key, value in trial.items():
                            if key == 'privacy_metrics':
                                processed_trial[key] = self._process_privacy_metrics(value)
                            else:
                                processed_trial[key] = value
                                
                        processed_trials.append(processed_trial)
                    
                    results[epsilon] = processed_trials
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing epsilon value {eps_str}: {str(e)}")
                    continue

            # Validate the converted results
            if not self._validate_results(results):
                logger.error("Loaded results failed validation")
                return {}

            return results

        except Exception as e:
            logger.error(f"Error loading results: {str(e)}")
            return {}

    def _process_privacy_metrics(self, metrics: Dict) -> Dict:
        """Process privacy metrics data, converting string values to appropriate types."""
        try:
            processed_metrics = {}
            
            # Process performance differences
            if 'performance_differences' in metrics:
                processed_metrics['performance_differences'] = {}
                for metric, value in metrics['performance_differences'].items():
                    if isinstance(value, (int, float)):
                        processed_metrics['performance_differences'][metric] = {
                            'difference_percent': float(value)
                        }
                    elif isinstance(value, dict):
                        processed_metrics['performance_differences'][metric] = {
                            'difference_percent': float(value.get('difference_percent', 0.0))
                        }
            
            # Process configuration differences
            if 'configuration_differences' in metrics:
                processed_metrics['configuration_differences'] = {}
                for param, value in metrics['configuration_differences'].items():
                    if isinstance(value, (int, float)):
                        processed_metrics['configuration_differences'][param] = {
                            'difference_percent': float(value)
                        }
                    elif isinstance(value, dict):
                        processed_metrics['configuration_differences'][param] = {
                            'difference_percent': float(value.get('difference_percent', 0.0))
                        }
            
            # Process privacy utility score
            if 'privacy_utility_score' in metrics:
                processed_metrics['privacy_utility_score'] = {}
                for score_type, value in metrics['privacy_utility_score'].items():
                    if isinstance(value, (int, float)):
                        processed_metrics['privacy_utility_score'][score_type] = float(value)
                    elif isinstance(value, dict):
                        processed_metrics['privacy_utility_score'][score_type] = float(value.get('score', 0.0))
            
            return processed_metrics
            
        except Exception as e:
            logger.error(f"Error processing privacy metrics: {str(e)}")
            return {}

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