from .visualization import EnhancedVisualization
import os
import json
from typing import Dict, List, Optional
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from .workload import WorkloadCharacteristics, WorkloadGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Data class for analysis results with validation."""
    metrics: Dict[str, float]
    configurations: Dict[str, Dict]
    workload_characteristics: Dict[str, float]
    timestamp: str = datetime.now().isoformat()
    
    def validate(self) -> bool:
        """Validate the analysis result data."""
        try:
            # Check required metrics
            required_metrics = ['throughput', 'latency', 'space_amplification']
            if not all(metric in self.metrics for metric in required_metrics):
                logger.error(f"Missing required metrics: {required_metrics}")
                return False
            
            # Check metric values
            for metric, value in self.metrics.items():
                if not isinstance(value, (int, float)):
                    logger.error(f"Invalid type for metric {metric}: {type(value)}")
                    return False
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"Invalid value for metric {metric}: {value}")
                    return False
                if value < 0:
                    logger.error(f"Negative value for metric {metric}: {value}")
                    return False
            
            # Check configurations
            if not self.configurations:
                logger.error("No configuration data provided")
                return False
            if 'original' not in self.configurations or 'private' not in self.configurations:
                logger.error("Missing required configurations: original and/or private")
                return False
            
            # Check workload characteristics
            if not self.workload_characteristics:
                logger.error("No workload characteristics provided")
                return False
            
            # Only check ratio values are between 0 and 1
            ratio_fields = ['read_ratio', 'write_ratio', 'hot_key_ratio']
            for field in ratio_fields:
                if field in self.workload_characteristics:
                    value = self.workload_characteristics[field]
                    if not 0 <= value <= 1:
                        logger.error(f"Invalid ratio value for {field} (must be between 0 and 1): {value}")
                        return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating results: {str(e)}")
            return False

class BaseAnalysis:
    """Base class for analysis with common functionality."""
    
    def __init__(self, results_dir: str):
        """Initialize base analysis class."""
        self.results_dir = results_dir
        self._setup_logging()
        self._setup_directories()
        self.visualizer = EnhancedVisualization(results_dir)
    
    def _setup_logging(self) -> None:
        """Setup file logging for the analysis."""
        log_file = os.path.join(self.results_dir, f"{self.__class__.__name__.lower()}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def _setup_directories(self) -> None:
        """Setup required directories."""
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {self.results_dir}: {str(e)}")
            raise
    
    def _validate_results(self, results: Dict) -> AnalysisResult:
        """Validate analysis results and return AnalysisResult object."""
        try:
            # Check required fields
            required_fields = ['metrics', 'configurations', 'workload_characteristics']
            if not all(field in results for field in required_fields):
                logger.error(f"Missing required fields: {required_fields}")
                return None
            
            # Create AnalysisResult object
            analysis_result = AnalysisResult(
                metrics=results['metrics'],
                configurations=results['configurations'],
                workload_characteristics=results['workload_characteristics']
            )
            
            # Validate the result
            if not analysis_result.validate():
                return None
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error validating results: {str(e)}")
            return None
    
    def _process_data(self, data: Dict) -> Dict:
        """Process data with standardized handling."""
        try:
            processed = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    processed[key] = self._process_data(value)
                elif isinstance(value, (int, float)):
                    processed[key] = float(value)  # Ensure numeric values are floats
                elif isinstance(value, list):
                    # Handle lists of numeric values
                    processed[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
                else:
                    processed[key] = value
            return processed
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def _save_results(self, results: Dict, filename: str) -> None:
        """Save analysis results with error handling."""
        try:
            # Process data to ensure numeric values are floats
            processed_results = self._process_data(results)
            
            results_with_metadata = {
                'timestamp': datetime.now().isoformat(),
                'data': processed_results,
                'validation_status': 'success'
            }
            
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(results_with_metadata, f, indent=2)
                
            # Verify the saved data
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
                if not isinstance(loaded_data.get('data', {}), dict):
                    logger.error("Saved data has invalid structure")
                    raise ValueError("Invalid data structure in saved file")
                
        except Exception as e:
            logger.error(f"Error saving results to {filename}: {str(e)}")
            raise
    
    def _handle_numeric_values(self, value: float) -> float:
        """Handle numeric values with edge cases."""
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Invalid numeric value: {value}")
            return 0.0
        return float(value)

class PrivacyAnalysis(BaseAnalysis):
    """Privacy analysis with edge case handling."""
    
    def __init__(self, results_dir: str, config: Optional[Dict] = None):
        """Initialize privacy analysis with configuration.
        
        Args:
            results_dir: Directory to store results
            config: Optional configuration dictionary with:
                - epsilon_values: List of epsilon values to test
                - num_trials: Number of trials per epsilon
                - workload_characteristics: Default workload characteristics
                - checkpoint_interval: Number of trials between checkpoints
        """
        super().__init__(results_dir)
        self.config = self._validate_config(config or {})
        self.checkpoint_file = os.path.join(results_dir, "privacy_analysis_checkpoint.json")
        self.current_trial = 0
        self.total_trials = len(self.config['epsilon_values']) * self.config['num_trials']
    
    def _validate_config(self, config: Dict) -> Dict:
        """Validate and set default configuration values."""
        default_config = {
            'epsilon_values': [0.1, 0.5, 1.0, 2.0, 5.0],
            'num_trials': 5,
            'workload_characteristics': {
                'read_ratio': 0.7,
                'write_ratio': 0.3,
                'key_size': 16,
                'value_size': 100,
                'operation_count': 1000000,
                'hot_key_ratio': 0.2,
                'hot_key_count': 100
            },
            'checkpoint_interval': 5
        }
        
        # Update with user config, keeping defaults for missing values
        validated_config = default_config.copy()
        validated_config.update(config)
        
        # Validate epsilon values
        if not all(isinstance(e, (int, float)) and e > 0 for e in validated_config['epsilon_values']):
            raise ValueError("Epsilon values must be positive numbers")
        
        # Validate number of trials
        if not isinstance(validated_config['num_trials'], int) or validated_config['num_trials'] < 1:
            raise ValueError("Number of trials must be a positive integer")
        
        return validated_config
    
    def run_analysis(self, results: Optional[Dict] = None) -> None:
        """Run privacy analysis with edge case handling and progress tracking."""
        try:
            # Load checkpoint if exists
            if os.path.exists(self.checkpoint_file):
                self._load_checkpoint()
            
            visualization_data = self._initialize_visualization_data()
            
            # Run trials for each epsilon value
            for epsilon in self.config['epsilon_values']:
                epsilon_key = float(epsilon)
                if epsilon_key not in visualization_data:
                    visualization_data[epsilon_key] = []
                
                # Run specified number of trials
                for trial in range(self.config['num_trials']):
                    if self.current_trial >= self.total_trials:
                        break
                        
                    try:
                        trial_data = self._run_single_trial(epsilon)
                        visualization_data[epsilon_key].append(trial_data)
                        
                        # Save checkpoint periodically
                        if (self.current_trial + 1) % self.config['checkpoint_interval'] == 0:
                            self._save_checkpoint(visualization_data)
                            
                        self.current_trial += 1
                        logger.info(f"Completed trial {self.current_trial}/{self.total_trials} for epsilon={epsilon}")
                        
                    except Exception as e:
                        logger.error(f"Error in trial {self.current_trial + 1} for epsilon={epsilon}: {str(e)}")
                        continue
            
            # Calculate statistical summaries
            self._add_statistical_summaries(visualization_data)
            
            # Save final results
            self._save_results(visualization_data, "privacy_analysis.json")
            
            # Generate visualizations
            self.visualizer.plot_privacy_performance_tradeoff(visualization_data)
            self.visualizer.plot_configuration_differences(visualization_data)
            
            # Clean up checkpoint file
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)
                
            logger.info("Privacy analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in privacy analysis: {str(e)}")
            raise
    
    def _initialize_visualization_data(self) -> Dict:
        """Initialize visualization data structure."""
        return {
            float(epsilon): [] for epsilon in self.config['epsilon_values']
        }
    
    def _run_single_trial(self, epsilon: float) -> Dict:
        """Run a single trial with the given epsilon value."""
        # Create workload generator with current epsilon
        workload_generator = WorkloadGenerator(epsilon=epsilon)
        
        # Generate workloads
        original_workload, private_workload = workload_generator.generate_workload(
            WorkloadCharacteristics(**self.config['workload_characteristics'])
        )
        
        # Calculate workload metrics
        original_metrics = workload_generator.calculate_workload_metrics(original_workload)
        private_metrics = workload_generator.calculate_workload_metrics(private_workload)
        
        # Create trial data
        trial_data = {
            'privacy_metrics': {
                'performance_differences': {},
                'configuration_differences': {},
                'privacy_utility_score': {
                    'performance_score': 0.0,
                    'configuration_score': 0.0,
                    'overall_score': 0.0
                }
            },
            'workload_characteristics': self.config['workload_characteristics'],
            'trial_number': self.current_trial + 1,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate performance differences
        for metric in ['throughput', 'latency', 'space_amplification']:
            if metric in original_metrics and metric in private_metrics:
                original_val = original_metrics[metric]
                private_val = private_metrics[metric]
                
                trial_data['privacy_metrics']['performance_differences'][metric] = {
                    'difference': abs(original_val - private_val),
                    'difference_percent': self._safe_percentage(original_val, private_val),
                    'impact': self._calculate_performance_impact(metric, original_val, private_val)
                }
        
        # Calculate configuration differences
        for param in ['read_ratio', 'write_ratio', 'hot_key_ratio']:
            if param in original_metrics and param in private_metrics:
                original_val = original_metrics[param]
                private_val = private_metrics[param]
                
                trial_data['privacy_metrics']['configuration_differences'][param] = {
                    'difference': abs(original_val - private_val),
                    'difference_percent': self._safe_percentage(original_val, private_val)
                }
        
        # Calculate privacy utility score
        trial_data['privacy_metrics']['privacy_utility_score'] = self._calculate_privacy_utility_score(
            trial_data['privacy_metrics']['configuration_differences'],
            trial_data['privacy_metrics']['performance_differences']
        )
        
        return trial_data
    
    def _add_statistical_summaries(self, visualization_data: Dict) -> None:
        """Add statistical summaries to the visualization data."""
        for epsilon, trials in visualization_data.items():
            if not trials:
                continue
                
            # Calculate means and standard deviations
            metrics = ['performance_score', 'configuration_score', 'overall_score']
            for metric in metrics:
                values = [trial['privacy_metrics']['privacy_utility_score'][metric] for trial in trials]
                mean = np.mean(values)
                std = np.std(values)
                
                visualization_data[epsilon].append({
                    'statistics': {
                        metric: {
                            'mean': mean,
                            'std_dev': std,
                            'min': min(values),
                            'max': max(values)
                        }
                    }
                })
    
    def _save_checkpoint(self, data: Dict) -> None:
        """Save checkpoint data."""
        checkpoint_data = {
            'current_trial': self.current_trial,
            'data': data
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def _load_checkpoint(self) -> None:
        """Load checkpoint data."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                self.current_trial = checkpoint_data['current_trial']
                return checkpoint_data['data']
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None
    
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

class SensitivityAnalysis(BaseAnalysis):
    """Sensitivity analysis with edge case handling."""
    
    def run_analysis(self, results: Dict) -> None:
        """Run sensitivity analysis with edge case handling."""
        try:
            # Validate and process results
            analysis_result = self._validate_results(results)
            if not analysis_result:
                raise ValueError("Invalid analysis results")
            
            # Process all data
            processed_data = self._process_data({
                'metrics': analysis_result.metrics,
                'configurations': analysis_result.configurations,
                'workload_characteristics': analysis_result.workload_characteristics
            })
            
            # Save results
            self._save_results(processed_data, "sensitivity_analysis.json")
            
            # Generate visualizations
            self.visualizer.plot_workload_sensitivity(processed_data)
            self.visualizer.plot_correlation_analysis(processed_data)
            
            logger.info("Sensitivity analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {str(e)}")
            raise

class PerformanceAnalysis(BaseAnalysis):
    """Performance analysis with edge case handling."""
    
    def run_analysis(self, results: Dict) -> None:
        """Run performance analysis with edge case handling."""
        try:
            # Validate and process results
            analysis_result = self._validate_results(results)
            if not analysis_result:
                raise ValueError("Invalid analysis results")
            
            # Process all data and ensure epsilon values are floats
            processed_data = self._process_data({
                'metrics': analysis_result.metrics,
                'configurations': analysis_result.configurations,
                'workload_characteristics': analysis_result.workload_characteristics
            })
            
            # Format data for visualization
            epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # From config
            visualization_data = {}
            
            for epsilon in epsilon_values:
                # Create trial data
                trial_data = {
                    'privacy_metrics': {
                        'performance_differences': {},
                        'configuration_differences': {},
                        'privacy_utility_score': {
                            'performance_score': 0.0,
                            'configuration_score': 0.0,
                            'overall_score': 0.0
                        }
                    },
                    'workload_characteristics': processed_data['workload_characteristics']
                }
                
                # Add performance differences
                for metric in ['throughput', 'latency', 'space_amplification']:
                    if metric in processed_data['metrics']:
                        trial_data['privacy_metrics']['performance_differences'][metric] = {
                            'difference': abs(processed_data['configurations']['original'][metric] - 
                                           processed_data['configurations']['private'][metric]),
                            'difference_percent': self._safe_percentage(
                                processed_data['configurations']['original'][metric],
                                processed_data['configurations']['private'][metric]
                            ),
                            'impact': self._calculate_performance_impact(
                                metric,
                                processed_data['configurations']['original'][metric],
                                processed_data['configurations']['private'][metric]
                            )
                        }
                
                # Add configuration differences
                for param in processed_data['configurations']['original'].keys():
                    if param in processed_data['configurations']['private']:
                        trial_data['privacy_metrics']['configuration_differences'][param] = {
                            'difference': abs(processed_data['configurations']['original'][param] - 
                                           processed_data['configurations']['private'][param]),
                            'difference_percent': self._safe_percentage(
                                processed_data['configurations']['original'][param],
                                processed_data['configurations']['private'][param]
                            )
                        }
                
                visualization_data[float(epsilon)] = [trial_data]  # List of trials for each epsilon
            
            # Save results
            self._save_results(visualization_data, "performance_analysis.json")
            
            # Generate visualizations
            self.visualizer.plot_privacy_performance_tradeoff(visualization_data)
            self.visualizer.plot_configuration_differences(visualization_data)
            
            logger.info("Performance analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
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