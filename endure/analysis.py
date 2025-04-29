from .visualization import EnhancedVisualization
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from .workload_generator import WorkloadCharacteristics, WorkloadGenerator
import shutil

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
    workload_characteristics: Dict[str, any]
    timestamp: str = datetime.now().isoformat()
    
    def validate(self) -> bool:
        """Validate the analysis result data."""
        try:
            # Check required metrics
            required_metrics = ['throughput', 'latency', 'space_amplification']
            if not all(metric in self.metrics for metric in required_metrics):
                logger.error(f"Missing required metrics: {required_metrics}")
                return False
            
            # Check metric values with reasonable ranges
            metric_ranges = {
                'throughput': (0, float('inf')),
                'latency': (0, 1000),  # Max 1000ms
                'space_amplification': (1, 10)  # Between 1x and 10x
            }
            
            for metric, value in self.metrics.items():
                if not isinstance(value, (int, float)):
                    logger.error(f"Invalid type for metric {metric}: {type(value)}")
                    return False
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"Invalid value for metric {metric}: {value}")
                    return False
                if metric in metric_ranges:
                    min_val, max_val = metric_ranges[metric]
                    if not min_val <= value <= max_val:
                        logger.warning(
                            f"Metric {metric} value {value} is outside expected range "
                            f"[{min_val}, {max_val}]"
                        )
            
            # Check configurations
            if not self.configurations:
                logger.error("No configuration data provided")
                return False
            if 'original' not in self.configurations or 'private' not in self.configurations:
                logger.error("Missing required configurations: original and/or private")
                return False
            
            # Validate configuration values
            for config_type in ['original', 'private']:
                config = self.configurations[config_type]
                for ratio in ['read_ratio', 'write_ratio', 'hot_key_ratio']:
                    if ratio in config:
                        value = config[ratio]
                        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                            logger.error(f"Invalid {ratio} in {config_type} config: {value}")
                            return False
            
            # Check workload characteristics
            if not isinstance(self.workload_characteristics, dict):
                logger.error("workload_characteristics must be a dictionary")
                return False
            
            # Validate workload characteristics with reasonable ranges
            wc_ranges = {
                'read_ratio': (0, 1),
                'write_ratio': (0, 1),
                'hot_key_ratio': (0, 1),
                'key_size': (1, 1024),  # Max 1KB
                'value_size': (1, 1048576),  # Max 1MB
                'operation_count': (1000, 100000000),  # Between 1K and 100M
                'hot_key_count': (1, 10000)  # Max 10K hot keys
            }
            
            for field, (min_val, max_val) in wc_ranges.items():
                if field in self.workload_characteristics:
                    value = self.workload_characteristics[field]
                    if not isinstance(value, (int, float)):
                        logger.error(f"Invalid type for {field}: {type(value)}")
                        return False
                    if not min_val <= value <= max_val:
                        logger.warning(
                            f"{field} value {value} is outside expected range [{min_val}, {max_val}]"
                        )
            
            # Validate ratio sum
            ratio_sum = (
                self.workload_characteristics.get('read_ratio', 0) +
                self.workload_characteristics.get('write_ratio', 0)
            )
            if abs(ratio_sum - 1.0) > 0.01:  # 1% tolerance
                logger.warning(f"read_ratio + write_ratio = {ratio_sum} (should be 1.0)")
            
            # Validate hot key count against operation count
            hot_key_count = self.workload_characteristics.get('hot_key_count', 0)
            operation_count = self.workload_characteristics.get('operation_count', 0)
            if hot_key_count > operation_count:
                logger.warning(
                    f"hot_key_count ({hot_key_count}) is greater than operation_count "
                    f"({operation_count})"
                )
            
            return True
        except Exception as e:
            logger.error(f"Error validating results: {str(e)}")
            return False

class BaseAnalysis:
    """Base class for analysis implementations."""
    
    def __init__(self, config: ConfigManager):
        """Initialize analysis with configuration."""
        self.config = config
        self.temp_files = []  # Track temporary files
        self._setup_logging()
        self._setup_directories()
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = os.path.join(self.config.analysis.results_dir, "analysis.log")
        logging.basicConfig(
            level=getattr(logging, self.config.analysis.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _setup_directories(self) -> None:
        """Set up required directories with validation."""
        try:
            os.makedirs(self.config.analysis.results_dir, exist_ok=True)
            if not os.access(self.config.analysis.results_dir, os.W_OK):
                raise PermissionError(f"No write permission for {self.config.analysis.results_dir}")
            
            # Check disk space
            disk = shutil.disk_usage(self.config.analysis.results_dir)
            free_gb = disk.free / (1024 * 1024 * 1024)
            if free_gb < self.config.analysis.max_disk_gb:
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

class PrivacyAnalysis(BaseAnalysis):
    """Privacy analysis implementation."""
    
    def run_privacy_sweep(self, workload: Dict[str, Any], epsilon_range: Tuple[float, float]) -> AnalysisResult:
        """Run privacy analysis with enhanced error handling and validation."""
        try:
            # Validate epsilon range
            if not isinstance(epsilon_range, tuple) or len(epsilon_range) != 2:
                raise ValueError("Invalid epsilon range format")
            if epsilon_range[0] < 0 or epsilon_range[1] <= epsilon_range[0]:
                raise ValueError("Invalid epsilon range values")
            
            # Validate workload
            if not isinstance(workload, dict):
                raise TypeError("Workload must be a dictionary")
            required_fields = ['read_ratio', 'write_ratio', 'key_size', 'value_size', 'operation_count']
            missing_fields = [field for field in required_fields if field not in workload]
            if missing_fields:
                raise ValueError(f"Missing required workload fields: {missing_fields}")
            
            # Create checkpoint file
            checkpoint_file = os.path.join(self.config.analysis.results_dir, "privacy_sweep_checkpoint.json")
            self.temp_files.append(checkpoint_file)
            
            results = []
            try:
                # Load checkpoint if exists
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r') as f:
                        results = json.load(f)
                    logging.info(f"Loaded {len(results)} results from checkpoint")
                
                # Run analysis for each epsilon value
                epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], num=10)
                for epsilon in epsilon_values:
                    if not any(r.get('epsilon') == epsilon for r in results):
                        result = self._run_single_privacy_analysis(workload, epsilon)
                        results.append(result)
                        # Save checkpoint
                        with open(checkpoint_file, 'w') as f:
                            json.dump(results, f)
                
                # Process and validate results
                processed_results = self._process_privacy_results(results)
                if not self._validate_privacy_results(processed_results):
                    raise ValueError("Invalid privacy analysis results")
                
                return AnalysisResult(
                    metrics=processed_results,
                    workload_characteristics=workload,
                    config={'epsilon_range': epsilon_range}
                )
            
            finally:
                # Clean up checkpoint file
                self.cleanup()
        
        except Exception as e:
            logging.error(f"Error in privacy sweep: {str(e)}")
            raise
    
    def _validate_privacy_results(self, results: Dict[str, Any]) -> bool:
        """Validate privacy analysis results."""
        try:
            required_metrics = ['privacy_loss', 'utility_loss', 'error_rate']
            if not all(metric in results for metric in required_metrics):
                logging.error("Missing required metrics in results")
                return False
            
            # Validate metric ranges
            if not (0 <= results['privacy_loss'] <= 1):
                logging.error("Privacy loss must be between 0 and 1")
                return False
            if not (0 <= results['utility_loss'] <= 1):
                logging.error("Utility loss must be between 0 and 1")
                return False
            if not (0 <= results['error_rate'] <= 1):
                logging.error("Error rate must be between 0 and 1")
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating privacy results: {str(e)}")
            return False

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
        """Safely calculate percentage difference between original and new values.
        
        Args:
            original: Original value
            new: New value to compare against
            
        Returns:
            float: Percentage difference (0-100)
        """
        try:
            if not isinstance(original, (int, float)) or not isinstance(new, (int, float)):
                logger.warning("Non-numeric values provided to _safe_percentage")
                return 0.0
                
            if original == 0:
                if new == 0:
                    return 0.0
                logger.warning("Original value is 0, cannot calculate percentage difference")
                return 100.0
                
            if np.isnan(original) or np.isnan(new) or np.isinf(original) or np.isinf(new):
                logger.warning("Invalid values (NaN or Inf) provided to _safe_percentage")
                return 0.0
                
            return abs(original - new) / original * 100
        except Exception as e:
            logger.error(f"Error calculating percentage difference: {str(e)}")
            return 0.0
        
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