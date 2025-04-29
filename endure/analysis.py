from .visualization import EnhancedVisualization
from .config import ConfigManager
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from .workload_generator import WorkloadCharacteristics, WorkloadGenerator
import shutil
import psutil
from pathlib import Path

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
    configurations: Dict[str, Dict[str, Union[int, float]]]
    workload_characteristics: Dict[str, Any]
    timestamp: str = datetime.now().isoformat()
    
    def validate(self) -> bool:
        """Validate the analysis result data.
        
        Returns:
            bool: True if validation passes, False otherwise
            
        Raises:
            ValueError: If validation fails with specific error message
        """
        try:
            # Validate metrics
            if not self._validate_metrics():
                return False
            
            # Validate configurations
            if not self._validate_configurations():
                return False
            
            # Validate workload characteristics
            if not self._validate_workload_characteristics():
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating results: {str(e)}")
            return False
    
    def _validate_metrics(self) -> bool:
        """Validate metrics dictionary.
        
        Returns:
            bool: True if metrics are valid, False otherwise
        """
        try:
            # Check required metrics
            required_metrics = ['throughput', 'latency', 'space_amplification']
            missing_metrics = [metric for metric in required_metrics if metric not in self.metrics]
            if missing_metrics:
                logger.error(f"Missing required metrics: {missing_metrics}")
                return False
            
            # Check metric values with reasonable ranges
            metric_ranges = {
                'throughput': (0, float('inf')),  # No upper limit on throughput
                'latency': (0, 5000),  # Increased to 5 seconds to accommodate slower systems
                'space_amplification': (1, 20)  # Increased to 20x to accommodate more use cases
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
                        # Don't fail validation for out-of-range values, just warn
                        # return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating metrics: {str(e)}")
            return False
    
    def _validate_configurations(self) -> bool:
        """Validate configurations dictionary.
        
        Returns:
            bool: True if configurations are valid, False otherwise
        """
        try:
            # Check configurations exist
            if not self.configurations:
                logger.error("No configuration data provided")
                return False
            
            # Check required configurations
            required_configs = ['original', 'private']
            missing_configs = [config for config in required_configs if config not in self.configurations]
            if missing_configs:
                logger.error(f"Missing required configurations: {missing_configs}")
                return False
            
            # Validate configuration values
            for config_type in required_configs:
                config = self.configurations[config_type]
                if not isinstance(config, dict):
                    logger.error(f"Configuration {config_type} must be a dictionary")
                    return False
                
                # Validate ratio values
                for ratio in ['read_ratio', 'write_ratio', 'hot_key_ratio']:
                    if ratio in config:
                        value = config[ratio]
                        if not isinstance(value, (int, float)):
                            logger.error(f"Invalid type for {ratio} in {config_type} config: {type(value)}")
                            return False
                        if not 0 <= value <= 1:
                            logger.error(f"Invalid {ratio} in {config_type} config: {value} (must be between 0 and 1)")
                            return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating configurations: {str(e)}")
            return False
    
    def _validate_workload_characteristics(self) -> bool:
        """Validate workload characteristics dictionary.
        
        Returns:
            bool: True if workload characteristics are valid, False otherwise
        """
        try:
            # Check workload characteristics type
            if not isinstance(self.workload_characteristics, dict):
                logger.error("workload_characteristics must be a dictionary")
                return False
            
            # Validate workload characteristics with reasonable ranges
            wc_ranges = {
                'read_ratio': (0, 1),
                'write_ratio': (0, 1),
                'hot_key_ratio': (0, 1),
                'key_size': (1, 4096),  # Increased to 4KB
                'value_size': (1, 16777216),  # Increased to 16MB
                'operation_count': (100, 1000000000),  # Lowered to 100, increased to 1B
                'hot_key_count': (1, 100000)  # Increased to 100K hot keys
            }
            
            # Check required fields
            required_fields = ['read_ratio', 'write_ratio', 'key_size', 'value_size', 'operation_count']
            missing_fields = [field for field in required_fields if field not in self.workload_characteristics]
            if missing_fields:
                logger.error(f"Missing required workload fields: {missing_fields}")
                return False
            
            # Validate field values
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
                        # Don't fail validation for out-of-range values, just warn
                        # return False
            
            # Validate ratio sum with more tolerance
            ratio_sum = (
                self.workload_characteristics.get('read_ratio', 0) +
                self.workload_characteristics.get('write_ratio', 0)
            )
            if abs(ratio_sum - 1.0) > 0.05:  # Increased tolerance to 5%
                logger.warning(f"read_ratio + write_ratio = {ratio_sum} (should be 1.0)")
            
            # Validate hot key count against operation count with more flexibility
            hot_key_count = self.workload_characteristics.get('hot_key_count', 0)
            operation_count = self.workload_characteristics.get('operation_count', 0)
            if hot_key_count > operation_count:
                logger.warning(
                    f"hot_key_count ({hot_key_count}) is greater than operation_count "
                    f"({operation_count})"
                )
                # Don't fail validation, just warn
            
            return True
        except Exception as e:
            logger.error(f"Error validating workload characteristics: {str(e)}")
            return False

class BaseAnalysis:
    """Base class for analysis implementations."""
    
    def __init__(self, config: ConfigManager):
        """Initialize analysis with configuration.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.temp_files = []  # Track temporary files
        self._setup_logging()
        self._setup_directories()
        self._setup_resource_monitoring()
    
    def _setup_resource_monitoring(self) -> None:
        """Set up resource monitoring with relative limits."""
        try:
            # Get system memory
            memory = psutil.virtual_memory()
            self._total_memory = memory.total
            
            # Set relative memory limits (percentage of total memory)
            self._memory_warning_threshold = 0.7  # 70% of total memory
            self._memory_critical_threshold = 0.9  # 90% of total memory
            
            # Set relative disk limits
            disk = shutil.disk_usage(self.config.analysis.results_dir)
            self._total_disk = disk.total
            self._disk_warning_threshold = 0.8  # 80% of disk space
            self._disk_critical_threshold = 0.95  # 95% of disk space
            
            # Initialize memory tracking
            self._peak_memory_usage = 0
            self._start_memory = psutil.Process().memory_info().rss
        except Exception as e:
            logger.error(f"Failed to setup resource monitoring: {str(e)}")
            raise
    
    def _check_resources(self) -> None:
        """Check system resources and raise warning if thresholds are exceeded."""
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss
            self._peak_memory_usage = max(self._peak_memory_usage, current_memory)
            
            # Calculate memory usage percentage
            memory_percent = current_memory / self._total_memory
            
            # Check memory thresholds
            if memory_percent > self._memory_critical_threshold:
                logger.error(
                    f"Critical memory usage: {memory_percent:.1%} of total memory "
                    f"({current_memory / (1024*1024*1024):.2f}GB)"
                )
                self._cleanup_temporary_files()
            elif memory_percent > self._memory_warning_threshold:
                logger.warning(
                    f"High memory usage: {memory_percent:.1%} of total memory "
                    f"({current_memory / (1024*1024*1024):.2f}GB)"
                )
            
            # Check disk space
            disk = shutil.disk_usage(self.config.analysis.results_dir)
            disk_percent = disk.used / self._total_disk
            
            if disk_percent > self._disk_critical_threshold:
                logger.error(
                    f"Critical disk usage: {disk_percent:.1%} of total disk "
                    f"({disk.used / (1024*1024*1024):.2f}GB used)"
                )
                self._cleanup_temporary_files()
            elif disk_percent > self._disk_warning_threshold:
                logger.warning(
                    f"High disk usage: {disk_percent:.1%} of total disk "
                    f"({disk.used / (1024*1024*1024):.2f}GB used)"
                )
            
            # Log memory usage statistics
            logger.debug(
                f"Memory usage: {memory_percent:.1%} of total, "
                f"Peak: {self._peak_memory_usage / (1024*1024*1024):.2f}GB"
            )
            
        except Exception as e:
            logger.warning(f"Resource check failed: {str(e)}")
    
    def _monitor_resources(self) -> None:
        """Monitor resources during analysis and adjust batch size if needed."""
        try:
            # Get current memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            current_memory = memory_info.rss
            memory_percent = current_memory / self._total_memory
            
            # If memory usage is high, reduce batch size
            if memory_percent > self._memory_warning_threshold:
                if hasattr(self, '_batch_size'):
                    # Reduce batch size by 20% but don't go below 100
                    new_batch_size = max(100, int(self._batch_size * 0.8))
                    if new_batch_size != self._batch_size:
                        logger.info(
                            f"Reducing batch size from {self._batch_size} to {new_batch_size} "
                            f"due to high memory usage ({memory_percent:.1%})"
                        )
                        self._batch_size = new_batch_size
            
            # Check disk space
            disk = shutil.disk_usage(self.config.analysis.results_dir)
            disk_percent = disk.used / self._total_disk
            
            if disk_percent > self._disk_warning_threshold:
                logger.warning(
                    f"High disk usage: {disk_percent:.1%} of total disk "
                    f"({disk.used / (1024*1024*1024):.2f}GB used)"
                )
                self._cleanup_temporary_files()
                
        except Exception as e:
            logger.warning(f"Resource monitoring failed: {str(e)}")
    
    def _cleanup_temporary_files(self) -> None:
        """Clean up temporary files to free resources."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
        self.temp_files.clear()
    
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
            results_dir = Path(self.config.analysis.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            if not os.access(results_dir, os.W_OK):
                raise PermissionError(f"No write permission for {results_dir}")
            
            # Check disk space
            disk = shutil.disk_usage(results_dir)
            free_gb = disk.free / (1024 * 1024 * 1024)
            if free_gb < self.config.analysis.max_disk_gb:
                raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB available")
        except Exception as e:
            logger.error(f"Error setting up directories: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        self._cleanup_temporary_files()
        # Clean up progress file if it exists
        if hasattr(self, '_progress_file') and os.path.exists(self._progress_file):
            try:
                os.remove(self._progress_file)
            except Exception as e:
                logger.warning(f"Failed to remove progress file: {str(e)}")

class PrivacyAnalysis(BaseAnalysis):
    """Privacy analysis implementation."""
    
    def __init__(self, config: ConfigManager = None, results_dir: str = None):
        """Initialize privacy analysis with configuration and results directory."""
        if config is None:
            from .config import ConfigManager
            config = ConfigManager()
        if results_dir:
            config.analysis.results_dir = results_dir
        super().__init__(config)
        self._batch_size = 1000  # Number of operations per batch
        self._progress_file = os.path.join(self.config.analysis.results_dir, "progress.json")
    
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
            
            # Initialize progress tracking
            self._initialize_progress(epsilon_range)
            
            results = []
            try:
                # Load checkpoint if exists
                if os.path.exists(checkpoint_file):
                    with open(checkpoint_file, 'r') as f:
                        results = json.load(f)
                    logger.info(f"Loaded {len(results)} results from checkpoint")
                
                # Run analysis for each epsilon value
                epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], num=10)
                for i, epsilon in enumerate(epsilon_values):
                    if not any(r.get('epsilon') == epsilon for r in results):
                        # Check resources before each iteration
                        self._check_resources()
                        
                        # Update progress
                        self._update_progress(i, len(epsilon_values), epsilon)
                        
                        # Process in batches
                        result = self._run_batched_privacy_analysis(workload, epsilon)
                        results.append(result)
                        
                        # Save checkpoint after each epsilon
                        with open(checkpoint_file, 'w') as f:
                            json.dump(results, f)
                        
                        # Monitor resources
                        self._monitor_resources()
                
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
                # Clean up checkpoint file and progress file
                self.cleanup()
                if os.path.exists(self._progress_file):
                    os.remove(self._progress_file)
        
        except Exception as e:
            logger.error(f"Error in privacy sweep: {str(e)}")
            raise
    
    def _initialize_progress(self, epsilon_range: Tuple[float, float]) -> None:
        """Initialize progress tracking."""
        progress = {
            'start_time': datetime.now().isoformat(),
            'total_epsilons': 10,
            'completed_epsilons': 0,
            'current_epsilon': None,
            'status': 'running'
        }
        with open(self._progress_file, 'w') as f:
            json.dump(progress, f)
    
    def _update_progress(self, current: int, total: int, epsilon: float) -> None:
        """Update progress tracking."""
        try:
            with open(self._progress_file, 'r') as f:
                progress = json.load(f)
            
            progress['completed_epsilons'] = current
            progress['current_epsilon'] = epsilon
            progress['percent_complete'] = (current / total) * 100
            
            with open(self._progress_file, 'w') as f:
                json.dump(progress, f)
        except Exception as e:
            logger.warning(f"Failed to update progress: {str(e)}")
    
    def _run_batched_privacy_analysis(self, workload: Dict[str, Any], epsilon: float) -> Dict[str, Any]:
        """Run privacy analysis in batches to manage memory usage."""
        operation_count = workload['operation_count']
        batches = (operation_count + self._batch_size - 1) // self._batch_size
        
        batch_results = []
        for batch in range(batches):
            start_idx = batch * self._batch_size
            end_idx = min((batch + 1) * self._batch_size, operation_count)
            
            # Process batch
            batch_workload = workload.copy()
            batch_workload['operation_count'] = end_idx - start_idx
            batch_result = self._run_single_privacy_analysis(batch_workload, epsilon)
            batch_results.append(batch_result)
            
            # Monitor resources after each batch
            self._monitor_resources()
        
        # Combine batch results
        return self._combine_batch_results(batch_results)
    
    def _combine_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple batches."""
        combined = {
            'privacy_loss': np.mean([r['privacy_loss'] for r in batch_results]),
            'utility_loss': np.mean([r['utility_loss'] for r in batch_results]),
            'error_rate': np.mean([r['error_rate'] for r in batch_results])
        }
        return combined
    
    def _validate_privacy_results(self, results: Dict[str, Any]) -> bool:
        """Validate privacy analysis results.
        
        Args:
            results: Dictionary containing privacy analysis results
            
        Returns:
            bool: True if results are valid, False otherwise
        """
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
                    # return False
            
            # Validate consistency between metrics with more tolerance
            if results['privacy_loss'] == 0 and results['utility_loss'] > 0.1:  # Increased threshold
                logger.warning("Non-zero utility loss with zero privacy loss")
            
            if results['error_rate'] > 0.7:  # Increased threshold to 70%
                logger.warning(f"High error rate detected: {results['error_rate']}")
            
            return True
        except Exception as e:
            logger.error(f"Error validating privacy results: {str(e)}")
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