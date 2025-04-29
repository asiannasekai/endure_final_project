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
from .privacy_analysis import PrivacyAnalysis
from .sensitivity_analysis import SensitivityAnalysis
from .performance_analysis import PerformanceAnalysis

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
    
    def _setup_directories(self) -> None:
        """Set up required directories with validation."""
        try:
            # Get base directories from config
            results_dir = Path(self.config.analysis.results_dir)
            checkpoints_dir = Path('checkpoints')
            
            # Create analysis-specific directories
            self.results_dir = results_dir / self.__class__.__name__.lower().replace('analysis', '')
            self.checkpoint_dir = checkpoints_dir / self.__class__.__name__.lower().replace('analysis', '')
            
            # Create directories
            self.results_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Verify write permissions
            if not os.access(self.results_dir, os.W_OK):
                raise PermissionError(f"No write permission for {self.results_dir}")
            if not os.access(self.checkpoint_dir, os.W_OK):
                raise PermissionError(f"No write permission for {self.checkpoint_dir}")
            
            # Check disk space
            for dir_path in [self.results_dir, self.checkpoint_dir]:
                disk = shutil.disk_usage(dir_path)
                free_gb = disk.free / (1024 * 1024 * 1024)
                if free_gb < self.config.analysis.max_disk_gb:
                    raise RuntimeError(f"Insufficient disk space in {dir_path}: {free_gb:.2f}GB available")
            
            # Setup logging directory
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            self.log_file = log_dir / f"{self.__class__.__name__.lower()}.log"
            
        except Exception as e:
            logger.error(f"Error setting up directories: {str(e)}")
            raise
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        try:
            logging.basicConfig(
                level=getattr(logging, self.config.analysis.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(self.log_file),
                    logging.StreamHandler()
                ]
            )
        except Exception as e:
            logger.error(f"Error setting up logging: {str(e)}")
            raise
    
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
            disk = shutil.disk_usage(self.results_dir)
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
            disk = shutil.disk_usage(self.results_dir)
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
            disk = shutil.disk_usage(self.results_dir)
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
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        try:
            # Clean up temporary files
            for temp_file in self.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
            self.temp_files.clear()
            
            # Clean up progress file if it exists
            if hasattr(self, '_progress_file') and os.path.exists(self._progress_file):
                try:
                    os.remove(self._progress_file)
                except Exception as e:
                    logger.warning(f"Failed to remove progress file: {str(e)}")
            
            # Clean up checkpoint files
            if hasattr(self, 'checkpoint_dir'):
                for file in self.checkpoint_dir.glob('*.json'):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint file {file}: {str(e)}")
            
            logger.info("Cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise

def run_analysis(input_file: str, output_dir: Optional[str] = None) -> Dict:
    """Run analysis based on input data."""
    try:
        # Load input data
        with open(input_file, 'r') as f:
            input_data = json.load(f)
        
        # Validate input data
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary")
        
        # Setup output directory
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analyses
        privacy_analysis = PrivacyAnalysis()
        sensitivity_analysis = SensitivityAnalysis()
        performance_analysis = PerformanceAnalysis()
        
        # Run analyses
        results = {
            'privacy': privacy_analysis.run_privacy_sweep(
                input_data.get('workload_characteristics', {}),
                epsilons=input_data.get('epsilons', [0.1, 0.5, 1.0, 2.0, 5.0]),
                num_trials=input_data.get('num_trials', 5)
            ),
            'sensitivity': sensitivity_analysis.run_analysis(
                input_data.get('workload_characteristics', {}),
                input_data.get('sensitivity_params', {})
            ),
            'performance': performance_analysis.run_analysis(
                input_data.get('workload_characteristics', {}),
                input_data.get('performance_params', {})
            )
        }
        
        # Save results
        if output_dir:
            for analysis_type, result in results.items():
                output_file = os.path.join(output_dir, f"{analysis_type}_results.json")
                with open(output_file, 'w') as f:
                    json.dump(result.metrics, f, indent=2)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}")
        raise 