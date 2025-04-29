"""
Command line interface for the Endure project.
"""

import argparse
import logging
import sys
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
from .types import WorkloadCharacteristics, AnalysisResults, Metrics
import psutil

# Get logger instance
logger = logging.getLogger(__name__)

@dataclass
class CLIError(Exception):
    """Custom exception for CLI errors."""
    message: str
    exit_code: int = 1

def setup_logging(log_level: str = 'info') -> None:
    """Setup logging configuration."""
    try:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    except Exception as e:
        raise CLIError(f"Error setting up logging: {str(e)}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments with improved validation."""
    try:
        parser = argparse.ArgumentParser(
            description="Endure: Privacy-Performance Analysis Tool"
        )
        
        # Analysis type
        parser.add_argument(
            'analysis_type',
            choices=['privacy', 'sensitivity', 'performance', 'all'],
            help="Type of analysis to run"
        )
        
        # Configuration
        parser.add_argument(
            '-c', '--config',
            type=str,
            help="Path to configuration file"
        )
        
        # Input data
        parser.add_argument(
            '-i', '--input',
            type=str,
            required=True,
            help="Path to input data file"
        )
        
        # Output directory
        parser.add_argument(
            '-o', '--output',
            type=str,
            help="Output directory for results"
        )
        
        # Log level
        parser.add_argument(
            '-l', '--log-level',
            choices=['debug', 'info', 'warning', 'error', 'critical'],
            default='info',
            help="Logging level"
        )
        
        return parser.parse_args()
    except Exception as e:
        raise CLIError(f"Error parsing arguments: {str(e)}")

def load_config(config_file: Optional[str] = None) -> Any:
    """Load and validate configuration."""
    try:
        from .config import ConfigManager
        config = ConfigManager(config_file)
        
        # Validate configuration
        if not config.validate():
            raise CLIError("Invalid configuration")
            
        return config
    except ImportError:
        raise CLIError("Failed to import configuration module")
    except Exception as e:
        raise CLIError(f"Error loading configuration: {str(e)}")

def load_input_data(input_file: str) -> Dict[str, Any]:
    """Load and validate input data with improved error handling."""
    try:
        if not os.path.exists(input_file):
            raise CLIError(f"Input file not found: {input_file}")
            
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, dict):
            raise CLIError("Input data must be a JSON object")
            
        # Initialize workload characteristics if missing
        if 'workload_characteristics' not in data:
            logger.info("No workload characteristics found, using defaults")
            data['workload_characteristics'] = {}
            
        # Convert to WorkloadCharacteristics object
        try:
            data['workload_characteristics'] = WorkloadCharacteristics.from_dict(
                data['workload_characteristics']
            )
        except Exception as e:
            raise CLIError(f"Invalid workload characteristics: {str(e)}")
            
        return data
    except json.JSONDecodeError as e:
        raise CLIError(f"Invalid JSON in input file: {str(e)}")
    except Exception as e:
        raise CLIError(f"Error loading input data: {str(e)}")

def validate_input_data(data: Dict[str, Any]) -> bool:
    """Validate input data structure with improved validation."""
    try:
        # Initialize workload characteristics with defaults
        wc = data.get('workload_characteristics', {})
        if not isinstance(wc, dict):
            logger.warning("workload_characteristics must be a dictionary, using defaults")
            wc = {}
            
        # Define default values with validation ranges
        defaults = {
            'read_ratio': {'value': 0.7, 'min': 0.0, 'max': 1.0},
            'write_ratio': {'value': 0.3, 'min': 0.0, 'max': 1.0},
            'key_size': {'value': 16, 'min': 1, 'max': 1024},
            'value_size': {'value': 100, 'min': 1, 'max': 1048576},  # 1MB max
            'operation_count': {'value': 100000, 'min': 1000, 'max': 100000000},
            'hot_key_ratio': {'value': 0.2, 'min': 0.0, 'max': 1.0},
            'hot_key_count': {'value': 100, 'min': 1, 'max': 10000}
        }
        
        # Validate and normalize ratio values
        for ratio in ['read_ratio', 'write_ratio', 'hot_key_ratio']:
            if ratio in wc:
                try:
                    value = float(wc[ratio])
                    min_val = defaults[ratio]['min']
                    max_val = defaults[ratio]['max']
                    if not min_val <= value <= max_val:
                        logger.warning(
                            f"Invalid {ratio} value: {value} (must be between {min_val} and {max_val}). "
                            f"Using default: {defaults[ratio]['value']}"
                        )
                        wc[ratio] = defaults[ratio]['value']
                    else:
                        wc[ratio] = value
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert {ratio} to float. Using default: {defaults[ratio]['value']}"
                    )
                    wc[ratio] = defaults[ratio]['value']
            else:
                logger.info(f"Using default value for {ratio}: {defaults[ratio]['value']}")
                wc[ratio] = defaults[ratio]['value']
        
        # Validate and normalize other values
        for field in ['key_size', 'value_size', 'operation_count', 'hot_key_count']:
            if field in wc:
                try:
                    value = int(wc[field])
                    min_val = defaults[field]['min']
                    max_val = defaults[field]['max']
                    if not min_val <= value <= max_val:
                        logger.warning(
                            f"Invalid {field} value: {value} (must be between {min_val} and {max_val}). "
                            f"Using default: {defaults[field]['value']}"
                        )
                        wc[field] = defaults[field]['value']
                    else:
                        wc[field] = value
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not convert {field} to integer. Using default: {defaults[field]['value']}"
                    )
                    wc[field] = defaults[field]['value']
            else:
                logger.info(f"Using default value for {field}: {defaults[field]['value']}")
                wc[field] = defaults[field]['value']
        
        # Update the input data with normalized values
        data['workload_characteristics'] = wc
        
        # Validate ratio sum with configurable tolerance
        ratio_sum = wc['read_ratio'] + wc['write_ratio']
        tolerance = 0.01  # 1% tolerance
        if abs(ratio_sum - 1.0) > tolerance:
            logger.warning(
                f"read_ratio + write_ratio = {ratio_sum} (should be 1.0 ± {tolerance}). Normalizing..."
            )
            total = wc['read_ratio'] + wc['write_ratio']
            if total == 0:
                logger.warning("Both ratios are 0, using default values")
                wc['read_ratio'] = defaults['read_ratio']['value']
                wc['write_ratio'] = defaults['write_ratio']['value']
            else:
                wc['read_ratio'] = wc['read_ratio'] / total
                wc['write_ratio'] = wc['write_ratio'] / total
        
        # Validate hot key count against operation count
        if wc['hot_key_count'] > wc['operation_count']:
            logger.warning(
                f"hot_key_count ({wc['hot_key_count']}) cannot be greater than operation_count "
                f"({wc['operation_count']}). Adjusting hot_key_count..."
            )
            wc['hot_key_count'] = min(wc['hot_key_count'], wc['operation_count'])
            
        # Validate hot key ratio against hot key count
        expected_hot_keys = int(wc['operation_count'] * wc['hot_key_ratio'])
        if wc['hot_key_count'] > expected_hot_keys:
            logger.warning(
                f"hot_key_count ({wc['hot_key_count']}) is larger than expected based on hot_key_ratio "
                f"({expected_hot_keys}). Adjusting hot_key_count..."
            )
            wc['hot_key_count'] = min(wc['hot_key_count'], expected_hot_keys)
            
        # Validate key_size and value_size relationship
        if wc['key_size'] > wc['value_size']:
            logger.warning(
                f"key_size ({wc['key_size']}) is larger than value_size ({wc['value_size']}). "
                "This may impact performance."
            )
            
        # Estimate memory usage
        estimated_memory = (
            wc['operation_count'] * (wc['key_size'] + wc['value_size']) +
            wc['hot_key_count'] * (wc['key_size'] + wc['value_size'])
        ) / (1024 * 1024)  # Convert to MB
        
        # Get system memory and set dynamic threshold
        memory = psutil.virtual_memory()
        total_memory_mb = memory.total / (1024 * 1024)
        memory_threshold_mb = min(4000, total_memory_mb * 0.2)  # 4GB or 20% of total, whichever is smaller
        
        if estimated_memory > memory_threshold_mb:
            logger.warning(
                f"Estimated memory usage ({estimated_memory:.2f}MB) exceeds threshold "
                f"({memory_threshold_mb:.2f}MB). This may impact system performance."
            )
        
        return True
    except Exception as e:
        logger.error(f"Error validating input data: {str(e)}")
        return False

def run_analysis(
    analysis_type: str,
    input_data: Dict[str, Any],
    config: Any,
    output_dir: Optional[str] = None
) -> None:
    """Run specified analysis type with improved error handling."""
    try:
        # Validate analysis type
        valid_types = ['privacy', 'sensitivity', 'performance', 'all']
        if analysis_type not in valid_types:
            raise CLIError(f"Invalid analysis type: {analysis_type}. Must be one of {valid_types}")
        
        # Setup output directory
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                
                # Verify write permissions
                test_file = os.path.join(output_dir, '.test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                
                # Check available disk space
                stat = os.statvfs(output_dir)
                free_space = stat.f_bavail * stat.f_frsize / (1024 * 1024)  # Convert to MB
                if free_space < 1000:  # Less than 1GB
                    logger.warning(f"Low disk space in output directory: {free_space:.2f}MB available")
                
            except Exception as e:
                raise CLIError(f"Error creating or accessing output directory: {str(e)}")
            config.analysis.results_dir = output_dir
        
        # Import analysis modules
        try:
            from .analysis import PrivacyAnalysis, SensitivityAnalysis, PerformanceAnalysis
            from .workload_generator import WorkloadGenerator, WorkloadCharacteristics
        except ImportError as e:
            raise CLIError(f"Failed to import required modules: {str(e)}")
        
        # Create workload characteristics
        try:
            wc = input_data['workload_characteristics']
            characteristics = WorkloadCharacteristics(
                read_ratio=wc['read_ratio'],
                write_ratio=wc['write_ratio'],
                key_size=wc['key_size'],
                value_size=wc['value_size'],
                operation_count=wc['operation_count'],
                hot_key_ratio=wc['hot_key_ratio'],
                hot_key_count=wc['hot_key_count']
            )
            
            # Validate characteristics
            if not characteristics.validate():
                logger.warning("Workload characteristics validation failed, but continuing with provided values")
                
        except Exception as e:
            raise CLIError(f"Error creating workload characteristics: {str(e)}")
        
        # Run analysis based on type
        try:
            if analysis_type in ['privacy', 'all']:
                logger.info("Running privacy analysis...")
                analysis = PrivacyAnalysis(results_dir=config.analysis.results_dir)
                
                # Validate privacy configuration
                if not hasattr(config, 'privacy'):
                    raise CLIError("Missing privacy configuration")
                    
                # Validate epsilon range
                if not hasattr(config.privacy, 'epsilon_range'):
                    raise CLIError("Missing epsilon_range in privacy configuration")
                
                try:
                    epsilons = [float(e) for e in config.privacy.epsilon_range]
                    if not all(e > 0 for e in epsilons):
                        raise ValueError("All epsilon values must be positive")
                    if len(epsilons) < 2:
                        raise ValueError("At least two epsilon values are required")
                except (ValueError, TypeError) as e:
                    raise CLIError(f"Invalid epsilon values: {str(e)}")
                
                # Validate num_trials
                if not hasattr(config.privacy, 'num_trials'):
                    raise CLIError("Missing num_trials in privacy configuration")
                try:
                    num_trials = int(config.privacy.num_trials)
                    if num_trials < 1:
                        raise ValueError("num_trials must be positive")
                    if num_trials > 1000:
                        logger.warning(f"Large number of trials ({num_trials}) may impact performance")
                except (ValueError, TypeError) as e:
                    raise CLIError(f"Invalid num_trials: {str(e)}")
                
                results = analysis.run_privacy_sweep(
                    characteristics={
                        'read_ratio': characteristics.read_ratio,
                        'write_ratio': characteristics.write_ratio,
                        'key_size': characteristics.key_size,
                        'value_size': characteristics.value_size,
                        'operation_count': characteristics.operation_count,
                        'hot_key_ratio': characteristics.hot_key_ratio,
                        'hot_key_count': characteristics.hot_key_count
                    },
                    epsilons=epsilons,
                    num_trials=num_trials
                )
                analysis.plot_results(results)
            
            if analysis_type in ['sensitivity', 'all']:
                logger.info("Running sensitivity analysis...")
                analysis = SensitivityAnalysis(results_dir=config.analysis.results_dir)
                analysis.run_analysis(input_data)
            
            if analysis_type in ['performance', 'all']:
                logger.info("Running performance analysis...")
                analysis = PerformanceAnalysis(results_dir=config.analysis.results_dir)
                analysis.run_analysis(input_data)
                
        except Exception as e:
            raise CLIError(f"Error during analysis: {str(e)}")
            
    except CLIError:
        raise
    except Exception as e:
        raise CLIError(f"Unexpected error in analysis: {str(e)}")

def main() -> None:
    """Main entry point with improved error handling."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Setup logging
        setup_logging(args.log_level)
        
        # Load configuration
        config = load_config(args.config)
        
        # Load input data
        input_data = load_input_data(args.input)
        
        # Run analysis
        run_analysis(args.analysis_type, input_data, config, args.output)
        
        logger.info("Analysis completed successfully")
        
    except CLIError as e:
        logger.error(e.message)
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 