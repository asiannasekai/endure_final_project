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
from .config import ConfigManager
from .visualization import EnhancedVisualization
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

def load_config(config_file: Optional[str] = None) -> ConfigManager:
    """Load and validate configuration."""
    try:
        config = ConfigManager(config_file)
        
        # Validate configuration
        if not config.validate():
            raise CLIError("Invalid configuration")
            
        return config
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
            
        # Validate and normalize workload characteristics
        wc = data['workload_characteristics']
        if not isinstance(wc, dict):
            raise CLIError("workload_characteristics must be a dictionary")
            
        # Define default values with validation ranges
        defaults = {
            'read_ratio': {'value': 0.7, 'min': 0.0, 'max': 1.0},
            'write_ratio': {'value': 0.3, 'min': 0.0, 'max': 1.0},
            'key_size': {'value': 16, 'min': 1, 'max': 1024},
            'value_size': {'value': 100, 'min': 1, 'max': 1048576},
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
        
        # Validate ratio sum
        ratio_sum = wc['read_ratio'] + wc['write_ratio']
        if abs(ratio_sum - 1.0) > 0.01:  # 1% tolerance
            logger.warning(
                f"read_ratio + write_ratio = {ratio_sum} (should be 1.0). Normalizing..."
            )
            total = wc['read_ratio'] + wc['write_ratio']
            wc['read_ratio'] = wc['read_ratio'] / total
            wc['write_ratio'] = wc['write_ratio'] / total
        
        # Update the input data with normalized values
        data['workload_characteristics'] = wc
        
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

def run_analysis(input_file: str, output_dir: str = None) -> None:
    """Run the analysis with improved error handling and validation."""
    try:
        # Load and validate input data
        input_data = load_input_data(input_file)
        if not isinstance(input_data, dict):
            raise CLIError("Input data must be a dictionary")
            
        # Extract and validate workload characteristics
        workload = input_data.get('workload_characteristics', {})
        if not isinstance(workload, dict):
            raise CLIError("Workload characteristics must be a dictionary")
            
        # Validate required fields
        required_fields = [
            'read_ratio', 'write_ratio', 'key_size', 'value_size',
            'operation_count', 'hot_key_ratio', 'hot_key_count'
        ]
        missing_fields = [field for field in required_fields if field not in workload]
        if missing_fields:
            raise CLIError(f"Missing required workload fields: {', '.join(missing_fields)}")
            
        # Create output directory if not specified
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_file), 'results')
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize analysis
        analysis = PrivacyAnalysis()
        
        # Run privacy sweep
        try:
            results = analysis.run_privacy_sweep(workload)
            if not results:
                raise CLIError("No results returned from privacy sweep")
                
            # Save results
            output_file = os.path.join(output_dir, 'privacy_analysis.json')
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Analysis completed successfully. Results saved to {output_file}")
            
        except Exception as e:
            raise CLIError(f"Error during privacy sweep: {str(e)}")
            
    except Exception as e:
        raise CLIError(f"Analysis failed: {str(e)}")

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
        run_analysis(args.input, args.output)
        
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