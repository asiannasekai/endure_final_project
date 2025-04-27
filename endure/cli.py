"""
Command line interface for the Endure project.
"""

import argparse
import logging
import sys
import json
import os
from typing import Optional
from pathlib import Path
from .config import ConfigManager
from .analysis import PrivacyAnalysis, SensitivityAnalysis, PerformanceAnalysis
from .visualization import EnhancedVisualization

def validate_input_data(data: dict) -> bool:
    """Validate input data structure."""
    required_fields = {
        'metrics': ['throughput', 'latency', 'space_amplification'],
        'configurations': ['original', 'private'],
        'workload_characteristics': ['read_ratio', 'write_ratio']
    }
    
    for section, fields in required_fields.items():
        if section not in data:
            logging.error(f"Missing required section: {section}")
            return False
        for field in fields:
            if field not in data[section]:
                logging.error(f"Missing required field: {field} in section {section}")
                return False
    return True

def setup_logging(log_level: str) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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

def load_input_data(input_file: str) -> dict:
    """Load input data from file."""
    try:
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            sys.exit(1)
            
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        if not validate_input_data(data):
            logging.error("Invalid input data structure")
            sys.exit(1)
            
        return data
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in input file: {input_file}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading input data: {str(e)}")
        sys.exit(1)

def run_analysis(
    analysis_type: str,
    input_data: dict,
    config: ConfigManager,
    output_dir: Optional[str] = None
) -> None:
    """Run specified analysis type."""
    try:
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                logging.error(f"Error creating output directory: {str(e)}")
                sys.exit(1)
            config.analysis.results_dir = output_dir
        
        # Convert epsilon values to floats if they exist in the config
        if hasattr(config, 'privacy') and hasattr(config.privacy, 'epsilon_range'):
            config.privacy.epsilon_range = [float(e) for e in config.privacy.epsilon_range]
        
        if analysis_type in ['privacy', 'all']:
            logging.info("Running privacy analysis...")
            analysis = PrivacyAnalysis(results_dir=config.analysis.results_dir)
            analysis.run_analysis(input_data)
        
        if analysis_type in ['sensitivity', 'all']:
            logging.info("Running sensitivity analysis...")
            analysis = SensitivityAnalysis(results_dir=config.analysis.results_dir)
            analysis.run_analysis(input_data)
        
        if analysis_type in ['performance', 'all']:
            logging.info("Running performance analysis...")
            analysis = PerformanceAnalysis(results_dir=config.analysis.results_dir)
            analysis.run_analysis(input_data)
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error in {analysis_type} analysis: {str(e)}")
        raise

def main() -> None:
    """Main entry point for the CLI."""
    try:
        args = parse_args()
        setup_logging(args.log_level)
        
        # Load configuration
        if args.config and not os.path.exists(args.config):
            logging.warning(f"Config file not found: {args.config}, using defaults")
            config = ConfigManager()
        else:
            config = ConfigManager(args.config)
        
        # Load input data
        input_data = load_input_data(args.input)
        
        # Run analysis
        run_analysis(args.analysis_type, input_data, config, args.output)
        logging.info("Analysis completed successfully")
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error running analysis: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 