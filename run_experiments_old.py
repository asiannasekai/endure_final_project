"""
Script to run experiments using the old system.
"""

import os
import json
import logging
import sys
from pathlib import Path
from endure.analysis import PrivacyAnalysis, SensitivityAnalysis, PerformanceAnalysis

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

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def setup_directories():
    """Setup required directories."""
    dirs = ['privacy_results', 'sensitivity_results', 'performance_results']
    for dir_name in dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directory {dir_name}: {str(e)}")
            sys.exit(1)

def run_experiments_old():
    """Run experiments using the old system."""
    logger = logging.getLogger(__name__)
    
    try:
        # Setup directories
        setup_directories()
        
        # Check input file exists
        if not os.path.exists('input_data.json'):
            logger.error("Input file not found: input_data.json")
            sys.exit(1)
            
        # Load and validate input data
        try:
            with open('input_data.json', 'r') as f:
                input_data = json.load(f)
            if not validate_input_data(input_data):
                logger.error("Invalid input data structure")
                sys.exit(1)
        except json.JSONDecodeError:
            logger.error("Invalid JSON in input file")
            sys.exit(1)
        
        # Run privacy analysis
        logger.info("Running privacy analysis...")
        privacy_analysis = PrivacyAnalysis(results_dir="privacy_results")
        privacy_analysis.run_analysis(input_data)
        
        # Run sensitivity analysis
        logger.info("Running sensitivity analysis...")
        sensitivity_analysis = SensitivityAnalysis(results_dir="sensitivity_results")
        sensitivity_analysis.run_analysis(input_data)
        
        # Run performance analysis
        logger.info("Running performance analysis...")
        performance_analysis = PerformanceAnalysis(results_dir="performance_results")
        performance_analysis.run_analysis(input_data)
        
        logger.info("All experiments completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running experiments: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    setup_logging()
    run_experiments_old() 