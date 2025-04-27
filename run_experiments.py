"""
Script to run experiments using the new system.
"""

import os
import json
import logging
import sys
from pathlib import Path
from endure.cli import main as cli_main
from endure.config import ConfigManager

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

def setup_directories():
    """Setup required directories."""
    dirs = ['results', 'privacy_results', 'sensitivity_results', 'performance_results']
    for dir_name in dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            logging.error(f"Error creating directory {dir_name}: {str(e)}")
            sys.exit(1)

def run_experiments():
    """Run all experiments."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
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
        
        # Setup directories
        setup_directories()
        logger.info("Directories setup complete")
        
        # Check config file exists
        if not os.path.exists('config.json'):
            logger.warning("Config file not found, using defaults")
            config_manager = ConfigManager()
        else:
            config_manager = ConfigManager('config.json')
        logger.info("Configuration loaded")
        
        # Run privacy analysis
        logger.info("Running privacy analysis...")
        cli_main(['privacy', '-i', 'input_data.json', '-c', 'config.json', '-o', 'privacy_results'])
        
        # Run sensitivity analysis
        logger.info("Running sensitivity analysis...")
        cli_main(['sensitivity', '-i', 'input_data.json', '-c', 'config.json', '-o', 'sensitivity_results'])
        
        # Run performance analysis
        logger.info("Running performance analysis...")
        cli_main(['performance', '-i', 'input_data.json', '-c', 'config.json', '-o', 'performance_results'])
        
        logger.info("All experiments completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running experiments: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    run_experiments() 