"""
Run experiments for the Endure project.
"""

import logging
import os
import json
import sys
from pathlib import Path
from endure.config import ConfigManager
from endure.cli import run_analysis, validate_input_data

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def validate_config(config: ConfigManager) -> bool:
    """Validate configuration settings."""
    try:
        if not config.analysis.validate():
            logging.error("Invalid analysis configuration")
            return False
        if not config.visualization.validate():
            logging.error("Invalid visualization configuration")
            return False
        if not config.privacy.validate():
            logging.error("Invalid privacy configuration")
            return False
        if not config.performance.validate():
            logging.error("Invalid performance configuration")
            return False
        return True
    except Exception as e:
        logging.error(f"Error validating configuration: {str(e)}")
        return False

def setup_directories() -> bool:
    """Setup required directories with error handling."""
    try:
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Create results directory and subdirectories
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/privacy_results', exist_ok=True)
        os.makedirs('results/sensitivity_results', exist_ok=True)
        os.makedirs('results/performance_results', exist_ok=True)
        return True
    except PermissionError:
        logging.error("Permission denied when creating directories")
        return False
    except OSError as e:
        logging.error(f"Error creating directories: {str(e)}")
        return False

def load_input_data() -> dict:
    """Load and validate input data with error handling."""
    try:
        input_file = 'data/input_data.json'
        if not os.path.exists(input_file):
            logging.error(f"Input file not found: {input_file}")
            return None
            
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        if not validate_input_data(data):
            logging.error("Invalid input data structure")
            return None
            
        return data
    except json.JSONDecodeError:
        logging.error("Invalid JSON in input file")
        return None
    except Exception as e:
        logging.error(f"Error loading input data: {str(e)}")
        return None

def main():
    """Main entry point for running experiments."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Create necessary directories
        logger.info("Setting up directories...")
        if not setup_directories():
            sys.exit(1)
        
        # Load and validate configuration
        logger.info("Loading configuration...")
        config_file = 'config.json' if os.path.exists('config.json') else None
        config = ConfigManager(config_file)
        if not validate_config(config):
            sys.exit(1)
        
        # Load and validate input data
        logger.info("Loading input data...")
        input_data = load_input_data()
        if input_data is None:
            sys.exit(1)
        
        # Run analyses
        try:
            # Run privacy analysis
            logger.info("Running privacy analysis...")
            run_analysis('privacy', input_data, config, 'results/privacy_results')
            
            # Run sensitivity analysis
            logger.info("Running sensitivity analysis...")
            run_analysis('sensitivity', input_data, config, 'results/sensitivity_results')
            
            # Run performance analysis
            logger.info("Running performance analysis...")
            run_analysis('performance', input_data, config, 'results/performance_results')
            
            logger.info("All experiments completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Analysis interrupted by user")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 