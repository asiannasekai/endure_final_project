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
from endure.workload_generator import WorkloadGenerator, WorkloadCharacteristics
from endure.endure_integration import EndureIntegration

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

def generate_experimental_data(epsilon: float = 1.0) -> dict:
    """Generate experimental data using workload generator."""
    # Define workload characteristics
    characteristics = WorkloadCharacteristics(
        read_ratio=0.7,
        write_ratio=0.3,
        key_size=16,
        value_size=100,
        operation_count=100000,
        hot_key_ratio=0.2,
        hot_key_count=100
    )
    
    # Initialize workload generator
    generator = WorkloadGenerator(epsilon=epsilon)
    
    # Generate workloads
    original_workload, private_workload = generator.generate_workload(characteristics)
    
    # Calculate metrics
    original_metrics = generator.calculate_workload_metrics(original_workload)
    private_metrics = generator.calculate_workload_metrics(private_workload)
    
    # Initialize Endure integration
    integration = EndureIntegration(epsilon)
    
    # Run experiments
    original_results = integration.run_endure_tuning(original_workload)
    private_results = integration.run_endure_tuning(private_workload)
    
    # Prepare input data
    input_data = {
        "metrics": {
            "throughput": original_results["performance_metrics"]["throughput"],
            "latency": original_results["performance_metrics"]["latency"],
            "space_amplification": original_results["performance_metrics"]["space_amplification"]
        },
        "configurations": {
            "original": original_results["performance_metrics"],
            "private": private_results["performance_metrics"]
        },
        "workload_characteristics": {
            "read_ratio": characteristics.read_ratio,
            "write_ratio": characteristics.write_ratio
        }
    }
    
    return input_data

def run_experiments():
    """Run all experiments."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
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
        
        # Generate experimental data
        logger.info("Generating experimental data...")
        input_data = generate_experimental_data()
        
        # Save input data
        with open('input_data.json', 'w') as f:
            json.dump(input_data, f, indent=2)
        logger.info("Experimental data saved to input_data.json")
        
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