"""
Direct analysis runner for the Endure project.
"""

import json
import logging
import os
from endure.privacy_analysis import PrivacyAnalysis
from endure.workload_generator import WorkloadCharacteristics

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_input_data(input_file: str = 'input.json'):
    """Load and validate input data."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Validate required fields
        if 'workload_characteristics' not in data:
            raise ValueError("Missing workload_characteristics in input data")
        
        wc = data['workload_characteristics']
        required_fields = [
            'read_ratio', 'write_ratio', 'key_size', 'value_size',
            'operation_count', 'hot_key_ratio', 'hot_key_count'
        ]
        
        for field in required_fields:
            if field not in wc:
                raise ValueError(f"Missing required field: {field}")
        
        return data
    except Exception as e:
        logging.error(f"Error loading input data: {str(e)}")
        raise

def main():
    """Main entry point."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Load input data
        logger.info("Loading input data...")
        input_data = load_input_data()
        
        # Create workload characteristics
        wc = input_data['workload_characteristics']
        characteristics = WorkloadCharacteristics(
            read_ratio=float(wc['read_ratio']),
            write_ratio=float(wc['write_ratio']),
            key_size=int(wc['key_size']),
            value_size=int(wc['value_size']),
            operation_count=int(wc['operation_count']),
            hot_key_ratio=float(wc['hot_key_ratio']),
            hot_key_count=int(wc['hot_key_count'])
        )
        
        # Run privacy analysis
        logger.info("Running privacy analysis...")
        analysis = PrivacyAnalysis()
        
        # Run privacy sweep with different epsilon values
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
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
            num_trials=5
        )
        
        # Plot results
        logger.info("Plotting results...")
        analysis.plot_results(results)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == '__main__':
    main() 