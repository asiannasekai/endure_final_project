import os
import logging
from endure.analysis import PrivacyAnalysis
from endure.config import ConfigManager
from endure.workload_generator import WorkloadCharacteristics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create test output directory
    output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config
    config = ConfigManager()
    config.analysis.results_dir = output_dir
    
    # Create workload characteristics
    workload = {
        'read_ratio': 0.7,
        'write_ratio': 0.3,
        'key_size': 16,
        'value_size': 100,
        'operation_count': 100000,
        'hot_key_ratio': 0.2,
        'hot_key_count': 100
    }
    
    # Create analysis instance
    analysis = PrivacyAnalysis(config=config)
    
    try:
        # Run privacy sweep
        logger.info("Starting privacy analysis...")
        results = analysis.run_privacy_sweep(
            workload=workload,
            epsilon_range=(0.1, 1.0)
        )
        
        # Print results
        logger.info("Analysis completed successfully!")
        logger.info(f"Results: {results}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 