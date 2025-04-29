import logging
from endure.types import WorkloadCharacteristics, MathUtils
from endure.workload_generator import WorkloadGenerator
from endure.visualization import EnhancedVisualization
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_workload_generation():
    """Test workload generation and validation."""
    try:
        # Test different workload characteristics
        test_cases = [
            {
                'name': 'Read-heavy workload',
                'characteristics': WorkloadCharacteristics(
                    read_ratio=0.8,
                    write_ratio=0.2,
                    key_size=16,
                    value_size=100,
                    operation_count=1000,
                    hot_key_ratio=0.2,
                    hot_key_count=10
                )
            },
            {
                'name': 'Write-heavy workload',
                'characteristics': WorkloadCharacteristics(
                    read_ratio=0.3,
                    write_ratio=0.7,
                    key_size=32,
                    value_size=200,
                    operation_count=2000,
                    hot_key_ratio=0.3,
                    hot_key_count=20
                )
            },
            {
                'name': 'Balanced workload',
                'characteristics': WorkloadCharacteristics(
                    read_ratio=0.5,
                    write_ratio=0.5,
                    key_size=64,
                    value_size=500,
                    operation_count=5000,
                    hot_key_ratio=0.4,
                    hot_key_count=50
                )
            }
        ]
        
        results = {}
        for test_case in test_cases:
            logging.info(f"\nTesting {test_case['name']}:")
            characteristics = test_case['characteristics']
            
            # Validate characteristics
            if not characteristics.validate():
                logging.error(f"Workload characteristics validation failed for {test_case['name']}")
                continue
            
            # Generate workload
            generator = WorkloadGenerator(epsilon=0.5)  # Set epsilon for privacy
            workload, private_workload = generator.generate_workload(characteristics)
            
            # Calculate metrics
            metrics = generator.calculate_workload_metrics(workload)
            private_metrics = generator.calculate_workload_metrics(private_workload)
            
            logging.info(f"Original workload metrics:")
            logging.info(f"- Total operations: {metrics['total_operations']}")
            logging.info(f"- Read ratio: {metrics['read_ratio']:.3f}")
            logging.info(f"- Write ratio: {metrics['write_ratio']:.3f}")
            logging.info(f"- Hot key ratio: {metrics['hot_key_ratio']:.3f}")
            
            logging.info(f"\nPrivate workload metrics:")
            logging.info(f"- Total operations: {private_metrics['total_operations']}")
            logging.info(f"- Read ratio: {private_metrics['read_ratio']:.3f}")
            logging.info(f"- Write ratio: {private_metrics['write_ratio']:.3f}")
            logging.info(f"- Hot key ratio: {private_metrics['hot_key_ratio']:.3f}")
            
            # Store results for visualization
            results[test_case['name']] = {
                'workload': workload,
                'private_workload': private_workload,
                'metrics': metrics,
                'private_metrics': private_metrics
            }
        
        return results
        
    except Exception as e:
        logging.error(f"Error in workload generation test: {str(e)}")
        return None

def test_visualization(results):
    """Test visualization with comprehensive data."""
    try:
        # Create visualization instance
        viz = EnhancedVisualization("test_results")
        
        # Create privacy-performance tradeoff data
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
        perf_results = {}
        
        for epsilon in epsilons:
            trials = []
            for _ in range(3):  # Multiple trials per epsilon
                performance_diffs = {
                    'throughput': {'difference_percent': np.random.normal(10 * epsilon, 2)},
                    'latency': {'difference_percent': np.random.normal(5 * epsilon, 1)},
                    'memory': {'difference_percent': np.random.normal(3 * epsilon, 0.5)}
                }
                
                config_diffs = {
                    'buffer_size': {'difference_percent': np.random.normal(2 * epsilon, 0.5)},
                    'cache_size': {'difference_percent': np.random.normal(3 * epsilon, 0.5)}
                }
                
                trials.append({
                    'privacy_metrics': {
                        'performance_differences': performance_diffs,
                        'configuration_differences': config_diffs
                    },
                    'workload_characteristics': {
                        'read_ratio': 0.7,
                        'write_ratio': 0.3,
                        'key_size': 16,
                        'value_size': 100,
                        'operation_count': 1000,
                        'hot_key_ratio': 0.2,
                        'hot_key_count': 10
                    }
                })
            perf_results[epsilon] = trials
        
        # Test all visualization methods
        logging.info("\nGenerating visualizations:")
        
        logging.info("1. Privacy-Performance Tradeoff")
        viz.plot_privacy_performance_tradeoff(perf_results, filename="privacy_performance_tradeoff.png")
        
        logging.info("2. Workload Sensitivity")
        viz.plot_workload_sensitivity({
            'workload_characteristics': {
                'read_ratio': 0.7,
                'write_ratio': 0.3,
                'key_size': 16,
                'value_size': 100,
                'operation_count': 1000,
                'hot_key_ratio': 0.2,
                'hot_key_count': 10
            }
        }, filename="workload_sensitivity.png")
        
        logging.info("3. Configuration Differences")
        viz.plot_configuration_differences(perf_results, filename="configuration_differences.png")
        
        logging.info("4. Correlation Analysis")
        viz.plot_correlation_analysis({
            'metrics': {
                'throughput': 1000,
                'latency': 50,
                'memory': 500
            },
            'workload_characteristics': {
                'read_ratio': 0.7,
                'write_ratio': 0.3,
                'key_size': 16,
                'value_size': 100,
                'operation_count': 1000,
                'hot_key_ratio': 0.2,
                'hot_key_count': 10
            }
        }, filename="correlation_analysis.png")
        
        logging.info("Visualization tests completed successfully")
        
    except Exception as e:
        logging.error(f"Error in visualization test: {str(e)}")

if __name__ == "__main__":
    logging.info("Starting comprehensive workload and visualization tests")
    results = test_workload_generation()
    if results:
        test_visualization(results)
    logging.info("Tests completed") 