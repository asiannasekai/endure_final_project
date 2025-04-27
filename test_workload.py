from endure.workload_generator import WorkloadGenerator, WorkloadCharacteristics
from endure.endure_experiment import EndureExperiment

def test_workload_generation():
    # Test workload characteristics
    characteristics = WorkloadCharacteristics(
        read_ratio=0.7,
        write_ratio=0.3,
        key_size=16,
        value_size=100,
        operation_count=1000,  # Small test size
        hot_key_ratio=0.2,
        hot_key_count=10
    )
    
    print("Testing workload generation...")
    print("\nInput Characteristics:")
    print(f"Read ratio: {characteristics.read_ratio}")
    print(f"Write ratio: {characteristics.write_ratio}")
    print(f"Hot key ratio: {characteristics.hot_key_ratio}")
    print(f"Operation count: {characteristics.operation_count}")
    print(f"Hot key count: {characteristics.hot_key_count}")
    
    # Initialize workload generator
    generator = WorkloadGenerator(epsilon=1.0)
    
    # Generate workloads
    original_workload, private_workload = generator.generate_workload(characteristics)
    
    # Calculate metrics
    original_metrics = generator.calculate_workload_metrics(original_workload)
    private_metrics = generator.calculate_workload_metrics(private_workload)
    
    print("\nOriginal Workload Metrics:")
    print(f"Read ratio: {original_metrics['read_ratio']:.3f}")
    print(f"Write ratio: {original_metrics['write_ratio']:.3f}")
    print(f"Hot key ratio: {original_metrics['hot_key_ratio']:.3f}")
    print(f"Total operations: {original_metrics['total_operations']}")
    
    print("\nPrivate Workload Metrics:")
    print(f"Read ratio: {private_metrics['read_ratio']:.3f}")
    print(f"Write ratio: {private_metrics['write_ratio']:.3f}")
    print(f"Hot key ratio: {private_metrics['hot_key_ratio']:.3f}")
    print(f"Total operations: {private_metrics['total_operations']}")
    
    # Run experiment
    print("\nRunning experiment...")
    experiment = EndureExperiment(epsilon=1.0)
    results = experiment.run_experiment(characteristics)
    
    print("\nExperiment Results:")
    print("Original Performance:")
    print(f"Operations/sec: {results['original_performance']['operations_per_second']:.2f}")
    print(f"Duration (s): {results['original_performance']['duration_seconds']:.2f}")
    
    print("\nPrivate Performance:")
    print(f"Operations/sec: {results['private_performance']['operations_per_second']:.2f}")
    print(f"Duration (s): {results['private_performance']['duration_seconds']:.2f}")

if __name__ == "__main__":
    test_workload_generation() 