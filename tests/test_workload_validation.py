import unittest
from endure.workload_generator import WorkloadGenerator, WorkloadCharacteristics
import logging

class TestWorkloadValidation(unittest.TestCase):
    def setUp(self):
        self.generator = WorkloadGenerator(epsilon=1.0)
        logging.basicConfig(level=logging.ERROR)  # Suppress validation warnings during tests

    def test_large_workload_validation(self):
        """Test validation with a large workload (>1M operations)"""
        characteristics = WorkloadCharacteristics(
            read_ratio=0.7,
            write_ratio=0.3,
            key_size=16,
            value_size=100,
            operation_count=2000000,  # 2M operations
            hot_key_ratio=0.2,
            hot_key_count=100
        )
        
        original_workload, private_workload = self.generator.generate_workload(characteristics)
        self.assertIsNotNone(original_workload)
        self.assertIsNotNone(private_workload)
        self.assertEqual(len(original_workload), characteristics.operation_count)

    def test_small_hot_key_ratio(self):
        """Test validation with very small hot key ratio (<1%)"""
        characteristics = WorkloadCharacteristics(
            read_ratio=0.7,
            write_ratio=0.3,
            key_size=16,
            value_size=100,
            operation_count=100000,
            hot_key_ratio=0.005,  # 0.5% hot key ratio
            hot_key_count=10
        )
        
        original_workload, private_workload = self.generator.generate_workload(characteristics)
        self.assertIsNotNone(original_workload)
        self.assertIsNotNone(private_workload)

    def test_edge_case_ratios(self):
        """Test validation with edge case ratios"""
        characteristics = WorkloadCharacteristics(
            read_ratio=0.99,
            write_ratio=0.01,
            key_size=16,
            value_size=100,
            operation_count=100000,
            hot_key_ratio=0.99,
            hot_key_count=100
        )
        
        original_workload, private_workload = self.generator.generate_workload(characteristics)
        self.assertIsNotNone(original_workload)
        self.assertIsNotNone(private_workload)

    def test_invalid_characteristics(self):
        """Test validation with invalid characteristics"""
        with self.assertRaises(ValueError):
            characteristics = WorkloadCharacteristics(
                read_ratio=1.1,  # Invalid ratio > 1
                write_ratio=0.3,
                key_size=16,
                value_size=100,
                operation_count=100000,
                hot_key_ratio=0.2,
                hot_key_count=100
            )
            self.generator.generate_workload(characteristics)

    def test_key_size_validation(self):
        """Test key size validation"""
        characteristics = WorkloadCharacteristics(
            read_ratio=0.7,
            write_ratio=0.3,
            key_size=8,  # Small key size
            value_size=100,
            operation_count=1000,
            hot_key_ratio=0.2,
            hot_key_count=10
        )
        
        original_workload, private_workload = self.generator.generate_workload(characteristics)
        self.assertIsNotNone(original_workload)
        self.assertIsNotNone(private_workload)
        
        # Verify key sizes
        for op in original_workload:
            self.assertLessEqual(len(op["key"].encode()), characteristics.key_size)

    def test_value_size_validation(self):
        """Test value size validation"""
        characteristics = WorkloadCharacteristics(
            read_ratio=0.7,
            write_ratio=0.3,
            key_size=16,
            value_size=50,  # Small value size
            operation_count=1000,
            hot_key_ratio=0.2,
            hot_key_count=10
        )
        
        original_workload, private_workload = self.generator.generate_workload(characteristics)
        self.assertIsNotNone(original_workload)
        self.assertIsNotNone(private_workload)
        
        # Verify value sizes
        for op in original_workload:
            if op["type"] == "write":
                self.assertEqual(len(op["value"]), characteristics.value_size)

if __name__ == '__main__':
    unittest.main() 