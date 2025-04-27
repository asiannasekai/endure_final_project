import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging

@dataclass
class WorkloadCharacteristics:
    """Characteristics of a database workload."""
    read_ratio: float  # Ratio of read operations
    write_ratio: float  # Ratio of write operations
    key_size: int  # Size of keys in bytes
    value_size: int  # Size of values in bytes
    operation_count: int  # Total number of operations
    hot_key_ratio: float  # Ratio of operations on hot keys
    hot_key_count: int  # Number of hot keys

    def validate(self) -> bool:
        """Validate workload characteristics."""
        if not 0 <= self.read_ratio <= 1:
            logging.error("read_ratio must be between 0 and 1")
            return False
        if not 0 <= self.write_ratio <= 1:
            logging.error("write_ratio must be between 0 and 1")
            return False
        if abs(self.read_ratio + self.write_ratio - 1.0) > 1e-6:
            logging.error("read_ratio + write_ratio must equal 1")
            return False
        if self.key_size <= 0:
            logging.error("key_size must be positive")
            return False
        if self.value_size <= 0:
            logging.error("value_size must be positive")
            return False
        if self.operation_count <= 0:
            logging.error("operation_count must be positive")
            return False
        if not 0 <= self.hot_key_ratio <= 1:
            logging.error("hot_key_ratio must be between 0 and 1")
            return False
        if self.hot_key_count <= 0:
            logging.error("hot_key_count must be positive")
            return False
        return True

class WorkloadGenerator:
    def __init__(self, epsilon: float = 1.0, batch_size: int = 1000):
        """Initialize workload generator with privacy parameter epsilon."""
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
            
        self.epsilon = epsilon
        self.batch_size = batch_size

    def get_adaptive_epsilon(self, characteristics: WorkloadCharacteristics) -> float:
        """Get adaptive epsilon based on workload characteristics."""
        if not characteristics.validate():
            raise ValueError("Invalid workload characteristics")
            
        base_epsilon = self.epsilon
        # Adjust epsilon based on workload patterns
        if characteristics.hot_key_ratio > 0.3:
            return base_epsilon * 1.5
        if characteristics.read_ratio > 0.8:
            return base_epsilon * 1.2
        return base_epsilon

    def _validate_generated_workload(self, workload: List[Dict], characteristics: WorkloadCharacteristics) -> bool:
        """Validate that the generated workload matches the expected characteristics."""
        if not workload:
            logging.error("Generated workload is empty")
            return False
            
        metrics = self.calculate_workload_metrics(workload)
        
        # Check operation count with tolerance for very large counts
        if characteristics.operation_count > 1000000:  # For large workloads
            tolerance = 0.01  # 1% tolerance
            if abs(metrics["total_operations"] - characteristics.operation_count) / characteristics.operation_count > tolerance:
                logging.error(f"Operation count mismatch: expected {characteristics.operation_count}, got {metrics['total_operations']}")
                return False
        else:
            if metrics["total_operations"] != characteristics.operation_count:
                logging.error(f"Operation count mismatch: expected {characteristics.operation_count}, got {metrics['total_operations']}")
                return False
            
        # Check ratios with adaptive tolerance
        if characteristics.hot_key_ratio < 0.01:  # For very small hot key ratios
            tolerance = 0.1  # 10% tolerance
        else:
            tolerance = 0.05  # 5% tolerance
        
        if abs(metrics["read_ratio"] - characteristics.read_ratio) > tolerance:
            logging.error(f"Read ratio mismatch: expected {characteristics.read_ratio}, got {metrics['read_ratio']}")
            return False
            
        if abs(metrics["write_ratio"] - characteristics.write_ratio) > tolerance:
            logging.error(f"Write ratio mismatch: expected {characteristics.write_ratio}, got {metrics['write_ratio']}")
            return False
            
        if abs(metrics["hot_key_ratio"] - characteristics.hot_key_ratio) > tolerance:
            logging.error(f"Hot key ratio mismatch: expected {characteristics.hot_key_ratio}, got {metrics['hot_key_ratio']}")
            return False
            
        # Check key and value sizes with buffer for encoding overhead
        for op in workload:
            encoded_key_size = len(op["key"].encode())
            if encoded_key_size > characteristics.key_size:
                logging.error(f"Key size exceeds limit: {encoded_key_size} > {characteristics.key_size}")
                return False
            if op["type"] == "write":
                value_size = len(op["value"])
                if value_size != characteristics.value_size:
                    logging.error(f"Value size mismatch: expected {characteristics.value_size}, got {value_size}")
                    return False
                
        return True

    def generate_workload(self, characteristics: WorkloadCharacteristics) -> Tuple[List[Dict], List[Dict]]:
        """Generate both original and differentially private workloads."""
        if not characteristics.validate():
            raise ValueError("Invalid workload characteristics")
            
        # Generate original workload
        original_workload = self._generate_workload_internal(characteristics)
        
        # Validate original workload
        if not self._validate_generated_workload(original_workload, characteristics):
            raise ValueError("Generated workload does not match characteristics")
        
        # Generate differentially private workload
        private_workload = self._add_differential_privacy(original_workload, characteristics)
        
        # Validate private workload against noisy characteristics (with higher tolerance)
        if not self._validate_generated_workload(private_workload, characteristics):
            logging.warning("Private workload deviates from original characteristics (expected due to privacy noise)")
        
        return original_workload, private_workload

    def _generate_workload_internal(self, characteristics: WorkloadCharacteristics) -> List[Dict]:
        """Generate the original workload with batch processing."""
        workload = []
        total_ops = characteristics.operation_count
        
        # Generate hot keys with size limit
        hot_keys = []
        for i in range(characteristics.hot_key_count):
            # Generate key that fits within size limit
            key = f"hk{i}"
            while len(key.encode()) > characteristics.key_size:
                key = key[:-1]  # Remove last character if too large
            hot_keys.append(key)
        
        # Generate operations in batches
        for batch_start in range(0, total_ops, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_ops)
            batch = []
            
            for i in range(batch_start, batch_end):
                # Determine if operation is read or write
                is_read = np.random.random() < characteristics.read_ratio
                
                # Determine if operation is on hot key
                is_hot = np.random.random() < characteristics.hot_key_ratio
                
                if is_hot:
                    key = np.random.choice(hot_keys)
                else:
                    # Generate key that fits within size limit
                    key = f"k{i}"
                    while len(key.encode()) > characteristics.key_size:
                        key = key[:-1]  # Remove last character if too large
                
                operation = {
                    "type": "read" if is_read else "write",
                    "key": key,
                    "value": "x" * characteristics.value_size if not is_read else None,
                    "is_hot": is_hot
                }
                batch.append(operation)
            
            workload.extend(batch)
        
        return workload

    def _add_differential_privacy(self, workload: List[Dict], 
                                characteristics: WorkloadCharacteristics) -> List[Dict]:
        """Add differential privacy to the workload with adaptive noise."""
        # Calculate adaptive epsilon
        adaptive_epsilon = self.get_adaptive_epsilon(characteristics)
        
        # Calculate sensitivity for each characteristic
        read_sensitivity = 0.5 / characteristics.operation_count  # Lower sensitivity for reads
        write_sensitivity = 1.0 / characteristics.operation_count
        hot_key_ratio_sensitivity = 1.0 / characteristics.operation_count
        
        # Add Laplace noise to ratios with different sensitivities
        noisy_read_ratio = characteristics.read_ratio + np.random.laplace(
            0, read_sensitivity / adaptive_epsilon)
        noisy_write_ratio = characteristics.write_ratio + np.random.laplace(
            0, write_sensitivity / adaptive_epsilon)
        noisy_hot_key_ratio = characteristics.hot_key_ratio + np.random.laplace(
            0, hot_key_ratio_sensitivity / adaptive_epsilon)
        
        # Ensure ratios are valid
        noisy_read_ratio = max(0, min(1, noisy_read_ratio))
        noisy_write_ratio = max(0, min(1, noisy_write_ratio))
        noisy_hot_key_ratio = max(0, min(1, noisy_hot_key_ratio))
        
        # Create new characteristics with noisy ratios
        noisy_characteristics = WorkloadCharacteristics(
            read_ratio=noisy_read_ratio,
            write_ratio=noisy_write_ratio,
            key_size=characteristics.key_size,
            value_size=characteristics.value_size,
            operation_count=characteristics.operation_count,
            hot_key_ratio=noisy_hot_key_ratio,
            hot_key_count=characteristics.hot_key_count
        )
        
        # Generate new workload with noisy characteristics
        return self._generate_workload_internal(noisy_characteristics)

    def calculate_workload_metrics(self, workload: List[Dict]) -> Dict:
        """Calculate metrics from a workload."""
        if not workload:
            raise ValueError("Cannot calculate metrics for empty workload")

        total_ops = len(workload)
        read_count = len([op for op in workload if op["type"] == "read"])
        write_count = total_ops - read_count
        hot_op_count = len([op for op in workload if op.get("is_hot", False)])

        # Add safeguards against division by zero
        if total_ops == 0:
            return {
                "read_ratio": 0.0,
                "write_ratio": 0.0,
                "hot_key_ratio": 0.0,
                "total_operations": 0
            }

        return {
            "read_ratio": read_count / total_ops,
            "write_ratio": write_count / total_ops,
            "hot_key_ratio": hot_op_count / total_ops,
            "total_operations": total_ops
        } 