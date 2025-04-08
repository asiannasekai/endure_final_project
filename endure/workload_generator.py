import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

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

class WorkloadGenerator:
    def __init__(self, epsilon: float = 1.0):
        """Initialize workload generator with privacy parameter epsilon."""
        self.epsilon = epsilon

    def generate_workload(self, characteristics: WorkloadCharacteristics) -> Tuple[List[Dict], List[Dict]]:
        """Generate both original and differentially private workloads."""
        # Generate original workload
        original_workload = self._generate_workload_internal(characteristics)
        
        # Generate differentially private workload
        private_workload = self._add_differential_privacy(original_workload, characteristics)
        
        return original_workload, private_workload

    def _generate_workload_internal(self, characteristics: WorkloadCharacteristics) -> List[Dict]:
        """Generate the original workload."""
        workload = []
        total_ops = characteristics.operation_count
        
        # Generate hot keys
        hot_keys = [f"hot_key_{i}" for i in range(characteristics.hot_key_count)]
        
        # Generate operations
        for i in range(total_ops):
            # Determine if operation is read or write
            is_read = np.random.random() < characteristics.read_ratio
            
            # Determine if operation is on hot key
            is_hot = np.random.random() < characteristics.hot_key_ratio
            
            if is_hot:
                key = np.random.choice(hot_keys)
            else:
                key = f"key_{i}"
            
            operation = {
                "type": "read" if is_read else "write",
                "key": key,
                "value": "x" * characteristics.value_size if not is_read else None,
                "is_hot": is_hot
            }
            workload.append(operation)
        
        return workload

    def _add_differential_privacy(self, workload: List[Dict], 
                                characteristics: WorkloadCharacteristics) -> List[Dict]:
        """Add differential privacy to the workload."""
        # Calculate sensitivity for each characteristic
        read_ratio_sensitivity = 1.0 / characteristics.operation_count
        write_ratio_sensitivity = 1.0 / characteristics.operation_count
        hot_key_ratio_sensitivity = 1.0 / characteristics.operation_count
        
        # Add Laplace noise to ratios
        noisy_read_ratio = characteristics.read_ratio + np.random.laplace(
            0, read_ratio_sensitivity / self.epsilon)
        noisy_write_ratio = characteristics.write_ratio + np.random.laplace(
            0, write_ratio_sensitivity / self.epsilon)
        noisy_hot_key_ratio = characteristics.hot_key_ratio + np.random.laplace(
            0, hot_key_ratio_sensitivity / self.epsilon)
        
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
        total_ops = len(workload)
        read_count = sum(1 for op in workload if op["type"] == "read")
        write_count = total_ops - read_count
        hot_op_count = sum(1 for op in workload if op["is_hot"])
        
        return {
            "read_ratio": read_count / total_ops,
            "write_ratio": write_count / total_ops,
            "hot_key_ratio": hot_op_count / total_ops,
            "total_operations": total_ops
        } 