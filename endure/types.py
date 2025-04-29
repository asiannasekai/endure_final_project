from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import math
import logging

@dataclass
class WorkloadCharacteristics:
    """Shared data structure for workload characteristics."""
    read_ratio: float  # Ratio of read operations (0.0 to 1.0)
    write_ratio: float  # Ratio of write operations (0.0 to 1.0)
    key_size: int  # Size of keys in bytes (1 to 4096)
    value_size: int  # Size of values in bytes (1 to 16777216)
    operation_count: int  # Total number of operations (100 to 1000000000)
    hot_key_ratio: float  # Ratio of operations on hot keys (0.0 to 1.0)
    hot_key_count: int  # Number of hot keys (1 to 100000)

    def validate(self) -> bool:
        """Validate workload characteristics with more lenient checks."""
        try:
            # Validate ratios in one pass
            try:
                self.read_ratio = MathUtils.validate_ratio(self.read_ratio, "read_ratio")
                self.write_ratio = MathUtils.validate_ratio(self.write_ratio, "write_ratio")
                self.hot_key_ratio = MathUtils.validate_ratio(self.hot_key_ratio, "hot_key_ratio")
                
                # Validate ratio sum with 5% tolerance
                ratios = MathUtils.validate_ratios_sum({
                    'read_ratio': self.read_ratio,
                    'write_ratio': self.write_ratio
                }, tolerance=0.05)
                self.read_ratio = ratios['read_ratio']
                self.write_ratio = ratios['write_ratio']
            except ValueError as e:
                logging.error(f"Ratio validation error: {str(e)}")
                return False
            
            # Validate sizes in one pass
            size_ranges = {
                'key_size': (1, 4096),
                'value_size': (1, 16777216),
                'operation_count': (100, 1000000000),
                'hot_key_count': (1, 100000)
            }
            
            for attr, (min_val, max_val) in size_ranges.items():
                value = getattr(self, attr)
                if not isinstance(value, int) or value <= 0:
                    logging.error(f"{attr} must be a positive integer")
                    return False
                if not min_val <= value <= max_val:
                    logging.warning(
                        f"{attr} value {value} is outside expected range [{min_val}, {max_val}]"
                    )
            
            # Validate hot key count against operation count
            if self.hot_key_count > self.operation_count:
                logging.warning(
                    f"hot_key_count ({self.hot_key_count}) cannot be greater than "
                    f"operation_count ({self.operation_count})"
                )
            
            # Validate hot key ratio against hot key count
            expected_hot_keys = MathUtils.calculate_hot_key_count(
                self.operation_count,
                self.hot_key_ratio
            )
            if self.hot_key_count > expected_hot_keys:
                logging.warning(
                    f"hot_key_count ({self.hot_key_count}) is larger than expected based on "
                    f"hot_key_ratio ({expected_hot_keys})"
                )
            
            # Estimate memory usage
            try:
                estimated_memory = MathUtils.estimate_memory_usage(
                    self.operation_count,
                    self.key_size,
                    self.value_size,
                    self.hot_key_count
                )
                
                if estimated_memory > 8000:  # 8GB threshold
                    logging.warning(
                        f"Estimated memory usage is {estimated_memory:.2f}MB. "
                        "This may impact system performance."
                    )
            except ValueError as e:
                logging.error(f"Memory estimation error: {str(e)}")
                return False
            
            return True
        except Exception as e:
            logging.error(f"Error validating workload characteristics: {str(e)}")
            return False

    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkloadCharacteristics':
        """Create instance from dictionary with validation."""
        defaults = {
            'read_ratio': 0.7,
            'write_ratio': 0.3,
            'key_size': 16,
            'value_size': 100,
            'operation_count': 100000,
            'hot_key_ratio': 0.2,
            'hot_key_count': 100
        }
        
        # Validate and normalize values
        validated = {}
        for key, default in defaults.items():
            value = data.get(key, default)
            if key in ['read_ratio', 'write_ratio', 'hot_key_ratio']:
                validated[key] = float(value)
            else:
                validated[key] = int(value)
        
        return cls(**validated)

@dataclass
class AnalysisResult:
    """Structure for analysis results."""
    epsilon: float
    privacy_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    workload_characteristics: WorkloadCharacteristics
    configuration: Dict[str, Any]

# Type aliases for common data structures
WorkloadData = List[Dict[str, Union[str, bytes, int, float]]]
AnalysisResults = Dict[float, List[AnalysisResult]]
Metrics = Dict[str, Union[float, int, str]]

class MathUtils:
    """Utility class for mathematical operations and validations."""
    
    @staticmethod
    def validate_ratio(value: float, name: str) -> float:
        """Validate and normalize a ratio value."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if value < 0 or value > 1:
            raise ValueError(f"{name} must be between 0 and 1")
        return float(value)
    
    @staticmethod
    def validate_ratios_sum(ratios: Dict[str, float], tolerance: float = 0.05) -> Dict[str, float]:
        """Validate that ratios sum to 1.0 within tolerance."""
        total = sum(ratios.values())
        if abs(total - 1.0) > tolerance:
            # Normalize ratios to sum to 1.0
            return {k: v/total for k, v in ratios.items()}
        return ratios
    
    @staticmethod
    def estimate_memory_usage(
        operation_count: int,
        key_size: int,
        value_size: int,
        hot_key_count: int
    ) -> float:
        """Estimate memory usage in MB."""
        try:
            # Estimate memory for keys and values
            key_memory = operation_count * key_size
            value_memory = operation_count * value_size
            
            # Add overhead for hot keys
            hot_key_overhead = hot_key_count * (key_size + value_size) * 2
            
            # Convert to MB
            total_memory = (key_memory + value_memory + hot_key_overhead) / (1024 * 1024)
            return total_memory
        except Exception as e:
            raise ValueError(f"Error estimating memory usage: {str(e)}")
    
    @staticmethod
    def calculate_hot_key_count(
        operation_count: int,
        hot_key_ratio: float
    ) -> int:
        """Calculate expected number of hot keys."""
        try:
            return max(1, int(operation_count * hot_key_ratio))
        except Exception as e:
            raise ValueError(f"Error calculating hot key count: {str(e)}")
    
    @staticmethod
    def smooth_data(x: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth data using moving average."""
        try:
            if len(x) != len(y):
                raise ValueError("x and y must have same length")
            if window_size < 1:
                raise ValueError("window_size must be positive")
            
            window = np.ones(window_size) / window_size
            y_smooth = np.convolve(y, window, mode='same')
            return x, y_smooth
        except Exception as e:
            raise ValueError(f"Error smoothing data: {str(e)}")
    
    @staticmethod
    def fit_curve(x: np.ndarray, y: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fit polynomial curve to data."""
        try:
            if len(x) != len(y):
                raise ValueError("x and y must have same length")
            if degree < 1:
                raise ValueError("degree must be positive")
            
            coeffs = np.polyfit(x, y, degree)
            y_fit = np.polyval(coeffs, x)
            return x, y_fit
        except Exception as e:
            raise ValueError(f"Error fitting curve: {str(e)}") 