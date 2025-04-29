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
    key_size: int  # Size of keys in bytes (1 to 1024)
    value_size: int  # Size of values in bytes (1 to 1048576)
    operation_count: int  # Total number of operations (1000 to 100000000)
    hot_key_ratio: float  # Ratio of operations on hot keys (0.0 to 1.0)
    hot_key_count: int  # Number of hot keys (1 to 10000)

    def validate(self) -> bool:
        """Validate workload characteristics with more lenient checks."""
        try:
            # Validate ratios in one pass
            try:
                self.read_ratio = MathUtils.validate_ratio(self.read_ratio, "read_ratio")
                self.write_ratio = MathUtils.validate_ratio(self.write_ratio, "write_ratio")
                self.hot_key_ratio = MathUtils.validate_ratio(self.hot_key_ratio, "hot_key_ratio")
                
                # Validate ratio sum
                ratios = MathUtils.validate_ratios_sum({
                    'read_ratio': self.read_ratio,
                    'write_ratio': self.write_ratio
                })
                self.read_ratio = ratios['read_ratio']
                self.write_ratio = ratios['write_ratio']
            except ValueError as e:
                logging.error(f"Ratio validation error: {str(e)}")
                return False
            
            # Validate sizes in one pass
            size_attrs = ['key_size', 'value_size', 'operation_count', 'hot_key_count']
            for attr in size_attrs:
                value = getattr(self, attr)
                if not isinstance(value, int) or value <= 0:
                    logging.error(f"{attr} must be a positive integer")
                    return False
            
            # Validate hot key count against operation count
            if self.hot_key_count > self.operation_count:
                logging.warning(
                    f"hot_key_count ({self.hot_key_count}) cannot be greater than "
                    f"operation_count ({self.operation_count})"
                )
                return False
            
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
                return False
            
            # Estimate memory usage
            try:
                estimated_memory = MathUtils.estimate_memory_usage(
                    self.operation_count,
                    self.key_size,
                    self.value_size,
                    self.hot_key_count
                )
                
                if estimated_memory > 1000:  # 1GB threshold
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
class VisualizationConfig:
    """Configuration for visualization settings."""
    style: str = "default"
    figure_size: Tuple[int, int] = (12, 8)
    font_size: int = 12
    color_palette: str = "viridis"
    dpi: int = 300
    smoothing_window: int = 3
    curve_degree: int = 2

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
    """Utility class for mathematical operations with validation."""
    
    @staticmethod
    def validate_ratio(value: float, name: str) -> float:
        """Validate and normalize a ratio value between 0 and 1."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        return max(0.0, min(1.0, float(value)))
    
    @staticmethod
    def validate_ratios_sum(ratios: Dict[str, float], tolerance: float = 0.01) -> Dict[str, float]:
        """Validate that ratios sum to 1.0 within tolerance."""
        ratio_sum = sum(ratios.values())
        if abs(ratio_sum - 1.0) > tolerance:
            if ratio_sum == 0:
                # If all ratios are 0, distribute evenly
                return {k: 1.0/len(ratios) for k in ratios}
            # Normalize ratios to sum to 1.0
            return {k: v/ratio_sum for k, v in ratios.items()}
        return ratios
    
    @staticmethod
    def estimate_memory_usage(
        operation_count: int,
        key_size: int,
        value_size: int,
        hot_key_count: int
    ) -> float:
        """Estimate memory usage in MB."""
        if any(v <= 0 for v in [operation_count, key_size, value_size, hot_key_count]):
            raise ValueError("All size parameters must be positive")
        return (
            operation_count * (key_size + value_size) +
            hot_key_count * (key_size + value_size)
        ) / (1024 * 1024)  # Convert to MB
    
    @staticmethod
    def calculate_hot_key_count(
        operation_count: int,
        hot_key_ratio: float
    ) -> int:
        """Calculate expected number of hot keys."""
        if operation_count <= 0:
            raise ValueError("operation_count must be positive")
        hot_key_ratio = MathUtils.validate_ratio(hot_key_ratio, "hot_key_ratio")
        return int(operation_count * hot_key_ratio)
    
    @staticmethod
    def smooth_data(x: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply moving average smoothing to data."""
        if len(x) != len(y):
            raise ValueError("x and y arrays must have same length")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if len(y) < window_size:
            return x, y
        
        kernel = np.ones(window_size) / window_size
        smoothed_y = np.convolve(y, kernel, mode='valid')
        smoothed_x = x[window_size-1:]
        return smoothed_x, smoothed_y
    
    @staticmethod
    def fit_curve(x: np.ndarray, y: np.ndarray, degree: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fit polynomial curve to data."""
        if len(x) != len(y):
            raise ValueError("x and y arrays must have same length")
        if degree < 0:
            raise ValueError("degree must be non-negative")
        if len(x) < degree + 1:
            return x, y
        
        z = np.polyfit(x, y, degree)
        p = np.poly1d(z)
        x_new = np.linspace(min(x), max(x), 100)
        y_new = p(x_new)
        return x_new, y_new 