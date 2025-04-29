import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
import os
import shutil
import psutil
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from .types import WorkloadCharacteristics, WorkloadData, MathUtils

class WorkloadGenerator:
    def __init__(self, epsilon: float = 1.0, batch_size: int = 1000, seed: int = None,
                 max_workers: int = 4, results_dir: str = "workload_results"):
        """Initialize workload generator with enhanced resource management."""
        self.results_dir = results_dir
        self.max_workers = max_workers
        self.temp_files: Set[str] = set()  # Track temporary files
        self._setup_logging()
        self._setup_directories()
        
        try:
            # Validate epsilon with reasonable range
            if not isinstance(epsilon, (int, float)):
                logging.error("epsilon must be numeric")
                epsilon = 1.0
            elif epsilon <= 0:
                logging.warning(f"Invalid epsilon value {epsilon}, using default 1.0")
                epsilon = 1.0
            elif epsilon > 10.0:
                logging.warning(f"Large epsilon value {epsilon} may impact privacy")
            
            # Validate batch size with reasonable range
            if not isinstance(batch_size, int):
                logging.error("batch_size must be integer")
                batch_size = 1000
            elif batch_size <= 0:
                logging.warning(f"Invalid batch_size {batch_size}, using default 1000")
                batch_size = 1000
            elif batch_size > 100000:
                logging.warning(f"Large batch_size {batch_size} may impact performance")
            
            self.epsilon = epsilon
            self.batch_size = batch_size
            
            # Initialize random number generator with validation
            if seed is not None:
                if not isinstance(seed, int):
                    logging.error("seed must be integer")
                    seed = None
                elif not 0 <= seed <= 2**32 - 1:
                    logging.warning(f"Invalid seed value {seed}, using random seed")
                    seed = None
            
            if seed is None:
                self.seed = np.random.randint(0, 2**32 - 1)
            else:
                self.seed = seed
                
            np.random.seed(self.seed)
            
        except Exception as e:
            logging.error(f"Error initializing WorkloadGenerator: {str(e)}")
            self.epsilon = 1.0
            self.batch_size = 1000
            self.seed = np.random.randint(0, 2**32 - 1)
            np.random.seed(self.seed)
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = os.path.join(self.results_dir, "workload_generator.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _setup_directories(self) -> None:
        """Set up required directories with validation."""
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            if not os.access(self.results_dir, os.W_OK):
                raise PermissionError(f"No write permission for {self.results_dir}")
            
            # Check disk space
            disk = shutil.disk_usage(self.results_dir)
            free_gb = disk.free / (1024 * 1024 * 1024)
            if free_gb < 1:  # Need at least 1GB free
                raise RuntimeError(f"Insufficient disk space: {free_gb:.2f}GB available")
        except Exception as e:
            logging.error(f"Error setting up directories: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logging.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
        self.temp_files.clear()

    def get_adaptive_epsilon(self, characteristics: WorkloadCharacteristics) -> float:
        """Get adaptive epsilon based on workload characteristics with enhanced validation."""
        try:
            if not characteristics.validate():
                logging.warning("Using default epsilon due to invalid characteristics")
                return self.epsilon
            
            # Check memory usage
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024 * 1024 * 1024)
            available_memory_gb = memory.available / (1024 * 1024 * 1024)
            
            # Adjust threshold based on total memory
            memory_threshold_gb = min(4.0, total_memory_gb * 0.2)  # 4GB or 20% of total, whichever is smaller
            
            if available_memory_gb < memory_threshold_gb:
                logging.warning(f"Low memory available ({available_memory_gb:.1f}GB < {memory_threshold_gb:.1f}GB), using conservative epsilon")
                return min(self.epsilon * 0.9, 1.0)  # Less aggressive reduction
            
            base_epsilon = self.epsilon
            adjustments = []
            
            # Adjust epsilon based on workload characteristics with bounds
            try:
                if characteristics.hot_key_ratio > 0.3:
                    adjustments.append(min(base_epsilon * 1.5, 10.0))  # Cap at 10.0
                if characteristics.read_ratio > 0.8:
                    adjustments.append(min(base_epsilon * 1.2, 5.0))  # Cap at 5.0
                if characteristics.operation_count > 1000000:
                    adjustments.append(min(base_epsilon * 1.1, 3.0))  # Cap at 3.0
                
                if adjustments:
                    adaptive_epsilon = min(adjustments)
                else:
                    adaptive_epsilon = base_epsilon
                
                # Validate final epsilon value
                if not 0 < adaptive_epsilon <= 10.0:
                    logging.warning(f"Invalid adaptive epsilon {adaptive_epsilon}, using base epsilon")
                    return base_epsilon
                
                return adaptive_epsilon
                
            except Exception as e:
                logging.error(f"Error calculating adaptive epsilon: {str(e)}")
                return base_epsilon
            
        except Exception as e:
            logging.error(f"Error in get_adaptive_epsilon: {str(e)}")
            return self.epsilon

    def _validate_generated_workload(self, workload: List[Dict], characteristics: WorkloadCharacteristics) -> bool:
        """Validate generated workload with more lenient checks."""
        try:
            if not isinstance(workload, list):
                logging.error("Workload must be a list")
                return False
            
            if not workload:
                logging.error("Workload is empty")
                return False
            
            # Check basic structure and count operations
            read_count = 0
            write_count = 0
            hot_key_count = 0
            valid_ops = 0
            
            for op in workload:
                if not isinstance(op, dict):
                    continue
                
                # Check for required fields with lenient validation
                has_type = "type" in op and op["type"] in ["read", "write"]
                has_key = "key" in op and isinstance(op["key"], (str, bytes))
                has_value = "value" in op and isinstance(op["value"], (str, bytes))
                
                if has_type and (has_key or has_value):
                    valid_ops += 1
                    if op["type"] == "read":
                        read_count += 1
                    else:
                        write_count += 1
                    
                    # Check key size
                    if has_key and len(op["key"]) > characteristics.key_size:
                        logging.warning(
                            f"Key size {len(op['key'])} exceeds specified key_size "
                            f"{characteristics.key_size}"
                        )
                    
                    # Check value size
                    if has_value and len(op["value"]) > characteristics.value_size:
                        logging.warning(
                            f"Value size {len(op['value'])} exceeds specified value_size "
                            f"{characteristics.value_size}"
                        )
            
            # Validate operation counts
            total_ops = read_count + write_count
            if total_ops != characteristics.operation_count:
                logging.warning(
                    f"Generated operation count {total_ops} does not match specified "
                    f"{characteristics.operation_count}"
                )
            
            # Validate read/write ratio
            actual_read_ratio = read_count / total_ops if total_ops > 0 else 0
            if abs(actual_read_ratio - characteristics.read_ratio) > 0.01:  # 1% tolerance
                logging.warning(
                    f"Actual read ratio {actual_read_ratio:.3f} differs from specified "
                    f"{characteristics.read_ratio:.3f}"
                )
            
            # Consider workload valid if most operations are valid
            return valid_ops / len(workload) > 0.9  # 90% valid operations
            
        except Exception as e:
            logging.error(f"Error validating generated workload: {str(e)}")
            return False

    def generate_workload(self, characteristics: WorkloadCharacteristics) -> Tuple[WorkloadData, WorkloadData]:
        """Generate workload with the given characteristics."""
        try:
            if not characteristics.validate():
                raise ValueError("Invalid workload characteristics")
            
            # Generate workload data
            workload = self._generate_workload_internal(characteristics)
            
            # Add differential privacy
            private_workload = self._add_differential_privacy(workload, characteristics)
            
            return workload, private_workload
            
        except Exception as e:
            logging.error(f"Error generating workload: {str(e)}")
            raise

    def _generate_unique_key(self, prefix: str, key_size: int, existing_keys: set) -> str:
        """Generate a unique key that fits within size constraints."""
        try:
            # Use a more efficient key generation strategy
            base_key = prefix[:min(len(prefix), key_size - 8)]  # Leave room for counter
            counter = len(existing_keys)  # Start from current count
            
            while True:
                # Generate key with counter
                key = f"{base_key}{counter:08d}"[:key_size]
                
                # Check uniqueness
                if key not in existing_keys:
                    existing_keys.add(key)
                    return key
                
                counter += 1
                
                # Prevent infinite loop
                if counter > 1_000_000:  # Much higher limit, but still safe
                    logging.error("Key space exhausted")
                    return f"{prefix}{np.random.randint(1000, 9999)}"
                    
        except Exception as e:
            logging.error(f"Error generating unique key: {str(e)}")
            return f"{prefix}{np.random.randint(1000, 9999)}"

    def _generate_value(self, size: int) -> bytes:
        """Generate a value of specified size with some variation."""
        try:
            if size <= 0:
                return b""
                
            # Generate some random bytes for variation
            random_bytes = np.random.bytes(min(size, 16))
            
            # Pad with pattern if needed
            if size > len(random_bytes):
                pattern = b"x" * (size - len(random_bytes))
                return random_bytes + pattern
            else:
                return random_bytes[:size]
                
        except Exception as e:
            logging.warning(f"Error generating value: {str(e)}")
            return b"x" * size

    def _generate_workload_internal(self, characteristics: WorkloadCharacteristics) -> List[Dict]:
        """Generate the original workload with batch processing."""
        try:
            workload = []
            total_ops = characteristics.operation_count
            existing_keys = set()
            
            # Generate hot keys with size limit and fallback
            hot_keys = []
            for i in range(characteristics.hot_key_count):
                try:
                    key = self._generate_unique_key("hk", characteristics.key_size, existing_keys)
                    hot_keys.append(key)
                except Exception as e:
                    logging.warning(f"Error generating hot key: {str(e)}")
                    continue
            
            if not hot_keys:
                logging.warning("No valid hot keys generated, using default")
                hot_keys = [self._generate_unique_key("hk", characteristics.key_size, existing_keys)]
            
            # Calculate exact number of hot key operations
            hot_ops_count = int(total_ops * characteristics.hot_key_ratio)
            cold_ops_count = total_ops - hot_ops_count
            
            # Process in batches
            for batch_start in range(0, total_ops, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_ops)
                batch_hot_ops = min(hot_ops_count - batch_start, batch_end - batch_start)
                batch_cold_ops = batch_end - batch_start - batch_hot_ops
                
                # Generate hot key operations
                for i in range(batch_hot_ops):
                    try:
                        is_read = np.random.random() < characteristics.read_ratio
                        key = np.random.choice(hot_keys)
                        operation = {
                            "type": "read" if is_read else "write",
                            "key": key,
                            "value": self._generate_value(characteristics.value_size) if not is_read else None,
                            "is_hot": True
                        }
                        workload.append(operation)
                    except Exception as e:
                        logging.warning(f"Error generating hot key operation: {str(e)}")
                        continue
                
                # Generate cold key operations
                for i in range(batch_cold_ops):
                    try:
                        is_read = np.random.random() < characteristics.read_ratio
                        key = self._generate_unique_key("k", characteristics.key_size, existing_keys)
                        operation = {
                            "type": "read" if is_read else "write",
                            "key": key,
                            "value": self._generate_value(characteristics.value_size) if not is_read else None,
                            "is_hot": False
                        }
                        workload.append(operation)
                    except Exception as e:
                        logging.warning(f"Error generating cold key operation: {str(e)}")
                        continue
            
            # Shuffle the workload to mix hot and cold operations
            if workload:
                np.random.shuffle(workload)
            
            return workload
        except Exception as e:
            logging.error(f"Error generating workload: {str(e)}")
            return []

    def _add_differential_privacy(self, workload: List[Dict], 
                                characteristics: WorkloadCharacteristics) -> List[Dict]:
        """Add differential privacy to the workload with enhanced error handling."""
        try:
            # Calculate adaptive epsilon with validation
            adaptive_epsilon = self.get_adaptive_epsilon(characteristics)
            if not 0 < adaptive_epsilon <= 10.0:
                raise ValueError(f"Invalid adaptive epsilon: {adaptive_epsilon}")
            
            # Calculate sensitivity for each characteristic with validation
            try:
                read_sensitivity = 0.5 / max(1, characteristics.operation_count)
                write_sensitivity = 1.0 / max(1, characteristics.operation_count)
                hot_key_ratio_sensitivity = 1.0 / max(1, characteristics.operation_count)
                
                # Add Laplace noise to ratios with different sensitivities
                noisy_read_ratio = characteristics.read_ratio + np.random.laplace(
                    0, read_sensitivity / adaptive_epsilon)
                noisy_write_ratio = characteristics.write_ratio + np.random.laplace(
                    0, write_sensitivity / adaptive_epsilon)
                noisy_hot_key_ratio = characteristics.hot_key_ratio + np.random.laplace(
                    0, hot_key_ratio_sensitivity / adaptive_epsilon)
                
                # Ensure ratios are valid and sum to 1
                noisy_read_ratio = max(0, min(1, noisy_read_ratio))
                noisy_write_ratio = max(0, min(1, noisy_write_ratio))
                noisy_hot_key_ratio = max(0, min(1, noisy_hot_key_ratio))
                
                # Normalize read and write ratios to sum to 1
                total = noisy_read_ratio + noisy_write_ratio
                if total > 0:
                    noisy_read_ratio /= total
                    noisy_write_ratio /= total
                else:
                    noisy_read_ratio = 0.5
                    noisy_write_ratio = 0.5
                
                # Apply noise to workload
                for op in workload:
                    if op["type"] == "read":
                        op["probability"] = noisy_read_ratio
                    elif op["type"] == "write":
                        op["probability"] = noisy_write_ratio
                
                return workload
                
            except Exception as e:
                logging.error(f"Error adding differential privacy: {str(e)}")
                return workload
            
        except Exception as e:
            logging.error(f"Error in _add_differential_privacy: {str(e)}")
            return workload

    def calculate_workload_metrics(self, workload: List[Dict]) -> Dict:
        """Calculate detailed workload metrics."""
        try:
            if not workload:
                return {
                    'total_operations': 0,
                    'read_ratio': 0.0,
                    'write_ratio': 0.0,
                    'hot_key_ratio': 0.0,
                    'operation_distribution': {},
                    'key_distribution': {},
                    'value_size_distribution': {},
                    'temporal_patterns': {}
                }
            
            # Basic metrics
            total_ops = len(workload)
            read_count = sum(1 for op in workload if op['type'] == 'read')
            write_count = total_ops - read_count
            hot_key_count = sum(1 for op in workload if op['is_hot'])
            
            # Key access patterns
            key_access_counts = {}
            for op in workload:
                key = op['key']
                key_access_counts[key] = key_access_counts.get(key, 0) + 1
            
            # Sort keys by access frequency
            sorted_keys = sorted(key_access_counts.items(), key=lambda x: x[1], reverse=True)
            hot_keys = sorted_keys[:10]  # Top 10 most accessed keys
            
            # Value size distribution
            value_sizes = {}
            for op in workload:
                if op['type'] == 'write':
                    size = len(op['value'])
                    value_sizes[size] = value_sizes.get(size, 0) + 1
            
            # Temporal patterns (operations per time window)
            time_windows = {}
            window_size = 1000  # operations per window
            for i, op in enumerate(workload):
                window = i // window_size
                time_windows[window] = time_windows.get(window, 0) + 1
            
            return {
                'total_operations': total_ops,
                'read_ratio': read_count / total_ops if total_ops > 0 else 0.0,
                'write_ratio': write_count / total_ops if total_ops > 0 else 0.0,
                'hot_key_ratio': hot_key_count / total_ops if total_ops > 0 else 0.0,
                'operation_distribution': {
                    'read': read_count,
                    'write': write_count
                },
                'key_distribution': {
                    'total_unique_keys': len(key_access_counts),
                    'hot_keys': dict(hot_keys),
                    'key_access_counts': key_access_counts
                },
                'value_size_distribution': value_sizes,
                'temporal_patterns': {
                    'window_size': window_size,
                    'operations_per_window': time_windows
                }
            }
        except Exception as e:
            logger.error(f"Error calculating workload metrics: {str(e)}")
            return {
                'total_operations': 0,
                'read_ratio': 0.0,
                'write_ratio': 0.0,
                'hot_key_ratio': 0.0,
                'operation_distribution': {},
                'key_distribution': {},
                'value_size_distribution': {},
                'temporal_patterns': {}
            } 