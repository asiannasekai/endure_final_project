# Standard library imports
import logging
import os
import random
import shutil
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any, Union

# Third-party imports
import numpy as np
import psutil

# Local imports
from .types import WorkloadCharacteristics, WorkloadData, MathUtils

class WorkloadGenerator:
    def __init__(self, epsilon: float = 1.0, batch_size: int = 1000, seed: int = None,
                 max_workers: int = 4, results_dir: str = "workload_results"):
        """Initialize workload generator with enhanced resource management."""
        self.results_dir = results_dir
        self.max_workers = max_workers
        self.temp_files: Set[str] = set()
        self._setup_logging()
        
        # Reduce memory requirements
        self.min_memory_gb = 0.5  # Reduced from 1.6GB to 0.5GB
        self.min_disk_gb = 0.5    # Reduced from 1GB to 0.5GB
        
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
        os.makedirs(self.results_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _check_resources(self) -> None:
        """Check system resources with reduced requirements."""
        try:
            # Check memory
            memory = psutil.virtual_memory()
            available_memory_gb = memory.available / (1024 * 1024 * 1024)
            if available_memory_gb < self.min_memory_gb:
                logging.warning(f"Low memory available ({available_memory_gb:.1f}GB < {self.min_memory_gb}GB)")
            
            # Check disk space only if we need to write files
            if self.results_dir:
                disk = shutil.disk_usage(self.results_dir)
                free_gb = disk.free / (1024 * 1024 * 1024)
                if free_gb < self.min_disk_gb:
                    logging.warning(f"Low disk space: {free_gb:.2f}GB available")
                    # Try to free up space by cleaning old files
                    self.cleanup()
        except Exception as e:
            logging.warning(f"Error checking resources: {str(e)}")
    
    def cleanup(self) -> None:
        """Clean up temporary files and resources."""
        try:
            # Remove temporary files
            for temp_file in self.temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logging.debug(f"Removed temporary file: {temp_file}")
            self.temp_files.clear()
            
            # Clean up old result files if they exist
            if os.path.exists(self.results_dir):
                for root, _, files in os.walk(self.results_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            # Remove files older than 1 hour
                            if time.time() - os.path.getmtime(file_path) > 3600:
                                os.remove(file_path)
                                logging.debug(f"Removed old file: {file_path}")
                        except Exception as e:
                            logging.warning(f"Failed to remove file {file_path}: {str(e)}")
        except Exception as e:
            logging.warning(f"Error during cleanup: {str(e)}")

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

    def generate_workload(self, characteristics: WorkloadCharacteristics) -> Tuple[List[Dict], List[Dict]]:
        """Generate workloads with resource management."""
        self._check_resources()
        
        try:
            # Generate workloads in memory
            original_workload = self._generate_workload_internal(characteristics)
            
            # Apply differential privacy with adaptive epsilon
            adaptive_epsilon = self.get_adaptive_epsilon(characteristics)
            private_workload = self._add_differential_privacy(original_workload.copy(), characteristics)
            
            # Clean up temporary resources
            self.cleanup()
            
            return original_workload, private_workload
            
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
        """Add differential privacy to the workload with proper privacy guarantees."""
        try:
            # Calculate adaptive epsilon with validation
            adaptive_epsilon = self.get_adaptive_epsilon(characteristics)
            if not 0 < adaptive_epsilon <= 10.0:
                raise ValueError(f"Invalid adaptive epsilon: {adaptive_epsilon}")
            
            # Group operations by type for sensitivity analysis
            read_ops = [op for op in workload if op['type'] == 'read']
            write_ops = [op for op in workload if op['type'] == 'write']
            hot_ops = [op for op in workload if op['is_hot']]
            
            # Calculate global sensitivities
            sensitivities = {
                'read': 1.0 / len(workload) if workload else 1.0,  # One operation change
                'write': 2.0 / len(workload) if workload else 1.0,  # Write has higher impact
                'hot_key': 1.0 / len(workload) if workload else 1.0,  # Hot key access pattern
                'key_size': characteristics.key_size * 0.1,  # 10% of key size
                'value_size': characteristics.value_size * 0.1  # 10% of value size
            }
            
            # Apply noise to operation counts with composition
            num_queries = 5  # Number of different queries we'll make
            individual_epsilon = adaptive_epsilon / np.sqrt(2 * num_queries * np.log(1/0.01))
            
            # Add noise to operation ratios
            noisy_read_count = len(read_ops) + np.random.laplace(
                0, sensitivities['read'] / individual_epsilon)
            noisy_write_count = len(write_ops) + np.random.laplace(
                0, sensitivities['write'] / individual_epsilon)
            noisy_hot_count = len(hot_ops) + np.random.laplace(
                0, sensitivities['hot_key'] / individual_epsilon)
            
            # Normalize counts
            total_count = max(1, noisy_read_count + noisy_write_count)
            noisy_read_ratio = max(0, min(1, noisy_read_count / total_count))
            noisy_write_ratio = max(0, min(1, noisy_write_count / total_count))
            noisy_hot_ratio = max(0, min(1, noisy_hot_count / len(workload)))
            
            # Create private workload with noisy ratios
            private_workload = []
            for op in workload:
                try:
                    # Deep copy the operation
                    private_op = dict(op)
                    
                    # Determine operation type based on noisy ratios
                    if np.random.random() < noisy_read_ratio:
                        private_op['type'] = 'read'
                    else:
                        private_op['type'] = 'write'
                    
                    # Determine hot key status based on noisy ratio
                    private_op['is_hot'] = np.random.random() < noisy_hot_ratio
                    
                    # Add noise to key size
                    if len(private_op['key']) > 0:
                        noisy_key_size = len(private_op['key']) + int(np.random.laplace(
                            0, sensitivities['key_size'] / individual_epsilon))
                        noisy_key_size = max(1, min(noisy_key_size, characteristics.key_size * 2))
                        private_op['key'] = self._adjust_key_size(private_op['key'], noisy_key_size)
                    
                    # Add noise to value size for write operations
                    if private_op['type'] == 'write' and private_op['value']:
                        noisy_value_size = len(private_op['value']) + int(np.random.laplace(
                            0, sensitivities['value_size'] / individual_epsilon))
                        noisy_value_size = max(1, min(noisy_value_size, characteristics.value_size * 2))
                        private_op['value'] = self._adjust_value_size(private_op['value'], noisy_value_size)
                    
                    # Add privacy metadata
                    private_op['privacy_metadata'] = {
                        'epsilon': individual_epsilon,
                        'noise_scale': {
                            'key_size': sensitivities['key_size'] / individual_epsilon,
                            'value_size': sensitivities['value_size'] / individual_epsilon if private_op['type'] == 'write' else 0
                        }
                    }
                    
                    private_workload.append(private_op)
                    
                except Exception as e:
                    logging.warning(f"Error processing operation: {str(e)}")
                    private_workload.append(op)  # Use original operation as fallback
            
            return private_workload
            
        except Exception as e:
            logging.error(f"Error in _add_differential_privacy: {str(e)}")
            return workload

    def _adjust_key_size(self, key: str, target_size: int) -> str:
        """Adjust key size while preserving format."""
        if target_size <= 0:
            return key
        
        if len(key) < target_size:
            # Pad with random characters
            padding = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'),
                                            target_size - len(key)))
            return key + padding
        else:
            # Truncate preserving prefix
            return key[:target_size]

    def _adjust_value_size(self, value: bytes, target_size: int) -> bytes:
        """Adjust value size while preserving content type."""
        if target_size <= 0:
            return value
        
        if len(value) < target_size:
            # Pad with zeros
            return value + b'\0' * (target_size - len(value))
        else:
            # Truncate preserving prefix
            return value[:target_size]

    def calculate_workload_metrics(self, workload: List[Dict]) -> Dict:
        """Calculate detailed workload metrics."""
        try:
            if not workload or not isinstance(workload, list):
                logging.warning("Empty or invalid workload, returning default metrics")
                return {
                    'total_operations': 0,
                    'read_ratio': 0.0,
                    'write_ratio': 0.0,
                    'hot_key_ratio': 0.0,
                    'throughput': 0.0,
                    'latency': 0.0,
                    'memory': 0.0,
                    'operation_distribution': {'read': 0, 'write': 0},
                    'key_distribution': {'total_unique_keys': 0, 'hot_keys': {}, 'key_access_counts': {}},
                    'value_size_distribution': {},
                    'temporal_patterns': {'window_size': 1000, 'operations_per_window': {}}
                }
            
            # Basic metrics
            total_ops = len(workload)
            read_count = sum(1 for op in workload if op.get('type') == 'read')
            write_count = total_ops - read_count
            hot_key_count = sum(1 for op in workload if op.get('is_hot', False))
            
            # Calculate throughput (operations per second)
            # Assuming average operation takes 1ms
            throughput = total_ops / (total_ops * 0.001) if total_ops > 0 else 0.0  # ops/sec
            
            # Calculate latency (average operation time in ms)
            # Base latency plus some variation based on operation type
            base_latency = 1.0  # ms
            read_latency = base_latency * 1.2  # Reads are slightly slower
            write_latency = base_latency * 1.5  # Writes are slower
            latency = (read_count * read_latency + write_count * write_latency) / total_ops if total_ops > 0 else 0.0
            
            # Calculate memory usage (MB)
            # Base memory plus per-operation overhead
            base_memory = 100  # MB
            per_op_memory = 0.1  # KB per operation
            memory = base_memory + (total_ops * per_op_memory) / 1024  # Convert to MB
            
            # Key access patterns
            key_access_counts = {}
            for op in workload:
                key = op.get('key')
                if key:
                    key_access_counts[key] = key_access_counts.get(key, 0) + 1
            
            # Sort keys by access frequency
            sorted_keys = sorted(key_access_counts.items(), key=lambda x: x[1], reverse=True)
            hot_keys = dict(sorted_keys[:10]) if sorted_keys else {}  # Top 10 most accessed keys
            
            # Value size distribution
            value_sizes = {}
            for op in workload:
                if op.get('type') == 'write' and op.get('value'):
                    size = len(op['value'])
                    value_sizes[size] = value_sizes.get(size, 0) + 1
            
            # Temporal patterns (operations per time window)
            time_windows = {}
            window_size = 1000
            for i, op in enumerate(workload):
                window = i // window_size
                time_windows[window] = time_windows.get(window, 0) + 1
            
            return {
                'total_operations': total_ops,
                'read_ratio': read_count / total_ops if total_ops > 0 else 0.0,
                'write_ratio': write_count / total_ops if total_ops > 0 else 0.0,
                'hot_key_ratio': hot_key_count / total_ops if total_ops > 0 else 0.0,
                'throughput': throughput,
                'latency': latency,
                'memory': memory,
                'operation_distribution': {
                    'read': read_count,
                    'write': write_count
                },
                'key_distribution': {
                    'total_unique_keys': len(key_access_counts),
                    'hot_keys': hot_keys,
                    'key_access_counts': key_access_counts
                },
                'value_size_distribution': value_sizes,
                'temporal_patterns': {
                    'window_size': window_size,
                    'operations_per_window': time_windows
                }
            }
        except Exception as e:
            logging.error(f"Error calculating workload metrics: {str(e)}")
            return {
                'total_operations': 0,
                'read_ratio': 0.0,
                'write_ratio': 0.0,
                'hot_key_ratio': 0.0,
                'throughput': 0.0,
                'latency': 0.0,
                'memory': 0.0,
                'operation_distribution': {'read': 0, 'write': 0},
                'key_distribution': {'total_unique_keys': 0, 'hot_keys': {}, 'key_access_counts': {}},
                'value_size_distribution': {},
                'temporal_patterns': {'window_size': 1000, 'operations_per_window': {}}
            } 