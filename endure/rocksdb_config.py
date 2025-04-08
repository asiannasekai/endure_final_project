from dataclasses import dataclass
from typing import Dict, Any
import json
import os

@dataclass
class RocksDBConfig:
    # Basic configuration
    db_path: str
    create_if_missing: bool = True
    error_if_exists: bool = False
    
    # Performance tuning
    max_background_jobs: int = 4
    max_subcompactions: int = 1
    max_open_files: int = -1  # -1 means unlimited
    
    # Memtable settings
    write_buffer_size: int = 64 * 1024 * 1024  # 64MB
    max_write_buffer_number: int = 3
    min_write_buffer_number_to_merge: int = 1
    
    # Level style compaction
    level0_file_num_compaction_trigger: int = 4
    level0_slowdown_writes_trigger: int = 20
    level0_stop_writes_trigger: int = 36
    max_bytes_for_level_base: int = 256 * 1024 * 1024  # 256MB
    target_file_size_base: int = 64 * 1024 * 1024  # 64MB
    
    # Compression
    compression_type: str = "snappy"  # Options: "none", "snappy", "zlib", "bzip2", "lz4", "lz4hc", "xpress", "zstd"
    
    # Bloom filter
    optimize_filters_for_hits: bool = False
    bloom_locality: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format for RocksDB options."""
        return {
            "create_if_missing": self.create_if_missing,
            "error_if_exists": self.error_if_exists,
            "max_background_jobs": self.max_background_jobs,
            "max_subcompactions": self.max_subcompactions,
            "max_open_files": self.max_open_files,
            "write_buffer_size": self.write_buffer_size,
            "max_write_buffer_number": self.max_write_buffer_number,
            "min_write_buffer_number_to_merge": self.min_write_buffer_number_to_merge,
            "level0_file_num_compaction_trigger": self.level0_file_num_compaction_trigger,
            "level0_slowdown_writes_trigger": self.level0_slowdown_writes_trigger,
            "level0_stop_writes_trigger": self.level0_stop_writes_trigger,
            "max_bytes_for_level_base": self.max_bytes_for_level_base,
            "target_file_size_base": self.target_file_size_base,
            "compression_type": self.compression_type,
            "optimize_filters_for_hits": self.optimize_filters_for_hits,
            "bloom_locality": self.bloom_locality
        }

    def save(self, filename: str) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load(self, filename: str) -> None:
        """Load configuration from file."""
        with open(filename, 'r') as f:
            config_dict = json.load(f)
            self.db_path = config_dict.get("db_path", "rocksdb_data")
            self.create_if_missing = config_dict.get("create_if_missing", True)
            self.error_if_exists = config_dict.get("error_if_exists", False)
            self.max_background_jobs = config_dict.get("max_background_jobs", 4)
            self.max_subcompactions = config_dict.get("max_subcompactions", 1)
            self.max_open_files = config_dict.get("max_open_files", -1)
            self.write_buffer_size = config_dict.get("write_buffer_size", 64 * 1024 * 1024)
            self.max_write_buffer_number = config_dict.get("max_write_buffer_number", 3)
            self.min_write_buffer_number_to_merge = config_dict.get("min_write_buffer_number_to_merge", 1)
            self.level0_file_num_compaction_trigger = config_dict.get("level0_file_num_compaction_trigger", 4)
            self.level0_slowdown_writes_trigger = config_dict.get("level0_slowdown_writes_trigger", 20)
            self.level0_stop_writes_trigger = config_dict.get("level0_stop_writes_trigger", 36)
            self.max_bytes_for_level_base = config_dict.get("max_bytes_for_level_base", 256 * 1024 * 1024)
            self.target_file_size_base = config_dict.get("target_file_size_base", 64 * 1024 * 1024)
            self.compression_type = config_dict.get("compression_type", "snappy")
            self.optimize_filters_for_hits = config_dict.get("optimize_filters_for_hits", False)
            self.bloom_locality = config_dict.get("bloom_locality", 0)

    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.to_dict()

# Default configurations for different workloads
DEFAULT_CONFIG = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/default"
)

# Perturbed versions of default config
DEFAULT_PERTURBED_1 = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/default_perturbed_1",
    write_buffer_size=32 * 1024 * 1024,  # 32MB (half of default)
    level0_file_num_compaction_trigger=2  # half of default
)

DEFAULT_PERTURBED_2 = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/default_perturbed_2",
    write_buffer_size=128 * 1024 * 1024,  # 128MB (double of default)
    level0_file_num_compaction_trigger=8  # double of default
)

# Write-intensive configuration
WRITE_INTENSIVE_CONFIG = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/write_intensive",
    write_buffer_size=128 * 1024 * 1024,  # 128MB
    max_write_buffer_number=6,
    level0_file_num_compaction_trigger=8,
    level0_slowdown_writes_trigger=32,
    level0_stop_writes_trigger=64
)

# Perturbed versions of write-intensive config
WRITE_INTENSIVE_PERTURBED_1 = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/write_intensive_perturbed_1",
    write_buffer_size=64 * 1024 * 1024,  # 64MB (half of write-intensive)
    max_write_buffer_number=3,  # half of write-intensive
    level0_file_num_compaction_trigger=4  # half of write-intensive
)

WRITE_INTENSIVE_PERTURBED_2 = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/write_intensive_perturbed_2",
    write_buffer_size=256 * 1024 * 1024,  # 256MB (double of write-intensive)
    max_write_buffer_number=12,  # double of write-intensive
    level0_file_num_compaction_trigger=16  # double of write-intensive
)

# Read-intensive configuration
READ_INTENSIVE_CONFIG = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/read_intensive",
    optimize_filters_for_hits=True,
    bloom_locality=1,
    max_open_files=5000
)

# Perturbed versions of read-intensive config
READ_INTENSIVE_PERTURBED_1 = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/read_intensive_perturbed_1",
    optimize_filters_for_hits=False,  # opposite of read-intensive
    bloom_locality=0,  # opposite of read-intensive
    max_open_files=2500  # half of read-intensive
)

READ_INTENSIVE_PERTURBED_2 = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/read_intensive_perturbed_2",
    optimize_filters_for_hits=True,
    bloom_locality=2,  # double of read-intensive
    max_open_files=10000  # double of read-intensive
)

# Balanced configuration
BALANCED_CONFIG = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/balanced",
    write_buffer_size=96 * 1024 * 1024,  # 96MB
    max_write_buffer_number=4,
    level0_file_num_compaction_trigger=6,
    optimize_filters_for_hits=True
)

# Perturbed versions of balanced config
BALANCED_PERTURBED_1 = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/balanced_perturbed_1",
    write_buffer_size=48 * 1024 * 1024,  # 48MB (half of balanced)
    max_write_buffer_number=2,  # half of balanced
    level0_file_num_compaction_trigger=3  # half of balanced
)

BALANCED_PERTURBED_2 = RocksDBConfig(
    db_path="/projectnb/cs561/students/asiannah/rocksdb_data/balanced_perturbed_2",
    write_buffer_size=192 * 1024 * 1024,  # 192MB (double of balanced)
    max_write_buffer_number=8,  # double of balanced
    level0_file_num_compaction_trigger=12  # double of balanced
)

def create_config_from_workload(workload_metrics: Dict[str, float]) -> Dict[str, Any]:
    """Create RocksDB configuration based on workload characteristics."""
    read_ratio = workload_metrics.get("read_ratio", 0.5)
    write_ratio = workload_metrics.get("write_ratio", 0.5)
    hot_key_ratio = workload_metrics.get("hot_key_ratio", 0.2)
    operation_count = workload_metrics.get("operation_count", 100000)
    
    # Base configuration
    config = {
        "db_path": "rocksdb_data",
        "create_if_missing": True,
        "error_if_exists": False,
        "paranoid_checks": True,
        "max_open_files": 5000,
        "use_direct_reads": True,
        "use_direct_io_for_flush_and_compaction": True,
        "allow_mmap_reads": True,
        "allow_mmap_writes": True,
        "is_fd_close_on_exec": True,
        "stats_dump_period_sec": 600,
        "max_background_jobs": 4,
        "max_subcompactions": 4,
        "use_fsync": False,
        "bytes_per_sync": 1048576,
        "compaction_style": "level",
        "compression": "lz4",
        "bottommost_compression": "zstd",
        "compression_opts": {
            "window_bits": -14,
            "level": 32767,
            "strategy": 0,
            "max_dict_bytes": 0
        },
        "bottommost_compression_opts": {
            "window_bits": -14,
            "level": 32767,
            "strategy": 0,
            "max_dict_bytes": 0
        },
        "level_compaction_dynamic_level_bytes": True,
        "optimize_filters_for_hits": True,
        "memtable_prefix_bloom_size_ratio": 0.1,
        "memtable_whole_key_filtering": True,
        "memtable_huge_page_size": 2 * 1024 * 1024,
        "max_write_buffer_number": 6,
        "min_write_buffer_number_to_merge": 2,
        "max_background_compactions": 4,
        "max_background_flushes": 2,
        "max_bytes_for_level_base": 256 * 1024 * 1024,
        "max_bytes_for_level_multiplier": 10,
        "target_file_size_base": 64 * 1024 * 1024,
        "target_file_size_multiplier": 1,
        "level0_file_num_compaction_trigger": 4,
        "level0_slowdown_writes_trigger": 20,
        "level0_stop_writes_trigger": 36,
        "soft_pending_compaction_bytes_limit": 64 * 1024 * 1024 * 1024,
        "hard_pending_compaction_bytes_limit": 256 * 1024 * 1024 * 1024,
        "max_compaction_bytes": 256 * 1024 * 1024,
        "compaction_pri": "kMinOverlappingRatio"
    }
    
    # Adjust configuration based on workload characteristics
    if read_ratio > 0.7:  # Read-intensive workload
        config.update({
            "max_open_files": 10000,
            "optimize_filters_for_hits": True,
            "memtable_prefix_bloom_size_ratio": 0.2,
            "max_write_buffer_number": 4,
            "level0_file_num_compaction_trigger": 8
        })
    elif write_ratio > 0.7:  # Write-intensive workload
        config.update({
            "max_write_buffer_number": 8,
            "min_write_buffer_number_to_merge": 1,
            "level0_file_num_compaction_trigger": 2,
            "max_bytes_for_level_base": 512 * 1024 * 1024
        })
    
    if hot_key_ratio > 0.3:  # High hot key ratio
        config.update({
            "memtable_prefix_bloom_size_ratio": 0.3,
            "optimize_filters_for_hits": True,
            "max_open_files": 10000
        })
    
    if operation_count > 1000000:  # Large operation count
        config.update({
            "max_background_jobs": 8,
            "max_subcompactions": 8,
            "max_bytes_for_level_base": 1024 * 1024 * 1024
        })
    
    return config

def compare_configurations(original_config: Dict[str, Any], private_config: Dict[str, Any]) -> Dict[str, float]:
    """Compare two configurations and calculate differences."""
    differences = {}
    for param in original_config:
        if param in private_config and isinstance(original_config[param], (int, float)):
            original_val = float(original_config[param])
            private_val = float(private_config[param])
            if original_val != 0:
                differences[param] = abs(original_val - private_val) / original_val
    return differences 