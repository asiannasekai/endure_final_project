"""
Run experiments for the Endure project with enhanced resource management and error recovery.
"""

import logging
import os
import json
import sys
import psutil
import shutil
import signal
import atexit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, List
from endure.config import ConfigManager
from endure.cli import validate_input_data
from endure.analysis import PrivacyAnalysis, SensitivityAnalysis, PerformanceAnalysis
from datetime import datetime

# Global state for cleanup
TEMP_FILES = set()
RUNNING_PROCESSES = set()

def cleanup_resources():
    """Cleanup temporary files and processes on exit."""
    for file in TEMP_FILES:
        try:
            if os.path.exists(file):
                if os.path.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)
        except Exception as e:
            logging.error(f"Error cleaning up {file}: {str(e)}")
    
    for process in RUNNING_PROCESSES:
        try:
            process.terminate()
        except Exception as e:
            logging.error(f"Error terminating process {process.pid}: {str(e)}")

def setup_logging():
    """Setup logging configuration with rotation."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'experiments.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_system_resources() -> bool:
    """Check if system has sufficient resources."""
    try:
        # Check disk space (need at least 1GB free)
        disk = shutil.disk_usage('.')
        if disk.free < 1_000_000_000:
            logging.error(f"Insufficient disk space: {disk.free / 1_000_000_000:.2f}GB free")
            return False
        
        # Check memory (need at least 1GB free)
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024 * 1024 * 1024)
        available_memory_gb = memory.available / (1024 * 1024 * 1024)
        
        # Adjust threshold based on total memory
        memory_threshold_gb = min(1.0, total_memory_gb * 0.15)  # 1GB or 15% of total, whichever is smaller
        
        if available_memory_gb < memory_threshold_gb:
            logging.error(f"Insufficient memory: {available_memory_gb:.1f}GB available (need {memory_threshold_gb:.1f}GB)")
            return False
        
        # Check CPU usage (should be below 80%)
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > 80:
            logging.error(f"High CPU usage: {cpu_percent}%")
            return False
            
        return True
    except Exception as e:
        logging.error(f"Error checking system resources: {str(e)}")
        return False

def validate_config(config: ConfigManager) -> bool:
    """Validate configuration settings with enhanced checks."""
    try:
        # Basic configuration validation
        if not hasattr(config, 'analysis') or not config.analysis.validate():
            logging.error("Invalid analysis configuration")
            return False
            
        if not hasattr(config, 'visualization') or not config.visualization.validate():
            logging.error("Invalid visualization configuration")
            return False
            
        if not hasattr(config, 'privacy') or not config.privacy.validate():
            logging.error("Invalid privacy configuration")
            return False
            
        if not hasattr(config, 'performance') or not config.performance.validate():
            logging.error("Invalid performance configuration")
            return False
            
        # Validate relationships between configurations
        if not hasattr(config.privacy, 'epsilon_range') or not (0 < config.privacy.epsilon_range[0] < config.privacy.epsilon_range[1]):
            logging.error(f"Invalid epsilon range: {config.privacy.epsilon_range}")
            return False
        
        # Validate visualization settings
        if not hasattr(config.visualization, 'style') or config.visualization.style not in ['default', 'whitegrid', 'darkgrid']:
            logging.error(f"Invalid visualization style: {config.visualization.style}")
            return False
            
        # Validate visualization output settings
        if not hasattr(config.visualization, 'output_dir'):
            config.visualization.output_dir = 'results/visualization'
            
        if not hasattr(config.visualization, 'figure_size'):
            config.visualization.figure_size = (12, 8)
            
        if not hasattr(config.visualization, 'file_format'):
            config.visualization.file_format = 'png'
            
        # Validate performance configuration
        if not hasattr(config.performance, 'operation_count'):
            config.performance.operation_count = 100000  # Default value
            
        return True
    except Exception as e:
        logging.error(f"Error validating configuration: {str(e)}")
        return False

def setup_directories() -> bool:
    """Setup required directories with error handling and cleanup."""
    try:
        # Define base directories
        base_dirs = {
            'data': 'data',
            'results': 'results',
            'checkpoints': 'checkpoints',
            'logs': 'logs',
            'temp': 'temp',
            'tests': 'tests',
            'backup': 'backup'
        }
        
        # Define subdirectories
        subdirs = {
            'results': [
                'privacy_results',
                'sensitivity_results',
                'performance_results',
                'visualization',
                'raw_data'
            ],
            'checkpoints': [
                'privacy',
                'sensitivity',
                'performance',
                'workload'
            ],
            'tests': [
                'unit',
                'integration',
                'performance',
                'results'
            ],
            'backup': [
                'results',
                'checkpoints',
                'logs'
            ]
        }
        
        # Create base directories
        for dir_name, dir_path in base_dirs.items():
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
            
            # Create subdirectories if they exist
            if dir_name in subdirs:
                for subdir in subdirs[dir_name]:
                    subdir_path = os.path.join(dir_path, subdir)
                    os.makedirs(subdir_path, exist_ok=True)
                    logger.info(f"Created subdirectory: {subdir_path}")
            
            # Verify write permissions
            if not os.access(dir_path, os.W_OK):
                raise PermissionError(f"No write permission for {dir_path}")
            
            # Check disk space for results and checkpoints
            if dir_name in ['results', 'checkpoints', 'backup']:
                disk = shutil.disk_usage(dir_path)
                free_gb = disk.free / (1024 * 1024 * 1024)
                if free_gb < 1:  # Need at least 1GB free
                    raise RuntimeError(f"Insufficient disk space in {dir_path}: {free_gb:.2f}GB available")
        
        # Register temp directory for cleanup
        TEMP_FILES.add('temp')
        return True
        
    except PermissionError:
        logger.error("Permission denied when creating directories")
        return False
    except OSError as e:
        logger.error(f"Error creating directories: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error setting up directories: {str(e)}")
        return False

def cleanup_directories() -> None:
    """Clean up temporary directories and files."""
    try:
        # Clean up temp directory
        temp_dir = 'temp'
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except Exception as e:
                        logger.warning(f"Failed to remove file {name}: {str(e)}")
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except Exception as e:
                        logger.warning(f"Failed to remove directory {name}: {str(e)}")
            try:
                os.rmdir(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temp directory: {str(e)}")
        
        # Clean up checkpoint files
        checkpoint_dir = 'checkpoints'
        if os.path.exists(checkpoint_dir):
            for root, dirs, files in os.walk(checkpoint_dir):
                for name in files:
                    if name.endswith('.json'):
                        try:
                            os.remove(os.path.join(root, name))
                        except Exception as e:
                            logger.warning(f"Failed to remove checkpoint file {name}: {str(e)}")
        
        # Move old results to backup
        backup_dir = 'backup'
        if os.path.exists(backup_dir):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_subdir = os.path.join(backup_dir, timestamp)
            os.makedirs(backup_subdir, exist_ok=True)
            
            # Move old results
            for old_dir in ['privacy_results', 'sensitivity_results', 'performance_results']:
                if os.path.exists(old_dir):
                    try:
                        shutil.move(old_dir, os.path.join(backup_subdir, old_dir))
                    except Exception as e:
                        logger.warning(f"Failed to move {old_dir} to backup: {str(e)}")
        
        logger.info("Directory cleanup completed")
    except Exception as e:
        logger.error(f"Error during directory cleanup: {str(e)}")

def setup_visualization_directories() -> bool:
    """Setup directories for visualization outputs."""
    try:
        vis_dirs = {
            'results/visualization': [
                'privacy',
                'sensitivity',
                'performance',
                'comparison'
            ]
        }
        
        for base_dir, subdirs in vis_dirs.items():
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"Created visualization directory: {base_dir}")
            
            for subdir in subdirs:
                subdir_path = os.path.join(base_dir, subdir)
                os.makedirs(subdir_path, exist_ok=True)
                logger.info(f"Created visualization subdirectory: {subdir_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error setting up visualization directories: {str(e)}")
        return False

def load_input_data(checkpoint: bool = True) -> Optional[dict]:
    """Load and validate input data with checkpoint support."""
    try:
        input_file = 'data/input_data.json'
        checkpoint_file = 'checkpoints/input_data.json'
        
        # Try loading from checkpoint first
        if checkpoint and os.path.exists(checkpoint_file):
            logging.info("Loading from checkpoint...")
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                if validate_input_data(data):
                    return data
                logging.warning("Checkpoint data invalid, falling back to input file")
        
        if not os.path.exists(input_file):
            logging.warning(f"Input file not found: {input_file}. Using default values.")
            data = {
                'workload_characteristics': {
                    'read_ratio': 0.7,
                    'write_ratio': 0.3,
                    'key_size': 16,
                    'value_size': 100,
                    'operation_count': 100000,
                    'hot_key_ratio': 0.2,
                    'hot_key_count': 100
                }
            }
        else:
            with open(input_file, 'r') as f:
                data = json.load(f)
        
        if not isinstance(data, dict):
            logging.error("Input data must be a JSON object")
            return None
        
        # Initialize workload characteristics if missing
        if 'workload_characteristics' not in data:
            logging.info("No workload characteristics found, using defaults")
            data['workload_characteristics'] = {}
        
        # Validate and normalize the data
        if not validate_input_data(data):
            logging.error("Failed to validate and normalize input data")
            return None
        
        # Save checkpoint
        if checkpoint:
            os.makedirs('checkpoints', exist_ok=True)
            with open(checkpoint_file, 'w') as f:
                json.dump(data, f)
        
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in input file: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Error loading input data: {str(e)}")
        return None

def run_analysis_with_retry(analysis_type: str, input_data: Dict, config: ConfigManager, 
                          output_dir: str, max_retries: int = 3) -> bool:
    """Run analysis with retry logic for transient failures."""
    for attempt in range(max_retries):
        try:
            # Create appropriate analysis instance
            if analysis_type == 'privacy':
                analysis = PrivacyAnalysis(config, output_dir)
                # Privacy analysis uses run_privacy_sweep
                result = analysis.run_privacy_sweep(
                    input_data['workload_characteristics'],
                    epsilon_range=(0.1, 5.0)  # Default epsilon range
                )
                if not result or not result.validate():
                    raise ValueError("Invalid privacy analysis results")
            elif analysis_type == 'sensitivity':
                analysis = SensitivityAnalysis(config)
                analysis.run_analysis(input_data)
            elif analysis_type == 'performance':
                analysis = PerformanceAnalysis(config)
                analysis.run_analysis(input_data)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Clean up analysis resources
            analysis.cleanup()
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                continue
            logging.error(f"All {max_retries} attempts failed for {analysis_type} analysis")
            return False

def main():
    """Main entry point with enhanced error handling and resource management."""
    try:
        # Register cleanup handler
        atexit.register(cleanup_resources)
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
        
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Check system resources
        logger.info("Checking system resources...")
        if not check_system_resources():
            raise Exception("Insufficient system resources")
            
        # Setup directories
        logger.info("Setting up directories...")
        if not setup_directories():
            raise Exception("Failed to setup directories")
            
        # Load input data
        logger.info("Loading input data...")
        input_data = load_input_data()
        if not input_data:
            raise Exception("Failed to load input data")
            
        # Load configuration
        logger.info("Loading configuration...")
        config = ConfigManager()
        if not validate_config(config):
            raise Exception("Invalid configuration")
            
        # Run analyses
        logger.info("Starting analyses...")
        
        # Run privacy analysis (cannot be parallelized with others)
        logger.info("Running privacy analysis...")
        if not run_analysis_with_retry('privacy', input_data, config, 'results/privacy_results'):
            raise Exception("Privacy analysis failed")
            
        # Run sensitivity and performance analyses in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_sensitivity = executor.submit(
                run_analysis_with_retry,
                'sensitivity',
                input_data,
                config,
                'results/sensitivity_results'
            )
            future_performance = executor.submit(
                run_analysis_with_retry,
                'performance',
                input_data,
                config,
                'results/performance_results'
            )
            
            # Wait for completion
            sensitivity_success = future_sensitivity.result()
            performance_success = future_performance.result()
            
            if not sensitivity_success:
                raise Exception("Sensitivity analysis failed")
            if not performance_success:
                raise Exception("Performance analysis failed")
                
        logger.info("All analyses completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 