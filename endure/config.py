"""
Configuration management module for the Endure project.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class AnalysisConfig:
    """Configuration for analysis settings."""
    results_dir: str = "results"
    log_level: str = "INFO"
    save_raw_data: bool = True
    validate_results: bool = True
    max_workers: int = 4
    
    def validate(self) -> bool:
        """Validate analysis configuration."""
        if not isinstance(self.max_workers, int) or self.max_workers < 1:
            logging.error("max_workers must be a positive integer")
            return False
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logging.error("Invalid log level")
            return False
        return True

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    style: str = "whitegrid"
    figure_size: tuple = (12, 8)
    font_size: int = 12
    color_palette: str = "husl"
    dpi: int = 300
    save_format: str = "png"
    
    def validate(self) -> bool:
        """Validate visualization configuration."""
        if not isinstance(self.figure_size, tuple) or len(self.figure_size) != 2:
            logging.error("figure_size must be a tuple of length 2")
            return False
        if not all(isinstance(x, (int, float)) and x > 0 for x in self.figure_size):
            logging.error("figure_size values must be positive numbers")
            return False
        if not isinstance(self.font_size, int) or self.font_size < 1:
            logging.error("font_size must be a positive integer")
            return False
        if not isinstance(self.dpi, int) or self.dpi < 1:
            logging.error("dpi must be a positive integer")
            return False
        return True

@dataclass
class PrivacyConfig:
    """Configuration for privacy settings."""
    epsilon_range: tuple = (0.1, 10.0)
    epsilon_steps: int = 10
    delta: float = 1e-5
    noise_scale: float = 1.0
    
    def validate(self) -> bool:
        """Validate privacy configuration."""
        if not isinstance(self.epsilon_range, tuple) or len(self.epsilon_range) != 2:
            logging.error("epsilon_range must be a tuple of length 2")
            return False
        if not all(isinstance(x, (int, float)) and x > 0 for x in self.epsilon_range):
            logging.error("epsilon_range values must be positive numbers")
            return False
        if not isinstance(self.epsilon_steps, int) or self.epsilon_steps < 1:
            logging.error("epsilon_steps must be a positive integer")
            return False
        if not isinstance(self.delta, (int, float)) or self.delta <= 0:
            logging.error("delta must be a positive number")
            return False
        if not isinstance(self.noise_scale, (int, float)) or self.noise_scale <= 0:
            logging.error("noise_scale must be a positive number")
            return False
        return True

@dataclass
class PerformanceConfig:
    """Configuration for performance settings."""
    metrics: tuple = ("throughput", "latency", "space_amplification")
    workload_types: tuple = ("read_heavy", "write_heavy", "balanced")
    sample_size: int = 1000
    
    def validate(self) -> bool:
        """Validate performance configuration."""
        if not isinstance(self.metrics, tuple) or not all(isinstance(x, str) for x in self.metrics):
            logging.error("metrics must be a tuple of strings")
            return False
        if not isinstance(self.workload_types, tuple) or not all(isinstance(x, str) for x in self.workload_types):
            logging.error("workload_types must be a tuple of strings")
            return False
        if not isinstance(self.sample_size, int) or self.sample_size < 1:
            logging.error("sample_size must be a positive integer")
            return False
        return True

class ConfigManager:
    """
    Configuration manager for the Endure project.
    
    This class manages all configuration settings for the project,
    including analysis, visualization, privacy, and performance settings.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file (Optional[str]): Path to configuration file
        """
        self.analysis = AnalysisConfig()
        self.visualization = VisualizationConfig()
        self.privacy = PrivacyConfig()
        self.performance = PerformanceConfig()
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from file.
        
        Args:
            config_file (str): Path to configuration file
        """
        try:
            if not os.path.exists(config_file):
                logging.error(f"Configuration file not found: {config_file}")
                return
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations
            if 'analysis' in config_data:
                self._update_config(self.analysis, config_data['analysis'])
            if 'visualization' in config_data:
                self._update_config(self.visualization, config_data['visualization'])
            if 'privacy' in config_data:
                self._update_config(self.privacy, config_data['privacy'])
            if 'performance' in config_data:
                self._update_config(self.performance, config_data['performance'])
            
            # Validate configurations
            if not all([
                self.analysis.validate(),
                self.visualization.validate(),
                self.privacy.validate(),
                self.performance.validate()
            ]):
                logging.error("Invalid configuration values")
                return
            
            logging.info(f"Configuration loaded from {config_file}")
            
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in configuration file: {config_file}")
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
    
    def save_config(self, config_file: str) -> None:
        """
        Save configuration to file.
        
        Args:
            config_file (str): Path to save configuration file
        """
        try:
            config_data = {
                'analysis': asdict(self.analysis),
                'visualization': asdict(self.visualization),
                'privacy': asdict(self.privacy),
                'performance': asdict(self.performance)
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logging.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            raise
    
    def _update_config(self, config_obj: Any, config_data: Dict) -> None:
        """
        Update configuration object with new data.
        
        Args:
            config_obj: Configuration object to update
            config_data: New configuration data
        """
        for key, value in config_data.items():
            if hasattr(config_obj, key):
                # Convert lists to tuples for specific fields
                if isinstance(value, list):
                    if key in ["epsilon_range", "metrics", "workload_types", "figure_size"]:
                        value = tuple(value)
                setattr(config_obj, key, value)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get all configuration settings.
        
        Returns:
            Dict[str, Any]: All configuration settings
        """
        return {
            'analysis': asdict(self.analysis),
            'visualization': asdict(self.visualization),
            'privacy': asdict(self.privacy),
            'performance': asdict(self.performance)
        } 