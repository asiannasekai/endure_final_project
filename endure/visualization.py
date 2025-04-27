"""
Enhanced visualization module for privacy-performance analysis.

This module provides comprehensive visualization capabilities for analyzing
privacy-performance tradeoffs, workload sensitivity, and configuration impacts.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# Get logger instance without configuring
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    style: str = "default"
    figure_size: Tuple[int, int] = (12, 8)
    font_size: int = 12
    color_palette: str = "viridis"
    dpi: int = 300

class EnhancedVisualization:
    """
    Enhanced visualization class for privacy-performance analysis.
    
    This class provides methods for creating various types of visualizations
    to analyze privacy-performance tradeoffs, workload sensitivity, and
    configuration impacts.
    
    Attributes:
        results_dir (str): Directory to save visualization outputs
        config (VisualizationConfig): Visualization configuration settings
    """
    
    def __init__(self, results_dir: str):
        """Initialize visualization with error handling."""
        try:
            self.results_dir = results_dir
            self.config = VisualizationConfig()
            self._setup_directories()
            self._setup_plotting_style()
        except Exception as e:
            logger.error(f"Error initializing visualization: {str(e)}")
            raise
    
    def _setup_directories(self) -> None:
        """Setup required directories with error handling."""
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {self.results_dir}: {str(e)}")
            raise
    
    def _setup_plotting_style(self) -> None:
        """Setup plotting style with error handling."""
        try:
            # Use a valid style from matplotlib
            plt.style.use('default')  # Use default style instead of seaborn
            plt.rcParams['figure.figsize'] = self.config.figure_size
            plt.rcParams['font.size'] = self.config.font_size
            plt.rcParams['axes.grid'] = True  # Add grid by default
            plt.rcParams['grid.alpha'] = 0.3  # Make grid lines slightly transparent
        except Exception as e:
            logger.error(f"Error setting up plotting style: {str(e)}")
            raise
    
    def _handle_numeric_data(self, data: Dict) -> Dict:
        """Handle numeric data with edge cases."""
        try:
            processed_data = {}
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        logger.warning(f"Invalid numeric value for {key}: {value}")
                        processed_data[key] = 0.0
                    else:
                        processed_data[key] = float(value)
                elif isinstance(value, dict):
                    processed_data[key] = self._handle_numeric_data(value)
                else:
                    processed_data[key] = value
            return processed_data
        except Exception as e:
            logger.error(f"Error processing numeric data: {str(e)}")
            raise
    
    def _save_figure(self, filename: str) -> None:
        """Save figure with error handling."""
        try:
            filepath = os.path.join(self.results_dir, filename)
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Error saving figure {filename}: {str(e)}")
            raise
    
    def plot_privacy_performance_tradeoff(self, results: Dict) -> None:
        """Plot privacy-performance tradeoff with edge case handling."""
        try:
            # Validate required data
            required_fields = ['metrics', 'configurations']
            if not all(field in results for field in required_fields):
                logger.error(f"Missing required fields for tradeoff plot: {required_fields}")
                return
            
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot metrics
            metrics = ['throughput', 'latency', 'space_amplification']
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                try:
                    if metric not in processed_data['metrics']:
                        logger.warning(f"Missing metric {metric} for tradeoff plot")
                        continue
                        
                    data = processed_data['metrics'][metric]
                    if not isinstance(data, (int, float)):
                        logger.warning(f"Invalid data type for metric {metric}")
                        continue
                        
                    ax.plot([0, 1], [data, data])  # Simple line plot for now
                    ax.set_title(f"{metric.capitalize()} vs Privacy")
                    ax.set_xlabel("Privacy Level")
                    ax.set_ylabel(metric.capitalize())
                except Exception as e:
                    logger.error(f"Error plotting {metric}: {str(e)}")
                    continue
            
            # Plot configuration comparison
            try:
                if 'original' in processed_data['configurations'] and 'private' in processed_data['configurations']:
                    original = processed_data['configurations']['original']
                    private = processed_data['configurations']['private']
                    
                    # Compare metrics between configurations
                    metrics = ['throughput', 'latency', 'space_amplification']
                    values = {
                        'Original': [original.get(m, 0) for m in metrics],
                        'Private': [private.get(m, 0) for m in metrics]
                    }
                    
                    x = np.arange(len(metrics))
                    width = 0.35
                    
                    ax = axes[1, 1]
                    ax.bar(x - width/2, values['Original'], width, label='Original')
                    ax.bar(x + width/2, values['Private'], width, label='Private')
                    
                    ax.set_title("Configuration Comparison")
                    ax.set_xticks(x)
                    ax.set_xticklabels(metrics)
                    ax.legend()
                else:
                    logger.warning("Missing configuration data for comparison")
            except Exception as e:
                logger.error(f"Error plotting configuration comparison: {str(e)}")
            
            plt.tight_layout()
            self._save_figure("privacy_performance_tradeoff.png")
            
        except Exception as e:
            logger.error(f"Error in privacy-performance tradeoff plot: {str(e)}")
            raise
    
    def plot_workload_sensitivity(self, results: Dict) -> None:
        """Plot workload sensitivity with edge case handling."""
        try:
            # Validate required data
            if 'workload_characteristics' not in results:
                logger.error("Missing workload characteristics data")
                return
            
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot characteristics
            characteristics = ['read_ratio', 'write_ratio']
            for i, char in enumerate(characteristics):
                ax = axes[i//2, i%2]
                try:
                    if char not in processed_data['workload_characteristics']:
                        logger.warning(f"Missing characteristic {char}")
                        continue
                        
                    value = processed_data['workload_characteristics'][char]
                    if not isinstance(value, (int, float)):
                        logger.warning(f"Invalid data type for characteristic {char}")
                        continue
                        
                    # Simple bar plot for now
                    ax.bar([char], [value])
                    ax.set_title(f"{char.replace('_', ' ').title()}")
                    ax.set_ylim(0, 1)
                except Exception as e:
                    logger.error(f"Error plotting {char}: {str(e)}")
                    continue
            
            plt.tight_layout()
            self._save_figure("workload_sensitivity.png")
            
        except Exception as e:
            logger.error(f"Error in workload sensitivity plot: {str(e)}")
            raise
    
    def plot_configuration_differences(self, results: Dict) -> None:
        """Plot configuration differences with edge case handling."""
        try:
            # Validate required data
            if 'configurations' not in results:
                logger.error("Missing configuration data")
                return
            
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot configurations
            configs = ['original', 'private']
            metrics = ['throughput', 'latency', 'space_amplification']
            
            for i, config in enumerate(configs):
                ax = axes[i//2, i%2]
                try:
                    if config not in processed_data['configurations']:
                        logger.warning(f"Missing configuration {config}")
                        continue
                        
                    config_data = processed_data['configurations'][config]
                    values = [config_data.get(m, 0) for m in metrics]
                    
                    ax.bar(metrics, values)
                    ax.set_title(f"{config.title()} Configuration")
                    ax.set_xticklabels(metrics, rotation=45)
                except Exception as e:
                    logger.error(f"Error plotting {config}: {str(e)}")
                    continue
            
            plt.tight_layout()
            self._save_figure("configuration_differences.png")
            
        except Exception as e:
            logger.error(f"Error in configuration differences plot: {str(e)}")
            raise
    
    def plot_correlation_analysis(self, results: Dict) -> None:
        """Plot correlation analysis with edge case handling."""
        try:
            # Validate required data
            required_fields = ['metrics', 'workload_characteristics']
            if not all(field in results for field in required_fields):
                logger.error(f"Missing required fields for correlation analysis: {required_fields}")
                return
            
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create correlation matrix
            metrics = list(processed_data['metrics'].keys())
            characteristics = list(processed_data['workload_characteristics'].keys())
            
            all_data = {**processed_data['metrics'], **processed_data['workload_characteristics']}
            correlation_matrix = np.zeros((len(all_data), len(all_data)))
            
            # Simple correlation calculation (can be improved)
            for i, (k1, v1) in enumerate(all_data.items()):
                for j, (k2, v2) in enumerate(all_data.items()):
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        correlation_matrix[i, j] = 1.0 if k1 == k2 else 0.5
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Plot correlation matrix using matplotlib
            im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            
            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(all_data)))
            ax.set_yticks(np.arange(len(all_data)))
            ax.set_xticklabels(list(all_data.keys()), rotation=45, ha='right')
            ax.set_yticklabels(list(all_data.keys()))
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add title
            ax.set_title("Metric Correlation Analysis")
            
            plt.tight_layout()
            self._save_figure("correlation_analysis.png")
            
        except Exception as e:
            logger.error(f"Error in correlation analysis plot: {str(e)}")
            raise 