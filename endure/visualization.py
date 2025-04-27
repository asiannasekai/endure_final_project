"""
Enhanced visualization module for privacy-performance analysis.

This module provides comprehensive visualization capabilities for analyzing
privacy-performance tradeoffs, workload sensitivity, and configuration impacts.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    style: str = "whitegrid"
    figure_size: Tuple[int, int] = (12, 8)
    font_size: int = 12
    color_palette: str = "husl"
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
            plt.style.use('seaborn')
            sns.set_context("paper")
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['font.size'] = 12
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
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot metrics
            metrics = ['throughput', 'latency', 'space_amplification']
            for i, metric in enumerate(metrics):
                ax = axes[i//2, i%2]
                try:
                    data = processed_data['metrics'][metric]
                    if not data:
                        logger.warning(f"No data for metric {metric}")
                        continue
                    ax.plot(data)
                    ax.set_title(f"{metric.capitalize()} vs Privacy")
                except Exception as e:
                    logger.error(f"Error plotting {metric}: {str(e)}")
                    continue
            
            # Plot heatmap
            try:
                sns.heatmap(processed_data['performance_impact'], ax=axes[1, 1])
                axes[1, 1].set_title("Performance Impact Heatmap")
            except Exception as e:
                logger.error(f"Error plotting heatmap: {str(e)}")
            
            plt.tight_layout()
            self._save_figure("privacy_performance_tradeoff.png")
            
        except Exception as e:
            logger.error(f"Error in privacy-performance tradeoff plot: {str(e)}")
            raise
    
    def plot_workload_sensitivity(self, results: Dict) -> None:
        """Plot workload sensitivity with edge case handling."""
        try:
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot characteristics
            characteristics = ['read_ratio', 'write_ratio']
            for i, char in enumerate(characteristics):
                ax = axes[i//2, i%2]
                try:
                    data = processed_data['workload_characteristics'][char]
                    if not data:
                        logger.warning(f"No data for characteristic {char}")
                        continue
                    ax.plot(data)
                    ax.set_title(f"{char.capitalize()} Sensitivity")
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
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot configurations
            configs = ['original', 'private']
            for i, config in enumerate(configs):
                ax = axes[i//2, i%2]
                try:
                    data = processed_data['configurations'][config]
                    if not data:
                        logger.warning(f"No data for configuration {config}")
                        continue
                    ax.bar(data.keys(), data.values())
                    ax.set_title(f"{config.capitalize()} Configuration")
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
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot correlation matrix
            try:
                sns.heatmap(processed_data['correlation_matrix'], ax=ax)
                ax.set_title("Metric Correlation Analysis")
            except Exception as e:
                logger.error(f"Error plotting correlation matrix: {str(e)}")
            
            plt.tight_layout()
            self._save_figure("correlation_analysis.png")
            
        except Exception as e:
            logger.error(f"Error in correlation analysis plot: {str(e)}")
            raise 