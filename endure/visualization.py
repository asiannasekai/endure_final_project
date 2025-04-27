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
            if not isinstance(data, dict):
                raise ValueError("Input data must be a dictionary")
                
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
                elif isinstance(value, list):
                    processed_data[key] = [
                        float(v) if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)) else 0.0
                        for v in value
                    ]
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
        """Plot privacy-performance tradeoff with detailed insights."""
        try:
            # Extract epsilon values and their corresponding metrics
            epsilons = sorted(results.keys())
            if not epsilons:
                logger.error("No results to plot")
                return
            
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 2)
            
            # Plot 1: Performance Metrics
            ax1 = fig.add_subplot(gs[0, 0])
            performance_metrics = {
                'throughput': [],
                'latency': [],
                'space_amplification': []
            }
            
            for epsilon in epsilons:
                trials = results[epsilon]
                if not trials:
                    continue
                    
                for metric in performance_metrics.keys():
                    try:
                        avg_diff = np.mean([
                            trial['privacy_metrics']['performance_differences'][metric]['difference_percent']
                            for trial in trials
                            if metric in trial['privacy_metrics']['performance_differences']
                        ])
                        performance_metrics[metric].append(avg_diff)
                    except (KeyError, TypeError):
                        performance_metrics[metric].append(0.0)
            
            for metric, values in performance_metrics.items():
                if values:  # Only plot if we have data
                    ax1.plot(epsilons, values, 'o-', label=metric.replace('_', ' ').title())
            
            ax1.set_title("Performance Impact vs Privacy Level")
            ax1.set_xlabel("Epsilon (ε)")
            ax1.set_ylabel("Average Performance Difference (%)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Configuration Changes
            ax2 = fig.add_subplot(gs[0, 1])
            config_metrics = {}
            
            for epsilon in epsilons:
                trials = results[epsilon]
                if not trials:
                    continue
                    
                for trial in trials:
                    try:
                        for param, diff in trial['privacy_metrics']['configuration_differences'].items():
                            if param not in config_metrics:
                                config_metrics[param] = []
                            config_metrics[param].append(diff['difference_percent'])
                    except (KeyError, TypeError):
                        continue
            
            if config_metrics:
                for param, values in config_metrics.items():
                    if values:  # Only plot if we have data
                        ax2.plot(epsilons, values, 'o-', label=param.replace('_', ' ').title())
                ax2.set_title("Configuration Changes vs Privacy Level")
            else:
                ax2.text(0.5, 0.5, 'No significant configuration changes',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=ax2.transAxes)
                ax2.set_title("Configuration Changes")
            
            ax2.set_xlabel("Epsilon (ε)")
            ax2.set_ylabel("Average Configuration Difference (%)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Privacy-Utility Tradeoff Score
            ax3 = fig.add_subplot(gs[1, :])
            scores = {
                'Performance': [],
                'Configuration': [],
                'Overall': []
            }
            
            for epsilon in epsilons:
                trials = results[epsilon]
                if not trials:
                    continue
                    
                for score_type in scores.keys():
                    try:
                        avg_score = np.mean([
                            trial['privacy_metrics']['privacy_utility_score'][f"{score_type.lower()}_score"]
                            for trial in trials
                        ])
                        scores[score_type].append(avg_score)
                    except (KeyError, TypeError):
                        scores[score_type].append(0.0)
            
            for score_type, values in scores.items():
                if values:  # Only plot if we have data
                    ax3.plot(epsilons, values, 'o-', label=score_type)
            
            ax3.set_title("Privacy-Utility Tradeoff Scores")
            ax3.set_xlabel("Epsilon (ε)")
            ax3.set_ylabel("Score (0-100)")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Plot 4: Performance Impact Levels
            ax4 = fig.add_subplot(gs[2, :])
            impact_levels = {
                'Negligible': [],
                'Minor': [],
                'Moderate': [],
                'Significant': []
            }
            
            for epsilon in epsilons:
                trials = results[epsilon]
                if not trials:
                    continue
                    
                for impact in impact_levels.keys():
                    try:
                        count = sum(
                            1 for trial in trials
                            for metric in trial['privacy_metrics']['performance_differences'].values()
                            if metric['impact'] == impact
                        )
                        impact_levels[impact].append(count / len(trials) * 100)
                    except (KeyError, TypeError):
                        impact_levels[impact].append(0.0)
            
            x = np.arange(len(epsilons))
            width = 0.2
            for i, (impact, values) in enumerate(impact_levels.items()):
                if values:  # Only plot if we have data
                    ax4.bar(x + i*width, values, width, label=impact)
            
            ax4.set_title("Distribution of Performance Impact Levels")
            ax4.set_xlabel("Epsilon (ε)")
            ax4.set_ylabel("Percentage of Trials (%)")
            ax4.set_xticks(x + width*1.5)
            ax4.set_xticklabels(epsilons)
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            self._save_figure("privacy_performance_tradeoff.png")
            
        except Exception as e:
            logger.error(f"Error in privacy-performance tradeoff plot: {str(e)}")
            raise
    
    def plot_workload_sensitivity(self, results: Dict) -> None:
        """Plot workload sensitivity with edge case handling."""
        try:
            # Validate required data
            if not isinstance(results, dict):
                raise ValueError("Results must be a dictionary")
                
            if 'workload_characteristics' not in results:
                logger.error("Missing workload characteristics data")
                return
            
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot characteristics
            characteristics = ['read_ratio', 'write_ratio', 'key_size', 'value_size']
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
                    ax.set_ylim(0, max(1, value * 1.1))  # Add 10% padding
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
            # Extract epsilon values and configuration parameters
            epsilons = sorted(results.keys())
            if not epsilons:
                logger.error("No results to plot")
                return
                
            config_params = set()
            
            # Get all unique configuration parameters
            for epsilon in epsilons:
                trials = results[epsilon]
                if not isinstance(trials, list):
                    logger.error(f"Invalid trials data for epsilon {epsilon}")
                    continue
                    
                for trial in trials:
                    if not isinstance(trial, dict):
                        continue
                    if 'comparison' not in trial or 'parameter_differences' not in trial['comparison']:
                        continue
                        
                    params = trial['comparison']['parameter_differences'].keys()
                    config_params.update(params)
            
            if not config_params:
                logger.warning("No configuration parameters found to plot")
                return
            
            # Prepare data for each parameter
            param_data = {param: [] for param in config_params}
            
            # Collect data for each epsilon and parameter
            for epsilon in epsilons:
                trials = results[epsilon]
                if not isinstance(trials, list):
                    continue
                    
                for param in config_params:
                    differences = []
                    for trial in trials:
                        if not isinstance(trial, dict):
                            continue
                        if 'comparison' not in trial or 'parameter_differences' not in trial['comparison']:
                            continue
                        if param not in trial['comparison']['parameter_differences']:
                            continue
                            
                        diff = trial['comparison']['parameter_differences'][param]
                        if isinstance(diff, dict) and 'difference_percent' in diff:
                            differences.append(diff['difference_percent'])
                    
                    if differences:
                        param_data[param].append(np.mean(differences))
                    else:
                        param_data[param].append(0.0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each parameter
            for param, values in param_data.items():
                if values:  # Only plot if we have data
                    ax.plot(epsilons, values, 'o-', label=param.replace('_', ' ').title())
            
            ax.set_title("Configuration Parameter Differences vs Privacy")
            ax.set_xlabel("Epsilon (ε)")
            ax.set_ylabel("Average Difference Percentage")
            ax.grid(True, alpha=0.3)
            
            # Only add legend if we have data to show
            if any(values for values in param_data.values()):
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            self._save_figure("configuration_differences.png")
            
        except Exception as e:
            logger.error(f"Error in configuration differences plot: {str(e)}")
            raise
    
    def plot_correlation_analysis(self, results: Dict) -> None:
        """Plot correlation analysis with edge case handling."""
        try:
            # Validate required data
            if not isinstance(results, dict):
                raise ValueError("Results must be a dictionary")
                
            required_fields = ['metrics', 'workload_characteristics']
            if not all(field in results for field in required_fields):
                logger.error(f"Missing required fields for correlation analysis: {required_fields}")
                return
            
            # Process data
            processed_data = self._handle_numeric_data(results)
            
            # Create correlation matrix
            metrics = list(processed_data['metrics'].keys())
            characteristics = list(processed_data['workload_characteristics'].keys())
            
            if not metrics or not characteristics:
                logger.error("No metrics or characteristics data available")
                return
            
            all_data = {**processed_data['metrics'], **processed_data['workload_characteristics']}
            correlation_matrix = np.zeros((len(all_data), len(all_data)))
            
            # Calculate correlations
            for i, (k1, v1) in enumerate(all_data.items()):
                for j, (k2, v2) in enumerate(all_data.items()):
                    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                        correlation_matrix[i, j] = 1.0 if k1 == k2 else 0.5  # Placeholder correlation
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Plot correlation matrix
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
    
    def plot_tuning_stability(self, results: Dict) -> None:
        """Plot tuning stability analysis with confidence intervals."""
        try:
            # Create analysis instance
            analysis = PrivacyAnalysis()
            stability_metrics = analysis._analyze_tuning_stability(results)
            
            if not stability_metrics:
                logger.warning("No stability metrics to plot")
                return
            
            # Create figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Configuration Stability
            ax1 = axes[0, 0]
            epsilons = sorted(stability_metrics.keys())
            config_stability = [
                stability_metrics[eps]['configuration_stability']
                for eps in epsilons
            ]
            
            means = [m['mean'] for m in config_stability]
            ci_lower = [m['ci_lower'] for m in config_stability]
            ci_upper = [m['ci_upper'] for m in config_stability]
            
            ax1.errorbar(epsilons, means, yerr=[means[i] - ci_lower[i] for i in range(len(means))],
                        fmt='o-', capsize=5, label='Mean with 95% CI')
            ax1.set_title("Configuration Stability vs Privacy Level")
            ax1.set_xlabel("Epsilon (ε)")
            ax1.set_ylabel("Average Configuration Difference (%)")
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Plot 2: Performance Stability
            ax2 = axes[0, 1]
            performance_metrics = ['throughput', 'latency', 'space_amplification']
            for metric in performance_metrics:
                means = []
                ci_lower = []
                ci_upper = []
                for eps in epsilons:
                    if metric in stability_metrics[eps]['performance_stability']:
                        stats = stability_metrics[eps]['performance_stability'][metric]
                        means.append(stats['mean'])
                        ci_lower.append(stats['ci_lower'])
                        ci_upper.append(stats['ci_upper'])
                    else:
                        means.append(0)
                        ci_lower.append(0)
                        ci_upper.append(0)
                
                ax2.errorbar(epsilons, means, yerr=[means[i] - ci_lower[i] for i in range(len(means))],
                            fmt='o-', capsize=5, label=metric.replace('_', ' ').title())
            
            ax2.set_title("Performance Stability vs Privacy Level")
            ax2.set_xlabel("Epsilon (ε)")
            ax2.set_ylabel("Average Performance Difference (%)")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Plot 3: Workload Pattern Analysis
            ax3 = axes[1, 0]
            pattern_metrics = analysis._analyze_workload_patterns(results)
            
            if pattern_metrics:
                patterns = ['read_write_ratio', 'hot_key_access']
                for pattern in patterns:
                    means = []
                    ci_lower = []
                    ci_upper = []
                    for eps in epsilons:
                        if pattern in pattern_metrics[eps]:
                            stats = pattern_metrics[eps][pattern]
                            means.append(stats['mean'])
                            ci_lower.append(stats['ci_lower'])
                            ci_upper.append(stats['ci_upper'])
                        else:
                            means.append(0)
                            ci_lower.append(0)
                            ci_upper.append(0)
                    
                    ax3.errorbar(epsilons, means, yerr=[means[i] - ci_lower[i] for i in range(len(means))],
                                fmt='o-', capsize=5, label=pattern.replace('_', ' ').title())
            
            ax3.set_title("Workload Pattern Stability vs Privacy Level")
            ax3.set_xlabel("Epsilon (ε)")
            ax3.set_ylabel("Pattern Metric Value")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Plot 4: Stability Summary
            ax4 = axes[1, 1]
            stability_scores = []
            for eps in epsilons:
                config_stab = stability_metrics[eps]['configuration_stability']['mean']
                perf_stab = np.mean([
                    stability_metrics[eps]['performance_stability'][metric]['mean']
                    for metric in performance_metrics
                    if metric in stability_metrics[eps]['performance_stability']
                ])
                stability_scores.append(100 - (config_stab + perf_stab) / 2)
            
            ax4.plot(epsilons, stability_scores, 'o-', label='Overall Stability Score')
            ax4.set_title("Overall Tuning Stability vs Privacy Level")
            ax4.set_xlabel("Epsilon (ε)")
            ax4.set_ylabel("Stability Score (0-100)")
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            self._save_figure("tuning_stability_analysis.png")
            
        except Exception as e:
            logger.error(f"Error in tuning stability plot: {str(e)}")
            raise 