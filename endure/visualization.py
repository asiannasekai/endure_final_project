"""
Enhanced visualization module for privacy-performance analysis.

This module provides comprehensive visualization capabilities for analyzing
privacy-performance tradeoffs, workload sensitivity, and configuration impacts.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import logging
from pathlib import Path
from scipy.interpolate import make_interp_spline
from .types import VisualizationConfig, AnalysisResults, WorkloadCharacteristics, Metrics, MathUtils

# Get logger instance without configuring
logger = logging.getLogger(__name__)

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
    
    def _handle_numeric_data(self, data: List[Dict], metric_key: str) -> Dict[str, float]:
        """Process numeric data for a given metric, calculating statistics.
        
        Args:
            data: List of dictionaries containing metric data
            metric_key: Key to extract from each dictionary
            
        Returns:
            Dictionary containing calculated statistics
        """
        try:
            if not data:
                raise ValueError(f"No data provided for metric {metric_key}")
            
            # Extract values, handling nested structure
            values = []
            for item in data:
                try:
                    if 'privacy_metrics' in item and 'performance_differences' in item['privacy_metrics']:
                        perf_diffs = item['privacy_metrics']['performance_differences']
                        if metric_key in perf_diffs and 'difference_percent' in perf_diffs[metric_key]:
                            values.append(perf_diffs[metric_key]['difference_percent'])
                except Exception as e:
                    logging.warning(f"Error extracting {metric_key} from item: {str(e)}")
                    continue
                
            if not values:
                raise ValueError(f"No valid values found for metric {metric_key}")
            
            # Convert to numpy array for calculations
            values = np.array(values)
            
            # Calculate statistics
            stats = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values)
            }
            
            return stats
        
        except Exception as e:
            logging.error(f"Error processing numeric data for {metric_key}: {str(e)}")
            raise
    
    def _save_figure(self, filename: str) -> None:
        """Save figure with robust error handling and cleanup."""
        filepath = os.path.join(self.results_dir, filename)
        
        # Create backup of existing file if it exists
        if os.path.exists(filepath):
            backup_path = f"{filepath}.bak"
            try:
                os.rename(filepath, backup_path)
            except Exception as e:
                logger.warning(f"Could not create backup: {str(e)}")
        
        # Save new figure with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
                # Small delay to allow file system to catch up
                import time
                time.sleep(0.1)
                
                # Verify the file was saved
                if os.path.exists(filepath):
                    # Remove backup if save was successful
                    if os.path.exists(f"{filepath}.bak"):
                        try:
                            os.remove(f"{filepath}.bak")
                        except Exception as e:
                            logger.warning(f"Could not remove backup: {str(e)}")
                    plt.close()
                    return
                
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to save figure on attempt {attempt + 1}, retrying...")
                    time.sleep(0.5)  # Longer delay between retries
                else:
                    raise IOError("Failed to save figure after multiple attempts")
                    
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Error saving figure {filename}: {str(e)}")
                    # Restore backup if it exists
                    if os.path.exists(f"{filepath}.bak"):
                        try:
                            os.rename(f"{filepath}.bak", filepath)
                        except Exception as e:
                            logger.error(f"Could not restore backup: {str(e)}")
                    plt.close()
                    raise
                else:
                    logger.warning(f"Error on attempt {attempt + 1}: {str(e)}")
                    continue
    
    def _validate_results_structure(self, results: Dict[float, List]) -> bool:
        """Validate results structure with more lenient checks."""
        try:
            if not isinstance(results, dict):
                logger.warning("Results must be a dictionary")
                return False

            if not results:
                logger.warning("Results dictionary is empty")
                return False

            valid_epsilons = []
            for epsilon, trials in results.items():
                try:
                    # Convert epsilon to float if it's a string
                    if isinstance(epsilon, str):
                        epsilon = float(epsilon)
                    
                    # More lenient epsilon validation
                    if not isinstance(epsilon, (int, float)):
                        logger.warning(f"Skipping invalid epsilon value type: {type(epsilon)}")
                        continue
                    if epsilon <= 0:
                        logger.warning(f"Skipping non-positive epsilon value: {epsilon}")
                        continue

                    # Validate trials list
                    if not isinstance(trials, list):
                        logger.warning(f"Invalid trials type for epsilon {epsilon}")
                        continue
                    
                    # Count valid trials
                    valid_trials = 0
                    for trial in trials:
                        if self._validate_trial_data(trial, epsilon, valid_trials):
                            valid_trials += 1
                    
                    if valid_trials > 0:
                        valid_epsilons.append(epsilon)
                    
                except Exception as e:
                    logger.warning(f"Error processing epsilon {epsilon}: {str(e)}")
                    continue

            # Consider results valid if we have at least one valid epsilon with trials
            return len(valid_epsilons) > 0

        except Exception as e:
            logger.warning(f"Error validating results structure: {str(e)}")
            return False

    def _validate_trial_data(self, trial: Dict, epsilon: float, trial_index: int) -> bool:
        """Validate individual trial data with more lenient checks."""
        try:
            if not isinstance(trial, dict):
                logger.warning(f"Trial {trial_index} for epsilon {epsilon} is not a dictionary")
                return False

            # Check for privacy metrics with lenient validation
            privacy_metrics = trial.get('privacy_metrics', {})
            if not isinstance(privacy_metrics, dict):
                logger.warning(f"Invalid privacy_metrics type in trial {trial_index} for epsilon {epsilon}")
                return False

            # Validate performance differences if present
            if 'performance_differences' in privacy_metrics:
                perf_diffs = privacy_metrics['performance_differences']
                if not isinstance(perf_diffs, dict):
                    logger.warning(f"Invalid performance_differences type in trial {trial_index}")
                    return False
                for metric, diff in perf_diffs.items():
                    if not isinstance(diff, dict):
                        continue
                    if 'difference_percent' in diff and not isinstance(diff['difference_percent'], (int, float)):
                        logger.warning(f"Invalid difference_percent type for {metric} in trial {trial_index}")
                        continue

            # Validate configuration differences if present
            if 'configuration_differences' in privacy_metrics:
                config_diffs = privacy_metrics['configuration_differences']
                if not isinstance(config_diffs, dict):
                    logger.warning(f"Invalid configuration_differences type in trial {trial_index}")
                    return False
                for param, diff in config_diffs.items():
                    if not isinstance(diff, dict):
                        continue
                    if 'difference_percent' in diff and not isinstance(diff['difference_percent'], (int, float)):
                        logger.warning(f"Invalid difference_percent type for {param} in trial {trial_index}")
                        continue

            # Validate workload characteristics if present
            workload_chars = trial.get('workload_characteristics', {})
            if not isinstance(workload_chars, dict):
                logger.warning(f"Invalid workload_characteristics type in trial {trial_index}")
                return False

            # If we have at least some valid data, consider the trial valid
            return True

        except Exception as e:
            logger.warning(f"Error validating trial {trial_index} data for epsilon {epsilon}: {str(e)}")
            return False

    def _smooth_data(self, x: np.ndarray, y: np.ndarray, window_size: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply smoothing to data points with robust error handling."""
        try:
            if window_size is None:
                window_size = self.config.smoothing_window
            return MathUtils.smooth_data(x, y, window_size)
        except Exception as e:
            logger.warning(f"Error smoothing data: {str(e)}")
            return x, y

    def _fit_curve(self, x: np.ndarray, y: np.ndarray, degree: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Fit polynomial curve to data with robust error handling."""
        try:
            if degree is None:
                degree = self.config.curve_degree
            return MathUtils.fit_curve(x, y, degree)
        except Exception as e:
            logger.warning(f"Error fitting curve: {str(e)}")
            return x, y

    def plot_privacy_performance_tradeoff(self, results: Dict[float, List[Dict]], metrics: List[str] = None, filename: str = "privacy_performance_tradeoff.png") -> None:
        """Plot privacy-performance tradeoff with comprehensive metrics."""
        try:
            if not self._validate_results_structure(results):
                raise ValueError("Invalid results structure")
            
            if metrics is None:
                metrics = ['throughput', 'latency', 'memory']
            
            # Create subplots for each metric
            fig, axes = plt.subplots(len(metrics), 1, figsize=self.config.figure_size)
            if len(metrics) == 1:
                axes = [axes]
            
            for i, metric in enumerate(metrics):
                # Calculate means and standard deviations
                epsilons = sorted(results.keys())
                means = []
                stds = []
                
                for epsilon in epsilons:
                    metric_data = self._handle_numeric_data(results[epsilon], metric)
                    means.append(metric_data['mean'])
                    stds.append(metric_data['std'])
                
                # Plot with error bars
                axes[i].errorbar(epsilons, means, yerr=stds, fmt='o-', capsize=5)
                
                # Add smoothed curve if enough points
                if len(epsilons) > 3:
                    x_smooth, y_smooth = self._smooth_data(np.array(epsilons), np.array(means))
                    axes[i].plot(x_smooth, y_smooth, '--', alpha=0.5)
                
                axes[i].set_xlabel('Epsilon')
                axes[i].set_ylabel(f'{metric.capitalize()} Difference (%)')
                axes[i].set_title(f'Privacy-Performance Tradeoff: {metric.capitalize()}')
                axes[i].grid(True)
            
            plt.tight_layout()
            self._save_figure(filename)
            
        except Exception as e:
            logger.error(f"Error plotting privacy-performance tradeoff: {str(e)}")
            raise
    
    def plot_workload_sensitivity(self, results: Dict, filename: str = "workload_sensitivity.png") -> None:
        """Plot workload sensitivity analysis."""
        try:
            # Implementation here
            self._save_figure(filename)
        except Exception as e:
            logger.error(f"Error plotting workload sensitivity: {str(e)}")
            raise
    
    def plot_configuration_differences(self, results: Dict[float, List], filename: str = "configuration_differences.png") -> None:
        """Plot configuration differences analysis."""
        try:
            # Implementation here
            self._save_figure(filename)
        except Exception as e:
            logger.error(f"Error plotting configuration differences: {str(e)}")
            raise
    
    def plot_correlation_analysis(self, results: Dict, filename: str = "correlation_analysis.png") -> None:
        """Plot correlation analysis."""
        try:
            # Implementation here
            self._save_figure(filename)
        except Exception as e:
            logger.error(f"Error plotting correlation analysis: {str(e)}")
            raise
    
    def plot_tuning_stability(self, results: Dict[float, List]) -> None:
        """Plot tuning stability analysis with confidence intervals."""
        try:
            if not results:
                logger.error("No results to plot")
                return

            if not self._validate_results_structure(results):
                logger.error("Invalid results structure for tuning stability plot")
                return

            # Create analysis instance
            analysis = PrivacyAnalysis()
            stability_metrics = analysis._analyze_tuning_stability(results)
            
            if not stability_metrics:
                logger.warning("No stability metrics to plot")
                return

            # Validate stability metrics structure
            for epsilon, metrics in stability_metrics.items():
                if not isinstance(metrics, dict):
                    logger.error(f"Invalid stability metrics format for epsilon {epsilon}")
                    continue
                if 'configuration_stability' not in metrics or 'performance_stability' not in metrics:
                    logger.error(f"Missing required stability metrics for epsilon {epsilon}")
                    continue

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
    
    def plot_workload_characteristics(self, results: Dict[float, List]) -> None:
        """Plot workload characteristics with detailed validation."""
        try:
            if not results:
                logger.error("No results to plot")
                return

            if not self._validate_results_structure(results):
                logger.error("Invalid results structure for workload characteristics plot")
                return

            # Create figure
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            epsilons = sorted(results.keys())
            if not epsilons:
                logger.error("No epsilon values found")
                return

            # Collect workload characteristics
            workload_metrics = {}
            for epsilon in epsilons:
                trials = results[epsilon]
                if not trials:
                    logger.warning(f"No trials found for epsilon {epsilon}")
                    continue

                for i, trial in enumerate(trials):
                    if not self._validate_trial_data(trial, epsilon, i):
                        logger.warning(f"Skipping invalid trial {i} for epsilon {epsilon}")
                        continue

                    try:
                        characteristics = trial['workload_characteristics']
                        if not isinstance(characteristics, dict):
                            logger.warning(f"Invalid workload characteristics format in trial {i} for epsilon {epsilon}")
                            continue

                        for char, value in characteristics.items():
                            if char not in workload_metrics:
                                workload_metrics[char] = []
                            if isinstance(value, (int, float)):
                                workload_metrics[char].append(value)
                            else:
                                logger.warning(f"Invalid value format for characteristic '{char}' in trial {i} for epsilon {epsilon}: {type(value)}")
                    except Exception as e:
                        logger.error(f"Error processing workload characteristics in trial {i} for epsilon {epsilon}: {str(e)}")
                        continue

            # Plot workload characteristics
            if workload_metrics:
                for char, values in workload_metrics.items():
                    if values:  # Only plot if we have data
                        ax.plot(epsilons, values, 'o-', label=char.replace('_', ' ').title())
            
                ax.set_title("Workload Characteristics vs Privacy Level")
                ax.set_xlabel("Epsilon (ε)")
                ax.set_ylabel("Characteristic Value")
                ax.grid(True, alpha=0.3)
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'No workload characteristics data available',
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax.transAxes)
                ax.set_title("Workload Characteristics")
                ax.set_xlabel("Epsilon (ε)")
                ax.set_ylabel("Characteristic Value")

            self._save_figure("workload_characteristics.png")
        except Exception as e:
            logger.error(f"Error in workload characteristics plot: {str(e)}")
            raise 