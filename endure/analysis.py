from .visualization import EnhancedVisualization
import os
import json
from typing import Dict, List, Optional
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Data class for analysis results with validation."""
    metrics: Dict[str, float]
    configurations: Dict[str, Dict]
    workload_characteristics: Dict[str, float]
    timestamp: str = datetime.now().isoformat()
    
    def validate(self) -> bool:
        """Validate the analysis result data."""
        try:
            # Check required metrics
            required_metrics = ['throughput', 'latency', 'space_amplification']
            if not all(metric in self.metrics for metric in required_metrics):
                logger.error(f"Missing required metrics: {required_metrics}")
                return False
            
            # Check metric values
            for metric, value in self.metrics.items():
                if not isinstance(value, (int, float)):
                    logger.error(f"Invalid type for metric {metric}: {type(value)}")
                    return False
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"Invalid value for metric {metric}: {value}")
                    return False
                if value < 0:
                    logger.error(f"Negative value for metric {metric}: {value}")
                    return False
            
            # Check configurations
            if not self.configurations:
                logger.error("No configuration data provided")
                return False
            if 'original' not in self.configurations or 'private' not in self.configurations:
                logger.error("Missing required configurations: original and/or private")
                return False
            
            # Check workload characteristics
            if not self.workload_characteristics:
                logger.error("No workload characteristics provided")
                return False
            if not all(0 <= v <= 1 for v in self.workload_characteristics.values()):
                logger.error("Invalid workload characteristics values (must be between 0 and 1)")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating results: {str(e)}")
            return False

class BaseAnalysis:
    """Base class for analysis with common functionality."""
    
    def __init__(self, results_dir: str):
        """Initialize base analysis class."""
        self.results_dir = results_dir
        self._setup_logging()
        self._setup_directories()
        self.visualizer = EnhancedVisualization(results_dir)
    
    def _setup_logging(self) -> None:
        """Setup file logging for the analysis."""
        log_file = os.path.join(self.results_dir, f"{self.__class__.__name__.lower()}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    def _setup_directories(self) -> None:
        """Setup required directories."""
        try:
            os.makedirs(self.results_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creating directory {self.results_dir}: {str(e)}")
            raise
    
    def _validate_results(self, results: Dict) -> AnalysisResult:
        """Validate analysis results and return AnalysisResult object."""
        try:
            # Check required fields
            required_fields = ['metrics', 'configurations', 'workload_characteristics']
            if not all(field in results for field in required_fields):
                logger.error(f"Missing required fields: {required_fields}")
                return None
            
            # Create AnalysisResult object
            analysis_result = AnalysisResult(
                metrics=results['metrics'],
                configurations=results['configurations'],
                workload_characteristics=results['workload_characteristics']
            )
            
            # Validate the result
            if not analysis_result.validate():
                return None
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error validating results: {str(e)}")
            return None
    
    def _process_data(self, data: Dict) -> Dict:
        """Process data with standardized handling."""
        try:
            processed = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    processed[key] = self._process_data(value)
                elif isinstance(value, (int, float)):
                    processed[key] = self._handle_numeric_values(value)
                else:
                    processed[key] = value
            return processed
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
    
    def _save_results(self, results: Dict, filename: str) -> None:
        """Save analysis results with error handling."""
        try:
            results_with_metadata = {
                'timestamp': datetime.now().isoformat(),
                'data': results,
                'validation_status': 'success'
            }
            
            filepath = os.path.join(self.results_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(results_with_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving results to {filename}: {str(e)}")
            raise
    
    def _handle_numeric_values(self, value: float) -> float:
        """Handle numeric values with edge cases."""
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Invalid numeric value: {value}")
            return 0.0
        return float(value)

class PrivacyAnalysis(BaseAnalysis):
    """Privacy analysis with edge case handling."""
    
    def run_analysis(self, results: Dict) -> None:
        """Run privacy analysis with edge case handling."""
        try:
            # Validate and process results
            analysis_result = self._validate_results(results)
            if not analysis_result:
                raise ValueError("Invalid analysis results")
            
            # Process all data
            processed_data = self._process_data({
                'metrics': analysis_result.metrics,
                'configurations': analysis_result.configurations,
                'workload_characteristics': analysis_result.workload_characteristics
            })
            
            # Save results
            self._save_results(processed_data, "privacy_analysis.json")
            
            # Generate visualizations
            self.visualizer.plot_privacy_performance_tradeoff(processed_data)
            self.visualizer.plot_configuration_differences(processed_data)
            
            logger.info("Privacy analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in privacy analysis: {str(e)}")
            raise

class SensitivityAnalysis(BaseAnalysis):
    """Sensitivity analysis with edge case handling."""
    
    def run_analysis(self, results: Dict) -> None:
        """Run sensitivity analysis with edge case handling."""
        try:
            # Validate and process results
            analysis_result = self._validate_results(results)
            if not analysis_result:
                raise ValueError("Invalid analysis results")
            
            # Process all data
            processed_data = self._process_data({
                'metrics': analysis_result.metrics,
                'configurations': analysis_result.configurations,
                'workload_characteristics': analysis_result.workload_characteristics
            })
            
            # Save results
            self._save_results(processed_data, "sensitivity_analysis.json")
            
            # Generate visualizations
            self.visualizer.plot_workload_sensitivity(processed_data)
            self.visualizer.plot_correlation_analysis(processed_data)
            
            logger.info("Sensitivity analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {str(e)}")
            raise

class PerformanceAnalysis(BaseAnalysis):
    """Performance analysis with edge case handling."""
    
    def run_analysis(self, results: Dict) -> None:
        """Run performance analysis with edge case handling."""
        try:
            # Validate and process results
            analysis_result = self._validate_results(results)
            if not analysis_result:
                raise ValueError("Invalid analysis results")
            
            # Process all data
            processed_data = self._process_data({
                'metrics': analysis_result.metrics,
                'configurations': analysis_result.configurations,
                'workload_characteristics': analysis_result.workload_characteristics
            })
            
            # Save results
            self._save_results(processed_data, "performance_analysis.json")
            
            # Generate visualizations
            self.visualizer.plot_privacy_performance_tradeoff(processed_data)
            self.visualizer.plot_configuration_differences(processed_data)
            
            logger.info("Performance analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {str(e)}")
            raise 