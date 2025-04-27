"""
Unit tests for the analysis module.
"""

import unittest
import os
import json
import tempfile
import shutil
from typing import Dict
import numpy as np
from endure.analysis import PrivacyAnalysis, SensitivityAnalysis, PerformanceAnalysis, AnalysisResult

class TestAnalysis(unittest.TestCase):
    """Test cases for analysis classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.sample_results = {
            'metrics': {
                'throughput': 1000.0,
                'latency': 50.0,
                'space_amplification': 2.0
            },
            'configurations': {
                'original': {'param1': 10, 'param2': 20},
                'private': {'param1': 15, 'param2': 25}
            },
            'workload_characteristics': {
                'read_ratio': 0.7,
                'write_ratio': 0.3
            }
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_analysis_result_validation(self):
        """Test AnalysisResult validation."""
        # Test valid results
        result = AnalysisResult(
            metrics=self.sample_results['metrics'],
            configurations=self.sample_results['configurations'],
            workload_characteristics=self.sample_results['workload_characteristics']
        )
        self.assertTrue(result.validate())
        
        # Test missing metrics
        invalid_metrics = self.sample_results['metrics'].copy()
        del invalid_metrics['throughput']
        result = AnalysisResult(
            metrics=invalid_metrics,
            configurations=self.sample_results['configurations'],
            workload_characteristics=self.sample_results['workload_characteristics']
        )
        self.assertFalse(result.validate())
        
        # Test empty configurations
        result = AnalysisResult(
            metrics=self.sample_results['metrics'],
            configurations={},
            workload_characteristics=self.sample_results['workload_characteristics']
        )
        self.assertFalse(result.validate())
    
    def test_privacy_analysis(self):
        """Test PrivacyAnalysis class."""
        analysis = PrivacyAnalysis(results_dir=self.test_dir)
        
        # Test successful analysis
        analysis.run_analysis(self.sample_results)
        
        # Verify output files
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "privacy_analysis.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "privacy_analysis.log")))
        
        # Test invalid results
        with self.assertRaises(ValueError):
            analysis.run_analysis({})
    
    def test_sensitivity_analysis(self):
        """Test SensitivityAnalysis class."""
        analysis = SensitivityAnalysis(results_dir=self.test_dir)
        
        # Test successful analysis
        analysis.run_analysis(self.sample_results)
        
        # Verify output files
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "sensitivity_analysis.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "sensitivity_analysis.log")))
        
        # Test invalid results
        with self.assertRaises(ValueError):
            analysis.run_analysis({})
    
    def test_performance_analysis(self):
        """Test PerformanceAnalysis class."""
        analysis = PerformanceAnalysis(results_dir=self.test_dir)
        
        # Test successful analysis
        analysis.run_analysis(self.sample_results)
        
        # Verify output files
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "performance_analysis.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "performance_analysis.log")))
        
        # Test invalid results
        with self.assertRaises(ValueError):
            analysis.run_analysis({})

if __name__ == '__main__':
    unittest.main() 