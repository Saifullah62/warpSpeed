import unittest
import numpy as np
from typing import Dict, Any
import pytest
from src.dataset_analysis.analyzers.energy_optimizer import EnergyOptimizer, EnergyOptimizationParameters

class TestEnergyOptimizer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        params = EnergyOptimizationParameters(
            learning_rate=0.01,
            max_iterations=100,
            convergence_threshold=1e-6,
            stability_threshold=0.95,
            energy_bounds=(0.0, 1e12),
            normalization_factor=1e10,
            field_size=32,
            target_energy=1e6,
            energy_tolerance=1e-6,
            max_field_energy=1e12,
            apply_smoothing=True
        )
        self.params = params
        self.optimizer = EnergyOptimizer(params)
        
    def generate_test_field(self, shape=(32, 32), noise_level=0.1) -> np.ndarray:
        """Generate test field data with controlled noise."""
        # Create base Gaussian field
        x = np.linspace(-2, 2, shape[0])
        y = np.linspace(-2, 2, shape[1])
        X, Y = np.meshgrid(x, y)
        field = np.exp(-(X**2 + Y**2))
        
        # Add controlled noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, field.shape)
            field += noise
            
        # Ensure positivity
        field = np.maximum(field, 0)
        
        return field
        
    def test_field_validation(self):
        """Test field data validation."""
        # Test valid field
        valid_field = self.generate_test_field()
        self.assertTrue(self.optimizer._validate_input(valid_field))
        
        # Test invalid fields
        invalid_cases = [
            (np.ones((32, 32)) * np.nan, "NaN field"),
            (np.ones((32, 32)) * np.inf, "Inf field"),
            (np.ones((32, 32)) * -1, "Negative field"),
            (np.ones((32, 32)) * (self.params.max_field_energy + 1), "Too high energy"),
            (np.ones((1, 1)), "Too small field")
        ]
        
        for invalid_field, case_name in invalid_cases:
            with self.subTest(case=case_name):
                self.assertFalse(self.optimizer._validate_input(invalid_field))
                
    def test_field_normalization(self):
        """Test field normalization."""
        test_field = self.generate_test_field()
        normalized_field = self.optimizer._normalize_field(test_field)
        
        self.assertTrue(np.all(normalized_field >= 0))
        self.assertTrue(np.all(normalized_field <= self.params.max_field_energy))
        self.assertAlmostEqual(np.sum(normalized_field), self.params.target_energy, delta=self.params.energy_tolerance)
        
    def test_geometry_optimization(self):
        """Test field geometry optimization."""
        test_field = self.generate_test_field()
        result = self.optimizer.optimize_energy_distribution(test_field)
        
        self.assertIsInstance(result, dict)
        self.assertIn('optimized_field', result)
        self.assertIn('metrics', result)
        self.assertTrue(self.optimizer._validate_input(result['optimized_field']))
        
    def test_energy_conservation(self):
        """Test energy conservation during optimization."""
        test_field = self.generate_test_field()
        initial_energy = np.sum(test_field)
        
        result = self.optimizer.optimize_energy_distribution(test_field)
        
        if result:
            final_energy = np.sum(result['optimized_field'])
            self.assertAlmostEqual(final_energy, initial_energy, delta=self.params.energy_tolerance)
            
    def test_stability_metrics(self):
        """Test stability metrics calculation."""
        test_field = self.generate_test_field()
        
        result = self.optimizer.optimize_energy_distribution(test_field)
        
        if result:
            metrics = result['metrics']
            self.assertIn('stability', metrics)
            self.assertGreaterEqual(metrics['stability'], 0.0)
            self.assertLessEqual(metrics['stability'], 1.0)
            
    def test_optimization_constraints(self):
        """Test optimization constraints."""
        test_field = self.generate_test_field()
        
        constraints = {
            'min_energy': 0.5 * self.params.target_energy,
            'max_energy': 1.5 * self.params.target_energy,
            'stability_threshold': 0.8
        }
        
        result = self.optimizer.optimize_energy_distribution(test_field, constraints)
        
        if result:
            optimized_field = result['optimized_field']
            total_energy = np.sum(optimized_field)
            self.assertGreaterEqual(total_energy, constraints['min_energy'])
            self.assertLessEqual(total_energy, constraints['max_energy'])
            self.assertGreaterEqual(result['metrics']['stability'], constraints['stability_threshold'])
            
    def test_error_handling(self):
        """Test error handling in critical functions."""
        # Test with invalid inputs
        invalid_cases = [
            (None, "None input"),
            (np.array([]), "Empty array"),
            (np.ones((32, 32, 32)), "Wrong dimensions"),
            ("not an array", "Invalid type")
        ]
        
        for invalid_input, case_name in invalid_cases:
            with self.subTest(case=case_name):
                # Should handle errors gracefully
                result = self.optimizer.optimize_energy_distribution(invalid_input)
                self.assertEqual(result, {})
                
    def test_optimization_statistics(self):
        """Test optimization statistics calculation."""
        # Perform multiple optimizations
        for _ in range(3):
            test_field = self.generate_test_field(noise_level=0.1)
            self.optimizer.optimize_energy_distribution(test_field)
            
        stats = self.optimizer.get_optimization_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('convergence_rate', stats)
        self.assertIn('metric_trends', stats)
        
    def test_noise_robustness(self):
        """Test optimizer robustness to different noise levels."""
        noise_levels = [0.0, 0.1, 0.2]
        for noise_level in noise_levels:
            test_field = self.generate_test_field(noise_level=noise_level)
            result = self.optimizer.optimize_energy_distribution(test_field)
            
            self.assertIsInstance(result, dict)
            if result:
                self.assertLessEqual(result['metrics'].get('noise', float('inf')), noise_level + 0.1)
                
    def test_boundary_conditions(self):
        """Test boundary condition handling."""
        test_field = self.generate_test_field()
        result = self.optimizer.optimize_energy_distribution(test_field)
        
        if result:
            boundary_check = self.optimizer._check_boundary_conditions(result['optimized_field'])
            self.assertGreaterEqual(boundary_check, 0.0)
            self.assertLessEqual(boundary_check, 1.0)
            
if __name__ == '__main__':
    unittest.main()
