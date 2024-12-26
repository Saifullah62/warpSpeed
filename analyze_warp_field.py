import os
import numpy as np
from src.dataset_analysis.analyzers.warp_field_analyzer import WarpFieldAnalyzer
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationError(Exception):
    pass

class WarpFieldAnalyzer:
    def __init__(self):
        self.validation_thresholds = {
            'energy_density_max': 1e20,  # J/m³
            'quantum_stability_min': 0.0,
            'quantum_stability_max': 1.0,
            'fidelity_threshold': 0.99,
            'coherence_min': 0.8
        }
        
    async def validate_quantum_metrics(self, metrics):
        """Validate quantum metrics with physical constraints."""
        try:
            if not all(isinstance(v, (int, float)) for v in metrics.values()):
                raise ValueError("All metrics must be numerical values")
                
            validations = {
                'energy_density': 0 <= metrics['energy_density'] <= self.validation_thresholds['energy_density_max'],
                'quantum_stability': self.validation_thresholds['quantum_stability_min'] <= metrics['stability'] <= self.validation_thresholds['quantum_stability_max'],
                'fidelity': 0 <= metrics['fidelity'] <= 1.0,
                'coherence': metrics['coherence'] >= self.validation_thresholds['coherence_min']
            }
            
            failed = [k for k, v in validations.items() if not v]
            if failed:
                raise ValidationError(f"Failed validations: {failed}")
                
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

async def analyze_warp_field_data():
    try:
        # Initialize analyzer
        analyzer = WarpFieldAnalyzer()
        
        # Create test warp field data
        grid_size = 20
        test_grid = np.zeros((grid_size, grid_size))
        
        # Add warp field characteristics
        # Central high-energy region
        center = grid_size // 2
        radius = grid_size // 4
        y, x = np.ogrid[-center:grid_size-center, -center:grid_size-center]
        mask = x*x + y*y <= radius*radius
        test_grid[mask] = 1.0
        
        # Add quantum fluctuations
        test_grid += 0.1 * np.random.randn(*test_grid.shape)
        
        # Add energy gradients
        gradient_x, gradient_y = np.meshgrid(
            np.linspace(-1, 1, grid_size),
            np.linspace(-1, 1, grid_size)
        )
        test_grid += 0.2 * np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize
        test_grid = (test_grid - test_grid.min()) / (test_grid.max() - test_grid.min())
        
        # Prepare analysis data
        warp_field_data = {
            'field_data': test_grid,
            'parameters': {
                'c': 299792458,  # Speed of light (m/s)
                'h_bar': 1.054571817e-34,  # Reduced Planck constant (J⋅s)
                'G': 6.67430e-11,  # Gravitational constant (m³/kg⋅s²)
                'energy_density_threshold': 1e15,  # Energy density threshold (J/m³)
                'stability_threshold': 0.95,  # Minimum stability factor
                'quantum_coupling': 0.1,  # Quantum coupling strength
                'field_strength': 1e10,  # Base field strength (N/m²)
                'optimization_parameters': {
                    'coherence_enhancement': True,
                    'harmonic_modulation': True,
                    'stochastic_balancing': True,
                    'geometry_optimization': True,
                    'vacuum_interaction': True,
                    'adaptive_control': True,
                    'subspace_geometry': True,
                    'learning_rate': 0.1,
                    'iterations': 100,
                    'convergence_threshold': 1e-6
                }
            }
        }
        
        # Run analysis
        logger.info("Starting warp field analysis...")
        results = await analyzer.analyze_warp_field(warp_field_data)
        
        # Validate quantum metrics
        if results.get('status') == 'success':
            metrics = results.get('field_metrics', {})
            validation_result = await analyzer.validate_quantum_metrics(metrics)
            if not validation_result:
                logger.error("Quantum metrics failed validation")
                return
        
        # Print results
        logger.info("\nAnalysis Results:")
        logger.info("----------------")
        
        if results.get('status') == 'success':
            # Field Metrics
            metrics = results.get('field_metrics', {})
            logger.info("\nField Characteristics:")
            logger.info(f"Energy Efficiency: {metrics.get('energy_efficiency', 0):.2%}")
            logger.info(f"Field Uniformity: {metrics.get('field_uniformity', 0):.2%}")
            logger.info(f"Quantum Stability: {metrics.get('quantum_stability', 0):.2%}")
            logger.info(f"Causality Preservation: {metrics.get('causality_preservation', 0):.2%}")
            
            # Optimization Results
            opt_results = results.get('optimization_result', {})
            improvements = opt_results.get('improvement_metrics', {})
            
            if improvements:
                logger.info("\nOptimization Improvements:")
                logger.info(f"Energy Reduction: {improvements.get('energy_reduction', 0):.2%}")
                logger.info(f"Uniformity Improvement: {improvements.get('uniformity_improvement', 0):.2%}")
                logger.info(f"Stability Improvement: {improvements.get('stability_improvement', 0):.2%}")
                
                # Enhanced metrics
                coherence = improvements.get('coherence_metrics', {})
                if coherence:
                    logger.info("\nQuantum Coherence Metrics:")
                    logger.info(f"Entanglement Density: {coherence.get('entanglement_density', 0):.2%}")
                    logger.info(f"Coherence Length: {coherence.get('coherence_length', 0):.2e} m")
                    logger.info(f"Quantum Fidelity: {coherence.get('quantum_fidelity', 0):.2%}")
                    logger.info(f"Phase Stability: {coherence.get('phase_stability', 0):.2%}")
                    logger.info(f"Entanglement Entropy: {coherence.get('entanglement_entropy', 0):.2f} bits")
                
                vacuum = improvements.get('vacuum_metrics', {})
                if vacuum:
                    logger.info("\nVacuum Interaction Metrics:")
                    logger.info(f"Vacuum Coupling: {vacuum.get('vacuum_coupling', 0):.2e}")
                    logger.info(f"Casimir Coupling: {vacuum.get('casimir_coupling', 0):.2e}")
                    logger.info(f"Polarization Strength: {vacuum.get('polarization_strength', 0):.2e}")
                
                control = improvements.get('control_metrics', {})
                if control:
                    logger.info("\nAdaptive Control Metrics:")
                    logger.info(f"Control Effectiveness: {control.get('effectiveness', 0):.2%}")
                    logger.info(f"Temporal Stability: {control.get('temporal_stability', 0):.2%}")
                    logger.info(f"Learning Rate: {control.get('learning_rate', 0):.2e}")
                
                geometry = improvements.get('geometry_metrics', {})
                if geometry:
                    logger.info("\nGeometry Optimization Metrics:")
                    logger.info(f"Curvature: {geometry.get('curvature', 0):.2e}")
                    logger.info(f"Subspace Uniformity: {geometry.get('uniformity', 0):.2%}")
                    logger.info(f"Correction Magnitude: {geometry.get('correction_magnitude', 0):.2e}")
            
            # Analysis Details
            final_analysis = opt_results.get('final_analysis', {})
            if final_analysis:
                topology = final_analysis.get('topology_metrics', {})
                quantum = final_analysis.get('quantum_metrics', {})
                
                logger.info("\nDetailed Analysis:")
                logger.info(f"Mean Energy Density: {topology.get('mean_energy', 0):.2e} J/m³")
                logger.info(f"Maximum Energy Density: {topology.get('max_energy', 0):.2e} J/m³")
                logger.info(f"Energy Variance: {topology.get('energy_variance', 0):.2e}")
                logger.info(f"Vacuum Energy: {quantum.get('vacuum_energy', 0):.2e} J")
                logger.info(f"Quantum Pressure: {quantum.get('quantum_pressure', 0):.2e} Pa")
                logger.info(f"Uncertainty: {quantum.get('uncertainty', 0):.2e} ℏ")
        else:
            logger.error(f"Analysis failed: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    asyncio.run(analyze_warp_field_data())
