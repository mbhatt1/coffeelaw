"""
Experimental protocols for Coffee Law verification
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

from .monte_carlo_runner import MonteCarloRunner, MonteCarloResults
from ..config import CONFIG

@dataclass
class ProtocolConfig:
    """Configuration for a specific protocol"""
    name: str
    description: str
    pe_ctx_range: Tuple[float, float]
    n_pe_variants: int
    samples_per_variant: int
    required_metrics: List[str]
    acceptance_criteria: Dict[str, Any]

class ExperimentProtocols:
    """
    Implementation of the three main Coffee Law verification protocols
    """
    
    def __init__(self, 
                 monte_carlo_runner: MonteCarloRunner,
                 config: Any = CONFIG):
        self.mc_runner = monte_carlo_runner
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Define protocol configurations
        self.protocols = self._define_protocols()
        
    def _define_protocols(self) -> Dict[str, ProtocolConfig]:
        """Define the three main verification protocols from README"""
        return {
            'sharpening': ProtocolConfig(
                name='Cube-root sharpening',
                description='Verify W/√D_eff ∝ Pe_ctx^(-1/3)',
                pe_ctx_range=(0.1, 10.0),  # >1 decade as required
                n_pe_variants=6,
                samples_per_variant=100,
                required_metrics=['W', 'D_eff'],
                acceptance_criteria={
                    'expected_slope': -1/3,
                    'tolerance': 0.07,
                    'min_r_squared': 0.8
                }
            ),
            'entropy': ProtocolConfig(
                name='Entropy slope verification',
                description='Verify H = a + b*ln(Pe_ctx) with b ≈ 2/3',
                pe_ctx_range=(0.1, 10.0),
                n_pe_variants=6,
                samples_per_variant=100,
                required_metrics=['H', 'W'],
                acceptance_criteria={
                    'expected_b': 2/3,
                    'tolerance_b': 0.10,
                    'identity_tolerance': 0.15,
                    'min_r_squared': 0.8
                }
            ),
            'diminishing_returns': ProtocolConfig(
                name='Logarithmic context scaling',
                description='Verify Pe_ctx(N) = a + b·ln(N)',
                pe_ctx_range=(1.0, 1.0),  # Not used for this test
                n_pe_variants=10,  # Actually N values
                samples_per_variant=50,
                required_metrics=['H', 'W', 'N_eff'],
                acceptance_criteria={
                    'expected_slope': -1/3,
                    'tolerance': 0.10,
                    'min_r_squared': 0.7
                }
            )
        }
    
    async def run_all_protocols(self, 
                              save_intermediate: bool = True) -> Dict[str, MonteCarloResults]:
        """
        Run all three verification protocols
        
        Returns dictionary mapping protocol name to results
        """
        self.logger.info("Starting full Coffee Law verification suite")
        
        results = {}
        
        # Protocol 1: Sharpening
        self.logger.info("Running Protocol 1: Cube-root sharpening")
        results['sharpening'] = await self.run_protocol_1_sharpening()
        
        # Protocol 2: Entropy  
        self.logger.info("Running Protocol 2: Entropy slope")
        results['entropy'] = await self.run_protocol_2_entropy()
        
        # Protocol 3: Logarithmic context scaling
        self.logger.info("Running Protocol 3: Logarithmic context scaling")
        results['diminishing_returns'] = await self.run_protocol_3_diminishing_returns()
        
        # Overall verification
        overall_passed = all(r.verification_passed for r in results.values())
        
        self.logger.info(f"Coffee Law verification {'PASSED' if overall_passed else 'FAILED'}")
        
        return results
    
    async def run_protocol_1_sharpening(self) -> MonteCarloResults:
        """
        Protocol 1: Verify W/√D_eff ∝ Pe_ctx^(-1/3)
        
        Creates 6 context variants across >1 decade of Pe_ctx
        Measures width and effective dimension
        Verifies power law with slope -1/3 ± 0.07
        """
        protocol = self.protocols['sharpening']
        
        # Run the sweep
        results = self.mc_runner.run_pe_ctx_sweep(
            protocol_name=protocol.name,
            n_pe_variants=protocol.n_pe_variants,
            samples_per_variant=protocol.samples_per_variant,
            pe_ctx_range=protocol.pe_ctx_range
        )
        
        # Additional analysis
        self._analyze_sharpening_results(results)
        
        return results
    
    async def run_protocol_2_entropy(self) -> MonteCarloResults:
        """
        Protocol 2: Verify entropy slope b ≈ 2/3
        Also verify identity: b ≈ -2 * slope_W
        """
        protocol = self.protocols['entropy']
        
        results = self.mc_runner.run_entropy_verification(
            protocol_name=protocol.name,
            n_pe_variants=protocol.n_pe_variants,
            samples_per_variant=protocol.samples_per_variant
        )
        
        # Check for diffusion floor effects
        self._check_diffusion_floor(results)
        
        return results
    
    async def run_protocol_3_diminishing_returns(self) -> MonteCarloResults:
        """
        Protocol 3: Verify Pe_ctx(N) = a + b·ln(N)
        
        Tests logarithmic scaling of Pe_ctx with number of context chunks
        """
        protocol = self.protocols['diminishing_returns']
        
        results = self.mc_runner.run_diminishing_returns_test(
            protocol_name=protocol.name,
            max_chunks=20,
            n_chunk_values=10,
            samples_per_n=protocol.samples_per_variant
        )
        
        return results
    
    def _analyze_sharpening_results(self, results: MonteCarloResults):
        """Additional analysis for sharpening protocol"""
        # Group by Pe_ctx
        from collections import defaultdict
        pe_groups = defaultdict(list)
        
        for r in results.results:
            pe_groups[r.pe_ctx].append(r.metrics['W_normalized'])
        
        # Calculate variance within each Pe group
        variances = {}
        for pe, values in pe_groups.items():
            variances[pe] = np.var(values) if len(values) > 1 else 0
        
        # Add to verification details
        results.verification_details['within_pe_variances'] = variances
        results.verification_details['mean_variance'] = np.mean(list(variances.values()))
    
    def _check_diffusion_floor(self, results: MonteCarloResults):
        """
        Check for diffusion floor effects using diagnostic from README
        
        If b << 2/3, estimate γ from: γ*Pe_ctx = (2/3)/b - 1
        """
        b_measured = results.verification_details.get('b_measured', 0)
        
        if b_measured > 0 and b_measured < 0.5:  # Significantly below 2/3
            # Average Pe_ctx
            pe_values = [r.pe_ctx for r in results.results]
            avg_pe = np.mean(pe_values)
            
            # Estimate diffusion floor parameter
            gamma_pe = (2/3) / b_measured - 1
            gamma = gamma_pe / avg_pe
            
            results.verification_details['diffusion_floor_detected'] = True
            results.verification_details['gamma_estimate'] = gamma
            results.verification_details['recommendation'] = (
                f"High diffusion floor detected (γ ≈ {gamma:.3f}). "
                "Consider: reducing paraphrases, fixing conflicts, "
                "moving key facts earlier, lowering temperature."
            )
        else:
            results.verification_details['diffusion_floor_detected'] = False
    
    def run_quick_diagnostic(self, n_samples: int = 20) -> Dict[str, float]:
        """
        Run a quick diagnostic test with minimal samples
        
        Useful for rapid iteration during development
        """
        self.logger.info("Running quick diagnostic")
        
        # Test 3 Pe values
        pe_values = [0.3, 1.0, 3.0]
        results = []
        
        for pe in pe_values:
            variant_results = self.mc_runner._run_variant_simulations(
                pe, n_samples, focus_metric='width'
            )
            results.extend(variant_results)
        
        # Quick slope calculation
        pe_list = [r.pe_ctx for r in results]
        w_norm_list = [r.metrics['W_normalized'] for r in results]
        
        if len(set(pe_list)) >= 2:
            from scipy import stats
            log_pe = np.log(pe_list)
            log_w = np.log(w_norm_list)
            slope, _, r_value, _, _ = stats.linregress(log_pe, log_w)
            
            return {
                'quick_slope': slope,
                'expected': -1/3,
                'error': abs(slope - (-1/3)),
                'r_squared': r_value**2,
                'n_samples': len(results)
            }
        
        return {'error': 'Insufficient data for diagnostic'}
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        Validate that the experimental setup is correct
        
        Checks:
        - Task dataset is loaded
        - LLM/embedding clients are configured
        - Pe_ctx calculator is working
        - Metrics calculators are initialized
        """
        checks = {
            'tasks_loaded': len(self.mc_runner.tasks) > 0,
            'llm_configured': self.mc_runner.llm is not None,
            'embedder_configured': self.mc_runner.embedder is not None,
            'pe_calculator_ready': True,
            'metrics_ready': True,
            'results_dir_exists': self.mc_runner.results_dir.exists()
        }
        
        # Test Pe calculation
        try:
            test_pe, _ = self.mc_runner.pe_calculator.calculate_pe_ctx()
            checks['pe_calculator_ready'] = test_pe > 0
        except:
            checks['pe_calculator_ready'] = False
        
        # Test metrics calculation
        try:
            test_embeddings = np.random.randn(10, 384)
            test_width = self.mc_runner.metrics_calculator.width_measurer.calculate_from_embeddings(
                test_embeddings
            )
            checks['metrics_ready'] = test_width > 0
        except:
            checks['metrics_ready'] = False
        
        all_valid = all(checks.values())
        
        if not all_valid:
            failed_checks = [k for k, v in checks.items() if not v]
            self.logger.warning(f"Setup validation failed: {failed_checks}")
        
        return checks