"""
Complete verification suite for Coffee Law
"""
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import pandas as pd

from .power_law_analyzer import PowerLawAnalyzer, PowerLawFit

@dataclass
class EntropyVerificationResult:
    """Results from entropy slope verification"""
    b_measured: float
    b_expected: float
    b_error: float
    b_passed: bool
    w_slope: float
    identity_ratio: float
    identity_passed: bool
    r_squared: float
    diffusion_floor_gamma: Optional[float]

@dataclass
class FullVerificationResult:
    """Combined results from all verification tests"""
    sharpening_passed: bool
    entropy_passed: bool
    diminishing_returns_passed: bool
    overall_passed: bool
    sharpening_details: Dict[str, Any]
    entropy_details: EntropyVerificationResult
    diminishing_details: Dict[str, Any]
    recommendations: List[str]

class VerificationSuite:
    """
    Complete verification suite for Coffee Law relationships
    
    Implements all three protocols from README:
    1. Sharpening: W/√D_eff ∝ Pe_ctx^(-1/3)
    2. Entropy: H = a + b*ln(Pe_ctx) with b ≈ 2/3
    3. Diminishing returns: α(N) ∼ N^(-1/3)
    """
    
    def __init__(self):
        self.power_law_analyzer = PowerLawAnalyzer()
        
    def verify_all(self, results_data: Dict[str, List]) -> FullVerificationResult:
        """
        Run all verification tests on experimental data
        
        Args:
            results_data: Dictionary containing:
                - 'pe_ctx': List of Pe_ctx values
                - 'w_normalized': List of W/√D_eff values
                - 'h_values': List of entropy H values
                - 'n_chunks': List of chunk counts N
                - 'alpha_values': List of coupling α values
                
        Returns:
            FullVerificationResult with pass/fail status and details
        """
        # Protocol 1: Sharpening
        sharpening_result = self.verify_sharpening(
            np.array(results_data['pe_ctx']),
            np.array(results_data['w_normalized'])
        )
        
        # Protocol 2: Entropy
        entropy_result = self.verify_entropy_slope(
            np.array(results_data['pe_ctx']),
            np.array(results_data['h_values']),
            np.array(results_data['w_normalized'])
        )
        
        # Protocol 3: Logarithmic scaling (new Law 3)
        diminishing_result = None
        diminishing_passed = True
        
        # Check for new format first
        if 'diminishing_details' in results_data:
            diminishing_result = results_data['diminishing_details']
            diminishing_passed = diminishing_result.get('passed', False)
        # Fallback to old format
        elif 'n_chunks' in results_data and 'alpha_values' in results_data:
            # Skip old verification if no valid alpha values
            alpha_values = np.array(results_data['alpha_values'])
            if len(alpha_values) > 0 and np.any(alpha_values > 0):
                diminishing_result = self.verify_diminishing_returns(
                    np.array(results_data['n_chunks']),
                    alpha_values
                )
                diminishing_passed = diminishing_result['within_tolerance']
            else:
                diminishing_result = {'error': 'No valid alpha values'}
                diminishing_passed = False
        
        # Overall assessment
        overall_passed = (
            sharpening_result['within_tolerance'] and
            entropy_result.b_passed and
            entropy_result.identity_passed and
            diminishing_passed
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            sharpening_result, entropy_result, diminishing_result
        )
        
        return FullVerificationResult(
            sharpening_passed=sharpening_result['within_tolerance'],
            entropy_passed=entropy_result.b_passed and entropy_result.identity_passed,
            diminishing_returns_passed=diminishing_passed,
            overall_passed=overall_passed,
            sharpening_details=sharpening_result,
            entropy_details=entropy_result,
            diminishing_details=diminishing_result,
            recommendations=recommendations
        )
    
    def verify_sharpening(self, 
                         pe_ctx: np.ndarray,
                         w_normalized: np.ndarray) -> Dict[str, Any]:
        """
        Protocol 1: Verify W/√D_eff ∝ Pe_ctx^(-1/3)
        """
        return self.power_law_analyzer.verify_coffee_law_sharpening(pe_ctx, w_normalized)
    
    def verify_entropy_slope(self,
                           pe_ctx: np.ndarray,
                           h_values: np.ndarray,
                           w_normalized: np.ndarray) -> EntropyVerificationResult:
        """
        Protocol 2: Verify H = a + b*ln(Pe_ctx) with b ≈ 2/3
        Also check identity: b ≈ -2 * slope_W
        """
        # Fit H vs ln(Pe_ctx)
        log_pe = np.log(pe_ctx)
        h_slope, h_intercept, h_r, h_p, h_stderr = stats.linregress(log_pe, h_values)
        
        # Get W slope for identity check
        log_w = np.log(w_normalized)
        w_slope, _, _, _, _ = stats.linregress(log_pe, log_w)
        
        # Expected values from README
        b_expected = 2/3
        b_tolerance = 0.10
        
        # Checks
        b_passed = abs(h_slope - b_expected) < b_tolerance
        
        # Identity check: b ≈ -2 * slope_W
        identity_ratio = h_slope / (-2 * w_slope) if w_slope != 0 else float('inf')
        identity_passed = abs(identity_ratio - 1.0) < 0.15
        
        # Diffusion floor diagnostic
        gamma = None
        if h_slope < 0.5:  # Significantly below 2/3
            avg_pe = np.mean(pe_ctx)
            gamma_pe = (b_expected / h_slope) - 1
            gamma = gamma_pe / avg_pe
        
        return EntropyVerificationResult(
            b_measured=h_slope,
            b_expected=b_expected,
            b_error=h_stderr,
            b_passed=b_passed,
            w_slope=w_slope,
            identity_ratio=identity_ratio,
            identity_passed=identity_passed,
            r_squared=h_r**2,
            diffusion_floor_gamma=gamma
        )
    
    def verify_diminishing_returns(self,
                                 n_chunks: np.ndarray,
                                 alpha_values: np.ndarray) -> Dict[str, Any]:
        """
        Protocol 3: Verify α(N) ∼ N^(-1/3) (legacy method)
        
        Note: This is kept for backward compatibility.
        The new Law 3 verifies Pe_ctx(N) = a + b·ln(N) directly in monte_carlo
        """
        try:
            return self.power_law_analyzer.verify_diminishing_returns(n_chunks, alpha_values)
        except ValueError as e:
            return {
                'within_tolerance': False,
                'error': str(e),
                'measured_exponent': None,
                'expected_exponent': -1/3,
                'r_squared': 0
            }
    
    def _generate_recommendations(self,
                                sharpening: Dict[str, Any],
                                entropy: EntropyVerificationResult,
                                diminishing: Optional[Dict]) -> List[str]:
        """
        Generate actionable recommendations based on verification results
        """
        recommendations = []
        
        # Sharpening issues
        if not sharpening['within_tolerance']:
            if sharpening['measured_exponent'] > -1/3 + 0.07:
                recommendations.append(
                    "Sharpening too weak: Increase template strength, improve front-loading, "
                    "or reduce redundancy/conflicts"
                )
            else:
                recommendations.append(
                    "Sharpening too strong: May indicate over-structured context or "
                    "insufficient variation in Pe_ctx"
                )
        
        # Entropy slope issues  
        if not entropy.b_passed:
            if entropy.b_measured < 0.5:
                recommendations.append(
                    f"Low entropy slope (b={entropy.b_measured:.3f}) indicates high diffusion. "
                    "Reduce paraphrase chains, resolve conflicts, canonicalize units/style"
                )
                if entropy.diffusion_floor_gamma:
                    recommendations.append(
                        f"Diffusion floor detected (γ≈{entropy.diffusion_floor_gamma:.3f}). "
                        "Consider lowering temperature and improving chunk quality"
                    )
        
        # Identity check failure
        if not entropy.identity_passed:
            recommendations.append(
                f"Identity check failed (ratio={entropy.identity_ratio:.3f}). "
                "This suggests inconsistency between width and entropy measurements"
            )
        
        # Diminishing returns / Logarithmic scaling
        if diminishing:
            # Handle new format (logarithmic scaling)
            if 'passed' in diminishing:
                if not diminishing['passed']:
                    if 'formula' in diminishing:
                        recommendations.append(
                            f"Logarithmic scaling failed: {diminishing.get('formula', 'N/A')}. "
                            f"R² = {diminishing.get('r_squared', 0):.3f}. "
                            "Check Pe_ctx calculation for different chunk counts."
                        )
                    else:
                        recommendations.append(
                            "Logarithmic scaling verification failed. "
                            "Check context quality scaling with chunk count."
                        )
            # Handle old format (power law)
            elif 'within_tolerance' in diminishing and not diminishing['within_tolerance']:
                recommendations.append(
                    "Diminishing returns not following N^(-1/3). "
                    "Check chunk selection algorithm and N_eff calculation"
                )
        
        # R-squared warnings
        if sharpening and 'r_squared' in sharpening and sharpening['r_squared'] < 0.7:
            recommendations.append(
                f"Low R² ({sharpening['r_squared']:.3f}) in sharpening fit suggests high variance. "
                "Increase sample size or improve measurement consistency"
            )
        
        return recommendations
    
    def create_verification_report(self, result: FullVerificationResult) -> str:
        """
        Create a formatted verification report
        """
        report = []
        report.append("="*60)
        report.append("COFFEE LAW VERIFICATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Overall status
        status = "✓ PASSED" if result.overall_passed else "✗ FAILED"
        report.append(f"Overall Verification: {status}")
        report.append("")
        
        # Protocol 1: Sharpening
        report.append("Protocol 1: Cube-root Sharpening")
        report.append("-" * 40)
        sharp = result.sharpening_details
        report.append(f"Expected: W/√D_eff ∝ Pe_ctx^(-1/3)")
        report.append(f"Measured exponent: {sharp['measured_exponent']:.4f} ± {sharp['fit'].exponent_error:.4f}")
        report.append(f"Status: {'✓ PASS' if result.sharpening_passed else '✗ FAIL'}")
        report.append(f"R²: {sharp['r_squared']:.4f}")
        report.append("")
        
        # Protocol 2: Entropy
        report.append("Protocol 2: Entropy Slope")
        report.append("-" * 40)
        ent = result.entropy_details
        report.append(f"Expected: H = a + b*ln(Pe_ctx) with b ≈ 2/3")
        report.append(f"Measured b: {ent.b_measured:.4f} ± {ent.b_error:.4f}")
        report.append(f"Status (b): {'✓ PASS' if ent.b_passed else '✗ FAIL'}")
        report.append(f"Identity check (b ≈ -2*slope_W): {ent.identity_ratio:.3f}")
        report.append(f"Status (identity): {'✓ PASS' if ent.identity_passed else '✗ FAIL'}")
        report.append(f"R²: {ent.r_squared:.4f}")
        
        if ent.diffusion_floor_gamma:
            report.append(f"⚠️  Diffusion floor detected: γ ≈ {ent.diffusion_floor_gamma:.3f}")
        report.append("")
        
        # Protocol 3: Diminishing Returns / Logarithmic Scaling
        if result.diminishing_details:
            dim = result.diminishing_details
            
            # Handle new logarithmic format
            if 'formula' in dim:
                report.append("Protocol 3: Logarithmic Context Scaling")
                report.append("-" * 40)
                report.append(f"Expected: Pe_ctx(N) = a + b·ln(N)")
                report.append(f"Measured: {dim['formula']}")
                report.append(f"Status: {'✓ PASS' if result.diminishing_returns_passed else '✗ FAIL'}")
                report.append(f"R²: {dim.get('r_squared', 0):.4f}")
                if 'b_slope' in dim:
                    report.append(f"b slope: {dim['b_slope']:.3f} (reasonable: {dim.get('b_reasonable', False)})")
            # Handle old power law format
            elif 'measured_exponent' in dim:
                report.append("Protocol 3: Diminishing Returns")
                report.append("-" * 40)
                report.append(f"Expected: α(N) ∼ N^(-1/3)")
                report.append(f"Measured exponent: {dim['measured_exponent']:.4f}")
                report.append(f"Status: {'✓ PASS' if result.diminishing_returns_passed else '✗ FAIL'}")
                report.append(f"R²: {dim.get('r_squared', 0):.4f}")
            else:
                report.append("Protocol 3: Failed")
                report.append("-" * 40)
                report.append(f"Error: {dim.get('error', 'Unknown error')}")
            
            report.append("")
        
        # Recommendations
        if result.recommendations:
            report.append("Recommendations:")
            report.append("-" * 40)
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)