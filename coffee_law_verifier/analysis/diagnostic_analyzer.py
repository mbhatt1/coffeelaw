"""
Diagnostic analysis tools for Coffee Law verification
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.decomposition import PCA
import pandas as pd
from dataclasses import dataclass

@dataclass
class DiagnosticResult:
    """Container for diagnostic analysis results"""
    test_name: str
    passed: bool
    value: float
    threshold: Optional[float]
    details: Dict[str, Any]
    recommendation: Optional[str]

class DiagnosticAnalyzer:
    """
    Advanced diagnostics for Coffee Law verification
    
    Implements additional checks beyond the three main protocols:
    - Diffusion floor analysis
    - Pe_ctx range validation
    - Sample size adequacy
    - Outlier detection
    - Consistency checks
    """
    
    def __init__(self):
        self.diagnostics_run = []
        
    def run_full_diagnostics(self, 
                           results_data: Dict[str, np.ndarray],
                           verbose: bool = True) -> List[DiagnosticResult]:
        """
        Run comprehensive diagnostic suite
        
        Args:
            results_data: Dictionary with experimental data
            verbose: Whether to print diagnostics as they run
            
        Returns:
            List of DiagnosticResult objects
        """
        diagnostics = []
        
        # Check Pe_ctx range
        if 'pe_ctx' in results_data:
            diag = self.check_pe_ctx_range(results_data['pe_ctx'])
            diagnostics.append(diag)
            if verbose:
                self._print_diagnostic(diag)
        
        # Check sample size adequacy
        diag = self.check_sample_size(results_data)
        diagnostics.append(diag)
        if verbose:
            self._print_diagnostic(diag)
        
        # Check for outliers
        for key in ['w_normalized', 'h_values']:
            if key in results_data:
                diag = self.check_outliers(results_data[key], metric_name=key)
                diagnostics.append(diag)
                if verbose:
                    self._print_diagnostic(diag)
        
        # Diffusion floor analysis
        if 'h_values' in results_data and 'pe_ctx' in results_data:
            diag = self.analyze_diffusion_floor(
                results_data['pe_ctx'],
                results_data['h_values']
            )
            diagnostics.append(diag)
            if verbose:
                self._print_diagnostic(diag)
        
        # Data quality checks
        diag = self.check_data_quality(results_data)
        diagnostics.append(diag)
        if verbose:
            self._print_diagnostic(diag)
        
        # Measurement consistency
        if all(k in results_data for k in ['w_normalized', 'h_values']):
            diag = self.check_measurement_consistency(
                results_data['w_normalized'],
                results_data['h_values']
            )
            diagnostics.append(diag)
            if verbose:
                self._print_diagnostic(diag)
        
        self.diagnostics_run = diagnostics
        return diagnostics
    
    def check_pe_ctx_range(self, pe_ctx: np.ndarray) -> DiagnosticResult:
        """
        Verify Pe_ctx spans >1 decade as required
        """
        # Convert to numpy array if needed
        pe_ctx = np.array(pe_ctx) if isinstance(pe_ctx, list) else pe_ctx
        
        pe_min = pe_ctx.min()
        pe_max = pe_ctx.max()
        decades_spanned = np.log10(pe_max / pe_min)
        
        passed = decades_spanned >= 1.0
        
        recommendation = None
        if not passed:
            recommendation = (
                f"Pe_ctx range only spans {decades_spanned:.2f} decades. "
                "Increase range to >1 decade for valid verification."
            )
        
        return DiagnosticResult(
            test_name="Pe_ctx Range Check",
            passed=passed,
            value=decades_spanned,
            threshold=1.0,
            details={
                'pe_min': pe_min,
                'pe_max': pe_max,
                'n_unique_values': len(np.unique(pe_ctx))
            },
            recommendation=recommendation
        )
    
    def check_sample_size(self, results_data: Dict[str, Any]) -> DiagnosticResult:
        """
        Check if sample sizes are adequate for reliable fitting
        """
        # Check for raw sample count first
        if 'n_raw_samples' in results_data:
            sample_size = results_data['n_raw_samples']
            min_required = 100  # For Monte Carlo simulations
            details = {"sample_size": sample_size, "is_raw": True}
        else:
            # Fallback to checking array lengths
            sample_sizes = {}
            for k, v in results_data.items():
                if v is None:
                    continue
                # Only check length for array-like objects
                if hasattr(v, '__len__') and not isinstance(v, (str, bytes)):
                    sample_sizes[k] = len(v)
                elif isinstance(v, (int, float, np.number)):
                    # Skip scalar values
                    continue
            
            if not sample_sizes:
                # No array data found
                return DiagnosticResult(
                    test_name="Sample Size Adequacy",
                    passed=True,
                    value=0.0,
                    threshold=0.0,
                    details={"note": "No array data found"},
                    recommendation=None
                )
            
            sample_size = min(sample_sizes.values())
            min_required = 20  # For aggregated data
            details = {"sample_size": sample_size, "is_raw": False, "array_sizes": sample_sizes}
        
        passed = sample_size >= min_required
        
        recommendation = None
        if not passed:
            recommendation = (
                f"Sample size ({sample_size}) below recommended minimum ({min_required}). "
                "Increase samples_per_variant for more reliable results."
            )
        
        return DiagnosticResult(
            test_name="Sample Size Adequacy",
            passed=passed,
            value=float(sample_size),
            threshold=float(min_required),
            details=details,
            recommendation=recommendation
        )
    
    def check_outliers(self, data: np.ndarray, metric_name: str) -> DiagnosticResult:
        """
        Detect outliers using IQR method
        """
        # Convert to numpy array if needed
        data = np.array(data) if isinstance(data, list) else data
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = (data < lower_bound) | (data > upper_bound)
        n_outliers = outliers.sum()
        outlier_fraction = n_outliers / len(data)
        
        # Pass if less than 5% outliers
        passed = outlier_fraction < 0.05
        
        recommendation = None
        if not passed:
            recommendation = (
                f"High outlier fraction ({outlier_fraction:.2%}) in {metric_name}. "
                "Consider robust regression or outlier removal."
            )
        
        return DiagnosticResult(
            test_name=f"Outlier Check - {metric_name}",
            passed=passed,
            value=outlier_fraction,
            threshold=0.05,
            details={
                'n_outliers': int(n_outliers),
                'outlier_indices': np.where(outliers)[0].tolist(),
                'bounds': (lower_bound, upper_bound)
            },
            recommendation=recommendation
        )
    
    def analyze_diffusion_floor(self, 
                              pe_ctx: np.ndarray,
                              h_values: np.ndarray) -> DiagnosticResult:
        """
        Detailed diffusion floor analysis
        
        From README: If b << 2/3, estimate γ from γ*Pe_ctx = (2/3)/b - 1
        """
        # Fit entropy slope
        log_pe = np.log(pe_ctx)
        b_measured, _, _, _, _ = stats.linregress(log_pe, h_values)
        
        expected_b = 2/3
        
        # Check if diffusion floor is affecting results
        if b_measured < 0.5:  # Significantly below expected
            avg_pe = np.mean(pe_ctx)
            gamma_pe = (expected_b / b_measured) - 1
            gamma = gamma_pe / avg_pe
            
            passed = False
            recommendation = (
                f"Strong diffusion floor detected (γ≈{gamma:.3f}). "
                "To improve: 1) Reduce redundancy/paraphrases, "
                "2) Fix contradictions, 3) Enforce consistent style/units, "
                "4) Lower temperature, 5) Move key facts earlier"
            )
            
            details = {
                'b_measured': b_measured,
                'b_expected': expected_b,
                'gamma': gamma,
                'gamma_pe_product': gamma_pe,
                'severity': 'high' if gamma > 1.0 else 'moderate'
            }
        else:
            passed = True
            gamma = 0.0
            recommendation = None
            details = {
                'b_measured': b_measured,
                'b_expected': expected_b,
                'gamma': gamma,
                'status': 'No significant diffusion floor detected'
            }
        
        return DiagnosticResult(
            test_name="Diffusion Floor Analysis",
            passed=passed,
            value=gamma,
            threshold=1.0,  # γ > 1 indicates severe floor
            details=details,
            recommendation=recommendation
        )
    
    def check_data_quality(self, results_data: Dict[str, np.ndarray]) -> DiagnosticResult:
        """
        Check for NaN, infinite values, or other data quality issues
        """
        issues = {}
        total_issues = 0
        
        for key, data in results_data.items():
            # Skip non-array data (including dicts)
            if isinstance(data, dict) or not hasattr(data, '__len__') or isinstance(data, (str, bytes)):
                continue
            
            # Skip scalar values
            if isinstance(data, (int, float, np.number)):
                continue
                
            # Convert to numpy array if needed
            try:
                data = np.array(data) if isinstance(data, list) else data
                
                # Skip if not a proper numpy array
                if not hasattr(data, 'astype'):
                    continue
                
                # Try to convert to float array
                data_numeric = data.astype(float)
                n_nan = np.isnan(data_numeric).sum()
                n_inf = np.isinf(data_numeric).sum()
                n_negative = (data_numeric < 0).sum() if key in ['w_normalized', 'h_values', 'pe_ctx'] else 0
            except (ValueError, TypeError, AttributeError):
                # Non-numeric data or conversion failed, skip
                continue
            
            if n_nan + n_inf + n_negative > 0:
                issues[key] = {
                    'nan': int(n_nan),
                    'inf': int(n_inf),
                    'negative': int(n_negative)
                }
                total_issues += n_nan + n_inf + n_negative
        
        passed = total_issues == 0
        
        recommendation = None
        if not passed:
            recommendation = (
                f"Data quality issues detected in {list(issues.keys())}. "
                "Check measurement procedures and data preprocessing."
            )
        
        return DiagnosticResult(
            test_name="Data Quality Check",
            passed=passed,
            value=float(total_issues),
            threshold=0.0,
            details=issues,
            recommendation=recommendation
        )
    
    def check_measurement_consistency(self,
                                    w_normalized: np.ndarray,
                                    h_values: np.ndarray) -> DiagnosticResult:
        """
        Check if W and H measurements are consistent
        
        They should be negatively correlated if Coffee Law holds
        """
        # Calculate correlation
        correlation = np.corrcoef(w_normalized, h_values)[0, 1]
        
        # Expect negative correlation
        passed = correlation < -0.3  # Moderate negative correlation
        
        recommendation = None
        if not passed:
            if correlation > 0:
                recommendation = (
                    "W and H show positive correlation, suggesting measurement issues. "
                    "Verify measurement procedures and check for confounding factors."
                )
            else:
                recommendation = (
                    "W and H correlation is weak. Increase Pe_ctx range or sample size."
                )
        
        return DiagnosticResult(
            test_name="W-H Consistency Check",
            passed=passed,
            value=correlation,
            threshold=-0.3,
            details={
                'correlation': correlation,
                'expected_sign': 'negative',
                'strength': self._correlation_strength(correlation)
            },
            recommendation=recommendation
        )
    
    def _correlation_strength(self, corr: float) -> str:
        """Categorize correlation strength"""
        abs_corr = abs(corr)
        if abs_corr < 0.3:
            return "weak"
        elif abs_corr < 0.7:
            return "moderate"
        else:
            return "strong"
    
    def _print_diagnostic(self, result: DiagnosticResult):
        """Pretty print a diagnostic result"""
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"\n{result.test_name}: {status}")
        print(f"  Value: {result.value:.4f}")
        if result.threshold is not None:
            print(f"  Threshold: {result.threshold:.4f}")
        if result.recommendation:
            print(f"  → {result.recommendation}")
    
    def generate_diagnostic_summary(self) -> str:
        """Generate summary of all diagnostics run"""
        if not self.diagnostics_run:
            return "No diagnostics have been run yet."
        
        summary = []
        summary.append("DIAGNOSTIC SUMMARY")
        summary.append("=" * 50)
        
        passed = sum(1 for d in self.diagnostics_run if d.passed)
        total = len(self.diagnostics_run)
        
        summary.append(f"Total tests: {total}")
        summary.append(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
        summary.append("")
        
        # Failed tests
        failed = [d for d in self.diagnostics_run if not d.passed]
        if failed:
            summary.append("Failed Tests:")
            summary.append("-" * 30)
            for diag in failed:
                summary.append(f"• {diag.test_name}")
                if diag.recommendation:
                    summary.append(f"  → {diag.recommendation}")
            summary.append("")
        
        # Critical issues
        critical = [d for d in failed if 'severity' in d.details and d.details['severity'] == 'high']
        if critical:
            summary.append("⚠️  CRITICAL ISSUES:")
            for diag in critical:
                summary.append(f"• {diag.test_name}: {diag.recommendation}")
        
        return "\n".join(summary)