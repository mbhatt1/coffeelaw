"""
Power law analysis for Coffee Law verification
"""
import numpy as np
from typing import Tuple, Dict, List, Optional
from scipy import stats, optimize
from dataclasses import dataclass
import warnings

@dataclass
class PowerLawFit:
    """Results from power law fitting"""
    exponent: float
    coefficient: float
    exponent_error: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    residuals: np.ndarray
    passed: bool
    expected_exponent: Optional[float]

class PowerLawAnalyzer:
    """
    Fit and verify power law relationships for Coffee Law
    
    Key relationships to verify:
    1. W/√D_eff ∝ Pe_ctx^(-1/3) (Law 1)
    2. H = H₀ + (2/3)ln(Pe_ctx) (Law 2)
    3. Pe_ctx(N) = a + b·ln(N) (Law 3)
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        
    def fit_power_law(self,
                     x: np.ndarray,
                     y: np.ndarray,
                     expected_exponent: Optional[float] = None,
                     tolerance: Optional[float] = None) -> PowerLawFit:
        """
        Fit power law y = a * x^b using log-log linear regression
        
        Args:
            x: Independent variable values
            y: Dependent variable values
            expected_exponent: Expected value of b (e.g., -1/3)
            tolerance: Tolerance for acceptance (e.g., 0.07)
            
        Returns:
            PowerLawFit object with results
        """
        # Remove any zero or negative values
        valid_mask = (x > 0) & (y > 0)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 3:
            raise ValueError("Insufficient valid data points for power law fitting")
        
        # Log transform
        log_x = np.log(x_valid)
        log_y = np.log(y_valid)
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        
        # Calculate confidence interval
        n = len(log_x)
        t_stat = stats.t.ppf((1 + self.confidence_level) / 2, n - 2)
        ci = (slope - t_stat * std_err, slope + t_stat * std_err)
        
        # Calculate residuals
        predicted_log_y = slope * log_x + intercept
        residuals = log_y - predicted_log_y
        
        # Check if passes expected value test
        passed = True
        if expected_exponent is not None:
            if tolerance is not None:
                passed = abs(slope - expected_exponent) <= tolerance
            else:
                # Check if expected value is within confidence interval
                passed = ci[0] <= expected_exponent <= ci[1]
        
        return PowerLawFit(
            exponent=slope,
            coefficient=np.exp(intercept),
            exponent_error=std_err,
            r_squared=r_value**2,
            p_value=p_value,
            confidence_interval=ci,
            residuals=residuals,
            passed=passed,
            expected_exponent=expected_exponent
        )
    
    def fit_robust_power_law(self,
                           x: np.ndarray,
                           y: np.ndarray,
                           expected_exponent: Optional[float] = None) -> PowerLawFit:
        """
        Fit power law using robust regression (Huber regression)
        
        More resistant to outliers than ordinary least squares
        """
        from sklearn.linear_model import HuberRegressor
        
        # Remove invalid values
        valid_mask = (x > 0) & (y > 0)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 5:
            # Fall back to regular fitting for small datasets
            return self.fit_power_law(x, y, expected_exponent)
        
        # Log transform
        log_x = np.log(x_valid).reshape(-1, 1)
        log_y = np.log(y_valid)
        
        # Robust regression
        huber = HuberRegressor()
        huber.fit(log_x, log_y)
        
        slope = huber.coef_[0]
        intercept = huber.intercept_
        
        # Calculate r-squared manually
        predicted = huber.predict(log_x)
        ss_res = np.sum((log_y - predicted) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Bootstrap for confidence intervals
        ci = self._bootstrap_confidence_interval(x_valid, y_valid, n_bootstrap=1000)
        
        # Simple pass/fail check
        passed = True
        if expected_exponent is not None:
            passed = ci[0] <= expected_exponent <= ci[1]
        
        return PowerLawFit(
            exponent=slope,
            coefficient=np.exp(intercept),
            exponent_error=(ci[1] - ci[0]) / (2 * 1.96),  # Approximate stderr
            r_squared=r_squared,
            p_value=0.0,  # Not computed for robust regression
            confidence_interval=ci,
            residuals=log_y - predicted,
            passed=passed,
            expected_exponent=expected_exponent
        )
    
    def verify_coffee_law_sharpening(self,
                                   pe_ctx: np.ndarray,
                                   w_normalized: np.ndarray) -> Dict[str, any]:
        """
        Verify the sharpening law: W/√D_eff ∝ Pe_ctx^(-1/3)
        
        From README: Pass if slope = -0.33 ± 0.07
        """
        fit = self.fit_power_law(
            pe_ctx, 
            w_normalized,
            expected_exponent=-1/3,
            tolerance=0.07
        )
        
        # Additional diagnostics
        diagnostics = {
            'fit': fit,
            'measured_exponent': fit.exponent,
            'expected_exponent': -1/3,
            'deviation': abs(fit.exponent - (-1/3)),
            'within_tolerance': fit.passed,
            'r_squared': fit.r_squared,
            'p_value': fit.p_value,
            'n_points': len(pe_ctx),
            'pe_range_decades': np.log10(pe_ctx.max() / pe_ctx.min())
        }
        
        # Check if we have >1 decade of Pe_ctx as required
        if diagnostics['pe_range_decades'] < 1.0:
            diagnostics['warning'] = f"Pe_ctx range only spans {diagnostics['pe_range_decades']:.2f} decades (>1 required)"
        
        return diagnostics
    
    def verify_diminishing_returns(self,
                                 n_chunks: np.ndarray,
                                 alpha: np.ndarray) -> Dict[str, any]:
        """
        Verify diminishing returns: α(N) ∼ N^(-1/3)
        
        From README: Pass if slope = -0.33 ± 0.10
        """
        fit = self.fit_power_law(
            n_chunks,
            np.abs(alpha),  # Use absolute value of coupling
            expected_exponent=-1/3,
            tolerance=0.10
        )
        
        diagnostics = {
            'fit': fit,
            'measured_exponent': fit.exponent,
            'expected_exponent': -1/3,
            'deviation': abs(fit.exponent - (-1/3)),
            'within_tolerance': fit.passed,
            'r_squared': fit.r_squared,
            'n_range': [n_chunks.min(), n_chunks.max()],
            'n_points': len(n_chunks)
        }
        
        return diagnostics
    
    def _bootstrap_confidence_interval(self,
                                     x: np.ndarray,
                                     y: np.ndarray,
                                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap
        """
        n_samples = len(x)
        exponents = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            # Fit power law
            try:
                log_x = np.log(x_boot)
                log_y = np.log(y_boot)
                slope, _, _, _, _ = stats.linregress(log_x, log_y)
                exponents.append(slope)
            except:
                continue
        
        # Calculate confidence interval
        if exponents:
            lower = np.percentile(exponents, (100 - self.confidence_level * 100) / 2)
            upper = np.percentile(exponents, 100 - (100 - self.confidence_level * 100) / 2)
            return (lower, upper)
        
        return (float('-inf'), float('inf'))
    
    def test_power_law_goodness_of_fit(self,
                                     x: np.ndarray,
                                     y: np.ndarray,
                                     fit: PowerLawFit) -> Dict[str, float]:
        """
        Additional goodness-of-fit tests for power law
        """
        # Kolmogorov-Smirnov test on residuals
        _, ks_p_value = stats.kstest(fit.residuals, 'norm')
        
        # Shapiro-Wilk test for normality of residuals
        _, sw_p_value = stats.shapiro(fit.residuals)
        
        # Durbin-Watson test for autocorrelation
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = durbin_watson(fit.residuals)
        
        # Calculate AIC and BIC
        n = len(x)
        log_likelihood = -0.5 * n * (np.log(2 * np.pi) + np.log(np.var(fit.residuals)) + 1)
        aic = 2 * 2 - 2 * log_likelihood  # 2 parameters (slope, intercept)
        bic = np.log(n) * 2 - 2 * log_likelihood
        
        return {
            'ks_test_p_value': ks_p_value,
            'shapiro_wilk_p_value': sw_p_value,
            'durbin_watson': dw_stat,
            'aic': aic,
            'bic': bic,
            'residual_std': np.std(fit.residuals)
        }