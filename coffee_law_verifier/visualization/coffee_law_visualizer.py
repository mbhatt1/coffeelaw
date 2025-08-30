"""
Main visualization module for Coffee Law verification results
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

class CoffeeLawVisualizer:
    """
    Create verification plots for Coffee Law experiments
    
    Generates all plots specified in the README verification protocol
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        plt.style.use(style)
        sns.set_palette("husl")
        
        # Set default figure parameters
        self.fig_params = {
            'figsize': (12, 10),
            'dpi': 150,
            'facecolor': 'white'
        }
        
        # Color scheme
        self.colors = {
            'data': '#2E86AB',
            'fit': '#E63946',
            'expected': '#1D3557',
            'confidence': '#F1FAEE',
            'pass': '#06D6A0',
            'fail': '#E63946'
        }
        
    def create_verification_report(self, 
                                 results: Dict[str, Any],
                                 save_path: Optional[Path] = None) -> Figure:
        """
        Create comprehensive verification report with all plots
        
        Args:
            results: Dictionary containing all experimental results
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 20))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: W/√D_eff vs Pe_ctx (Protocol 1)
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_sharpening_law(ax1, results)
        
        # Plot 2: H vs ln(Pe_ctx) (Protocol 2)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_entropy_scaling(ax2, results)
        
        # Plot 3: Identity check
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_identity_check(ax3, results)
        
        # Plot 4: α(N) vs N (Protocol 3)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_diminishing_returns(ax4, results)
        
        # Plot 5: Residuals analysis
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_residuals(ax5, results)
        
        # Plot 6: Summary dashboard
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_summary_dashboard(ax6, results)
        
        # Main title
        fig.suptitle('Coffee Law Verification Report', fontsize=20, y=0.99)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def _plot_sharpening_law(self, ax: plt.Axes, results: Dict[str, Any]):
        """
        Plot W/√D_eff vs Pe_ctx in log-log space
        Expected: slope = -1/3
        """
        # Convert to numpy arrays if needed
        pe_ctx = np.array(results['pe_ctx']) if isinstance(results['pe_ctx'], list) else results['pe_ctx']
        w_normalized = np.array(results['w_normalized']) if isinstance(results['w_normalized'], list) else results['w_normalized']
        fit = results.get('sharpening_fit', {})
        
        # Data points
        ax.loglog(pe_ctx, w_normalized, 'o', 
                 color=self.colors['data'], markersize=8, 
                 label='Data', alpha=0.7)
        
        # Fitted line
        if 'fitted_values' in fit:
            ax.loglog(pe_ctx, fit['fitted_values'], '-',
                     color=self.colors['fit'], linewidth=2.5,
                     label=f"Fit: slope = {fit['slope']:.3f} ± {fit['error']:.3f}")
        
        # Expected slope line
        pe_min, pe_max = np.min(pe_ctx), np.max(pe_ctx)
        pe_range = np.array([pe_min, pe_max])
        expected_values = pe_range ** (-1/3) * w_normalized[0] * (pe_ctx[0] ** (1/3))
        ax.loglog(pe_range, expected_values, '--',
                 color=self.colors['expected'], linewidth=2,
                 label='Expected: slope = -1/3')
        
        # Confidence band
        if 'confidence_band' in fit:
            ax.fill_between(pe_ctx, 
                          fit['confidence_band']['lower'],
                          fit['confidence_band']['upper'],
                          color=self.colors['confidence'], alpha=0.3)
        
        ax.set_xlabel('$Pe_{ctx}$', fontsize=14)
        ax.set_ylabel('$W/\\sqrt{D_{eff}}$', fontsize=14)
        ax.set_title('Protocol 1: Cube-root Sharpening Law', fontsize=16)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add pass/fail indicator
        passed = fit.get('passed', False)
        status_color = self.colors['pass'] if passed else self.colors['fail']
        status_text = '✓ PASS' if passed else '✗ FAIL'
        ax.text(0.95, 0.95, status_text, transform=ax.transAxes,
               fontsize=14, fontweight='bold', color=status_color,
               ha='right', va='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
    
    def _plot_entropy_scaling(self, ax: plt.Axes, results: Dict[str, Any]):
        """
        Plot H vs ln(Pe_ctx)
        Expected: slope b ≈ 2/3
        """
        # Convert to numpy arrays if needed
        pe_ctx = np.array(results['pe_ctx']) if isinstance(results['pe_ctx'], list) else results['pe_ctx']
        h_values = np.array(results['h_values']) if isinstance(results['h_values'], list) else results['h_values']
        fit = results.get('entropy_fit', {})
        
        log_pe = np.log(pe_ctx)
        
        # Data points
        ax.scatter(log_pe, h_values, s=60, 
                  color=self.colors['data'], alpha=0.7,
                  label='Data')
        
        # Fitted line
        if 'slope' in fit and 'intercept' in fit:
            fit_line = fit['slope'] * log_pe + fit['intercept']
            ax.plot(log_pe, fit_line, '-',
                   color=self.colors['fit'], linewidth=2.5,
                   label=f"Fit: b = {fit['slope']:.3f} ± {fit['error']:.3f}")
        
        # Expected slope reference
        ax.text(0.05, 0.95, f"Expected: b = 2/3 = 0.667",
               transform=ax.transAxes, fontsize=12,
               va='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8))
        
        ax.set_xlabel('$\\ln(Pe_{ctx})$', fontsize=14)
        ax.set_ylabel('$H$ (Entropy)', fontsize=14)
        ax.set_title('Protocol 2: Entropy Scaling', fontsize=16)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Pass/fail indicator
        passed = abs(fit.get('slope', 0) - 2/3) < 0.1
        status_color = self.colors['pass'] if passed else self.colors['fail']
        status_text = '✓ PASS' if passed else '✗ FAIL'
        ax.text(0.95, 0.05, status_text, transform=ax.transAxes,
               fontsize=14, fontweight='bold', color=status_color,
               ha='right', va='bottom')
    
    def _plot_identity_check(self, ax: plt.Axes, results: Dict[str, Any]):
        """
        Plot identity verification: b ≈ -2 * slope_W
        """
        b = results.get('entropy_fit', {}).get('slope', 0)
        slope_w = results.get('sharpening_fit', {}).get('slope', 0)
        identity_ratio = b / (-2 * slope_w) if slope_w != 0 else 0
        
        # Bar plot showing the ratio
        ax.bar(['b / (-2 × slope_W)'], [identity_ratio], 
              color=self.colors['data'], width=0.5)
        
        # Expected value line
        ax.axhline(y=1.0, color=self.colors['expected'], 
                  linestyle='--', linewidth=2,
                  label='Expected = 1.0')
        
        # Tolerance bands
        ax.axhline(y=0.85, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=1.15, color='gray', linestyle=':', alpha=0.5)
        ax.fill_between([-0.5, 0.5], 0.85, 1.15, 
                       color='gray', alpha=0.1,
                       label='±15% tolerance')
        
        ax.set_ylim(0, 1.5)
        ax.set_ylabel('Identity Ratio', fontsize=14)
        ax.set_title('Identity Check: b ≈ -2 × slope_W', fontsize=16)
        ax.legend(loc='best')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value text
        ax.text(0, identity_ratio + 0.05, f'{identity_ratio:.3f}',
               ha='center', fontsize=12, fontweight='bold')
        
        # Pass/fail
        passed = abs(identity_ratio - 1.0) < 0.15
        status_color = self.colors['pass'] if passed else self.colors['fail']
        status_text = '✓ PASS' if passed else '✗ FAIL'
        ax.text(0.95, 0.95, status_text, transform=ax.transAxes,
               fontsize=14, fontweight='bold', color=status_color,
               ha='right', va='top')
    
    def _plot_diminishing_returns(self, ax: plt.Axes, results: Dict[str, Any]):
        """
        Plot α(N) vs N in log-log space
        Expected: slope = -1/3
        """
        if 'n_chunks' not in results or 'alpha_values' not in results:
            ax.text(0.5, 0.5, 'No chunk count data available',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14)
            ax.set_title('Protocol 3: Diminishing Returns (No Data)', fontsize=16)
            return
        
        # Convert to numpy arrays if needed
        n_chunks = np.array(results['n_chunks']) if isinstance(results['n_chunks'], list) else results['n_chunks']
        alpha_values = np.array(results['alpha_values']) if isinstance(results['alpha_values'], list) else results['alpha_values']
        alpha_values = np.abs(alpha_values)
        fit = results.get('diminishing_fit', {})
        
        # Data points
        ax.loglog(n_chunks, alpha_values, 'o',
                 color=self.colors['data'], markersize=8,
                 label='Data', alpha=0.7)
        
        # Fitted line
        if 'slope' in fit:
            n_min, n_max = np.min(n_chunks), np.max(n_chunks)
            n_range = np.array([n_min, n_max])
            fitted_alpha = n_range ** fit['slope'] * alpha_values[0] * (n_chunks[0] ** (-fit['slope']))
            ax.loglog(n_range, fitted_alpha, '-',
                     color=self.colors['fit'], linewidth=2.5,
                     label=f"Fit: slope = {fit['slope']:.3f}")
        
        ax.set_xlabel('N (Number of chunks)', fontsize=14)
        ax.set_ylabel('|α(N)|', fontsize=14)
        ax.set_title('Protocol 3: Diminishing Returns', fontsize=16)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Pass/fail
        passed = abs(fit.get('slope', 0) - (-1/3)) < 0.1
        status_color = self.colors['pass'] if passed else self.colors['fail']
        status_text = '✓ PASS' if passed else '✗ FAIL'
        ax.text(0.95, 0.95, status_text, transform=ax.transAxes,
               fontsize=14, fontweight='bold', color=status_color,
               ha='right', va='top')
    
    def _plot_residuals(self, ax: plt.Axes, results: Dict[str, Any]):
        """
        Plot residual analysis for model fits
        """
        # Get residuals from sharpening fit
        fit = results.get('sharpening_fit', {})
        if 'residuals' not in fit:
            ax.text(0.5, 0.5, 'No residual data available',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14)
            ax.set_title('Residual Analysis', fontsize=16)
            return
        
        residuals = fit['residuals']
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        
        ax.set_title('Residual Q-Q Plot', fontsize=16)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.grid(True, alpha=0.3)
        
        # Add normality test result
        _, p_value = stats.shapiro(residuals)
        ax.text(0.05, 0.95, f'Shapiro-Wilk p={p_value:.3f}',
               transform=ax.transAxes, fontsize=12,
               va='top', bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.8))
    
    def _plot_summary_dashboard(self, ax: plt.Axes, results: Dict[str, Any]):
        """
        Summary dashboard with key metrics and status
        """
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.95, 'Verification Summary', 
               fontsize=18, fontweight='bold',
               ha='center', va='top', transform=ax.transAxes)
        
        # Create summary table
        protocols = [
            ('Protocol 1: Sharpening', results.get('protocol1_passed', False)),
            ('Protocol 2: Entropy', results.get('protocol2_passed', False)),
            ('Protocol 3: Logarithmic Scaling', results.get('protocol3_passed', False)),
            ('Overall', results.get('overall_passed', False))
        ]
        
        y_pos = 0.8
        for name, passed in protocols:
            status_color = self.colors['pass'] if passed else self.colors['fail']
            status_text = '✓ PASS' if passed else '✗ FAIL'
            
            ax.text(0.3, y_pos, name, fontsize=14,
                   ha='right', va='center', transform=ax.transAxes)
            ax.text(0.35, y_pos, status_text, fontsize=14,
                   fontweight='bold', color=status_color,
                   ha='left', va='center', transform=ax.transAxes)
            
            y_pos -= 0.15
        
        # Key metrics
        metrics_text = []
        if 'sharpening_fit' in results:
            slope = results['sharpening_fit'].get('slope', 0)
            metrics_text.append(f"W/√D_eff slope: {slope:.3f} (expected: -0.333)")
        
        if 'entropy_fit' in results:
            b = results['entropy_fit'].get('slope', 0)
            metrics_text.append(f"Entropy slope b: {b:.3f} (expected: 0.667)")
        
        if metrics_text:
            ax.text(0.7, 0.5, '\n'.join(metrics_text),
                   fontsize=12, ha='center', va='center',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    def create_pe_ctx_sweep_plot(self,
                               pe_values: List[float],
                               metric_values: Dict[str, List[float]],
                               metric_name: str = 'W_normalized',
                               save_path: Optional[Path] = None) -> Figure:
        """
        Create a single plot for Pe_ctx sweep analysis
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot data
        ax.loglog(pe_values, metric_values[metric_name], 'o-',
                 color=self.colors['data'], markersize=10,
                 linewidth=2, label=f'{metric_name} vs Pe_ctx')
        
        # Add error bars if available
        if f'{metric_name}_std' in metric_values:
            ax.errorbar(pe_values, metric_values[metric_name],
                       yerr=metric_values[f'{metric_name}_std'],
                       fmt='none', ecolor=self.colors['data'], alpha=0.5)
        
        ax.set_xlabel('$Pe_{ctx}$', fontsize=14)
        ax.set_ylabel(metric_name.replace('_', ' '), fontsize=14)
        ax.set_title(f'{metric_name} vs Pe_ctx', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def create_diagnostic_plots(self, 
                              diagnostic_results: List[Any],
                              save_path: Optional[Path] = None) -> Figure:
        """
        Create diagnostic visualization plots
        """
        n_diagnostics = len(diagnostic_results)
        n_cols = 2
        n_rows = (n_diagnostics + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6*n_rows))
        axes = axes.flatten() if n_diagnostics > 1 else [axes]
        
        for i, diag in enumerate(diagnostic_results):
            ax = axes[i]
            
            # Simple bar chart for pass/fail
            color = self.colors['pass'] if diag.passed else self.colors['fail']
            ax.bar([diag.test_name], [diag.value], color=color)
            
            if diag.threshold is not None:
                ax.axhline(y=diag.threshold, color='black', 
                          linestyle='--', label='Threshold')
            
            ax.set_ylabel('Value')
            ax.set_title(diag.test_name)
            ax.legend()
        
        # Hide unused axes
        for i in range(n_diagnostics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig