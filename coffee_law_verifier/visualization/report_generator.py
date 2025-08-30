"""
Report generator for Coffee Law verification experiments
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from jinja2 import Template

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

class ReportGenerator:
    """
    Generate comprehensive reports from Coffee Law verification results
    
    Creates:
    - HTML reports with embedded plots
    - PDF summaries
    - CSV data exports
    - JSON result archives
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./reports")
        self.output_dir.mkdir(exist_ok=True)
        
        # HTML template for reports
        self.html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Coffee Law Verification Report - {{ timestamp }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            background-color: #2E86AB;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .summary {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .pass { color: #06D6A0; font-weight: bold; }
        .fail { color: #E63946; font-weight: bold; }
        .protocol {
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric {
            display: inline-block;
            margin: 10px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }
        .plot {
            text-align: center;
            margin: 20px 0;
        }
        .recommendations {
            background-color: #FFF3CD;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #FFA500;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #2E86AB;
            color: white;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Coffee Law Verification Report</h1>
        <p>{{ timestamp }}</p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <p>Overall Verification: <span class="{{ 'pass' if overall_passed else 'fail' }}">
            {{ 'PASSED' if overall_passed else 'FAILED' }}
        </span></p>
        
        <div class="metric">
            <strong>Total Simulations:</strong> {{ n_simulations }}
        </div>
        <div class="metric">
            <strong>Pe_ctx Range:</strong> {{ pe_range }}
        </div>
        <div class="metric">
            <strong>Duration:</strong> {{ duration }}
        </div>
    </div>
    
    <!-- Protocol 1: Sharpening -->
    <div class="protocol">
        <h2>Protocol 1: Cube-root Sharpening Law</h2>
        <p><strong>Hypothesis:</strong> W/√D_eff ∝ Pe_ctx^(-1/3)</p>
        <p><strong>Status:</strong> <span class="{{ 'pass' if protocol1_passed else 'fail' }}">
            {{ 'PASSED' if protocol1_passed else 'FAILED' }}
        </span></p>
        
        <table>
            <tr>
                <th>Metric</th>
                <th>Measured</th>
                <th>Expected</th>
                <th>Tolerance</th>
            </tr>
            <tr>
                <td>Exponent</td>
                <td>{{ sharpening.measured_slope | default(0) | round(4) }}</td>
                <td>-0.3333</td>
                <td>±0.07</td>
            </tr>
            <tr>
                <td>R²</td>
                <td>{{ sharpening.r_squared | default(0) | round(4) }}</td>
                <td>&gt; 0.8</td>
                <td>-</td>
            </tr>
        </table>
        
        <div class="plot">
            <img src="plots/sharpening_law.png" alt="Sharpening Law Plot" width="800">
        </div>
    </div>
    
    <!-- Protocol 2: Entropy -->
    <div class="protocol">
        <h2>Protocol 2: Entropy Scaling</h2>
        <p><strong>Hypothesis:</strong> H = a + b*ln(Pe_ctx) with b ≈ 2/3</p>
        <p><strong>Status:</strong> <span class="{{ 'pass' if protocol2_passed else 'fail' }}">
            {{ 'PASSED' if protocol2_passed else 'FAILED' }}
        </span></p>
        
        <table>
            <tr>
                <th>Metric</th>
                <th>Measured</th>
                <th>Expected</th>
                <th>Tolerance</th>
            </tr>
            <tr>
                <td>Slope b</td>
                <td>{{ entropy.b_measured | round(4) }}</td>
                <td>0.6667</td>
                <td>±0.10</td>
            </tr>
            <tr>
                <td>Identity Ratio</td>
                <td>{{ entropy.identity_ratio | round(4) }}</td>
                <td>1.0</td>
                <td>±0.15</td>
            </tr>
        </table>
        
        <div class="plot">
            <img src="plots/entropy_scaling.png" alt="Entropy Scaling Plot" width="800">
        </div>
    </div>
    
    <!-- Recommendations -->
    {% if recommendations %}
    <div class="recommendations">
        <h3>Recommendations</h3>
        <ul>
        {% for rec in recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
    
    <!-- Diagnostics -->
    <div class="protocol">
        <h2>Diagnostic Results</h2>
        {{ diagnostics_html | safe }}
    </div>
    
    <div class="footer">
        <p>Generated by Coffee Law Verifier | {{ version }}</p>
    </div>
</body>
</html>
        """)
    
    def generate_full_report(self,
                           results: Dict[str, Any],
                           experiment_name: str = "coffee_law_verification") -> Path:
        """
        Generate comprehensive report package
        
        Args:
            results: Complete results dictionary from experiments
            experiment_name: Name for the report
            
        Returns:
            Path to generated report directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"{experiment_name}_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (report_dir / "plots").mkdir(exist_ok=True)
        (report_dir / "data").mkdir(exist_ok=True)
        
        # Generate plots
        self._generate_plots(results, report_dir / "plots")
        
        # Export data
        self._export_data(results, report_dir / "data")
        
        # Generate HTML report
        html_path = self._generate_html_report(results, report_dir)
        
        # Generate summary JSON
        self._generate_json_summary(results, report_dir)
        
        # Generate CSV exports
        self._generate_csv_exports(results, report_dir / "data")
        
        return report_dir
    
    def _generate_plots(self, results: Dict[str, Any], plots_dir: Path):
        """Generate and save all plots"""
        from .coffee_law_visualizer import CoffeeLawVisualizer
        
        visualizer = CoffeeLawVisualizer()
        
        # Main verification plots
        fig = visualizer.create_verification_report(results)
        fig.savefig(plots_dir / "full_verification_report.png", dpi=300, bbox_inches='tight')
        
        # Individual protocol plots
        # These would be saved as separate files referenced in the HTML
        plt.close('all')
    
    def _export_data(self, results: Dict[str, Any], data_dir: Path):
        """Export raw data in various formats"""
        # Raw results as JSON
        with open(data_dir / "raw_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Simulation results as CSV
        if 'simulation_results' in results:
            df = pd.DataFrame(results['simulation_results'])
            df.to_csv(data_dir / "simulation_results.csv", index=False)
    
    def _generate_html_report(self, results: Dict[str, Any], report_dir: Path) -> Path:
        """Generate HTML report"""
        # Prepare template variables
        template_vars = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'overall_passed': results.get('overall_passed', False),
            'n_simulations': results.get('n_simulations', 0),
            'pe_range': f"{results.get('pe_min', 0):.2f} - {results.get('pe_max', 0):.2f}",
            'duration': results.get('duration', 'N/A'),
            'protocol1_passed': results.get('protocol1_passed', False),
            'protocol2_passed': results.get('protocol2_passed', False),
            'sharpening': results.get('sharpening_details', {}),
            'entropy': results.get('entropy_details', {}),
            'recommendations': results.get('recommendations', []),
            'diagnostics_html': self._format_diagnostics_table(results.get('diagnostics', [])),
            'version': '1.0.0'
        }
        
        # Render HTML
        html_content = self.html_template.render(**template_vars)
        
        # Save HTML
        html_path = report_dir / "report.html"
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def _generate_json_summary(self, results: Dict[str, Any], report_dir: Path):
        """Generate JSON summary file"""
        summary = {
            'experiment_date': datetime.now().isoformat(),
            'overall_passed': results.get('overall_passed', False),
            'protocols': {
                'sharpening': {
                    'passed': results.get('protocol1_passed', False),
                    'measured_exponent': results.get('sharpening_details', {}).get('measured_exponent', None),
                    'expected_exponent': -1/3,
                    'tolerance': 0.07
                },
                'entropy': {
                    'passed': results.get('protocol2_passed', False),
                    'measured_b': results.get('entropy_details', {}).get('b_measured', None),
                    'expected_b': 2/3,
                    'tolerance': 0.10
                },
                'diminishing_returns': {
                    'passed': results.get('protocol3_passed', False),
                    'measured_exponent': results.get('diminishing_details', {}).get('measured_exponent', None),
                    'expected_exponent': -1/3,
                    'tolerance': 0.10
                }
            },
            'recommendations': results.get('recommendations', [])
        }
        
        with open(report_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    def _generate_csv_exports(self, results: Dict[str, Any], data_dir: Path):
        """Generate CSV exports for key metrics"""
        # Pe_ctx sweep data
        if all(k in results for k in ['pe_ctx', 'w_normalized', 'h_values']):
            sweep_df = pd.DataFrame({
                'pe_ctx': results['pe_ctx'],
                'w_normalized': results['w_normalized'],
                'h_values': results['h_values'],
                'd_eff': results.get('d_eff_values', []),
                'n_eff': results.get('n_eff_values', [])
            })
            sweep_df.to_csv(data_dir / "pe_ctx_sweep.csv", index=False)
        
        # Fit parameters
        fit_params = []
        
        if 'sharpening_fit' in results:
            fit_params.append({
                'protocol': 'sharpening',
                'parameter': 'exponent',
                'value': results['sharpening_fit'].get('slope', None),
                'error': results['sharpening_fit'].get('error', None),
                'r_squared': results['sharpening_fit'].get('r_squared', None)
            })
        
        if 'entropy_fit' in results:
            fit_params.append({
                'protocol': 'entropy',
                'parameter': 'b',
                'value': results['entropy_fit'].get('slope', None),
                'error': results['entropy_fit'].get('error', None),
                'r_squared': results['entropy_fit'].get('r_squared', None)
            })
        
        if fit_params:
            pd.DataFrame(fit_params).to_csv(data_dir / "fit_parameters.csv", index=False)
    
    def _format_diagnostics_table(self, diagnostics: List[Dict]) -> str:
        """Format diagnostics as HTML table"""
        if not diagnostics:
            return "<p>No diagnostic results available.</p>"
        
        html = "<table>"
        html += "<tr><th>Test</th><th>Status</th><th>Value</th><th>Threshold</th></tr>"
        
        for diag in diagnostics:
            # DiagnosticResult is a dataclass, not a dict
            status = '<span class="pass">PASS</span>' if diag.passed else '<span class="fail">FAIL</span>'
            value = f"{diag.value:.4f}" if isinstance(diag.value, (int, float)) else str(diag.value)
            threshold = f"{diag.threshold:.4f}" if diag.threshold is not None else 'N/A'
            
            html += f"<tr><td>{diag.test_name}</td>"
            html += f"<td>{status}</td>"
            html += f"<td>{value}</td>"
            html += f"<td>{threshold}</td></tr>"
        
        html += "</table>"
        return html
    
    def create_summary_table(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Create summary DataFrame for quick analysis"""
        summary_data = []
        
        # Extract key metrics
        protocols = ['sharpening', 'entropy', 'diminishing_returns']
        
        for protocol in protocols:
            if f'{protocol}_details' in results:
                details = results[f'{protocol}_details']
                summary_data.append({
                    'Protocol': protocol,
                    'Passed': details.get('passed', False),
                    'Measured': details.get('measured_value', None),
                    'Expected': details.get('expected_value', None),
                    'Error': details.get('error', None),
                    'R_squared': details.get('r_squared', None)
                })
        
        return pd.DataFrame(summary_data)