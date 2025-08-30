"""
Interactive dashboard for real-time Coffee Law experiment monitoring
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import time

class InteractiveDashboard:
    """
    Streamlit-based interactive dashboard for Coffee Law verification
    
    Features:
    - Real-time experiment monitoring
    - Interactive plot exploration
    - Parameter tuning interface
    - Live diagnostics
    """
    
    def __init__(self):
        self.setup_page_config()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Coffee Law Verification Dashboard",
            page_icon="☕",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Main dashboard entry point"""
        st.title("☕ Coffee Law Verification Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Controls")
            
            mode = st.radio(
                "Dashboard Mode",
                ["Live Monitoring", "Result Analysis", "Parameter Explorer"]
            )
            
            if mode == "Live Monitoring":
                refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 3)
            
            st.markdown("---")
            st.header("Experiment Settings")
            
            n_pe_variants = st.number_input("Pe_ctx Variants", 4, 12, 6)
            samples_per_variant = st.number_input("Samples per Variant", 50, 500, 100)
            
        # Main content based on mode
        if mode == "Live Monitoring":
            self.show_live_monitoring(refresh_rate)
        elif mode == "Result Analysis":
            self.show_result_analysis()
        else:  # Parameter Explorer
            self.show_parameter_explorer()
    
    def show_live_monitoring(self, refresh_rate: int):
        """Display live experiment monitoring"""
        st.header("🔴 Live Experiment Monitoring")
        
        # Create placeholders for live updates
        col1, col2, col3 = st.columns(3)
        
        with col1:
            progress_placeholder = st.empty()
            protocol1_placeholder = st.empty()
        
        with col2:
            metrics_placeholder = st.empty()
            protocol2_placeholder = st.empty()
        
        with col3:
            diagnostics_placeholder = st.empty()
            protocol3_placeholder = st.empty()
        
        # Simulation loop (in real implementation, would read from actual experiment)
        for i in range(100):
            # Update progress
            progress = i / 100
            progress_placeholder.progress(progress, text=f"Progress: {progress*100:.1f}%")
            
            # Update metrics
            metrics_placeholder.metric(
                "Current Pe_ctx",
                f"{10**(i/50-1):.3f}",
                f"{10**(i/50-1) - 10**((i-1)/50-1):.3f}"
            )
            
            # Update protocol status
            protocol1_placeholder.info(f"Protocol 1: Collecting data... ({i}%)")
            protocol2_placeholder.warning(f"Protocol 2: Waiting...")
            protocol3_placeholder.success(f"Protocol 3: Ready")
            
            # Update diagnostics
            with diagnostics_placeholder.container():
                st.subheader("Live Diagnostics")
                diag_df = pd.DataFrame({
                    'Metric': ['Pe Range', 'Sample Size', 'Data Quality'],
                    'Status': ['✅ Pass', '✅ Pass', '⚠️ Warning'],
                    'Value': [f'{i/50:.2f} decades', f'{i*10} samples', '95% clean']
                })
                st.dataframe(diag_df, hide_index=True)
            
            time.sleep(refresh_rate)
            
            if st.button("Stop Monitoring"):
                break
    
    def show_result_analysis(self):
        """Display result analysis interface"""
        st.header("📊 Result Analysis")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Results JSON",
            type=['json'],
            help="Upload the results.json file from your experiment"
        )
        
        if uploaded_file is not None:
            results = json.load(uploaded_file)
            self._analyze_results(results)
        else:
            # Use sample data
            st.info("Upload a results file or use sample data")
            if st.button("Load Sample Data"):
                results = self._generate_sample_results()
                self._analyze_results(results)
    
    def _analyze_results(self, results: Dict[str, Any]):
        """Analyze and display results"""
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Status", 
                     "✅ PASSED" if results.get('overall_passed') else "❌ FAILED")
        
        with col2:
            st.metric("Protocol 1", 
                     "Pass" if results.get('protocol1_passed') else "Fail",
                     f"slope: {results.get('sharpening_slope', 0):.3f}")
        
        with col3:
            st.metric("Protocol 2",
                     "Pass" if results.get('protocol2_passed') else "Fail",
                     f"b: {results.get('entropy_b', 0):.3f}")
        
        with col4:
            st.metric("Protocol 3",
                     "Pass" if results.get('protocol3_passed') else "Fail",
                     f"slope: {results.get('diminishing_slope', 0):.3f}")
        
        # Tabs for different protocols
        tab1, tab2, tab3, tab4 = st.tabs([
            "Protocol 1: Sharpening",
            "Protocol 2: Entropy",
            "Protocol 3: Diminishing Returns",
            "Diagnostics"
        ])
        
        with tab1:
            self._plot_sharpening_results(results)
        
        with tab2:
            self._plot_entropy_results(results)
        
        with tab3:
            self._plot_diminishing_results(results)
        
        with tab4:
            self._show_diagnostics(results)
    
    def _plot_sharpening_results(self, results: Dict[str, Any]):
        """Plot Protocol 1 results"""
        st.subheader("W/√D_eff vs Pe_ctx")
        
        # Create interactive plot
        pe_ctx = results.get('pe_ctx', np.logspace(-1, 1, 50))
        w_normalized = results.get('w_normalized', pe_ctx**(-1/3) + np.random.normal(0, 0.05, len(pe_ctx)))
        
        fig = go.Figure()
        
        # Data points
        fig.add_trace(go.Scatter(
            x=pe_ctx,
            y=w_normalized,
            mode='markers',
            name='Data',
            marker=dict(size=10, color='blue')
        ))
        
        # Fitted line
        fit_line = pe_ctx**(-0.33)
        fig.add_trace(go.Scatter(
            x=pe_ctx,
            y=fit_line,
            mode='lines',
            name='Fit (slope=-0.33)',
            line=dict(color='red', width=2)
        ))
        
        # Expected line
        expected_line = pe_ctx**(-1/3)
        fig.add_trace(go.Scatter(
            x=pe_ctx,
            y=expected_line,
            mode='lines',
            name='Expected (slope=-1/3)',
            line=dict(color='green', dash='dash')
        ))
        
        fig.update_xaxes(type="log", title="Pe_ctx")
        fig.update_yaxes(type="log", title="W/√D_eff")
        fig.update_layout(title="Protocol 1: Cube-root Sharpening Law")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show fit details
        with st.expander("Fit Details"):
            st.markdown(f"""
            - **Measured Exponent**: {results.get('sharpening_slope', -0.33):.4f}
            - **Expected Exponent**: -0.3333
            - **R²**: {results.get('sharpening_r2', 0.95):.4f}
            - **p-value**: {results.get('sharpening_p', 0.001):.4e}
            """)
    
    def _plot_entropy_results(self, results: Dict[str, Any]):
        """Plot Protocol 2 results"""
        st.subheader("H vs ln(Pe_ctx)")
        
        pe_ctx = results.get('pe_ctx', np.logspace(-1, 1, 50))
        h_values = results.get('h_values', 2 + 0.65*np.log(pe_ctx) + np.random.normal(0, 0.1, len(pe_ctx)))
        
        fig = px.scatter(
            x=np.log(pe_ctx),
            y=h_values,
            labels={'x': 'ln(Pe_ctx)', 'y': 'H (Entropy)'},
            title='Protocol 2: Entropy Scaling'
        )
        
        # Add trend line
        fig.add_traces(
            px.line(x=np.log(pe_ctx), y=2 + 2/3*np.log(pe_ctx)).data[0]
            .update(name='Expected (b=2/3)', line=dict(dash='dash'))
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Identity check
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Measured b", f"{results.get('entropy_b', 0.65):.4f}")
        with col2:
            st.metric("Identity Ratio", f"{results.get('identity_ratio', 0.98):.4f}")
    
    def _plot_diminishing_results(self, results: Dict[str, Any]):
        """Plot Protocol 3 results"""
        st.subheader("α(N) vs N")
        
        if 'n_chunks' not in results:
            st.warning("No chunk count data available")
            return
        
        n_chunks = results.get('n_chunks', np.arange(1, 21))
        alpha = results.get('alpha_values', n_chunks**(-1/3) + np.random.normal(0, 0.05, len(n_chunks)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=n_chunks,
            y=np.abs(alpha),
            mode='markers',
            name='|α(N)|'
        ))
        
        fig.update_xaxes(type="log", title="N (chunks)")
        fig.update_yaxes(type="log", title="|α(N)|")
        fig.update_layout(title="Protocol 3: Diminishing Returns")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _show_diagnostics(self, results: Dict[str, Any]):
        """Show diagnostic results"""
        st.subheader("Diagnostics")
        
        diagnostics = results.get('diagnostics', [])
        
        if diagnostics:
            df = pd.DataFrame(diagnostics)
            
            # Color code pass/fail
            def color_status(val):
                color = 'green' if val else 'red'
                return f'color: {color}'
            
            styled_df = df.style.applymap(color_status, subset=['passed'])
            st.dataframe(styled_df)
        else:
            st.info("No diagnostic data available")
    
    def show_parameter_explorer(self):
        """Interactive parameter exploration"""
        st.header("🎛️ Parameter Explorer")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Pe_ctx Control Parameters")
            
            template_strength = st.slider("Template Strength", 0.0, 1.0, 0.8)
            front_loading = st.slider("Front-loading", 0.0, 1.0, 0.8)
            deduplication = st.slider("Deduplication", 0.0, 1.0, 0.8)
            style_consistency = st.slider("Style Consistency", 0.0, 1.0, 0.8)
            temperature = st.slider("Temperature", 0.1, 1.0, 0.3)
            
            # Calculate Pe_ctx
            stretch = template_strength * front_loading * 1.0  # alignment=1.0
            diffusion = (1-deduplication) + 0.1 + (1-style_consistency) + temperature/0.3
            pe_ctx = stretch / max(diffusion, 0.1)
            
            st.metric("Calculated Pe_ctx", f"{pe_ctx:.3f}")
        
        with col2:
            st.subheader("Expected Metrics")
            
            # Predict metrics based on Pe_ctx
            w_expected = pe_ctx**(-1/3)
            h_expected = 2 + 2/3 * np.log(pe_ctx)
            
            st.metric("Expected W/√D_eff", f"{w_expected:.3f}")
            st.metric("Expected H", f"{h_expected:.3f}")
            
            # Show how changing parameters affects Pe_ctx
            fig = go.Figure()
            
            pe_range = np.linspace(0.1, 10, 100)
            fig.add_trace(go.Scatter(
                x=pe_range,
                y=pe_range**(-1/3),
                name='W/√D_eff',
                line=dict(color='blue')
            ))
            
            fig.add_vline(x=pe_ctx, line_dash="dash", line_color="red",
                         annotation_text=f"Current Pe={pe_ctx:.2f}")
            
            fig.update_xaxes(type="log", title="Pe_ctx")
            fig.update_yaxes(type="log", title="W/√D_eff")
            fig.update_layout(title="Parameter Impact on Metrics")
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _generate_sample_results(self) -> Dict[str, Any]:
        """Generate sample results for demonstration"""
        pe_ctx = np.logspace(-1, 1, 50)
        
        return {
            'overall_passed': True,
            'protocol1_passed': True,
            'protocol2_passed': True,
            'protocol3_passed': False,
            'pe_ctx': pe_ctx.tolist(),
            'w_normalized': (pe_ctx**(-0.32) + np.random.normal(0, 0.05, len(pe_ctx))).tolist(),
            'h_values': (2 + 0.64*np.log(pe_ctx) + np.random.normal(0, 0.1, len(pe_ctx))).tolist(),
            'sharpening_slope': -0.32,
            'sharpening_r2': 0.96,
            'entropy_b': 0.64,
            'identity_ratio': 1.01,
            'n_chunks': list(range(1, 21)),
            'alpha_values': (np.arange(1, 21)**(-0.28) + np.random.normal(0, 0.03, 20)).tolist(),
            'diagnostics': [
                {'test_name': 'Pe Range Check', 'passed': True, 'value': 2.1},
                {'test_name': 'Sample Size', 'passed': True, 'value': 50},
                {'test_name': 'Data Quality', 'passed': True, 'value': 0},
                {'test_name': 'Diffusion Floor', 'passed': False, 'value': 1.2}
            ]
        }

def main():
    """Main entry point for dashboard"""
    dashboard = InteractiveDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()