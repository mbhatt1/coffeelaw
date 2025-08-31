"""
Main script to run Coffee Law verification experiments
"""
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import json
import argparse
from typing import Dict, Any, Optional, List
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from coffee_law_verifier.config import CONFIG, DATA_DIR, RESULTS_DIR, PLOTS_DIR
from coffee_law_verifier.monte_carlo import MonteCarloRunner, ExperimentProtocols, TaskGenerator
from coffee_law_verifier.analysis import VerificationSuite, DiagnosticAnalyzer
from coffee_law_verifier.visualization import CoffeeLawVisualizer, ReportGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('coffee_law_verification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CoffeeLawVerifier:
    """
    Main orchestrator for Coffee Law verification experiments
    """
    
    def __init__(self, config: Any = CONFIG, embedding_client: Optional[Any] = None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embedding_client = embedding_client
        
        # Initialize components
        self._setup_components()
        
    def _setup_components(self):
        """Initialize all verification components"""
        self.logger.info("Initializing Coffee Law Verifier components...")
        
        # Generate or load task dataset
        self.task_generator = TaskGenerator(seed=self.config.random_seed)
        self.tasks = self._load_or_generate_tasks()
        
        # Create mock clients
        self.llm_client = MockLLMClient()
        # Use provided embedding client or default to mock
        if self.embedding_client is None:
            self.embedding_client = MockEmbeddingClient()
        
        # Initialize Monte Carlo runner
        self.mc_runner = MonteCarloRunner(
            self.tasks,
            self.llm_client,
            self.embedding_client,
            self.config
        )
        
        # Initialize protocols
        self.protocols = ExperimentProtocols(self.mc_runner, self.config)
        
        # Initialize analysis suite
        self.verification_suite = VerificationSuite()
        self.diagnostic_analyzer = DiagnosticAnalyzer()
        
        # Initialize visualization
        self.visualizer = CoffeeLawVisualizer()
        self.report_generator = ReportGenerator()
        
        self.logger.info("All components initialized successfully")
    
    def _load_or_generate_tasks(self) -> list:
        """Load existing tasks or generate new ones"""
        task_file = DATA_DIR / "tasks.json"
        
        if task_file.exists():
            self.logger.info(f"Loading tasks from {task_file}")
            return self.task_generator.load_dataset(task_file)
        else:
            self.logger.info("Generating new task dataset...")
            tasks = self.task_generator.create_balanced_dataset(n_per_type=100)
            self.task_generator.save_dataset(tasks, task_file)
            return tasks
    
    async def run_full_verification(self,
                                  save_results: bool = True,
                                  generate_report: bool = True,
                                  samples_per_variant: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete Coffee Law verification suite
        
        Args:
            save_results: Whether to save results to disk
            generate_report: Whether to generate HTML/PDF reports
            samples_per_variant: Override the number of samples per variant
            
        Returns:
            Dictionary with all verification results
        """
        self.logger.info("="*60)
        self.logger.info("Starting Coffee Law Verification")
        self.logger.info("="*60)
        
        start_time = datetime.now()
        
        # Validate setup
        self.logger.info("Validating experimental setup...")
        validation = self.protocols.validate_setup()
        if not all(validation.values()):
            self.logger.error(f"Setup validation failed: {validation}")
            raise RuntimeError("Experimental setup validation failed")
        
        # Override samples per variant if specified
        if samples_per_variant:
            self.logger.info(f"Overriding samples_per_variant to {samples_per_variant}")
            # Update protocol configurations
            for protocol in self.protocols.protocols.values():
                protocol.samples_per_variant = samples_per_variant
        
        # Run all protocols
        self.logger.info("Running experimental protocols...")
        protocol_results = await self.protocols.run_all_protocols()
        
        # Compile results
        all_results = self._compile_results(protocol_results)
        
        # Run diagnostics
        self.logger.info("Running diagnostics...")
        diagnostics = self.diagnostic_analyzer.run_full_diagnostics(all_results)
        all_results['diagnostics'] = diagnostics
        
        # Run verification analysis
        self.logger.info("Running verification analysis...")
        verification_result = self.verification_suite.verify_all(all_results)
        
        # Generate verification report text
        report_text = self.verification_suite.create_verification_report(verification_result)
        self.logger.info("\n" + report_text)
        
        # Add metadata
        all_results['metadata'] = {
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration': str(datetime.now() - start_time),
            'config': {
                'n_pe_variants': self.config.n_pe_ctx_variants,
                'samples_per_variant': self.config.samples_per_variant,
                'n_embedding_samples': self.config.n_embedding_samples
            },
            'verification_result': verification_result.__dict__
        }
        
        # Save results
        if save_results:
            self._save_results(all_results)
        
        # Generate reports
        if generate_report:
            self._generate_reports(all_results)
        
        self.logger.info("="*60)
        self.logger.info(f"Verification Complete: {'PASSED' if verification_result.overall_passed else 'FAILED'}")
        self.logger.info("="*60)
        
        return all_results
    
    def _compile_results(self, protocol_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile results from all protocols into unified format"""
        # Extract data from protocol results
        compiled = {
            'pe_ctx': [],
            'w_normalized': [],
            'h_values': [],
            'd_eff_values': [],
            'n_eff_values': [],
            'n_chunks': [],
            'alpha_values': []
        }
        
        # Aggregate from sharpening protocol
        if 'sharpening' in protocol_results:
            for result in protocol_results['sharpening'].results:
                compiled['pe_ctx'].append(result.pe_ctx)
                compiled['w_normalized'].append(result.metrics['W_normalized'])
                compiled['h_values'].append(result.metrics['H'])
                compiled['d_eff_values'].append(result.metrics['D_eff'])
                compiled['n_eff_values'].append(result.metrics['N_eff'])
        
        # Aggregate from diminishing returns protocol (now logarithmic scaling)
        if 'diminishing_returns' in protocol_results:
            verification_details = protocol_results['diminishing_returns'].verification_details
            
            # Extract n_values and pe_values from the new Law 3 verification
            if 'n_values' in verification_details and 'pe_values' in verification_details:
                compiled['n_chunks'] = verification_details['n_values']
                compiled['pe_ctx_values'] = verification_details['pe_values']
                
                # Keep alpha_values for backward compatibility, but fill with zeros
                compiled['alpha_values'] = [0] * len(compiled['n_chunks'])
            else:
                # Fallback for old-style data
                n_chunks_alpha_map = {}
                for result in protocol_results['diminishing_returns'].results:
                    n_chunks = result.metrics.get('n_chunks', 0)
                    coupling_alpha = result.metrics.get('coupling_alpha', 0)
                    if n_chunks > 0 and coupling_alpha > 0:
                        if n_chunks not in n_chunks_alpha_map:
                            n_chunks_alpha_map[n_chunks] = []
                        n_chunks_alpha_map[n_chunks].append(coupling_alpha)
                
                for n_chunks, alpha_list in n_chunks_alpha_map.items():
                    if len(alpha_list) > 0:
                        compiled['n_chunks'].append(n_chunks)
                        compiled['alpha_values'].append(np.mean(alpha_list))
        
        # Add protocol-specific results
        compiled['protocol1_passed'] = protocol_results.get('sharpening', {}).verification_passed
        compiled['protocol2_passed'] = protocol_results.get('entropy', {}).verification_passed
        compiled['protocol3_passed'] = protocol_results.get('diminishing_returns', {}).verification_passed
        
        compiled['sharpening_details'] = protocol_results.get('sharpening', {}).verification_details
        compiled['entropy_details'] = protocol_results.get('entropy', {}).verification_details
        compiled['diminishing_details'] = protocol_results.get('diminishing_returns', {}).verification_details
        
        compiled['overall_passed'] = all([
            compiled['protocol1_passed'],
            compiled['protocol2_passed'],
            compiled['protocol3_passed']
        ])
        
        # Add summary stats
        compiled['n_simulations'] = sum(
            len(p.results) for p in protocol_results.values()
        )
        
        # Add raw sample count for diagnostics
        compiled['n_raw_samples'] = compiled['n_simulations']
        
        if compiled['pe_ctx']:
            compiled['pe_min'] = min(compiled['pe_ctx'])
            compiled['pe_max'] = max(compiled['pe_ctx'])
            compiled['n_pe_variants'] = len(set(compiled['pe_ctx']))
        
        return compiled
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_file = RESULTS_DIR / f"verification_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _generate_reports(self, results: Dict[str, Any]):
        """Generate visualization reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate plots
        plot_file = PLOTS_DIR / f"verification_plots_{timestamp}.png"
        fig = self.visualizer.create_verification_report(results, save_path=plot_file)
        
        # Generate full report
        report_dir = self.report_generator.generate_full_report(results)
        
        self.logger.info(f"Reports generated in {report_dir}")
    
    def run_quick_test(self) -> Dict[str, Any]:
        """Run a quick diagnostic test"""
        self.logger.info("Running quick diagnostic test...")
        
        diagnostic = self.protocols.run_quick_diagnostic(n_samples=20)
        
        self.logger.info(f"Quick test results: {diagnostic}")
        
        return diagnostic


class MockLLMClient:
    """Mock LLM client for testing"""
    async def generate(self, prompt: str, **kwargs):
        import random
        await asyncio.sleep(0.01)  # Simulate API delay
        return f"Response to: {prompt[:50]}... Value: {random.uniform(10, 100):.2f}"

class MockEmbeddingClient:
    """Mock embedding client for testing"""
    async def embed(self, text: str):
        import numpy as np
        await asyncio.sleep(0.005)  # Simulate API delay
        return np.random.randn(384)  # Standard embedding size
    
    async def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[np.ndarray]:
        """Generate mock embeddings for multiple texts"""
        import numpy as np
        embeddings = []
        for text in texts:
            await asyncio.sleep(0.005)  # Simulate API delay per text
            embeddings.append(np.random.randn(384))
        return embeddings
    
    def get_dimension(self) -> int:
        """Get the dimension of mock embeddings"""
        return 384


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Run Coffee Law Verification")
    parser.add_argument('--quick', action='store_true', help='Run quick diagnostic test')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--samples', type=int, help='Override samples per variant')
    parser.add_argument('--use-openai', action='store_true', help='Use OpenAI embeddings instead of mock')
    
    # New arguments for model selection
    parser.add_argument('--llm-provider', type=str,
                       choices=['mock', 'openai', 'anthropic', 'gemini'],
                       default='mock',
                       help='LLM provider to use (default: mock)')
    parser.add_argument('--llm-model', type=str,
                       help='Specific LLM model to use (provider-dependent)')
    parser.add_argument('--embedding-provider', type=str,
                       choices=['mock', 'openai', 'anthropic', 'gemini', 'vertex'],
                       default='mock',
                       help='Embedding provider to use (default: mock)')
    parser.add_argument('--embedding-model', type=str,
                       help='Specific embedding model to use (provider-dependent)')
    
    args = parser.parse_args()
    
    # Handle legacy --use-openai flag
    if args.use_openai and args.embedding_provider == 'mock':
        args.embedding_provider = 'openai'
    
    # Set up LLM client based on provider
    llm_client = None
    if args.llm_provider == 'mock':
        llm_client = MockLLMClient()
        print("Using mock LLM")
    elif args.llm_provider == 'openai':
        try:
            from coffee_law_verifier.measurement.openai_embedding_client import OpenAILLMClient
            llm_client = OpenAILLMClient(model=args.llm_model)
            print(f"Using OpenAI LLM: {args.llm_model or 'default'}")
        except Exception as e:
            print(f"Failed to initialize OpenAI LLM client: {e}")
            print("Falling back to mock LLM")
            llm_client = MockLLMClient()
    elif args.llm_provider == 'anthropic':
        try:
            from coffee_law_verifier.measurement.anthropic_embedding_client import AnthropicLLMClient
            llm_client = AnthropicLLMClient(model=args.llm_model)
            print(f"Using Anthropic LLM: {args.llm_model or 'claude-3-haiku-20240307'}")
        except Exception as e:
            print(f"Failed to initialize Anthropic LLM client: {e}")
            print("Falling back to mock LLM")
            llm_client = MockLLMClient()
    elif args.llm_provider == 'gemini':
        try:
            from coffee_law_verifier.measurement.gemini_embedding_client import GeminiLLMClient
            llm_client = GeminiLLMClient(model=args.llm_model)
            print(f"Using Gemini LLM: {args.llm_model or 'gemini-pro'}")
        except Exception as e:
            print(f"Failed to initialize Gemini LLM client: {e}")
            print("Falling back to mock LLM")
            llm_client = MockLLMClient()
    
    # Set up embedding client based on provider
    embedding_client = None
    if args.embedding_provider == 'mock':
        embedding_client = MockEmbeddingClient()
        print("Using mock embeddings")
    elif args.embedding_provider == 'openai':
        try:
            from coffee_law_verifier.measurement.openai_embedding_client import OpenAIEmbeddingClient
            embedding_client = OpenAIEmbeddingClient(model=args.embedding_model)
            print(f"Using OpenAI embeddings: {args.embedding_model or 'text-embedding-3-small'}")
        except Exception as e:
            print(f"Failed to initialize OpenAI embedding client: {e}")
            print("Falling back to mock embeddings")
            embedding_client = MockEmbeddingClient()
    elif args.embedding_provider == 'anthropic':
        try:
            from coffee_law_verifier.measurement.anthropic_embedding_client import AnthropicEmbeddingClient
            embedding_client = AnthropicEmbeddingClient()
            print("Using Anthropic embeddings (simulated)")
        except Exception as e:
            print(f"Failed to initialize Anthropic embedding client: {e}")
            print("Falling back to mock embeddings")
            embedding_client = MockEmbeddingClient()
    elif args.embedding_provider == 'gemini':
        try:
            from coffee_law_verifier.measurement.gemini_embedding_client import GeminiEmbeddingClient
            embedding_client = GeminiEmbeddingClient(model=args.embedding_model)
            print(f"Using Gemini embeddings: {args.embedding_model or 'models/embedding-001'}")
        except Exception as e:
            print(f"Failed to initialize Gemini embedding client: {e}")
            print("Falling back to mock embeddings")
            embedding_client = MockEmbeddingClient()
    elif args.embedding_provider == 'vertex':
        try:
            from coffee_law_verifier.measurement.gemini_embedding_client import VertexAIEmbeddingClient
            embedding_client = VertexAIEmbeddingClient(model=args.embedding_model)
            print(f"Using Vertex AI embeddings: {args.embedding_model or 'textembedding-gecko@003'}")
        except Exception as e:
            print(f"Failed to initialize Vertex AI embedding client: {e}")
            print("Falling back to mock embeddings")
            embedding_client = MockEmbeddingClient()
    
    # Initialize verifier with both clients
    verifier = CoffeeLawVerifier(config=CONFIG, embedding_client=embedding_client)
    # Set the LLM client
    verifier.llm_client = llm_client
    
    if args.quick:
        # Run quick test
        results = verifier.run_quick_test()
        print(f"\nQuick test completed: {results}")
    else:
        # Run full verification
        try:
            results = asyncio.run(
                verifier.run_full_verification(
                    save_results=True,
                    generate_report=not args.no_report,
                    samples_per_variant=args.samples
                )
            )
            
            # Print summary
            print("\n" + "="*60)
            print("VERIFICATION SUMMARY")
            print("="*60)
            print(f"Overall: {'PASSED' if results['overall_passed'] else 'FAILED'}")
            print(f"Protocol 1 (Sharpening): {'PASSED' if results['protocol1_passed'] else 'FAILED'}")
            print(f"Protocol 2 (Entropy): {'PASSED' if results['protocol2_passed'] else 'FAILED'}")
            print(f"Protocol 3 (Diminishing Returns): {'PASSED' if results['protocol3_passed'] else 'FAILED'}")
            print(f"Total simulations: {results['n_simulations']}")
            print("="*60)
            
        except Exception as e:
            logger.error(f"Verification failed: {e}", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    main()