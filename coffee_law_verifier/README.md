# Coffee Law Verifier

A production-grade Monte Carlo simulation framework for verifying the Coffee Law relationships in context engineering.

## Overview

This framework implements comprehensive verification of the three key Coffee Law claims:

1. **Law 1 - Cube-root Sharpening**: W/√D_eff = α · Pe_ctx^(-1/3)
   - Response width (normalized by effective diffusion) scales inversely with the cube root of context quality
   
2. **Law 2 - Entropy Scaling**: H = H₀ + (2/3)ln(Pe_ctx)
   - Response entropy increases logarithmically with context quality at rate 2/3
   
3. **Law 3 - Logarithmic Context Scaling**: Pe_ctx(N) = a + b·ln(N)
   - Context quality itself scales logarithmically with the number of chunks N

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/coffee-law/verifier.git
cd coffee-law-verifier

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Running Verification

```bash
# Run full verification suite with mock embeddings (default)
python -m coffee_law_verifier.run_verification

# Run with real OpenAI embeddings (requires OPENAI_API_KEY)
python -m coffee_law_verifier.run_verification --use-openai

# Run quick diagnostic test
python -m coffee_law_verifier.run_verification --quick

# Skip report generation
python -m coffee_law_verifier.run_verification --no-report
```

### Interactive Dashboard

```bash
# Launch the interactive dashboard
streamlit run coffee_law_verifier/visualization/interactive_dashboard.py
```

## Architecture

### Core Components

```
coffee_law_verifier/
├── context_engine/          # Pe_ctx manipulation
│   ├── context_variator.py  # Context variation control
│   ├── chunk_processor.py   # Chunk analysis & selection
│   └── pe_calculator.py     # Pe_ctx calculation
│
├── measurement/             # Metric measurement
│   ├── metrics_calculator.py # Main metrics orchestrator
│   ├── width_measurer.py    # W (ambiguity width)
│   ├── entropy_measurer.py  # H (coarse entropy)
│   └── embedding_analyzer.py # D_eff (participation ratio)
│
├── monte_carlo/             # Simulation framework
│   ├── monte_carlo_runner.py # Main simulation runner
│   ├── experiment_protocols.py # Protocol implementations
│   └── task_generator.py    # Task dataset generation
│
├── analysis/                # Statistical analysis
│   ├── power_law_analyzer.py # Power law fitting
│   ├── verification_suite.py # Verification tests
│   └── diagnostic_analyzer.py # Diagnostic checks
│
└── visualization/           # Reporting & visualization
    ├── coffee_law_visualizer.py # Plot generation
    ├── report_generator.py   # HTML/PDF reports
    └── interactive_dashboard.py # Live monitoring
```

### Key Concepts

#### Pe_ctx Control

Pe_ctx is controlled through two factors:

**Stretch factors** (numerator):
- Template strength
- Front-loading (relevance ordering)
- Alignment quality

**Diffusion factors** (denominator):
- Redundancy
- Conflicts
- Style/unit drift
- Decoding temperature

Formula: `Pe_ctx ≈ (alignment × schema × front-loading) / (redundancy + conflict + style drift + decoding noise)`

#### Metrics

- **W (Width)**: Standard deviation of response embeddings from centroid
- **H (Entropy)**: Coarse entropy of PCA-whitened embeddings  
- **D_eff**: Participation ratio (tr C)² / tr(C²)
- **N_eff**: Effective number of independent chunks

## Experimental Protocols

### Protocol 1: Cube-root Sharpening

- Creates 6 context variants across >1 decade of Pe_ctx
- Measures W/√D_eff for each variant
- Verifies power law with slope -1/3 ± 0.07

### Protocol 2: Entropy Scaling

- Measures H vs ln(Pe_ctx)
- Verifies slope b ≈ 2/3 ± 0.10
- Checks identity: b ≈ -2 × slope_W = -2 × (-1/3) = 2/3

### Protocol 3: Logarithmic Context Scaling

- Varies number of chunks N
- Measures Pe_ctx for each N
- Verifies logarithmic relationship Pe_ctx(N) = a + b·ln(N)

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class VerifierConfig:
    # Monte Carlo settings
    n_pe_ctx_variants: int = 6
    samples_per_variant: int = 100
    n_embedding_samples: int = 16
    
    # Statistical thresholds
    w_slope_expected: float = -1/3
    w_slope_tolerance: float = 0.07
    entropy_slope_expected: float = 2/3
    entropy_slope_tolerance: float = 0.10
    
    # API settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = "gpt-4-turbo-preview"
    embedding_model: str = "text-embedding-3-small"
```

## API Usage

```python
from coffee_law_verifier import CoffeeLawVerifier
from coffee_law_verifier.measurement.openai_embedding_client import OpenAIEmbeddingClient

# Initialize verifier with mock embeddings (default)
verifier = CoffeeLawVerifier()

# Or initialize with OpenAI embeddings
embedding_client = OpenAIEmbeddingClient()  # Uses text-embedding-3-small
verifier = CoffeeLawVerifier(embedding_client=embedding_client)

# Run full verification
results = await verifier.run_full_verification()

# Check results
if results['overall_passed']:
    print("Coffee Law verified!")
else:
    print("Verification failed:", results['recommendations'])

# Run quick diagnostic
diagnostic = verifier.run_quick_test()
```

## Embeddings and Metrics

The verifier uses embeddings to calculate three key metrics:

### Embedding Sources

1. **Mock Embeddings** (default): 384-dimensional random vectors for testing
2. **OpenAI Embeddings**: Real semantic embeddings (1536-dim for text-embedding-3-small)

### Metrics Calculated from Embeddings

- **W (Ambiguity Width)**: Standard deviation of response embeddings from their centroid
- **H (Coarse Entropy)**: Entropy of PCA-whitened embeddings after discretization
- **D_eff (Effective Dimension)**: Participation ratio (tr C)² / tr(C²)

### Parameter Relationships

The Coffee Laws reveal deep mathematical relationships:

1. **Identity Relationship**: b ≈ -2 × slope_W
   - Since slope_W = -1/3, we get b = 2/3
   - This connects the sharpening and entropy laws

2. **Universal -1/3 Exponent**: Appears in both sharpening and diminishing returns laws

3. **Inverse W-H Relationship**: As Pe_ctx increases, W decreases while H increases

## Output Files

The verifier generates several output files:

```
results/
├── verification_results_TIMESTAMP.json  # Raw results
├── plots/
│   └── verification_plots_TIMESTAMP.png # All plots
└── reports/
    ├── report.html                     # Interactive HTML report
    ├── summary.json                    # Summary statistics
    └── data/
        ├── pe_ctx_sweep.csv           # Pe_ctx sweep data
        └── fit_parameters.csv         # Fitted parameters
```

## Diagnostics

The framework includes comprehensive diagnostics:

- **Pe_ctx Range Check**: Ensures >1 decade span
- **Sample Size Adequacy**: Verifies sufficient samples
- **Outlier Detection**: IQR-based outlier analysis
- **Diffusion Floor Analysis**: Detects excess diffusion
- **Data Quality Checks**: NaN/Inf detection
- **Consistency Checks**: W-H correlation analysis

## Troubleshooting

### Common Issues

1. **Low entropy slope (b << 2/3)**
   - Indicates high diffusion floor
   - Solutions: Reduce redundancy, fix conflicts, lower temperature

2. **Poor R² values**
   - Insufficient Pe_ctx range or sample size
   - Solutions: Increase n_pe_variants or samples_per_variant

3. **Identity check failure**
   - Inconsistency between W and H measurements
   - Solutions: Check measurement procedures

### Debug Mode

Enable verbose logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{coffee_law_verifier,
  title = {Coffee Law Verifier: Production Monte Carlo for Context Engineering},
  author = {Coffee Law Research Team},
  year = {2024},
  url = {https://github.com/coffee-law/verifier}
}
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

- Documentation: https://coffee-law.github.io/verifier
- Issues: https://github.com/coffee-law/verifier/issues
- Discussions: https://github.com/coffee-law/verifier/discussions