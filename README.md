# Coffee Law: A Mathematical Framework for Context Engineering in LLMs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Coffee Law proposes fundamental mathematical relationships between context quality and LLM performance, analogous to fluid dynamics principles. This repository provides a production-grade verification framework with Monte Carlo simulations.

## 🎯 Key Concepts

**Context Peclet Number (Pe_ctx)**: A dimensionless quantity measuring context quality as the ratio of stretch factors to diffusion factors.

```
Pe_ctx = (alignment × schema × front_loading) / (redundancy + conflict + style_drift + decoding_noise)
```

### The Three Laws

1. **Cube-Root Sharpening**: `W/√D_eff ∝ Pe_ctx^(-1/3)`
   - Width-to-diffusion ratio follows inverse cube-root scaling
   
2. **Entropy Scaling**: `H = a + b*ln(Pe_ctx)` where `b ≈ 2/3`
   - Context entropy grows logarithmically with quality
   
3. **Diminishing Returns**: `α(N) ∼ N^(-1/3)`
   - Coupling strength decreases with context size

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/coffee_law.git
cd coffee_law

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Usage

```bash
# Run verification with default settings (mock embeddings)
cd coffee_law_verifier
python run_verification.py

# Run with OpenAI embeddings (requires OPENAI_API_KEY environment variable)
export OPENAI_API_KEY=your-api-key-here
python run_verification.py --use-openai

# Run large-scale verification (50,000+ simulations)
python run_16k_verification.py
```

### Python API

```python
from coffee_law_verifier import CoffeeLawVerifier, PeContextCalculator

# Calculate Pe_ctx for your context
calculator = PeContextCalculator()
pe_ctx = calculator.calculate(
    context="Your optimized context here",
    stretch_factors={'alignment': 0.8, 'schema': 0.9, 'front_loading': 0.7},
    diffusion_factors={'redundancy': 0.2, 'conflict': 0.1, 'style_drift': 0.15, 'decoding_noise': 0.05}
)

# Run verification
verifier = CoffeeLawVerifier()
results = verifier.verify_all_protocols()
```

## 📊 Verification Results

Latest verification run (50,000+ simulations):

| Protocol | Status | Measured | Expected | p-value |
|----------|--------|----------|----------|---------|
| Cube-root sharpening | ✅ PASSED | -0.329 | -0.333 | 0.742 |
| Entropy scaling | ✅ PASSED | 0.651 | 0.667 | 0.831 |
| Diminishing returns | ❌ FAILED | 0.007 | -0.333 | <0.001 |

*Note: Protocol 3 implementation is under investigation*

## 🏗️ Architecture

```
coffee_law/
├── coffee_law_verifier/
│   ├── context_engine/      # Pe_ctx calculation and context variation
│   ├── measurement/         # Width, entropy, and coupling metrics
│   ├── monte_carlo/         # Large-scale simulation runner
│   ├── analysis/            # Statistical analysis and power law fitting
│   └── visualization/       # Plotting and reporting
├── tests/                   # Test suite
├── docs/                    # Documentation
└── examples/               # Usage examples
```

## 🔬 How It Works

1. **Context Variation**: Generate contexts with controlled Pe_ctx values (0.1 to 10.0)
2. **Response Generation**: Create multiple LLM responses for each context variant
3. **Embedding**: Convert responses to embeddings (mock 384-dim or OpenAI 1536-dim)
4. **Measurement**: Calculate width (W), entropy (H), effective dimension (D_eff) from embeddings
5. **Monte Carlo**: Run thousands of simulations across parameter space
6. **Analysis**: Fit power laws and verify theoretical predictions
7. **Reporting**: Generate comprehensive verification reports with visualizations

### Key Metrics from Embeddings

- **W (Ambiguity Width)**: Standard deviation of response embeddings from centroid
- **H (Coarse Entropy)**: Entropy of PCA-whitened embeddings
- **D_eff (Effective Dimension)**: Participation ratio measuring embedding space usage

### Mathematical Relationships

The three laws are connected through the identity: **b ≈ -2 × slope_W**
- Since slope_W = -1/3 (Law 1), we get b = 2/3 (Law 2)
- The universal -1/3 exponent appears in both Laws 1 and 3

## 📈 Engineering Applications

See [ENGINEERING_GUIDE.md](coffee_law_verifier/ENGINEERING_GUIDE.md) for practical applications:

- Context optimization strategies
- Pe_ctx calculation for your prompts
- ROI analysis for context improvements
- Integration patterns

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black coffee_law_verifier

# Type checking
mypy coffee_law_verifier
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🔗 Links

- [Documentation](https://coffee-law.readthedocs.io) (coming soon)
- [Paper](https://arxiv.org/abs/xxxx.xxxxx) (coming soon)
- [Blog Post](https://medium.com/@coffee-law/introduction) (coming soon)

## ✨ Acknowledgments

Special thanks to all contributors and the LLM research community.

---

*"Context engineering is to LLMs what fluid dynamics is to aeronautics"*
