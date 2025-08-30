# Contributing to Coffee Law

First off, thank you for considering contributing to Coffee Law! It's people like you that make Coffee Law such a great tool for understanding and optimizing LLM context engineering.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Making Changes](#making-changes)
5. [Coding Standards](#coding-standards)
6. [Testing](#testing)
7. [Documentation](#documentation)
8. [Submitting Changes](#submitting-changes)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to coffee-law@example.com.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue to identify the problem
* **Describe the exact steps which reproduce the problem** in as many details as possible
* **Provide specific examples to demonstrate the steps**
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior
* **Explain which behavior you expected to see instead and why**
* **Include screenshots and animated GIFs** if possible
* **Include your configuration** (Python version, OS, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title** for the issue to identify the suggestion
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible
* **Provide specific examples to demonstrate the steps** or point out the part of Coffee Law where the suggestion is related to
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why
* **Explain why this enhancement would be useful** to most Coffee Law users

### Contributing Code

#### Areas of Interest

We're particularly interested in contributions in these areas:

1. **Protocol 3 Fix**: Debug and fix the diminishing returns verification (currently failing)
2. **Performance Optimizations**: Make simulations run faster
3. **Additional Metrics**: Implement new measurement approaches
4. **Real-world Examples**: Add examples using actual LLM APIs
5. **Documentation**: Improve guides and API documentation
6. **Testing**: Increase test coverage

## Development Setup

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/coffee_law.git
   cd coffee_law
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Making Changes

1. **Create a new branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Add or update tests** as needed

4. **Run the test suite**:
   ```bash
   pytest
   ```

5. **Check code quality**:
   ```bash
   # Format code
   black coffee_law_verifier
   
   # Check linting
   flake8 coffee_law_verifier
   
   # Type checking
   mypy coffee_law_verifier
   ```

## Coding Standards

### Python Style Guide

We follow PEP 8 with these specific conventions:

* **Line length**: 88 characters (Black default)
* **Imports**: Use isort for organizing imports
* **Docstrings**: Use Google style docstrings
* **Type hints**: Required for all public functions

### Example Code Style

```python
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


def calculate_pe_ctx(
    stretch_factors: Dict[str, float],
    diffusion_factors: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Calculate the context Peclet number.
    
    Args:
        stretch_factors: Dictionary of stretch factor values.
        diffusion_factors: Dictionary of diffusion factor values.
        weights: Optional weights for factors.
        
    Returns:
        The calculated Pe_ctx value.
        
    Raises:
        ValueError: If factors are invalid.
    """
    # Implementation here
    pass
```

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Example:
```
Add entropy measurement optimization

- Implement caching for repeated calculations
- Reduce numpy array allocations
- Add performance benchmarks

Fixes #123
```

## Testing

### Writing Tests

* Place tests in the `tests/` directory mirroring the source structure
* Use pytest for all tests
* Aim for >80% code coverage
* Include both unit tests and integration tests

### Test Structure

```python
import pytest
from coffee_law_verifier.context_engine import PeContextCalculator


class TestPeContextCalculator:
    """Test suite for PeContextCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create a calculator instance for testing."""
        return PeContextCalculator()
    
    def test_calculate_basic(self, calculator):
        """Test basic Pe_ctx calculation."""
        result = calculator.calculate(
            context="Test context",
            stretch_factors={'alignment': 0.8},
            diffusion_factors={'redundancy': 0.2}
        )
        assert 0 < result < 100
    
    def test_invalid_factors(self, calculator):
        """Test handling of invalid factors."""
        with pytest.raises(ValueError):
            calculator.calculate(
                context="",
                stretch_factors={},
                diffusion_factors={}
            )
```

## Documentation

### Docstring Requirements

All public modules, functions, classes, and methods must have docstrings. We use Google style:

```python
def function_with_docstring(param1: str, param2: int) -> bool:
    """Brief description of function.
    
    Longer description if needed, explaining the purpose
    and any important details about the function.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: When validation fails.
        
    Example:
        >>> function_with_docstring("test", 42)
        True
    """
```

### Documentation Updates

When adding new features or changing APIs:

1. Update the relevant `.md` files
2. Add usage examples
3. Update the API documentation
4. Include in the changelog

## Submitting Changes

### Pull Request Process

1. **Update the CHANGELOG.md** with details of your changes
2. **Update the README.md** if needed
3. **Ensure all tests pass** and coverage hasn't decreased
4. **Update documentation** as needed
5. **Submit the pull request** with a clear title and description

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All tests pass locally
- [ ] Added new tests for changes
- [ ] Coverage remains above 80%

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings

## Related Issues
Fixes #(issue number)
```

### Review Process

1. At least one maintainer review is required
2. All CI checks must pass
3. No merge conflicts
4. Documentation is updated

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Special thanks section

## Questions?

Feel free to open an issue with the label "question" or reach out to the maintainers.

Thank you for contributing to Coffee Law! 🚀☕