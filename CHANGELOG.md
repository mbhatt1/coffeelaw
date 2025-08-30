# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- OpenAI embedding support with `--use-openai` flag
- Engineering guide with practical implementation patterns
- ROI calculator for context optimization
- Support for custom sample sizes in verification

### Changed
- Improved Pe_ctx calculation to properly vary across different context parameters
- Enhanced diagnostic analyzer to handle raw sample counts
- Updated visualization to handle various data types

### Fixed
- Pe_ctx calculation now correctly varies across parameter space
- Fixed parameter mapping in context variator
- Resolved JSON serialization issues with numpy types
- Fixed visualization compatibility with list inputs

## [1.0.0] - 2024-08-28

### Added
- Initial release of Coffee Law verification framework
- Monte Carlo simulation engine for Coffee Law verification
- Context variation engine with Pe_ctx control
- Measurement infrastructure (W, H, D_eff, N_eff)
- Statistical analysis suite with power law fitting
- Visualization and reporting system
- Mock LLM and embedding clients for testing

### Verified
- Cube-root sharpening law: W/√D_eff ∝ Pe_ctx^(-1/3) ✓
- Entropy scaling law: H = a + b·ln(Pe_ctx) with b ≈ 2/3 ✓
- Diminishing returns law: α(N) ∼ N^(-1/3) (needs debugging)

## [0.1.0] - 2024-08-20

### Added
- Initial proof of concept
- Basic Pe_ctx calculator
- Simple context variator

[Unreleased]: https://github.com/yourusername/coffee_law/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/yourusername/coffee_law/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/yourusername/coffee_law/releases/tag/v0.1.0