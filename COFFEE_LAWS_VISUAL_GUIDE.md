# Coffee Laws Visual Guide 📊

## Overview

This guide provides visual representations of the Coffee Laws and their implementation in the verification framework.

## The Three Coffee Laws

### 📐 Law 1: Cube-root Sharpening

```
W/√D_eff = α · Pe_ctx^(-1/3)

       Response Width (normalized)
               │
         1.0 ──┤ ●
               │   ●
         0.8 ──┤     ●
               │       ●
         0.6 ──┤         ●●
               │            ●●
         0.4 ──┤               ●●●
               │                   ●●●●
         0.2 ──┤                        ●●●●●●●
               └─────┬──────┬──────┬──────┬────────
                    0.1     1.0    10    100   Pe_ctx
                         (log scale)

Key Insight: Doubling Pe_ctx → Width reduces by only 1.26×
```

### 📈 Law 2: Entropy Scaling

```
H = H₀ + (2/3)ln(Pe_ctx)

       Response Entropy
               │
         5.0 ──┤                              ●●●●●
               │                         ●●●●●
         4.5 ──┤                    ●●●●●
               │               ●●●●●
         4.0 ──┤          ●●●●●
               │     ●●●●●
         3.5 ──┤●●●●●
               │
         3.0 ──┤H₀
               └─────┬──────┬──────┬──────┬────────
                    0.1     1.0    10    100   Pe_ctx
                         (log scale)

Key Insight: Entropy grows logarithmically at rate 2/3
```

### 📊 Law 3: Logarithmic Context Scaling

```
Pe_ctx(N) = a + b·ln(N)

       Context Quality (Pe_ctx)
               │
         5.0 ──┤                              ●●●●●
               │                         ●●●●●
         4.0 ──┤                    ●●●●●
               │               ●●●●●
         3.0 ──┤          ●●●●●
               │     ●●●●●
         2.0 ──┤●●●●●
               │
         1.0 ──┤
               └─────┬──────┬──────┬──────┬────────
                     1      5     10     20    N chunks

Key Insight: Each chunk adds ~1/N information
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Coffee Law Verifier                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐       │
│  │   Context    │  │  Measurement │  │    Analysis    │       │
│  │   Engine     │  │   Engine     │  │    Engine      │       │
│  │             │  │              │  │                │       │
│  │ • Pe_ctx    │→│ • Width (W)  │→│ • Power Law    │       │
│  │ • Variator  │  │ • Entropy(H) │  │ • Statistical  │       │
│  │ • Chunks    │  │ • D_eff      │  │ • Validation   │       │
│  └─────────────┘  └──────────────┘  └────────────────┘       │
│         ↓                 ↓                   ↓                │
│  ┌─────────────────────────────────────────────────┐          │
│  │              Monte Carlo Runner                  │          │
│  │  • 1000s of simulations                        │          │
│  │  • Parameter sweeps                            │          │
│  │  • Statistical robustness                      │          │
│  └─────────────────────────────────────────────────┘          │
│                           ↓                                    │
│  ┌─────────────────────────────────────────────────┐          │
│  │           Visualization & Reporting              │          │
│  └─────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Pe_ctx Calculation Flow

```
                    ┌──────────────┐
                    │  Task Input  │
                    └──────┬───────┘
                           ↓
        ┌──────────────────┴──────────────────┐
        │         Calculate Stretch           │
        │                                     │
        │  ┌─────────┐ ┌─────────┐ ┌───────┐│
        │  │Alignment│×│ Schema  │×│Front- ││ = Stretch
        │  │  (S_a)  │ │  (S_s)  │ │loading││
        │  └─────────┘ └─────────┘ └───────┘│
        └─────────────────────────────────────┘
                           ↓
        ┌─────────────────────────────────────┐
        │        Calculate Diffusion          │
        │                                     │
        │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐  │
        │  │Redun│+│Conf-│+│Style│+│Temp │  │ = Diffusion
        │  │dancy│ │lict │ │Drift│ │Noise│  │
        │  └─────┘ └─────┘ └─────┘ └─────┘  │
        └─────────────────────────────────────┘
                           ↓
                   ┌─────────────┐
                   │   Pe_ctx =  │
                   │   Stretch/  │
                   │  Diffusion  │
                   └─────────────┘
```

## Embedding Flow Diagram

```
Mock Mode (Default):                    OpenAI Mode (--use-openai):
┌─────────────┐                        ┌─────────────┐
│ LLM Response│                        │ LLM Response│
└──────┬──────┘                        └──────┬──────┘
       ↓                                      ↓
┌─────────────┐                        ┌─────────────────┐
│Random Vector│                        │ OpenAI API Call │
│  384-dim    │                        │text-embedding-3 │
└──────┬──────┘                        └────────┬────────┘
       ↓                                        ↓
┌─────────────┐                        ┌─────────────┐
│  Metrics    │                        │  Metrics    │
│ Calculator  │                        │ Calculator  │
└──────┬──────┘                        └──────┬──────┘
       ↓                                      ↓
  ┌────┴────┐                            ┌────┴────┐
  │W, H, D_eff│                          │W, H, D_eff│
  └─────────┘                            └─────────┘
```

## Metric Calculations from Embeddings

### Width (W) Calculation
```
Embeddings: e₁, e₂, ..., eₙ
                ↓
         Calculate Centroid
         ē = mean(e₁...eₙ)
                ↓
      Calculate Distances
      dᵢ = ||eᵢ - ē||
                ↓
         Width = std(d)
```

### Entropy (H) Calculation
```
Embeddings: e₁, e₂, ..., eₙ
                ↓
          PCA Whitening
                ↓
         Discretization
         (√n bins)
                ↓
      H = -Σ p·log(p)
```

### D_eff Calculation
```
Embeddings: e₁, e₂, ..., eₙ
                ↓
      Covariance Matrix C
                ↓
    D_eff = (tr C)²/tr(C²)
    (Participation Ratio)
```

## Mathematical Relationships

```
                    Law 1: W/√D_eff ∝ Pe_ctx^(-1/3)
                              ↓
                         slope_W = -1/3
                              ↓
                    ┌─────────┴─────────┐
                    │ Identity Relation │
                    │   b = -2×slope_W  │
                    │   b = -2×(-1/3)   │
                    │   b = 2/3         │
                    └─────────┬─────────┘
                              ↓
                    Law 2: H = H₀ + (2/3)ln(Pe_ctx)
```

## Verification Process Flow

```
START
  │
  ├─→ Protocol 1: Cube-root Sharpening
  │     │
  │     ├─→ Vary Pe_ctx (0.1 to 10)
  │     ├─→ Measure W/√D_eff
  │     ├─→ Fit power law
  │     └─→ Verify slope = -1/3 ± 0.07
  │
  ├─→ Protocol 2: Entropy Scaling
  │     │
  │     ├─→ Vary Pe_ctx (0.1 to 10)
  │     ├─→ Measure H
  │     ├─→ Fit H = a + b·ln(Pe_ctx)
  │     ├─→ Verify b = 2/3 ± 0.10
  │     └─→ Check identity: b ≈ -2×slope_W
  │
  └─→ Protocol 3: Logarithmic Context Scaling
        │
        ├─→ Vary N chunks (1 to 20)
        ├─→ Measure Pe_ctx for each N
        ├─→ Fit Pe_ctx = a + b·ln(N)
        └─→ Verify R² > 0.9
```

## Usage Examples

### Basic Verification
```bash
# With mock embeddings (fast, no API costs)
python run_verification.py

# Output flow:
Task Generation → Context Creation → Mock Embeddings → Metrics → Analysis
     (100s)           (varied)          (384-dim)      (W,H,D)   (Laws)
```

### Production Verification
```bash
# With OpenAI embeddings (accurate, requires API key)
export OPENAI_API_KEY=sk-...
python run_verification.py --use-openai

# Output flow:
Task Generation → Context Creation → OpenAI API → Metrics → Analysis
     (100s)           (varied)       (1536-dim)   (W,H,D)   (Laws)
```

## Key Insights Visualization

```
┌─────────────────────────────────────────────────┐
│              Diminishing Returns                │
├─────────────────────────────────────────────────┤
│                                                 │
│  Effort to      │  Improvement in               │
│  improve Pe_ctx │  Response Quality              │
│  ─────────────  │  ─────────────────             │
│       2×        │      1.26×  (not 2×!)         │
│      10×        │      2.15×  (not 10×!)        │
│     100×        │      4.64×  (not 100×!)       │
│                                                 │
│  This is why context engineering has limits!   │
└─────────────────────────────────────────────────┘
```

## Pe_ctx Ranges and Interpretation

```
Pe_ctx Range    Quality        Typical Characteristics
─────────────────────────────────────────────────────
  < 0.5        Poor      High noise, low relevance, conflicts
0.5 - 2.0      Moderate  Usable but significant room to improve
2.0 - 5.0      Good      Well-optimized, coherent, aligned
  > 5.0        Excellent Minimal noise, highly relevant, structured
```

## Common Patterns in Results

```
Good Verification:              Poor Verification:
     │                               │
  W  │●                           W  │  ●
     │ ●●                            │●   ●
     │   ●●●                         │  ●  ●
     │      ●●●●                     │ ●  ● ●
     │          ●●●●                 │● ● ●  ●
     └────────────                   └────────────
       Pe_ctx (log)                    Pe_ctx (log)
    
    R² > 0.9                        R² < 0.5
    Clear power law                 No clear pattern
```

## Troubleshooting Guide

```
┌─────────────────────────────────────────────────┐
│ Issue: Low entropy slope (b << 2/3)            │
├─────────────────────────────────────────────────┤
│ Diagnosis:                                      │
│   Pe_ctx → │ → → → → → → → │ → Metrics         │
│            └─ Bottleneck ──┘                    │
│                                                 │
│ Solution: Reduce diffusion factors              │
│ • Lower redundancy                              │
│ • Fix conflicts                                 │
│ • Reduce temperature                            │
└─────────────────────────────────────────────────┘
```

## Summary

The Coffee Laws reveal fundamental mathematical relationships in how LLMs process context:

1. **Quality improvements follow cube-root scaling** - massive effort yields modest gains
2. **Information content grows logarithmically** - entropy increases predictably
3. **Context additions have diminishing returns** - each chunk adds ~1/N value

These laws provide quantitative foundations for optimal context engineering strategies.