# OpenAI Embeddings Integration Summary

## Overview

The Coffee Law Verifier uses embeddings to measure how context engineering affects LLM response consistency. This document summarizes the OpenAI embedding integration, fixes applied, and mathematical relationships between parameters.

## Embedding Integration

### Two Modes of Operation

1. **Mock Mode (Default)**
   - Uses 384-dimensional random vectors
   - No API costs, suitable for testing
   - Activated by default

2. **OpenAI Mode**
   - Uses real OpenAI embeddings (text-embedding-3-small: 1536 dimensions)
   - Requires OPENAI_API_KEY environment variable
   - Activated with `--use-openai` flag

### Usage Examples

```bash
# Mock embeddings (default)
python run_verification.py

# OpenAI embeddings
export OPENAI_API_KEY=your-key-here
python run_verification.py --use-openai
```

## How Embeddings Are Used

### Workflow

1. **Generate Responses**: For each context variant, generate 16 LLM responses
2. **Convert to Embeddings**: Each response → embedding vector
3. **Calculate Metrics**:
   - **W (Ambiguity Width)**: Standard deviation from centroid
   - **H (Coarse Entropy)**: Entropy of PCA-whitened embeddings
   - **D_eff (Effective Dimension)**: Participation ratio

### Code Flow

```
MetricsCalculator.calculate_metrics()
    ↓
_generate_responses_with_embeddings()
    ↓
_get_embeddings_batch()  # Fixed to use real embeddings
    ↓
embedding_client.embed_batch()  # Calls OpenAI API if --use-openai
```

## Mathematical Relationships Between Parameters

### The Three Coffee Laws

1. **Cube-root Sharpening**: W/√D_eff ∝ Pe_ctx^(-1/3)
2. **Entropy Scaling**: H = a + b*ln(Pe_ctx) with b ≈ 2/3
3. **Diminishing Returns**: α(N) ∝ N^(-1/3)

### Key Relationships

#### 1. Identity Relationship
```
b ≈ -2 × slope_W
```
- Since slope_W = -1/3 (from Law 1)
- Therefore: b = -2 × (-1/3) = 2/3 (confirms Law 2)

#### 2. Universal -1/3 Exponent
- Appears in Law 1 (cube-root sharpening)
- Represents fundamental scaling in high-dimensional spaces
- Related to cube-root volume scaling

#### 3. Logarithmic Scaling
- Appears in Laws 2 and 3
- Law 2: Entropy scales with ln(Pe_ctx)
- Law 3: Pe_ctx scales with ln(N)

#### 3. Inverse W-H Relationship
- As Pe_ctx ↑: W ↓ (less ambiguous) and H ↑ (more diverse)
- Physical interpretation: Better context allows sharper yet more varied responses

#### 4. D_eff Normalization
- W is normalized by √D_eff to account for embedding space dimensionality
- Ensures laws hold across different embedding models

## Fixes Applied

### 1. metrics_calculator.py
- Fixed `_get_embeddings_batch()` to actually call `embedder.embed_batch()`
- Added proper fallback handling for mock vs real embeddings

### 2. chunk_processor.py
- Fixed async/sync embedding calls to avoid event loop conflicts
- Added dimension detection from embedding client

### 3. run_verification.py
- Enhanced `MockEmbeddingClient` with `embed_batch()` and `get_dimension()`
- Fixed missing `List` import

### 4. Documentation
- Updated both README files with accurate usage instructions
- Added embedding model configuration to examples
- Documented parameter relationships

## Verification Output

When using OpenAI embeddings, the system:
1. Generates more accurate W measurements (semantic spread)
2. Produces meaningful H values (true diversity)
3. Calculates valid D_eff (actual dimension usage)
4. Verifies the mathematical relationships hold with real data

## Important Notes

- OpenAI embeddings are 4x larger than mock (1536 vs 384 dimensions)
- The identity relationship (b ≈ -2 × slope_W) is a key validation
- Real embeddings capture semantic similarity, not just random variance
- The -1/3 exponent suggests deep geometric properties of language models

## Future Improvements

1. Support for other embedding models (text-embedding-3-large, ada-002)
2. Async refactoring of chunk_processor for better performance
3. Caching of embeddings to reduce API costs
4. Batch size optimization for OpenAI API calls