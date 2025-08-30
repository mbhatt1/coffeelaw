# How OpenAI Embeddings Are Used in Coffee Law Verification

## Complete Workflow

### 1. **Initialization** (when `--use-openai` flag is set)
```python
# In run_verification.py
embedding_client = OpenAIEmbeddingClient()  # Uses text-embedding-3-small by default
verifier = CoffeeLawVerifier(embedding_client=embedding_client)
```

### 2. **Monte Carlo Simulation Loop**
The system runs multiple experiments varying the "Pe_ctx" (effective perplexity of context):

```python
# For each Pe_ctx variant (e.g., 10, 30, 100, 300, 1000, 3000):
for pe_ctx in pe_ctx_values:
    # Generate varied context using context_variator
    modified_prompt = vary_context(original_prompt, pe_ctx)
    
    # Calculate metrics for this context
    metrics = await calculate_metrics(modified_prompt)
```

### 3. **Response Generation & Embedding** 
For each context variant, the system:

```python
# In metrics_calculator.py
async def calculate_metrics(prompt, n_samples=16):
    # Step 1: Generate multiple LLM responses
    responses = []
    for i in range(n_samples):  # Default: 16 samples
        response = await llm.generate(prompt, temperature=0.3)
        responses.append(response)
    
    # Step 2: Convert responses to embeddings using OpenAI
    embeddings = await embedding_client.embed_batch(responses)
    # Returns: List of 1536-dimensional vectors (for text-embedding-3-small)
```

### 4. **Metric Calculation from Embeddings**

The embeddings are used to calculate three key metrics:

#### a) **Ambiguity Width (W)**
```python
# In width_measurer.py
def calculate_from_embeddings(embeddings):
    # Calculate centroid of all response embeddings
    centroid = embeddings.mean(axis=0)
    
    # Measure distances from centroid
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    
    # Width = standard deviation of distances
    W = distances.std()
    return W
```

#### b) **Coarse Entropy (H)**
```python
# In entropy_measurer.py
def calculate_entropy(embeddings):
    # 1. Apply PCA whitening to embeddings
    whitened = pca_whiten(embeddings)
    
    # 2. Discretize into bins
    n_bins = int(sqrt(len(embeddings)))
    binned = discretize(whitened, n_bins)
    
    # 3. Calculate entropy of distribution
    H = -sum(p * log(p) for p in probabilities)
    return H
```

#### c) **Effective Dimension (D_eff)**
```python
# In embedding_analyzer.py
def calculate_d_effective(embeddings):
    # Calculate covariance matrix
    centered = embeddings - embeddings.mean(axis=0)
    cov = (centered.T @ centered) / (n_samples - 1)
    
    # Participation ratio: (tr C)² / tr(C²)
    trace_cov = np.trace(cov)
    trace_cov_squared = np.trace(cov @ cov)
    
    D_eff = (trace_cov ** 2) / trace_cov_squared
    return D_eff
```

### 5. **Power Law Verification**

The system collects (Pe_ctx, W, H, D_eff) tuples and verifies:

1. **Cube-root sharpening**: W/√D_eff ∝ Pe_ctx^(-1/3)
2. **Entropy scaling**: H = a + b*ln(Pe_ctx) where b ≈ 2/3
3. **Identity check**: b ≈ -2 × slope_W

## Data Flow Summary

```
User Input with --use-openai flag
    ↓
Initialize OpenAIEmbeddingClient
    ↓
For each Pe_ctx variant:
    ↓
    Generate modified context
    ↓
    Generate 16 LLM responses
    ↓
    Convert to OpenAI embeddings (1536-dim)
    ↓
    Calculate W, H, D_eff from embeddings
    ↓
Fit power laws to verify Coffee Law relationships
    ↓
Generate verification report
```

## Key Benefits of Using Real Embeddings

1. **Semantic Understanding**: Real embeddings capture semantic similarity between responses
2. **Accurate Width Measurement**: W reflects actual semantic spread, not random noise
3. **Meaningful Entropy**: H measures true response diversity in semantic space
4. **Valid D_eff**: Effective dimension reflects actual usage of embedding space

Without real embeddings (mock mode), the measurements are based on random vectors and don't reflect the actual semantic properties of the LLM responses.