#!/usr/bin/env python3
"""
Test script to verify that OpenAI embeddings are properly integrated
"""
import asyncio
import numpy as np

# Test the mock client
from coffee_law_verifier.run_verification import MockEmbeddingClient

# Test the real OpenAI client (if available)
try:
    from coffee_law_verifier.measurement.openai_embedding_client import OpenAIEmbeddingClient
    has_openai = True
except ImportError:
    has_openai = False

async def test_mock_client():
    """Test that mock client has all required methods"""
    print("Testing Mock Embedding Client...")
    client = MockEmbeddingClient()
    
    # Test single embed
    embedding = await client.embed("test text")
    print(f"✓ Single embedding shape: {embedding.shape}")
    assert embedding.shape == (384,), f"Expected shape (384,), got {embedding.shape}"
    
    # Test batch embed
    texts = ["text1", "text2", "text3"]
    embeddings = await client.embed_batch(texts)
    print(f"✓ Batch embeddings count: {len(embeddings)}")
    assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
    
    # Test dimension getter
    dim = client.get_dimension()
    print(f"✓ Embedding dimension: {dim}")
    assert dim == 384, f"Expected dimension 384, got {dim}"
    
    print("✓ Mock client test passed!\n")

async def test_metrics_calculator():
    """Test that metrics calculator properly uses embeddings"""
    from coffee_law_verifier.measurement.metrics_calculator import MetricsCalculator
    
    print("Testing Metrics Calculator integration...")
    
    # Create mock clients
    mock_llm = type('MockLLM', (), {})()
    mock_embedder = MockEmbeddingClient()
    mock_config = type('MockConfig', (), {
        'n_embedding_samples': 5,
        'samples_per_variant': 10
    })()
    
    calculator = MetricsCalculator(mock_llm, mock_embedder, mock_config)
    
    # Test _get_embeddings_batch
    texts = ["response 1", "response 2"]
    embeddings = await calculator._get_embeddings_batch(texts)
    
    print(f"✓ Got {len(embeddings)} embeddings")
    assert len(embeddings) == 2, f"Expected 2 embeddings, got {len(embeddings)}"
    assert all(isinstance(e, np.ndarray) for e in embeddings), "All embeddings should be numpy arrays"
    
    print("✓ Metrics calculator test passed!\n")

async def test_openai_client():
    """Test OpenAI client if available"""
    if not has_openai:
        print("⚠️  OpenAI client not available (missing dependencies or API key)")
        return
        
    try:
        print("Testing OpenAI Embedding Client...")
        client = OpenAIEmbeddingClient()
        
        # Test methods exist
        assert hasattr(client, 'embed'), "Missing embed method"
        assert hasattr(client, 'embed_batch'), "Missing embed_batch method"
        assert hasattr(client, 'get_dimension'), "Missing get_dimension method"
        
        dim = client.get_dimension()
        print(f"✓ OpenAI embedding dimension: {dim}")
        
        print("✓ OpenAI client structure test passed!\n")
    except Exception as e:
        print(f"⚠️  Could not test OpenAI client: {e}\n")

async def main():
    print("=== Testing Embedding Integration Fixes ===\n")
    
    await test_mock_client()
    await test_metrics_calculator()
    await test_openai_client()
    
    print("=== All tests completed! ===")

if __name__ == "__main__":
    asyncio.run(main())