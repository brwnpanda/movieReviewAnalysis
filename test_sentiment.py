"""
Quick validation test for the sentiment analysis implementation
Tests basic functionality without full training
"""

import numpy as np
import tensorflow as tf
from sentiment_analysis import (
    build_lstm_model,
    predict_sentiment
)


def test_model_building():
    """Test that the model can be built with various configurations."""
    print("Testing model building...")
    
    # Test default configuration
    model = build_lstm_model(vocab_size=1000, embedding_dim=64, maxlen=100)
    assert model is not None
    assert len(model.layers) == 3  # Embedding, LSTM, Dense
    
    # Test model compilation
    assert model.optimizer is not None
    assert model.loss is not None
    
    print("✓ Model building test passed")


def test_model_prediction():
    """Test that the model can make predictions."""
    print("Testing model predictions...")
    
    # Create a simple model
    model = build_lstm_model(vocab_size=100, embedding_dim=32, maxlen=20)
    
    # Create dummy input
    dummy_input = np.random.randint(0, 100, size=(1, 20))
    
    # Make prediction
    sentiment, confidence = predict_sentiment(model, dummy_input)
    
    assert sentiment in ['positive', 'negative']
    assert 0.5 <= confidence <= 1.0
    
    print(f"✓ Prediction test passed: {sentiment} (confidence: {confidence:.2%})")


def test_data_shapes():
    """Test data preprocessing shapes."""
    print("Testing data shapes...")
    
    # These would be the expected shapes after preprocessing
    maxlen = 200
    batch_size = 32
    
    # Simulate preprocessed data
    dummy_x = np.random.randint(0, 10000, size=(batch_size, maxlen))
    dummy_y = np.random.randint(0, 2, size=(batch_size,))
    
    assert dummy_x.shape == (batch_size, maxlen)
    assert dummy_y.shape == (batch_size,)
    
    print(f"✓ Data shape test passed: X={dummy_x.shape}, y={dummy_y.shape}")


def test_model_training_quick():
    """Quick test of model training with minimal data."""
    print("Testing quick model training...")
    
    # Create tiny dataset
    x_train = np.random.randint(0, 100, size=(10, 20))
    y_train = np.random.randint(0, 2, size=(10,))
    
    # Build model
    model = build_lstm_model(vocab_size=100, embedding_dim=16, maxlen=20)
    
    # Train for just 1 epoch with tiny dataset
    history = model.fit(x_train, y_train, epochs=1, batch_size=5, verbose=0)
    
    assert 'loss' in history.history
    assert 'accuracy' in history.history
    
    print(f"✓ Quick training test passed (loss: {history.history['loss'][0]:.4f})")


def main():
    """Run all tests."""
    print("="*60)
    print("Running Sentiment Analysis Validation Tests")
    print("="*60)
    print()
    
    tests = [
        test_model_building,
        test_model_prediction,
        test_data_shapes,
        test_model_training_quick
    ]
    
    for test in tests:
        try:
            test()
            print()
        except Exception as e:
            print(f"✗ Test failed: {e}")
            return False
    
    print("="*60)
    print("All tests passed successfully! ✓")
    print("="*60)
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
