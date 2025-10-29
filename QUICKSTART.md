# Quick Start Guide

## Overview
This project implements a simplified sentiment analysis system for movie reviews using LSTM neural networks.

## Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python sentiment_analysis.py

# Run interactive demo
python demo.py

# Run validation tests
python test_sentiment.py
```

## Example Usage

### Training a Model

```python
from sentiment_analysis import *

# Load data
x_train, y_train, x_test, y_test = load_and_preprocess_data()

# Build model
model = build_lstm_model()

# Train
history = train_model(model, x_train, y_train, x_test, y_test)

# Evaluate
test_loss, test_accuracy = evaluate_model(model, x_test, y_test)

# Save
model.save('my_model.h5')
```

### Making Predictions

```python
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load trained model
model = keras.models.load_model('sentiment_model.h5')

# Prepare your review (example with word indices)
review = [1, 14, 22, 16, 43, 530, 973]  # Word indices
review_padded = pad_sequences([review], maxlen=200, padding='post')

# Predict
from sentiment_analysis import predict_sentiment
sentiment, confidence = predict_sentiment(model, review_padded)

print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2%}")
```

## Expected Results

- **Training time**: 5-10 minutes on CPU, 2-3 minutes on GPU
- **Test accuracy**: ~88%
- **Model size**: ~4 MB
- **Inference time**: <1 second per review

## Architecture

```
Input Layer
    ↓
Embedding Layer (10,000 → 128 dimensions)
    ↓
LSTM Layer (64 units, dropout=0.2)
    ↓
Dense Layer (1 unit, sigmoid activation)
    ↓
Output (0=negative, 1=positive)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: Training is very slow
**Solution**: TensorFlow will use GPU automatically if available. On CPU, training 5 epochs may take 5-10 minutes, which is normal.

### Issue: "Model file not found"
**Solution**: Run `python sentiment_analysis.py` first to train and save the model before running the demo.

## Customization

### Change Model Parameters

Edit the configuration in `sentiment_analysis.py`:

```python
NUM_WORDS = 10000      # Vocabulary size
MAXLEN = 200           # Sequence length
EMBEDDING_DIM = 128    # Embedding dimension
EPOCHS = 5             # Training epochs
BATCH_SIZE = 32        # Batch size
```

### Use Your Own Data

To use custom data, modify the `load_and_preprocess_data()` function:

```python
def load_custom_data():
    # Load your data
    x_train = your_training_reviews
    y_train = your_training_labels
    x_test = your_test_reviews
    y_test = your_test_labels
    
    # Preprocess (tokenize, pad, etc.)
    # ...
    
    return x_train, y_train, x_test, y_test
```

## Performance Tips

1. **GPU Acceleration**: Install tensorflow-gpu for faster training
2. **Batch Size**: Increase batch size if you have more memory
3. **Epochs**: More epochs may improve accuracy but risk overfitting
4. **Vocabulary**: Larger vocabulary captures more words but increases model size

## Next Steps

- Experiment with different architectures (bidirectional LSTM, GRU)
- Try different hyperparameters
- Add attention mechanisms
- Build a web interface for predictions
- Train on custom movie review datasets
