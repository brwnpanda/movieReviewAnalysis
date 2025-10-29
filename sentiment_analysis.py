"""
Simplified Movie Review Sentiment Analysis using LSTM RNN
This implementation provides a straightforward approach to sentiment classification
with minimal complexity while maintaining good performance.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_and_preprocess_data(num_words=10000, maxlen=200):
    """
    Load and preprocess the IMDB movie review dataset.
    
    Args:
        num_words: Maximum number of words to keep in vocabulary
        maxlen: Maximum length of each review (for padding/truncation)
    
    Returns:
        Tuple of (x_train, y_train, x_test, y_test)
    """
    print("Loading IMDB dataset...")
    # Load pre-processed IMDB dataset
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Pad sequences to ensure uniform length
    print("Padding sequences...")
    x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')
    
    return x_train, y_train, x_test, y_test


def build_lstm_model(vocab_size=10000, embedding_dim=128, maxlen=200):
    """
    Build a simple LSTM-based RNN model for sentiment classification.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_dim: Dimension of word embeddings
        maxlen: Maximum sequence length
    
    Returns:
        Compiled Keras model
    """
    print("Building LSTM model...")
    
    model = keras.Sequential([
        # Embedding layer converts word indices to dense vectors
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        
        # LSTM layer for sequence processing
        layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        
        # Output layer for binary classification
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model with binary crossentropy for binary classification
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=32):
    """
    Train the sentiment analysis model.
    
    Args:
        model: Compiled Keras model
        x_train, y_train: Training data
        x_test, y_test: Test data
        epochs: Number of training epochs
        batch_size: Batch size for training
    
    Returns:
        Training history
    """
    print("Training model...")
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model
        x_test, y_test: Test data
    
    Returns:
        Test loss and accuracy
    """
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return test_loss, test_accuracy


def predict_sentiment(model, text_sequence):
    """
    Predict sentiment for a given text sequence.
    
    Args:
        model: Trained Keras model
        text_sequence: Preprocessed text sequence (padded)
    
    Returns:
        Prediction (0 for negative, 1 for positive) and confidence score
    """
    prediction = model.predict(text_sequence, verbose=0)[0][0]
    sentiment = "positive" if prediction > 0.5 else "negative"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    return sentiment, confidence


def main():
    """
    Main function to run the complete sentiment analysis pipeline.
    """
    # Configuration
    NUM_WORDS = 10000  # Vocabulary size
    MAXLEN = 200       # Maximum sequence length
    EMBEDDING_DIM = 128
    EPOCHS = 5
    BATCH_SIZE = 32
    
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_data(
        num_words=NUM_WORDS,
        maxlen=MAXLEN
    )
    
    # Build model
    model = build_lstm_model(
        vocab_size=NUM_WORDS,
        embedding_dim=EMBEDDING_DIM,
        maxlen=MAXLEN
    )
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    history = train_model(
        model, x_train, y_train, x_test, y_test,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    
    # Evaluate model
    test_loss, test_accuracy = evaluate_model(model, x_test, y_test)
    
    # Save model
    model.save('sentiment_model.h5')
    print("\nModel saved as 'sentiment_model.h5'")
    
    # Display sample predictions
    print("\nSample predictions:")
    for i in range(3):
        sample = x_test[i:i+1]
        sentiment, confidence = predict_sentiment(model, sample)
        actual = "positive" if y_test[i] == 1 else "negative"
        print(f"  Predicted: {sentiment} (confidence: {confidence:.2%}), Actual: {actual}")


if __name__ == "__main__":
    main()
