"""
Demo script for testing the trained sentiment analysis model
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences


def decode_review(encoded_review, reverse_word_index):
    """
    Decode an encoded review back to text.
    
    Args:
        encoded_review: List of word indices
        reverse_word_index: Dictionary mapping indices to words
    
    Returns:
        Decoded text string
    """
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review if i > 0])


def encode_text(text, word_index, maxlen=200):
    """
    Encode a text string into a sequence of word indices.
    
    Args:
        text: Input text string
        word_index: Dictionary mapping words to indices
        maxlen: Maximum sequence length
    
    Returns:
        Padded sequence of word indices
    """
    words = text.lower().split()
    encoded = [word_index.get(word, 2) for word in words]  # 2 is OOV (out of vocabulary)
    padded = pad_sequences([encoded], maxlen=maxlen, padding='post', truncating='post')
    return padded


def main():
    """
    Demo function to test the trained model with sample reviews.
    """
    print("Loading model...")
    try:
        model = keras.models.load_model('sentiment_model.h5')
    except:
        print("Error: Model file 'sentiment_model.h5' not found.")
        print("Please run 'sentiment_analysis.py' first to train and save the model.")
        return
    
    # Load word index for encoding/decoding
    word_index = imdb.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    # Load some test samples
    (_, _), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    print("\n" + "="*60)
    print("Movie Review Sentiment Analysis Demo")
    print("="*60)
    
    # Test with actual reviews from the dataset
    print("\nTesting with samples from IMDB dataset:\n")
    for i in range(5):
        # Decode and display review
        review_text = decode_review(x_test[i], reverse_word_index)
        actual_sentiment = "positive" if y_test[i] == 1 else "negative"
        
        # Prepare and predict
        padded_review = pad_sequences([x_test[i]], maxlen=200, padding='post', truncating='post')
        prediction = model.predict(padded_review, verbose=0)[0][0]
        predicted_sentiment = "positive" if prediction > 0.5 else "negative"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        print(f"Review {i+1}:")
        print(f"  Text: {review_text[:150]}...")
        print(f"  Actual: {actual_sentiment}")
        print(f"  Predicted: {predicted_sentiment} (confidence: {confidence:.2%})")
        print()
    
    # Interactive mode
    print("\n" + "="*60)
    print("Try your own movie review!")
    print("(Type 'quit' to exit)")
    print("="*60)
    
    while True:
        user_input = input("\nEnter a movie review: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Encode and predict
        encoded = encode_text(user_input, word_index, maxlen=200)
        prediction = model.predict(encoded, verbose=0)[0][0]
        sentiment = "positive" if prediction > 0.5 else "negative"
        confidence = prediction if prediction > 0.5 else 1 - prediction
        
        print(f"  Sentiment: {sentiment} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
