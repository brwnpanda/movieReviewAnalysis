# Movie Review Sentiment Analysis

A simplified implementation of sentiment analysis for movie reviews using LSTM-based Recurrent Neural Networks with TensorFlow/Keras.

## Project Overview

This project demonstrates text sentiment classification using deep learning. The model achieves approximately **88% accuracy** on the IMDB movie review dataset through a clean and straightforward implementation.

### Key Features

- **LSTM-based RNN Architecture**: Uses Long Short-Term Memory networks for effective sequence processing
- **Simple Text Preprocessing**: Includes tokenization, padding, and embedding layers
- **Easy to Use**: Minimal code with clear structure and documentation
- **Pre-trained Dataset**: Uses the IMDB movie review dataset (25,000 training and 25,000 test reviews)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/brwnpanda/movieReviewAnalysis.git
cd movieReviewAnalysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the main sentiment analysis script to train the model:

```bash
python sentiment_analysis.py
```

This will:
- Load and preprocess the IMDB dataset
- Build an LSTM model with embedding layer
- Train for 5 epochs (takes ~5-10 minutes depending on hardware)
- Evaluate on test data
- Save the trained model as `sentiment_model.h5`

### Testing with Demo

After training, you can test the model with the demo script:

```bash
python demo.py
```

The demo provides:
- Sample predictions on test data
- Interactive mode to test your own movie reviews

## Model Architecture

The model uses a simple yet effective architecture:

```
Input (word indices)
    ↓
Embedding Layer (128-dimensional word vectors)
    ↓
LSTM Layer (64 units with dropout)
    ↓
Dense Layer (sigmoid activation)
    ↓
Output (sentiment: 0=negative, 1=positive)
```

### Hyperparameters

- **Vocabulary Size**: 10,000 most frequent words
- **Embedding Dimension**: 128
- **Sequence Length**: 200 (padded/truncated)
- **LSTM Units**: 64
- **Dropout**: 0.2
- **Epochs**: 5
- **Batch Size**: 32

## Implementation Details

### Text Preprocessing Pipeline

1. **Tokenization**: Converts text to sequences of word indices
2. **Vocabulary Filtering**: Keeps only the 10,000 most common words
3. **Padding**: Ensures all sequences have uniform length (200)
4. **Embedding**: Converts word indices to dense 128-dimensional vectors

### Model Training

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Validation**: 25,000 test samples

## Performance

Expected results:
- **Training Accuracy**: ~90-95%
- **Test Accuracy**: ~88%
- **Training Time**: ~5-10 minutes (CPU) or ~2-3 minutes (GPU)

## File Structure

```
movieReviewAnalysis/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── sentiment_analysis.py     # Main training script
├── demo.py                   # Demo/testing script
└── sentiment_model.h5        # Saved model (after training)
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy 1.21+
- scikit-learn 1.0+

## Future Improvements

- Add support for custom datasets
- Implement bidirectional LSTM
- Add attention mechanisms
- Create web interface for predictions
- Add visualization of training metrics

## License

This project is open source and available for educational purposes.

## Author

Personal project demonstrating NLP and deep learning skills.