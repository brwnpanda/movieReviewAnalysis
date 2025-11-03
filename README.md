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
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── sentiment_analysis.py        # Main training script
├── demo.py                      # Demo/testing script
├── sentiment_model.h5           # Saved model (after training)
├── trading_backtest.py          # Algorithmic trading backtester
├── event.py                     # Event system for backtesting
├── data_handler.py              # Market data handler
├── strategy.py                  # Trading strategies
├── portfolio.py                 # Portfolio management
├── execution.py                 # Order execution simulator
├── backtest.py                  # Main backtesting engine
├── test_sentiment.py            # Tests for sentiment analysis
└── test_trading_backtest.py    # Tests for trading backtester
```

## Requirements

- Python 3.7+
- TensorFlow 2.10+
- NumPy 1.21+
- scikit-learn 1.0+
- pandas 1.3+

## Algorithmic Trading Strategy Backtester

In addition to sentiment analysis, this repository includes a **discrete event-based backtesting engine** for quantitative trading strategies.

### Features

- **Event-Driven Architecture**: Uses a priority queue (min-heap) to manage time-stamped events
- **Event Types**: Market data, signals, orders, and fills
- **Momentum Strategy**: Custom momentum-based strategy using moving average crossovers
- **Performance Metrics**: Sharpe ratio, maximum drawdown, win rate, total return
- **Synthetic Data**: Generates realistic market data for testing when CSV files aren't available

### Running the Backtester

```bash
python trading_backtest.py
```

This will:
- Run a momentum strategy backtest on synthetic market data
- Compare momentum strategy against buy-and-hold benchmark
- Display performance metrics and results

### Key Components

1. **Event System** (`event.py`): Time-stamped events (MarketEvent, SignalEvent, OrderEvent, FillEvent)
2. **Data Handler** (`data_handler.py`): Loads and streams historical market data
3. **Strategy** (`strategy.py`): Implements momentum-based and buy-and-hold strategies
4. **Portfolio** (`portfolio.py`): Manages positions, cash, and generates orders
5. **Execution** (`execution.py`): Simulates order execution with optional slippage
6. **Backtest Engine** (`backtest.py`): Priority queue-based discrete event simulation

### Testing the Backtester

```bash
python test_trading_backtest.py
```

This runs comprehensive unit tests for all backtesting components.

## Future Improvements

- Add support for custom datasets
- Implement bidirectional LSTM
- Add attention mechanisms
- Create web interface for predictions
- Add visualization of training metrics
- Add more trading strategies (mean reversion, pairs trading)
- Support for real market data (CSV, API integration)
- Add visualization of equity curves and performance

## License

This project is open source and available for educational purposes.

## Author

Personal project demonstrating NLP, deep learning, and quantitative finance skills.