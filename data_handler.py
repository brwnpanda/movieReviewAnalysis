"""
Data handler for loading and streaming historical market data.
Generates MarketEvent objects from historical data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from event import MarketEvent


class DataHandler:
    """
    Base class for handling market data.
    Subclasses implement specific data sources (CSV, database, API, etc.).
    """
    
    def __init__(self, symbol_list):
        """
        Initialize the DataHandler.
        
        Args:
            symbol_list: list of ticker symbols to track
        """
        self.symbol_list = symbol_list
        self.symbol_data = {}
        self.current_idx = {}
        self.continue_backtest = True
    
    def get_latest_bars(self, symbol, n=1):
        """
        Get the last n bars for a symbol.
        
        Args:
            symbol: ticker symbol
            n: number of bars to retrieve
            
        Returns:
            List of the last n bars
        """
        raise NotImplementedError("Should implement get_latest_bars()")
    
    def update_bars(self):
        """
        Push the next bar for each symbol to the event queue.
        
        Returns:
            List of MarketEvent objects
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricCSVDataHandler(DataHandler):
    """
    Reads CSV files containing historical OHLCV data.
    Expected format: Date, Open, High, Low, Close, Volume
    """
    
    def __init__(self, csv_dir, symbol_list):
        """
        Initialize the CSV data handler.
        
        Args:
            csv_dir: path to directory containing CSV files
            symbol_list: list of ticker symbols
        """
        super().__init__(symbol_list)
        self.csv_dir = csv_dir
        self._load_data()
    
    def _load_data(self):
        """Load CSV data for each symbol."""
        for symbol in self.symbol_list:
            try:
                # Try to load CSV file
                filepath = f"{self.csv_dir}/{symbol}.csv"
                df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
                df = df.sort_index()
                self.symbol_data[symbol] = df
                self.current_idx[symbol] = 0
            except FileNotFoundError:
                print(f"Warning: CSV file for {symbol} not found. Using synthetic data.")
                self.symbol_data[symbol] = self._generate_synthetic_data(symbol)
                self.current_idx[symbol] = 0
    
    def _generate_synthetic_data(self, symbol, days=252):
        """
        Generate synthetic market data for testing.
        Simulates realistic price movements with trend and volatility.
        
        Args:
            symbol: ticker symbol
            days: number of trading days to generate
            
        Returns:
            DataFrame with OHLCV data
        """
        # Start date and initial price
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(days)]
        
        # Generate price series with random walk + trend
        np.random.seed(hash(symbol) % (2**32))
        returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
        
        # Add momentum/trend component
        momentum = np.zeros(days)
        lookback = 20
        for i in range(lookback, days):
            momentum[i] = np.mean(returns[i-lookback:i]) * 0.5
        
        returns = returns + momentum
        
        # Calculate prices
        initial_price = 100.0
        closes = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = []
        for i, date in enumerate(dates):
            close = closes[i]
            # Open is previous close (or initial for first bar)
            open_price = closes[i-1] if i > 0 else initial_price
            
            # High and Low around close with some randomness
            daily_range = abs(close - open_price) + np.random.uniform(0.5, 2.0)
            high = max(open_price, close) + np.random.uniform(0, daily_range * 0.3)
            low = min(open_price, close) - np.random.uniform(0, daily_range * 0.3)
            
            # Volume
            volume = int(np.random.uniform(1000000, 5000000))
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('Date')
        return df
    
    def get_latest_bars(self, symbol, n=1):
        """
        Get the last n bars for a symbol.
        
        Args:
            symbol: ticker symbol
            n: number of bars to retrieve
            
        Returns:
            List of dictionaries containing bar data
        """
        try:
            idx = self.current_idx[symbol]
            if idx < n:
                # Not enough historical bars yet
                bars_list = self.symbol_data[symbol].iloc[0:idx].to_dict('records')
            else:
                bars_list = self.symbol_data[symbol].iloc[idx-n:idx].to_dict('records')
            
            # Convert to list of dicts with date
            result = []
            for bar in bars_list:
                result.append(bar)
            return result
        except (KeyError, IndexError):
            return []
    
    def update_bars(self):
        """
        Generate the next MarketEvent for each symbol.
        
        Returns:
            List of MarketEvent objects (one per symbol)
        """
        events = []
        
        for symbol in self.symbol_list:
            idx = self.current_idx[symbol]
            
            # Check if we've reached the end of data
            if idx >= len(self.symbol_data[symbol]):
                self.continue_backtest = False
                continue
            
            # Get current bar
            bar = self.symbol_data[symbol].iloc[idx]
            timestamp = self.symbol_data[symbol].index[idx]
            
            # Create MarketEvent
            event = MarketEvent(
                timestamp=timestamp,
                symbol=symbol,
                open_price=bar['Open'],
                high=bar['High'],
                low=bar['Low'],
                close=bar['Close'],
                volume=bar['Volume']
            )
            events.append(event)
            
            # Increment index
            self.current_idx[symbol] += 1
        
        return events
