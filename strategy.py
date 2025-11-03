"""
Trading strategies for the backtesting system.
Implements momentum-based strategy and provides base class for custom strategies.
"""

import numpy as np
from event import SignalEvent


class Strategy:
    """
    Base class for trading strategies.
    Strategies generate SignalEvents based on MarketEvents.
    """
    
    def __init__(self, data_handler):
        """
        Initialize the Strategy.
        
        Args:
            data_handler: DataHandler instance for accessing market data
        """
        self.data_handler = data_handler
        self.symbol_list = data_handler.symbol_list
    
    def calculate_signals(self, event):
        """
        Calculate trading signals based on market data.
        
        Args:
            event: MarketEvent object
            
        Returns:
            List of SignalEvent objects (can be empty)
        """
        raise NotImplementedError("Should implement calculate_signals()")


class MomentumStrategy(Strategy):
    """
    Momentum-based trading strategy.
    
    Goes long when recent returns are positive (upward momentum).
    Goes short when recent returns are negative (downward momentum).
    Uses a simple moving average crossover approach.
    """
    
    def __init__(self, data_handler, short_window=20, long_window=50):
        """
        Initialize the Momentum Strategy.
        
        Args:
            data_handler: DataHandler instance
            short_window: lookback period for short-term momentum
            long_window: lookback period for long-term momentum
        """
        super().__init__(data_handler)
        self.short_window = short_window
        self.long_window = long_window
        self.positions = {symbol: 0 for symbol in self.symbol_list}  # 0, 1 (long), -1 (short)
    
    def calculate_signals(self, event):
        """
        Generate signals based on momentum.
        
        Strategy logic:
        - Calculate short-term and long-term moving averages
        - Go long when short MA crosses above long MA (positive momentum)
        - Go short when short MA crosses below long MA (negative momentum)
        - Exit when momentum reverses
        
        Args:
            event: MarketEvent object
            
        Returns:
            List of SignalEvent objects
        """
        signals = []
        
        if event.type.name != 'MARKET':
            return signals
        
        symbol = event.symbol
        
        # Get historical bars
        bars = self.data_handler.get_latest_bars(symbol, n=self.long_window)
        
        # Need enough data for long-term average
        if len(bars) < self.long_window:
            return signals
        
        # Calculate moving averages
        closes = [bar['Close'] for bar in bars]
        short_ma = np.mean(closes[-self.short_window:])
        long_ma = np.mean(closes[-self.long_window:])
        
        # Get previous MAs to detect crossovers
        if len(bars) >= self.long_window + 1:
            prev_bars = self.data_handler.get_latest_bars(symbol, n=self.long_window + 1)
            if len(prev_bars) >= self.long_window + 1:
                prev_closes = [bar['Close'] for bar in prev_bars[:-1]]
                prev_short_ma = np.mean(prev_closes[-self.short_window:])
                prev_long_ma = np.mean(prev_closes[-self.long_window:])
            else:
                # Not enough history yet
                return signals
        else:
            # Not enough history yet
            return signals
        
        # Calculate signal strength based on momentum magnitude
        momentum = (short_ma - long_ma) / long_ma
        strength = min(abs(momentum) * 10, 1.0)  # Scale and cap at 1.0
        
        current_position = self.positions[symbol]
        
        # Detect crossovers and generate signals
        # Bullish crossover: short MA crosses above long MA
        if prev_short_ma <= prev_long_ma and short_ma > long_ma:
            if current_position != 1:
                signal = SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type='LONG',
                    strength=strength
                )
                signals.append(signal)
                self.positions[symbol] = 1
        
        # Bearish crossover: short MA crosses below long MA
        elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
            if current_position != -1:
                signal = SignalEvent(
                    timestamp=event.timestamp,
                    symbol=symbol,
                    signal_type='SHORT',
                    strength=strength
                )
                signals.append(signal)
                self.positions[symbol] = -1
        
        return signals


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy-and-hold strategy for benchmarking.
    Buys on the first bar and holds until the end.
    """
    
    def __init__(self, data_handler):
        """Initialize the Buy and Hold Strategy."""
        super().__init__(data_handler)
        self.bought = {symbol: False for symbol in self.symbol_list}
    
    def calculate_signals(self, event):
        """
        Generate a single buy signal at the start.
        
        Args:
            event: MarketEvent object
            
        Returns:
            List containing a single LONG SignalEvent on first bar
        """
        signals = []
        
        if event.type.name != 'MARKET':
            return signals
        
        symbol = event.symbol
        
        # Buy on the first bar
        if not self.bought[symbol]:
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=symbol,
                signal_type='LONG',
                strength=1.0
            )
            signals.append(signal)
            self.bought[symbol] = True
        
        return signals
