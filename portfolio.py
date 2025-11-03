"""
Portfolio management system for the backtesting engine.
Tracks positions, generates orders from signals, and manages capital.
"""

from event import OrderEvent, FillEvent
from collections import defaultdict


class Portfolio:
    """
    Manages the portfolio of positions and generates orders from signals.
    Tracks cash, holdings, and portfolio value over time.
    """
    
    def __init__(self, data_handler, initial_capital=100000.0):
        """
        Initialize the Portfolio.
        
        Args:
            data_handler: DataHandler instance for accessing market data
            initial_capital: starting cash in dollars
        """
        self.data_handler = data_handler
        self.symbol_list = data_handler.symbol_list
        self.initial_capital = initial_capital
        self.current_cash = initial_capital
        
        # Track positions (number of shares held)
        self.positions = {symbol: 0 for symbol in self.symbol_list}
        
        # Track average cost basis for each position
        self.cost_basis = {symbol: 0.0 for symbol in self.symbol_list}
        
        # Track portfolio value history
        self.equity_curve = []
        self.all_holdings = []
        self.current_holdings = self._initialize_current_holdings()
    
    def _initialize_current_holdings(self):
        """Initialize a dictionary to track current holdings."""
        holdings = {symbol: 0.0 for symbol in self.symbol_list}
        holdings['cash'] = self.current_cash
        holdings['commission'] = 0.0
        holdings['total'] = self.current_cash
        return holdings
    
    def update_timeindex(self, event):
        """
        Update portfolio holdings based on latest market data.
        
        Args:
            event: MarketEvent object with latest prices
        """
        if event.type.name == 'MARKET':
            # Update holdings value based on current prices
            holdings = {symbol: 0.0 for symbol in self.symbol_list}
            holdings['datetime'] = event.timestamp
            
            for symbol in self.symbol_list:
                # Get latest price
                bars = self.data_handler.get_latest_bars(symbol, n=1)
                if bars:
                    market_value = self.positions[symbol] * bars[0]['Close']
                    holdings[symbol] = market_value
            
            holdings['cash'] = self.current_cash
            holdings['commission'] = self.current_holdings['commission']
            holdings['total'] = self.current_cash + sum(
                holdings[symbol] for symbol in self.symbol_list
            )
            
            self.current_holdings = holdings
            self.all_holdings.append(holdings.copy())
    
    def update_signal(self, signal_event):
        """
        Generate orders from signals based on portfolio logic.
        
        Args:
            signal_event: SignalEvent object
            
        Returns:
            OrderEvent object or None
        """
        symbol = signal_event.symbol
        direction = signal_event.signal_type
        strength = signal_event.strength
        
        # Get current position
        current_position = self.positions[symbol]
        
        # Get current price
        bars = self.data_handler.get_latest_bars(symbol, n=1)
        if not bars:
            return None
        
        current_price = bars[0]['Close']
        
        # Calculate order quantity based on signal strength and available capital
        # Risk a percentage of portfolio per trade
        risk_per_trade = 0.1  # 10% of portfolio per trade
        position_size = (self.current_holdings['total'] * risk_per_trade * strength) / current_price
        quantity = int(position_size)
        
        if quantity == 0:
            return None
        
        order = None
        
        # Generate order based on signal type and current position
        if direction == 'LONG':
            if current_position <= 0:
                # Close short position if any, then go long
                if current_position < 0:
                    # First close the short
                    order = OrderEvent(
                        timestamp=signal_event.timestamp,
                        symbol=symbol,
                        order_type='MARKET',
                        quantity=abs(current_position),
                        direction='BUY'
                    )
                else:
                    # Open long position
                    order = OrderEvent(
                        timestamp=signal_event.timestamp,
                        symbol=symbol,
                        order_type='MARKET',
                        quantity=quantity,
                        direction='BUY'
                    )
        
        elif direction == 'SHORT':
            if current_position >= 0:
                # Close long position if any, then go short
                if current_position > 0:
                    # First close the long
                    order = OrderEvent(
                        timestamp=signal_event.timestamp,
                        symbol=symbol,
                        order_type='MARKET',
                        quantity=current_position,
                        direction='SELL'
                    )
                else:
                    # Open short position
                    order = OrderEvent(
                        timestamp=signal_event.timestamp,
                        symbol=symbol,
                        order_type='MARKET',
                        quantity=quantity,
                        direction='SELL'
                    )
        
        elif direction == 'EXIT':
            # Close any open position
            if current_position > 0:
                order = OrderEvent(
                    timestamp=signal_event.timestamp,
                    symbol=symbol,
                    order_type='MARKET',
                    quantity=current_position,
                    direction='SELL'
                )
            elif current_position < 0:
                order = OrderEvent(
                    timestamp=signal_event.timestamp,
                    symbol=symbol,
                    order_type='MARKET',
                    quantity=abs(current_position),
                    direction='BUY'
                )
        
        return order
    
    def update_fill(self, fill_event):
        """
        Update portfolio based on filled order.
        
        Args:
            fill_event: FillEvent object
        """
        symbol = fill_event.symbol
        direction = fill_event.direction
        quantity = fill_event.quantity
        fill_cost = fill_event.fill_cost
        commission = fill_event.commission
        
        # Update position
        if direction == 'BUY':
            self.positions[symbol] += quantity
            # Update cost basis
            total_cost = (self.cost_basis[symbol] * (self.positions[symbol] - quantity) + 
                         fill_cost * quantity)
            if self.positions[symbol] != 0:
                self.cost_basis[symbol] = total_cost / self.positions[symbol]
        else:  # SELL
            self.positions[symbol] -= quantity
            # Update cost basis (if position is closed, reset to 0)
            if self.positions[symbol] == 0:
                self.cost_basis[symbol] = 0.0
        
        # Update cash
        if direction == 'BUY':
            self.current_cash -= (fill_cost * quantity + commission)
        else:
            self.current_cash += (fill_cost * quantity - commission)
        
        # Update commission total
        self.current_holdings['commission'] += commission
    
    def get_holdings(self):
        """Return current holdings."""
        return self.current_holdings
    
    def get_positions(self):
        """Return current positions."""
        return self.positions
    
    def create_equity_curve(self):
        """
        Create a pandas DataFrame representing the equity curve.
        
        Returns:
            DataFrame with datetime index and equity values
        """
        import pandas as pd
        
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        
        return curve
