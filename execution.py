"""
Execution handler for simulating order execution in backtesting.
In live trading, this would interface with a broker API.
"""

from event import FillEvent


class ExecutionHandler:
    """
    Base class for execution handlers.
    Simulates order execution and generates FillEvents.
    """
    
    def __init__(self, data_handler):
        """
        Initialize the ExecutionHandler.
        
        Args:
            data_handler: DataHandler instance for accessing market data
        """
        self.data_handler = data_handler
    
    def execute_order(self, order_event):
        """
        Execute an order and generate a FillEvent.
        
        Args:
            order_event: OrderEvent to execute
            
        Returns:
            FillEvent representing the filled order
        """
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulates order execution for backtesting.
    Assumes all market orders are filled at the current close price.
    """
    
    def __init__(self, data_handler):
        """
        Initialize the Simulated Execution Handler.
        
        Args:
            data_handler: DataHandler instance
        """
        super().__init__(data_handler)
    
    def execute_order(self, order_event):
        """
        Simulate order execution.
        
        For market orders in backtesting, we assume:
        - Orders are filled at the next available close price
        - No slippage (in reality, there would be slippage)
        - Sufficient liquidity (all orders can be filled)
        
        Args:
            order_event: OrderEvent to execute
            
        Returns:
            FillEvent or None if execution fails
        """
        if order_event.type.name != 'ORDER':
            return None
        
        symbol = order_event.symbol
        
        # Get current market price (latest close)
        bars = self.data_handler.get_latest_bars(symbol, n=1)
        if not bars:
            print(f"Warning: No market data available for {symbol}")
            return None
        
        # Use close price as fill price (simplified model)
        fill_price = bars[0]['Close']
        
        # Create FillEvent
        fill_event = FillEvent(
            timestamp=order_event.timestamp,
            symbol=symbol,
            exchange='SIMULATED',
            quantity=order_event.quantity,
            direction=order_event.direction,
            fill_cost=fill_price,
            commission=None  # Will be calculated by FillEvent
        )
        
        return fill_event


class RealisticExecutionHandler(ExecutionHandler):
    """
    More realistic execution handler with slippage and market impact.
    """
    
    def __init__(self, data_handler, slippage_pct=0.001):
        """
        Initialize the Realistic Execution Handler.
        
        Args:
            data_handler: DataHandler instance
            slippage_pct: percentage slippage per trade (e.g., 0.001 = 0.1%)
        """
        super().__init__(data_handler)
        self.slippage_pct = slippage_pct
    
    def execute_order(self, order_event):
        """
        Simulate order execution with slippage.
        
        Args:
            order_event: OrderEvent to execute
            
        Returns:
            FillEvent with slippage applied
        """
        if order_event.type.name != 'ORDER':
            return None
        
        symbol = order_event.symbol
        
        # Get current market price
        bars = self.data_handler.get_latest_bars(symbol, n=1)
        if not bars:
            return None
        
        base_price = bars[0]['Close']
        
        # Apply slippage
        # Buy orders slip up, sell orders slip down
        if order_event.direction == 'BUY':
            fill_price = base_price * (1 + self.slippage_pct)
        else:  # SELL
            fill_price = base_price * (1 - self.slippage_pct)
        
        # Create FillEvent
        fill_event = FillEvent(
            timestamp=order_event.timestamp,
            symbol=symbol,
            exchange='SIMULATED',
            quantity=order_event.quantity,
            direction=order_event.direction,
            fill_cost=fill_price,
            commission=None
        )
        
        return fill_event
