"""
Main backtesting engine with discrete event-based simulation.
Uses a priority queue (min-heap) to process events in chronological order.
"""

import heapq
import numpy as np
from datetime import datetime


class Backtest:
    """
    Event-driven backtesting engine.
    
    Uses a priority queue (min-heap) to manage time-stamped events and
    processes them in chronological order. This discrete event simulation
    approach ensures events are handled in the correct sequence.
    """
    
    def __init__(self, data_handler, strategy, portfolio, execution_handler):
        """
        Initialize the Backtest engine.
        
        Args:
            data_handler: DataHandler instance for market data
            strategy: Strategy instance for generating signals
            portfolio: Portfolio instance for position management
            execution_handler: ExecutionHandler for order execution
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
        
        # Priority queue (min-heap) for events
        # Events are ordered by timestamp, ensuring chronological processing
        self.event_queue = []
        
        # Statistics tracking
        self.signals_generated = 0
        self.orders_generated = 0
        self.fills_executed = 0
    
    def _push_event(self, event):
        """
        Push an event onto the priority queue.
        
        Args:
            event: Event object to add to the queue
        """
        heapq.heappush(self.event_queue, event)
    
    def _pop_event(self):
        """
        Pop the next event from the priority queue.
        
        Returns:
            Event object with the earliest timestamp
        """
        if self.event_queue:
            return heapq.heappop(self.event_queue)
        return None
    
    def _process_market_event(self, event):
        """
        Process a MarketEvent.
        
        1. Update portfolio with new market data
        2. Generate signals from strategy
        
        Args:
            event: MarketEvent object
        """
        # Update portfolio holdings with new market data
        self.portfolio.update_timeindex(event)
        
        # Generate signals from strategy
        signals = self.strategy.calculate_signals(event)
        
        # Add signals to event queue
        for signal in signals:
            self._push_event(signal)
            self.signals_generated += 1
    
    def _process_signal_event(self, event):
        """
        Process a SignalEvent.
        
        Generate orders from signals via portfolio management.
        
        Args:
            event: SignalEvent object
        """
        # Generate order from signal
        order = self.portfolio.update_signal(event)
        
        if order:
            self._push_event(order)
            self.orders_generated += 1
    
    def _process_order_event(self, event):
        """
        Process an OrderEvent.
        
        Execute the order and generate a FillEvent.
        
        Args:
            event: OrderEvent object
        """
        # Execute order and get fill
        fill = self.execution_handler.execute_order(event)
        
        if fill:
            self._push_event(fill)
    
    def _process_fill_event(self, event):
        """
        Process a FillEvent.
        
        Update portfolio with the executed trade.
        
        Args:
            event: FillEvent object
        """
        # Update portfolio with fill
        self.portfolio.update_fill(event)
        self.fills_executed += 1
    
    def run(self):
        """
        Run the backtest.
        
        Main event loop:
        1. Generate market data events from data handler
        2. Process events from priority queue in chronological order
        3. Continue until no more market data
        """
        print("Starting backtest...")
        print(f"Initial capital: ${self.portfolio.initial_capital:,.2f}")
        print()
        
        # Main backtest loop
        iteration = 0
        while self.data_handler.continue_backtest:
            iteration += 1
            
            # Generate new market events (one per symbol)
            market_events = self.data_handler.update_bars()
            
            # Add market events to priority queue
            for event in market_events:
                self._push_event(event)
            
            # Process all events in the queue in chronological order
            while self.event_queue:
                event = self._pop_event()
                
                if event is None:
                    break
                
                # Route event to appropriate handler
                if event.type.name == 'MARKET':
                    self._process_market_event(event)
                elif event.type.name == 'SIGNAL':
                    self._process_signal_event(event)
                elif event.type.name == 'ORDER':
                    self._process_order_event(event)
                elif event.type.name == 'FILL':
                    self._process_fill_event(event)
            
            # Optional: print progress every N iterations
            if iteration % 50 == 0:
                holdings = self.portfolio.get_holdings()
                print(f"Iteration {iteration}: Portfolio value = ${holdings['total']:,.2f}")
        
        print()
        print("Backtest complete!")
        print(f"Total iterations: {iteration}")
        print(f"Signals generated: {self.signals_generated}")
        print(f"Orders placed: {self.orders_generated}")
        print(f"Fills executed: {self.fills_executed}")
    
    def get_results(self):
        """
        Generate backtest results and performance metrics.
        
        Returns:
            Dictionary containing performance statistics
        """
        import numpy as np
        
        # Create equity curve
        equity_curve = self.portfolio.create_equity_curve()
        
        # Calculate statistics
        total_return = (
            (self.portfolio.current_holdings['total'] - self.portfolio.initial_capital) /
            self.portfolio.initial_capital
        )
        
        returns = equity_curve['returns'].dropna()
        
        if len(returns) > 0:
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown(equity_curve['total'])
            win_rate = self._calculate_win_rate(returns)
        else:
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            win_rate = 0.0
        
        results = {
            'initial_capital': self.portfolio.initial_capital,
            'final_portfolio_value': self.portfolio.current_holdings['total'],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'total_trades': self.fills_executed,
            'total_commission': self.portfolio.current_holdings['commission'],
            'equity_curve': equity_curve
        }
        
        return results
    
    def _calculate_sharpe_ratio(self, returns, periods=252):
        """
        Calculate Sharpe ratio (annualized).
        
        Args:
            returns: pandas Series of returns
            periods: number of trading periods per year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        return np.sqrt(periods) * (returns.mean() / returns.std())
    
    def _calculate_max_drawdown(self, equity_curve):
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: pandas Series of portfolio values
            
        Returns:
            Maximum drawdown as a decimal (e.g., 0.15 for 15%)
        """
        if len(equity_curve) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return abs(drawdown.min())
    
    def _calculate_win_rate(self, returns):
        """
        Calculate win rate (percentage of positive return periods).
        
        Args:
            returns: pandas Series of returns
            
        Returns:
            Win rate as a decimal (e.g., 0.55 for 55%)
        """
        if len(returns) == 0:
            return 0.0
        
        winning_periods = (returns > 0).sum()
        total_periods = len(returns)
        
        return winning_periods / total_periods
    
    def print_results(self):
        """Print formatted backtest results."""
        results = self.get_results()
        
        print()
        print("=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital:        ${results['initial_capital']:>15,.2f}")
        print(f"Final Portfolio Value:  ${results['final_portfolio_value']:>15,.2f}")
        print(f"Total Return:           {results['total_return_pct']:>15.2f}%")
        print(f"Sharpe Ratio:           {results['sharpe_ratio']:>15.2f}")
        print(f"Maximum Drawdown:       {results['max_drawdown_pct']:>15.2f}%")
        print(f"Win Rate:               {results['win_rate']*100:>15.2f}%")
        print(f"Total Trades:           {results['total_trades']:>15}")
        print(f"Total Commission:       ${results['total_commission']:>15,.2f}")
        print("=" * 60)
