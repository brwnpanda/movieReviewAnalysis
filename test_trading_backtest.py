"""
Unit tests for the algorithmic trading backtester.
Tests the event system, priority queue, and backtesting components.
"""

import unittest
import numpy as np
from datetime import datetime, timedelta
from event import Event, MarketEvent, SignalEvent, OrderEvent, FillEvent, EventType
from data_handler import HistoricCSVDataHandler
from strategy import MomentumStrategy, BuyAndHoldStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
from backtest import Backtest
import heapq


class TestEventSystem(unittest.TestCase):
    """Test the event classes and priority queue ordering."""
    
    def test_event_ordering(self):
        """Test that events are ordered by timestamp in priority queue."""
        # Create events with different timestamps
        t1 = datetime(2023, 1, 1, 10, 0)
        t2 = datetime(2023, 1, 1, 9, 0)
        t3 = datetime(2023, 1, 1, 11, 0)
        
        event1 = MarketEvent(t1, 'AAPL', 100, 101, 99, 100.5, 1000000)
        event2 = MarketEvent(t2, 'AAPL', 98, 99, 97, 98.5, 1000000)
        event3 = MarketEvent(t3, 'AAPL', 102, 103, 101, 102.5, 1000000)
        
        # Add to priority queue
        queue = []
        heapq.heappush(queue, event1)
        heapq.heappush(queue, event2)
        heapq.heappush(queue, event3)
        
        # Pop in order - should be t2, t1, t3
        first = heapq.heappop(queue)
        second = heapq.heappop(queue)
        third = heapq.heappop(queue)
        
        self.assertEqual(first.timestamp, t2)
        self.assertEqual(second.timestamp, t1)
        self.assertEqual(third.timestamp, t3)
    
    def test_market_event_creation(self):
        """Test MarketEvent creation and attributes."""
        timestamp = datetime(2023, 1, 1)
        event = MarketEvent(timestamp, 'AAPL', 100, 101, 99, 100.5, 1000000)
        
        self.assertEqual(event.type, EventType.MARKET)
        self.assertEqual(event.symbol, 'AAPL')
        self.assertEqual(event.close, 100.5)
        self.assertEqual(event.timestamp, timestamp)
    
    def test_signal_event_creation(self):
        """Test SignalEvent creation and attributes."""
        timestamp = datetime(2023, 1, 1)
        event = SignalEvent(timestamp, 'AAPL', 'LONG', 0.8)
        
        self.assertEqual(event.type, EventType.SIGNAL)
        self.assertEqual(event.symbol, 'AAPL')
        self.assertEqual(event.signal_type, 'LONG')
        self.assertEqual(event.strength, 0.8)
    
    def test_order_event_creation(self):
        """Test OrderEvent creation and attributes."""
        timestamp = datetime(2023, 1, 1)
        event = OrderEvent(timestamp, 'AAPL', 'MARKET', 100, 'BUY')
        
        self.assertEqual(event.type, EventType.ORDER)
        self.assertEqual(event.symbol, 'AAPL')
        self.assertEqual(event.quantity, 100)
        self.assertEqual(event.direction, 'BUY')
    
    def test_fill_event_creation(self):
        """Test FillEvent creation and commission calculation."""
        timestamp = datetime(2023, 1, 1)
        event = FillEvent(timestamp, 'AAPL', 'NYSE', 100, 'BUY', 100.0)
        
        self.assertEqual(event.type, EventType.FILL)
        self.assertEqual(event.quantity, 100)
        self.assertGreater(event.commission, 0)  # Should have commission


class TestDataHandler(unittest.TestCase):
    """Test the data handler and synthetic data generation."""
    
    def test_synthetic_data_generation(self):
        """Test that synthetic data is generated correctly."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        
        # Should have data for TEST symbol
        self.assertIn('TEST', data_handler.symbol_data)
        
        # Should have generated data
        df = data_handler.symbol_data['TEST']
        self.assertGreater(len(df), 0)
        self.assertIn('Close', df.columns)
    
    def test_market_event_generation(self):
        """Test that market events are generated correctly."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        
        # Get first market event
        events = data_handler.update_bars()
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, EventType.MARKET)
        self.assertEqual(events[0].symbol, 'TEST')
    
    def test_get_latest_bars(self):
        """Test retrieving latest bars."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        
        # Update a few bars
        data_handler.update_bars()
        data_handler.update_bars()
        data_handler.update_bars()
        
        # Get latest bars
        bars = data_handler.get_latest_bars('TEST', n=2)
        
        self.assertEqual(len(bars), 2)
        self.assertIn('Close', bars[0])


class TestStrategy(unittest.TestCase):
    """Test trading strategies."""
    
    def test_buy_and_hold_strategy(self):
        """Test buy-and-hold strategy generates signal on first bar."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        strategy = BuyAndHoldStrategy(data_handler)
        
        # Generate first market event
        events = data_handler.update_bars()
        market_event = events[0]
        
        # Should generate a LONG signal
        signals = strategy.calculate_signals(market_event)
        
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].signal_type, 'LONG')
    
    def test_momentum_strategy_initialization(self):
        """Test momentum strategy initialization."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        strategy = MomentumStrategy(data_handler, short_window=10, long_window=20)
        
        self.assertEqual(strategy.short_window, 10)
        self.assertEqual(strategy.long_window, 20)
        self.assertIn('TEST', strategy.positions)


class TestPortfolio(unittest.TestCase):
    """Test portfolio management."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        portfolio = Portfolio(data_handler, initial_capital=100000.0)
        
        self.assertEqual(portfolio.initial_capital, 100000.0)
        self.assertEqual(portfolio.current_cash, 100000.0)
        self.assertEqual(portfolio.positions['TEST'], 0)
    
    def test_portfolio_fill_update(self):
        """Test portfolio update after fill."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        portfolio = Portfolio(data_handler, initial_capital=100000.0)
        
        # Create a fill event
        timestamp = datetime(2023, 1, 1)
        fill = FillEvent(timestamp, 'TEST', 'NYSE', 100, 'BUY', 50.0)
        
        # Update portfolio
        initial_cash = portfolio.current_cash
        portfolio.update_fill(fill)
        
        # Check position and cash
        self.assertEqual(portfolio.positions['TEST'], 100)
        self.assertLess(portfolio.current_cash, initial_cash)


class TestBacktest(unittest.TestCase):
    """Test the backtesting engine."""
    
    def test_backtest_runs(self):
        """Test that backtest runs without errors."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        strategy = BuyAndHoldStrategy(data_handler)
        portfolio = Portfolio(data_handler, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(data_handler)
        
        backtest = Backtest(data_handler, strategy, portfolio, execution)
        
        # Run backtest (should complete without errors)
        backtest.run()
        
        # Check that some events were processed
        self.assertGreater(backtest.signals_generated, 0)
    
    def test_backtest_results(self):
        """Test that backtest generates results."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        strategy = BuyAndHoldStrategy(data_handler)
        portfolio = Portfolio(data_handler, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(data_handler)
        
        backtest = Backtest(data_handler, strategy, portfolio, execution)
        backtest.run()
        
        results = backtest.get_results()
        
        # Check results structure
        self.assertIn('initial_capital', results)
        self.assertIn('final_portfolio_value', results)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
    
    def test_priority_queue_ordering(self):
        """Test that events are processed in chronological order."""
        data_handler = HistoricCSVDataHandler('./data', ['TEST'])
        strategy = BuyAndHoldStrategy(data_handler)
        portfolio = Portfolio(data_handler, initial_capital=100000.0)
        execution = SimulatedExecutionHandler(data_handler)
        
        backtest = Backtest(data_handler, strategy, portfolio, execution)
        
        # Add events with different timestamps
        t1 = datetime(2023, 1, 1, 10, 0)
        t2 = datetime(2023, 1, 1, 9, 0)
        t3 = datetime(2023, 1, 1, 11, 0)
        
        backtest._push_event(MarketEvent(t1, 'TEST', 100, 101, 99, 100, 1000000))
        backtest._push_event(MarketEvent(t2, 'TEST', 98, 99, 97, 98, 1000000))
        backtest._push_event(MarketEvent(t3, 'TEST', 102, 103, 101, 102, 1000000))
        
        # Pop events - should be in chronological order
        e1 = backtest._pop_event()
        e2 = backtest._pop_event()
        e3 = backtest._pop_event()
        
        self.assertEqual(e1.timestamp, t2)
        self.assertEqual(e2.timestamp, t1)
        self.assertEqual(e3.timestamp, t3)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEventSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestDataHandler))
    suite.addTests(loader.loadTestsFromTestCase(TestStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolio))
    suite.addTests(loader.loadTestsFromTestCase(TestBacktest))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
