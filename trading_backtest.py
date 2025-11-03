"""
Main entry point for running algorithmic trading strategy backtests.
Demonstrates the event-driven backtesting engine with momentum strategy.
"""

import os
from data_handler import HistoricCSVDataHandler
from strategy import MomentumStrategy, BuyAndHoldStrategy
from portfolio import Portfolio
from execution import SimulatedExecutionHandler
from backtest import Backtest


def run_backtest(symbol_list=['AAPL'], initial_capital=100000.0, 
                 strategy_type='momentum', short_window=20, long_window=50):
    """
    Run a backtest with the specified parameters.
    
    Args:
        symbol_list: list of ticker symbols to trade
        initial_capital: starting capital in dollars
        strategy_type: 'momentum' or 'buy_and_hold'
        short_window: short-term lookback period for momentum
        long_window: long-term lookback period for momentum
    """
    # Initialize components
    print("Initializing backtesting components...")
    
    # Data handler - will generate synthetic data if CSV files not found
    data_handler = HistoricCSVDataHandler(
        csv_dir='./data',
        symbol_list=symbol_list
    )
    
    # Strategy
    if strategy_type == 'momentum':
        strategy = MomentumStrategy(
            data_handler,
            short_window=short_window,
            long_window=long_window
        )
        print(f"Strategy: Momentum (short={short_window}, long={long_window})")
    else:
        strategy = BuyAndHoldStrategy(data_handler)
        print("Strategy: Buy and Hold")
    
    # Portfolio
    portfolio = Portfolio(data_handler, initial_capital=initial_capital)
    
    # Execution handler
    execution_handler = SimulatedExecutionHandler(data_handler)
    
    # Backtest engine
    backtest_engine = Backtest(
        data_handler=data_handler,
        strategy=strategy,
        portfolio=portfolio,
        execution_handler=execution_handler
    )
    
    # Run backtest
    print()
    backtest_engine.run()
    
    # Print results
    backtest_engine.print_results()
    
    return backtest_engine


def compare_strategies(symbol_list=['AAPL', 'MSFT']):
    """
    Compare momentum strategy against buy-and-hold benchmark.
    
    Args:
        symbol_list: list of ticker symbols to trade
    """
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    
    initial_capital = 100000.0
    
    # Run momentum strategy
    print("\n1. MOMENTUM STRATEGY")
    print("-" * 60)
    momentum_backtest = run_backtest(
        symbol_list=symbol_list,
        initial_capital=initial_capital,
        strategy_type='momentum'
    )
    momentum_results = momentum_backtest.get_results()
    
    # Run buy-and-hold strategy
    print("\n2. BUY AND HOLD STRATEGY (BENCHMARK)")
    print("-" * 60)
    buyhold_backtest = run_backtest(
        symbol_list=symbol_list,
        initial_capital=initial_capital,
        strategy_type='buy_and_hold'
    )
    buyhold_results = buyhold_backtest.get_results()
    
    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Momentum':>15} {'Buy & Hold':>15}")
    print("-" * 60)
    print(f"{'Total Return':<25} {momentum_results['total_return_pct']:>14.2f}% {buyhold_results['total_return_pct']:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {momentum_results['sharpe_ratio']:>15.2f} {buyhold_results['sharpe_ratio']:>15.2f}")
    print(f"{'Max Drawdown':<25} {momentum_results['max_drawdown_pct']:>14.2f}% {buyhold_results['max_drawdown_pct']:>14.2f}%")
    print(f"{'Win Rate':<25} {momentum_results['win_rate']*100:>14.2f}% {buyhold_results['win_rate']*100:>14.2f}%")
    print(f"{'Total Trades':<25} {momentum_results['total_trades']:>15} {buyhold_results['total_trades']:>15}")
    print("=" * 60)
    
    # Determine winner
    if momentum_results['total_return'] > buyhold_results['total_return']:
        print("\n✓ Momentum strategy outperformed buy-and-hold!")
    else:
        print("\n✗ Momentum strategy underperformed buy-and-hold.")


def main():
    """
    Main function demonstrating the algorithmic trading backtester.
    """
    print("=" * 60)
    print("ALGORITHMIC TRADING STRATEGY BACKTESTER")
    print("=" * 60)
    print()
    print("This backtesting system features:")
    print("- Discrete event-based simulation engine")
    print("- Priority queue (min-heap) for time-stamped events")
    print("- Momentum-based trading strategy")
    print("- Performance analysis and metrics")
    print()
    
    # Single strategy backtest
    print("\n" + "=" * 60)
    print("SINGLE STRATEGY BACKTEST")
    print("=" * 60)
    backtest = run_backtest(
        symbol_list=['AAPL'],
        initial_capital=100000.0,
        strategy_type='momentum',
        short_window=20,
        long_window=50
    )
    
    # Multi-symbol strategy comparison
    print("\n\n" + "=" * 60)
    print("MULTI-SYMBOL STRATEGY COMPARISON")
    print("=" * 60)
    compare_strategies(symbol_list=['AAPL', 'MSFT'])
    
    print("\n" + "=" * 60)
    print("Backtest Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
