"""
Alpaca trading client wrapper for live trading.

Handles:
- Order submission (market orders)
- Portfolio sync (cash, positions)
- Price fetching (for verification)
- Paper vs live trading modes
"""

import os
import sys
from typing import Optional, Dict, Tuple
from datetime import datetime

import pandas as pd

# Add parent paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from config import settings


class AlpacaTrader:
    """Wrapper for Alpaca trading operations."""
    
    def __init__(self, api_key: str = None, secret_key: str = None, paper: bool = True):
        """
        Initialize Alpaca trading client.
        
        Args:
            api_key: Alpaca API key (or from env ALPACA_API_KEY)
            secret_key: Alpaca secret key (or from env ALPACA_SECRET_KEY)
            paper: Use paper trading (True) or live trading (False)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper = paper
        self.client = None
        self.data_client = None
        
        if self.api_key and self.secret_key:
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.data.historical import StockHistoricalDataClient
                
                self.client = TradingClient(self.api_key, self.secret_key, paper=paper)
                self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
                
                # Test connection
                account = self.client.get_account()
                print(f"✅ Alpaca connected: {'Paper' if paper else 'LIVE'} trading")
                print(f"   Account: {account.account_number}")
                print(f"   Cash: ${float(account.cash):,.2f}")
                
            except Exception as e:
                print(f"❌ Failed to connect to Alpaca: {e}")
                self.client = None
                self.data_client = None
        else:
            print("⚠️ No Alpaca credentials provided. Trading disabled.")
    
    def is_connected(self) -> bool:
        """Check if Alpaca client is connected."""
        return self.client is not None
    
    def get_portfolio(self) -> Dict:
        """
        Get current portfolio state from Alpaca.
        
        Returns:
            Dict with cash, position, entry_price, etc.
        """
        if not self.is_connected():
            return {
                "cash": 0.0,
                "position": 0.0,
                "entry_price": 0.0,
                "portfolio_value": 0.0,
                "unrealized_pnl": 0.0,
            }
        
        try:
            # Get account info
            account = self.client.get_account()
            raw_cash = float(account.cash)
            
            # Cap to trading budget
            available_cash = min(raw_cash, settings.LIVE_TRADING_BUDGET)
            
            # Get position
            try:
                position = self.client.get_open_position(settings.SYMBOL)
                current_shares = float(position.qty)
                entry_price = float(position.avg_entry_price)
                unrealized_pnl_pct = float(position.unrealized_plpc) if position.unrealized_plpc else 0.0
                unrealized_pnl_dollar = float(position.unrealized_pl) if position.unrealized_pl else 0.0
            except Exception:
                # No position
                current_shares = 0.0
                entry_price = 0.0
                unrealized_pnl_pct = 0.0
                unrealized_pnl_dollar = 0.0
            
            portfolio_value = available_cash + (current_shares * entry_price if current_shares > 0 else 0)
            
            return {
                "cash": available_cash,
                "raw_cash": raw_cash,  # Full Alpaca balance
                "position": current_shares,
                "entry_price": entry_price,
                "portfolio_value": portfolio_value,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "unrealized_pnl_dollar": unrealized_pnl_dollar,
            }
            
        except Exception as e:
            print(f"❌ Error fetching portfolio: {e}")
            return {
                "cash": 0.0,
                "position": 0.0,
                "entry_price": 0.0,
                "portfolio_value": 0.0,
                "unrealized_pnl": 0.0,
            }
    
    def get_current_price(self, symbol: str = None) -> Optional[float]:
        """
        Get current price from Alpaca for verification.
        
        Args:
            symbol: Stock symbol (default from settings)
            
        Returns:
            Current price or None if error
        """
        if not self.is_connected() or self.data_client is None:
            return None
        
        symbol = symbol or settings.SYMBOL
        
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            if symbol in quotes:
                # Use midpoint of bid/ask
                bid = float(quotes[symbol].bid_price)
                ask = float(quotes[symbol].ask_price)
                midpoint = (bid + ask) / 2
                return midpoint
            
            return None
            
        except Exception as e:
            print(f"⚠️ Could not fetch Alpaca price: {e}")
            return None
    
    def verify_price(self, expected_price: float, max_diff_pct: float = 0.005) -> Tuple[bool, float, str]:
        """
        Verify that Alpaca price matches expected price (from yfinance).
        
        Args:
            expected_price: Price we expect (from yfinance/analysis)
            max_diff_pct: Maximum acceptable difference (default 0.5%)
            
        Returns:
            (is_valid, alpaca_price, message)
        """
        alpaca_price = self.get_current_price()
        
        if alpaca_price is None:
            return False, 0.0, "Could not fetch Alpaca price for verification"
        
        diff_pct = abs(alpaca_price - expected_price) / expected_price
        
        if diff_pct > max_diff_pct:
            return (
                False,
                alpaca_price,
                f"Price mismatch: Expected ${expected_price:.2f}, Alpaca shows ${alpaca_price:.2f} ({diff_pct:.2%} diff)"
            )
        
        return True, alpaca_price, f"Price verified: ${alpaca_price:.2f} (diff: {diff_pct:.2%})"
    
    def submit_order(
        self,
        action: int,
        action_space: str,
        current_price: float,
        current_shares: float,
        available_cash: float,
    ) -> Dict:
        """
        Submit order to Alpaca.
        
        Args:
            action: Action index (0-4)
            action_space: "discrete_3" or "discrete_5"
            current_price: Current stock price
            current_shares: Current position size
            available_cash: Available cash
            
        Returns:
            Dict with order status, message, etc.
        """
        if not self.is_connected():
            return {
                "success": False,
                "status": "error",
                "message": "Alpaca client not connected",
            }
        
        try:
            from alpaca.trading.requests import MarketOrderRequest
            from alpaca.trading.enums import OrderSide, TimeInForce
            
            result = {
                "success": False,
                "status": "unknown",
                "message": "",
                "order_id": None,
                "quantity": 0,
            }
            
            # Determine order details based on action
            if action_space == "discrete_3":
                if action == 1:  # Buy All
                    investment = available_cash * settings.CASH_RISK_FRACTION
                    if investment < 10.0:
                        return {"success": False, "status": "skipped", "message": "Investment too small"}
                    
                    order = MarketOrderRequest(
                        symbol=settings.SYMBOL,
                        notional=investment,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                    result["message"] = f"BUY_ALL: ${investment:.2f}"
                    result["quantity"] = int(investment / current_price) if current_price > 0 else 0
                    
                elif action == 0:  # Sell All
                    if current_shares <= 0:
                        return {"success": False, "status": "skipped", "message": "No position to sell"}
                    
                    order = MarketOrderRequest(
                        symbol=settings.SYMBOL,
                        qty=current_shares,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    result["message"] = f"SELL_ALL: {current_shares:.4f} shares"
                    result["quantity"] = int(current_shares)
                    
                else:  # Hold
                    return {"success": True, "status": "skipped", "message": "HOLD - no action needed"}
            
            else:  # discrete_5
                if action == 4:  # Buy 100%
                    investment = available_cash * settings.CASH_RISK_FRACTION
                    if investment < 10.0:
                        return {"success": False, "status": "skipped", "message": "Investment too small"}
                    
                    order = MarketOrderRequest(
                        symbol=settings.SYMBOL,
                        notional=investment,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                    result["message"] = f"BUY_100: ${investment:.2f}"
                    result["quantity"] = int(investment / current_price) if current_price > 0 else 0
                    
                elif action == 3:  # Buy 50%
                    investment = available_cash * 0.5 * settings.CASH_RISK_FRACTION
                    if investment < 10.0:
                        return {"success": False, "status": "skipped", "message": "Investment too small"}
                    
                    order = MarketOrderRequest(
                        symbol=settings.SYMBOL,
                        notional=investment,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                    result["message"] = f"BUY_50: ${investment:.2f}"
                    result["quantity"] = int(investment / current_price) if current_price > 0 else 0
                    
                elif action == 0:  # Sell 100%
                    if current_shares <= 0:
                        return {"success": False, "status": "skipped", "message": "No position to sell"}
                    
                    order = MarketOrderRequest(
                        symbol=settings.SYMBOL,
                        qty=current_shares,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    result["message"] = f"SELL_100: {current_shares:.4f} shares"
                    result["quantity"] = int(current_shares)
                    
                elif action == 1:  # Sell 50%
                    if current_shares <= 0:
                        return {"success": False, "status": "skipped", "message": "No position to sell"}
                    
                    shares_to_sell = current_shares * 0.5
                    order = MarketOrderRequest(
                        symbol=settings.SYMBOL,
                        qty=shares_to_sell,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                    result["message"] = f"SELL_50: {shares_to_sell:.4f} shares"
                    result["quantity"] = int(shares_to_sell)
                    
                else:  # Hold
                    return {"success": True, "status": "skipped", "message": "HOLD - no action needed"}
            
            # Submit order
            submitted_order = self.client.submit_order(order)
            
            result["success"] = True
            result["status"] = "submitted"
            result["order_id"] = str(submitted_order.id)
            
            print(f"✅ Order submitted: {result['message']} (ID: {result['order_id']})")
            
            return result
            
        except Exception as e:
            error_msg = f"Order submission failed: {e}"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "status": "error",
                "message": error_msg,
            }
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get status of a submitted order."""
        if not self.is_connected():
            return {"status": "unknown", "message": "Not connected"}
        
        try:
            order = self.client.get_order_by_id(order_id)
            return {
                "status": order.status,
                "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
                "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}


def get_alpaca_trader() -> AlpacaTrader:
    """Factory function to get Alpaca trader instance."""
    return AlpacaTrader()
