"""
Telegram notification utilities for trade alerts
"""

import os
import requests
from typing import Optional
from datetime import datetime


class TelegramNotifier:
    """Send trade notifications via Telegram Bot API"""
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token) and bool(self.chat_id)
        
    def send_message(self, message: str) -> bool:
        """Send a message to Telegram"""
        if not self.enabled:
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"[Telegram] Failed to send message: {e}")
            return False
    
    def notify_order(self, symbol: str, action: str, quantity: int, 
                     price: float, mode: str, pnl: Optional[float] = None) -> bool:
        """Send order execution notification"""
        if not self.enabled:
            return False
            
        # Determine emoji based on action
        action_emojis = {
            "buy": "🟢",
            "sell": "🔴",
            "hold": "⚪",
        }
        
        mode_emojis = {
            "simulate": "🔍 SIMULATE",
            "secure": "🛡️ SECURE",
            "autopilot": "🤖 AUTOPILOT",
        }
        
        emoji = action_emojis.get(action.lower(), "⚪")
        mode_display = mode_emojis.get(mode.lower(), mode.upper())
        
        # Build message
        lines = [
            f"🦊 *Fox of Wallstreet - Trade Alert*",
            f"",
            f"{emoji} *{action.upper()}* {symbol}",
            f"📊 Mode: {mode_display}",
            f"📈 Quantity: {quantity} shares",
            f"💵 Price: ${price:.2f}",
            f"💰 Total: ${quantity * price:,.2f}",
        ]
        
        if pnl is not None:
            pnl_emoji = "📈" if pnl >= 0 else "📉"
            lines.append(f"{pnl_emoji} P&L: ${pnl:,.2f}")
        
        lines.append(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        message = "\n".join(lines)
        return self.send_message(message)
    
    def notify_error(self, symbol: str, error_msg: str) -> bool:
        """Send error notification"""
        if not self.enabled:
            return False
            
        message = (
            f"⚠️ *Trade Error*\n\n"
            f"Symbol: {symbol}\n"
            f"Error: {error_msg}\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return self.send_message(message)
    
    def notify_daily_summary(self, symbol: str, trades: int, total_pnl: float,
                             win_rate: float, current_shares: int) -> bool:
        """Send daily summary notification"""
        if not self.enabled:
            return False
            
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        message = (
            f"📊 *Daily Trading Summary - {symbol}*\n\n"
            f"🔄 Trades: {trades}\n"
            f"{pnl_emoji} Total P&L: ${total_pnl:,.2f}\n"
            f"🏆 Win Rate: {win_rate:.1%}\n"
            f"📈 Position: {current_shares} shares\n\n"
            f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        return self.send_message(message)


# Global notifier instance
_notifier = None

def get_notifier() -> TelegramNotifier:
    """Get or create Telegram notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
