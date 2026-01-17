"""
Telegram Alert Service - Critical notifications for trading bot.

Sends alerts for:
- Circuit breaker triggered
- Emergency close executed
- PnL threshold breach
- Bot crash/restart
"""

import asyncio
import aiohttp
import logging
from typing import Optional
from config.settings import settings

logger = logging.getLogger("Utils.TelegramAlerts")


class TelegramAlerter:
    """
    Simple async Telegram alerter for critical trading events.
    
    Usage:
        alerter = TelegramAlerter()
        await alerter.send("🚨 Circuit breaker triggered!")
    """
    
    API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        """
        Initialize with Telegram bot token and chat ID.
        
        Get these by:
        1. Create a bot via @BotFather -> get token
        2. Send /start to your bot
        3. Visit https://api.telegram.org/bot<TOKEN>/getUpdates -> get chat_id
        """
        self.token = token or getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
        self.chat_id = chat_id or getattr(settings, 'TELEGRAM_CHAT_ID', None)
        self.enabled = bool(self.token and self.chat_id)
        
        if self.enabled:
            logger.info("✅ Telegram alerts enabled")
        else:
            logger.debug("Telegram alerts disabled (no token/chat_id configured)")
    
    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to the configured Telegram chat.
        
        Args:
            message: Text to send (supports HTML formatting)
            parse_mode: "HTML" or "Markdown"
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug(f"Telegram (disabled): {message}")
            return False
        
        url = self.API_URL.format(token=self.token)
        payload = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": parse_mode,
            "disable_notification": False
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        logger.debug(f"Telegram sent: {message[:50]}...")
                        return True
                    else:
                        logger.warning(f"Telegram API error: {response.status}")
                        return False
        except asyncio.TimeoutError:
            logger.warning("Telegram send timed out")
            return False
        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False
    
    # Convenience methods for common alerts
    
    async def alert_circuit_breaker(self, reason: str):
        """Alert when circuit breaker is triggered."""
        await self.send(
            f"🛑 <b>CIRCUIT BREAKER</b>\n\n"
            f"Trading paused: {reason}\n\n"
            f"Manual intervention required."
        )
    
    async def alert_emergency_close(self, symbol: str, side: str, qty: float, exchange: str):
        """Alert when emergency close is executed."""
        await self.send(
            f"🚨 <b>EMERGENCY CLOSE</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"Qty: {qty}\n"
            f"Exchange: {exchange}\n\n"
            f"Legged position was closed with MARKET order."
        )
    
    async def alert_pnl_breach(self, current_pnl: float, threshold: float):
        """Alert when PnL drops below threshold."""
        await self.send(
            f"⚠️ <b>PnL ALERT</b>\n\n"
            f"Current PnL: ${current_pnl:+.2f}\n"
            f"Threshold: ${threshold:.2f}\n\n"
            f"Review trading activity immediately."
        )
    
    async def alert_bot_started(self):
        """Alert when bot starts."""
        mode = "TESTNET" if settings.TESTNET else "⚠️ MAINNET"
        await self.send(
            f"✅ <b>BOT STARTED</b>\n\n"
            f"Mode: {mode}\n"
            f"Capital: ${settings.INITIAL_CAPITAL:.2f}"
        )
    
    async def alert_bot_stopped(self, reason: str = "Normal shutdown"):
        """Alert when bot stops."""
        await self.send(
            f"🛑 <b>BOT STOPPED</b>\n\n"
            f"Reason: {reason}"
        )
    
    async def alert_error(self, error: str):
        """Alert on critical error."""
        await self.send(
            f"❌ <b>CRITICAL ERROR</b>\n\n"
            f"<code>{error[:500]}</code>"
        )


# Global instance
_alerter: Optional[TelegramAlerter] = None


def get_telegram_alerter() -> TelegramAlerter:
    """Get or create the global Telegram alerter."""
    global _alerter
    if _alerter is None:
        _alerter = TelegramAlerter()
    return _alerter


# Quick send function for one-off alerts
async def send_telegram_alert(message: str) -> bool:
    """Quick way to send a Telegram alert."""
    return await get_telegram_alerter().send(message)


class TelegramCommandHandler:
    """
    Telegram command handler for remote bot control.
    
    Supports commands:
    - /stop: Create STOP_SIGNAL file to halt trading
    - /pause: Create PAUSE_SIGNAL file to pause trading
    - /resume: Remove signal files to resume trading
    - /status: Get current bot status
    
    Usage:
        handler = TelegramCommandHandler()
        asyncio.create_task(handler.start_polling())
    """
    
    GET_UPDATES_URL = "https://api.telegram.org/bot{token}/getUpdates"
    STOP_SIGNAL_FILE = "data/STOP_SIGNAL"
    PAUSE_SIGNAL_FILE = "data/PAUSE_SIGNAL"
    
    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or getattr(settings, 'TELEGRAM_BOT_TOKEN', None)
        self.chat_id = chat_id or getattr(settings, 'TELEGRAM_CHAT_ID', None)
        self.enabled = bool(self.token and self.chat_id)
        self.alerter = TelegramAlerter(self.token, self.chat_id)
        self.last_update_id = 0
        self._running = False
        
        if self.enabled:
            logger.info("✅ Telegram command handler enabled")
    
    async def start_polling(self, poll_interval: float = 2.0):
        """Start polling for commands in background."""
        if not self.enabled:
            logger.debug("Telegram commands disabled - no token configured")
            return
            
        self._running = True
        logger.info("🎧 Starting Telegram command listener...")
        
        while self._running:
            try:
                await self._poll_updates()
            except Exception as e:
                logger.warning(f"Telegram poll error: {e}")
            await asyncio.sleep(poll_interval)
    
    def stop(self):
        """Stop the polling loop."""
        self._running = False
    
    async def _poll_updates(self):
        """Poll for new Telegram messages."""
        url = self.GET_UPDATES_URL.format(token=self.token)
        params = {"offset": self.last_update_id + 1, "timeout": 1}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=5) as response:
                    if response.status != 200:
                        return
                    
                    data = await response.json()
                    if not data.get("ok"):
                        return
                    
                    for update in data.get("result", []):
                        self.last_update_id = update["update_id"]
                        message = update.get("message", {})
                        text = message.get("text", "")
                        chat_id = str(message.get("chat", {}).get("id", ""))
                        
                        # Only process commands from authorized chat
                        if chat_id == self.chat_id:
                            await self._handle_command(text)
        except asyncio.TimeoutError:
            pass
    
    async def _handle_command(self, text: str):
        """Handle incoming command."""
        text = text.strip().lower()
        
        if text == "/stop":
            await self._cmd_stop()
        elif text == "/pause":
            await self._cmd_pause()
        elif text == "/resume":
            await self._cmd_resume()
        elif text == "/status":
            await self._cmd_status()
    
    async def _cmd_stop(self):
        """Handle /stop command - halt trading completely."""
        import os
        from datetime import datetime
        
        with open(self.STOP_SIGNAL_FILE, "w") as f:
            f.write(f"STOP:{datetime.now().isoformat()}")
        
        await self.alerter.send("🛑 <b>STOP COMMAND RECEIVED</b>\n\nTrading halted. Send /resume to restart.")
        logger.warning("📱 STOP command received via Telegram")
    
    async def _cmd_pause(self):
        """Handle /pause command - pause trading temporarily."""
        import os
        from datetime import datetime
        
        with open(self.PAUSE_SIGNAL_FILE, "w") as f:
            f.write(f"PAUSE:{datetime.now().isoformat()}")
        
        await self.alerter.send("⏸️ <b>PAUSE COMMAND RECEIVED</b>\n\nTrading paused. Send /resume to continue.")
        logger.info("📱 PAUSE command received via Telegram")
    
    async def _cmd_resume(self):
        """Handle /resume command - resume trading."""
        import os
        
        for f in [self.STOP_SIGNAL_FILE, self.PAUSE_SIGNAL_FILE]:
            if os.path.exists(f):
                os.remove(f)
        
        await self.alerter.send("▶️ <b>RESUME COMMAND RECEIVED</b>\n\nTrading resumed.")
        logger.info("📱 RESUME command received via Telegram")
    
    async def _cmd_status(self):
        """Handle /status command - report current status."""
        import os
        
        if os.path.exists(self.STOP_SIGNAL_FILE):
            status = "🛑 STOPPED"
        elif os.path.exists(self.PAUSE_SIGNAL_FILE):
            status = "⏸️ PAUSED"
        else:
            status = "✅ ACTIVE"
        
        await self.alerter.send(f"📊 <b>BOT STATUS</b>\n\nStatus: {status}")


# Global command handler instance
_command_handler: Optional[TelegramCommandHandler] = None


def get_telegram_command_handler() -> TelegramCommandHandler:
    """Get or create the global Telegram command handler."""
    global _command_handler
    if _command_handler is None:
        _command_handler = TelegramCommandHandler()
    return _command_handler
