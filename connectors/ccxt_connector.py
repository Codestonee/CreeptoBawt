"""
Multi-Exchange Connector - Unified CCXT-based connector for multiple CEXes.

Supports: Bybit, OKX, KuCoin, MEXC, Gate.io, Bitget
Uses CCXT Pro for WebSocket connectivity where available.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

import structlog

from connectors.base import BaseConnector, ConnectionConfig
from core.events import MarketEvent, OrderBook, OrderBookLevel

log = structlog.get_logger()


@dataclass
class CCXTConfig(ConnectionConfig):
    """CCXT connector configuration."""
    exchange_id: str = "bybit"
    use_testnet: bool = False
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None  # For OKX
    sandbox: bool = False
    
    # Rate limiting
    rate_limit: int = 100  # Requests per minute
    
    # Extra options per exchange
    options: Dict[str, Any] = field(default_factory=dict)


# Exchange-specific symbol mappings
EXCHANGE_SYMBOL_MAPS = {
    "bybit": {
        "BTCUSDT": "BTC-USDT",
        "ETHUSDT": "ETH-USDT",
        "SOLUSDT": "SOL-USDT",
        "XRPUSDT": "XRP-USDT",
    },
    "okx": {
        "BTC-USDT": "BTC-USDT",
        "ETH-USDT": "ETH-USDT",
        "SOL-USDT": "SOL-USDT",
    },
    "kucoin": {
        "BTC-USDT": "BTC-USDT",
        "ETH-USDT": "ETH-USDT",
    },
    "mexc": {
        "BTCUSDT": "BTC-USDT",
        "ETHUSDT": "ETH-USDT",
    },
    "gateio": {
        "BTC_USDT": "BTC-USDT",
        "ETH_USDT": "ETH-USDT",
    },
    "bitget": {
        "BTCUSDT": "BTC-USDT",
        "ETHUSDT": "ETH-USDT",
    },
}


class CCXTConnector(BaseConnector):
    """
    CCXT-based multi-exchange connector.
    
    Uses CCXT Pro for WebSocket where available, falls back to polling.
    Supports: bybit, okx, kucoin, mexc, gateio, bitget
    """
    
    def __init__(self, config: Optional[CCXTConfig] = None) -> None:
        self._ccxt_config = config or CCXTConfig()
        
        # Set symbol map based on exchange
        self.SYMBOL_MAP = EXCHANGE_SYMBOL_MAPS.get(
            self._ccxt_config.exchange_id, {}
        )
        
        super().__init__(self._ccxt_config.exchange_id, self._ccxt_config)
        
        self._exchange = None
        self._watch_tasks: Dict[str, asyncio.Task] = {}
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert normalized symbol to exchange format."""
        exchange_id = self._ccxt_config.exchange_id
        
        # Reverse lookup
        for ex_sym, norm_sym in self.SYMBOL_MAP.items():
            if norm_sym == symbol:
                return ex_sym
        
        # Exchange-specific formatting
        base, quote = symbol.split("-") if "-" in symbol else (symbol[:3], symbol[3:])
        
        if exchange_id == "gateio":
            return f"{base}_{quote}"
        elif exchange_id in ("bybit", "mexc", "bitget"):
            return f"{base}{quote}"
        else:
            return f"{base}-{quote}"
    
    async def _connect(self) -> None:
        """Initialize CCXT exchange."""
        try:
            import ccxt.async_support as ccxt
            
            exchange_class = getattr(ccxt, self._ccxt_config.exchange_id)
            
            config = {
                "apiKey": self._ccxt_config.api_key,
                "secret": self._ccxt_config.api_secret,
                "enableRateLimit": True,
                "options": self._ccxt_config.options,
            }
            
            if self._ccxt_config.passphrase:
                config["password"] = self._ccxt_config.passphrase
            
            if self._ccxt_config.use_testnet:
                config["sandbox"] = True
            
            self._exchange = exchange_class(config)
            
            # Load markets
            await self._exchange.load_markets()
            
            log.info(
                "ccxt_connected",
                exchange=self._ccxt_config.exchange_id,
                markets_count=len(self._exchange.markets),
            )
            
        except ImportError:
            log.error("ccxt_not_installed")
            raise
        except Exception as e:
            log.error("ccxt_connection_failed", error=str(e))
            raise
    
    async def _disconnect(self) -> None:
        """Close CCXT connection."""
        # Cancel watch tasks
        for task in self._watch_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._watch_tasks.clear()
        
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
    
    async def _subscribe_symbols(self, symbols: List[str]) -> None:
        """Subscribe to symbols using watch methods."""
        if not self._exchange:
            return
        
        for symbol in symbols:
            ex_symbol = self.denormalize_symbol(symbol)
            
            # Start watch tasks
            task_key = f"trades_{symbol}"
            if task_key not in self._watch_tasks:
                self._watch_tasks[task_key] = asyncio.create_task(
                    self._watch_trades(symbol, ex_symbol)
                )
            
            task_key = f"orderbook_{symbol}"
            if task_key not in self._watch_tasks:
                self._watch_tasks[task_key] = asyncio.create_task(
                    self._watch_orderbook(symbol, ex_symbol)
                )
    
    async def _unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from symbols."""
        for symbol in symbols:
            for prefix in ["trades_", "orderbook_"]:
                task_key = f"{prefix}{symbol}"
                if task_key in self._watch_tasks:
                    self._watch_tasks[task_key].cancel()
                    del self._watch_tasks[task_key]
    
    async def _watch_trades(self, normalized_symbol: str, exchange_symbol: str) -> None:
        """Watch trades for a symbol."""
        while self._running and self._exchange:
            try:
                trades = await self._exchange.watch_trades(exchange_symbol)
                
                for trade in trades:
                    event = self._parse_ccxt_trade(trade, normalized_symbol)
                    if event:
                        for handler in self._handlers:
                            handler(event)
                        self._stats.messages_processed += 1
                
                self._stats.messages_received += len(trades)
                self._stats.last_message_time = time.time()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(
                    "watch_trades_error",
                    symbol=normalized_symbol,
                    error=str(e),
                )
                await asyncio.sleep(1.0)
    
    async def _watch_orderbook(self, normalized_symbol: str, exchange_symbol: str) -> None:
        """Watch order book for a symbol."""
        while self._running and self._exchange:
            try:
                ob = await self._exchange.watch_order_book(exchange_symbol, limit=10)
                
                event = self._parse_ccxt_orderbook(ob, normalized_symbol)
                if event:
                    for handler in self._handlers:
                        handler(event)
                    self._stats.messages_processed += 1
                
                self._stats.messages_received += 1
                self._stats.last_message_time = time.time()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(
                    "watch_orderbook_error",
                    symbol=normalized_symbol,
                    error=str(e),
                )
                await asyncio.sleep(1.0)
    
    def _parse_ccxt_trade(self, trade: Dict[str, Any], symbol: str) -> MarketEvent:
        """Parse CCXT trade format."""
        return MarketEvent(
            event_type="trade",
            exchange=self._ccxt_config.exchange_id,
            symbol=symbol,
            timestamp_exchange=int(trade.get("timestamp", 0) * 1000),
            timestamp_received=int(time.time() * 1_000_000),
            price=Decimal(str(trade.get("price", "0"))),
            quantity=Decimal(str(trade.get("amount", "0"))),
            side=trade.get("side", "buy"),
            trade_id=str(trade.get("id", "")),
        )
    
    def _parse_ccxt_orderbook(self, ob: Dict[str, Any], symbol: str) -> MarketEvent:
        """Parse CCXT order book format."""
        bids = [
            OrderBookLevel(
                price=Decimal(str(level[0])),
                quantity=Decimal(str(level[1])),
            )
            for level in ob.get("bids", [])[:10]
        ]
        
        asks = [
            OrderBookLevel(
                price=Decimal(str(level[0])),
                quantity=Decimal(str(level[1])),
            )
            for level in ob.get("asks", [])[:10]
        ]
        
        book = OrderBook(bids=bids, asks=asks, timestamp=int(time.time() * 1_000_000))
        
        return MarketEvent(
            event_type="book_update",
            exchange=self._ccxt_config.exchange_id,
            symbol=symbol,
            timestamp_exchange=int(ob.get("timestamp", 0) * 1000),
            timestamp_received=int(time.time() * 1_000_000),
            price=book.mid_price or Decimal("0"),
            quantity=Decimal("0"),
            side="buy",
            book_snapshot=book,
        )
    
    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        """Not used - we use watch tasks instead."""
        await asyncio.sleep(1.0)
        return None
    
    def _parse_message(self, message: Dict[str, Any]) -> Optional[MarketEvent]:
        """Not used - parsing in watch tasks."""
        return None
    
    async def _send_ping(self) -> None:
        """Keep-alive handled by CCXT."""
        pass


# Factory functions for specific exchanges
def create_bybit_connector(testnet: bool = False) -> CCXTConnector:
    """Create Bybit connector."""
    return CCXTConnector(CCXTConfig(exchange_id="bybit", use_testnet=testnet))


def create_okx_connector(testnet: bool = False) -> CCXTConnector:
    """Create OKX connector."""
    return CCXTConnector(CCXTConfig(exchange_id="okx", use_testnet=testnet))


def create_kucoin_connector(testnet: bool = False) -> CCXTConnector:
    """Create KuCoin connector."""
    return CCXTConnector(CCXTConfig(exchange_id="kucoin", use_testnet=testnet))


def create_mexc_connector() -> CCXTConnector:
    """Create MEXC connector."""
    return CCXTConnector(CCXTConfig(exchange_id="mexc"))


def create_gateio_connector() -> CCXTConnector:
    """Create Gate.io connector."""
    return CCXTConnector(CCXTConfig(exchange_id="gateio"))


def create_bitget_connector() -> CCXTConnector:
    """Create Bitget connector."""
    return CCXTConnector(CCXTConfig(exchange_id="bitget"))
