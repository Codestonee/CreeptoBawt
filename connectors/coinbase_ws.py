"""
Coinbase WebSocket Connector - Coinbase Advanced Trade market data.

Features:
- Match and level2 channels
- Symbol normalization (BTC-USD -> BTC-USDT)
- Authentication support for user channels
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog

from connectors.base import BaseConnector, ConnectionConfig
from core.events import MarketEvent, OrderBook, OrderBookLevel

log = structlog.get_logger()


@dataclass
class CoinbaseConfig(ConnectionConfig):
    """Coinbase-specific configuration."""
    ws_url: str = "wss://advanced-trade-ws.coinbase.com"
    
    # Authentication (optional, for user channels)
    api_key: Optional[str] = None
    api_secret: Optional[str] = None


class CoinbaseWebSocket(BaseConnector):
    """
    Coinbase Advanced Trade WebSocket connector.
    
    Channels:
    - matches: Trade executions
    - level2: Order book updates
    - ticker: Price updates
    """
    
    SYMBOL_MAP = {
        "BTC-USD": "BTC-USDT",
        "ETH-USD": "ETH-USDT",
        "SOL-USD": "SOL-USDT",
        "DOGE-USD": "DOGE-USDT",
        "XRP-USD": "XRP-USDT",
        "ADA-USD": "ADA-USDT",
        "LINK-USD": "LINK-USDT",
        "AVAX-USD": "AVAX-USDT",
        "DOT-USD": "DOT-USDT",
        "MATIC-USD": "MATIC-USDT",
    }
    
    def __init__(self, config: Optional[CoinbaseConfig] = None) -> None:
        self._coinbase_config = config or CoinbaseConfig()
        super().__init__("coinbase", self._coinbase_config)
        self._ws = None
    
    def denormalize_symbol(self, symbol: str) -> str:
        """Convert normalized symbol to Coinbase format."""
        # Reverse lookup
        for cb_sym, norm_sym in self.SYMBOL_MAP.items():
            if norm_sym == symbol:
                return cb_sym
        
        # Default: already in format BASE-QUOTE
        return symbol
    
    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        import websockets
        
        self._ws = await websockets.connect(
            self._coinbase_config.ws_url,
            ping_interval=self.config.ping_interval_seconds,
            ping_timeout=self.config.pong_timeout_seconds,
        )
        
        log.info("coinbase_connected", url=self._coinbase_config.ws_url)
    
    async def _disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._ws = None
    
    async def _subscribe_symbols(self, symbols: List[str]) -> None:
        """Subscribe to channels for symbols."""
        if not self._ws:
            return
        
        product_ids = [self.denormalize_symbol(s) for s in symbols]
        
        message = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channel": "matches",  # Trade matches
        }
        
        await self._ws.send(json.dumps(message))
        
        # Also subscribe to level2 for order book
        message["channel"] = "level2"
        await self._ws.send(json.dumps(message))
        
        log.info("coinbase_subscribed", products=product_ids)
    
    async def _unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from channels."""
        if not self._ws:
            return
        
        product_ids = [self.denormalize_symbol(s) for s in symbols]
        
        message = {
            "type": "unsubscribe",
            "product_ids": product_ids,
            "channel": "matches",
        }
        
        await self._ws.send(json.dumps(message))
    
    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive WebSocket message."""
        if not self._ws:
            return None
        
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
            return json.loads(raw)
        except asyncio.TimeoutError:
            return None
    
    def _parse_message(self, message: Dict[str, Any]) -> Optional[MarketEvent]:
        """Parse Coinbase message."""
        msg_type = message.get("type", "")
        channel = message.get("channel", "")
        
        # Skip subscription confirmations
        if msg_type in ("subscriptions", "subscribe", "unsubscribe"):
            return None
        
        # Handle matches channel
        if channel == "matches" or msg_type == "match":
            return self._parse_match(message)
        
        # Handle level2 updates
        if channel == "level2" or msg_type == "l2update":
            return self._parse_level2(message)
        
        return None
    
    def _parse_match(self, data: Dict[str, Any]) -> Optional[MarketEvent]:
        """Parse match/trade message."""
        events = data.get("events", [data])
        
        for event in events:
            trades = event.get("trades", [event])
            for trade in trades:
                product_id = trade.get("product_id", "")
                if not product_id:
                    continue
                
                return MarketEvent(
                    event_type="trade",
                    exchange="coinbase",
                    symbol=self.normalize_symbol(product_id),
                    timestamp_exchange=int(time.time() * 1_000_000),  # CB doesn't always provide
                    timestamp_received=int(time.time() * 1_000_000),
                    price=Decimal(str(trade.get("price", "0"))),
                    quantity=Decimal(str(trade.get("size", "0"))),
                    side=trade.get("side", "buy").lower(),
                    trade_id=str(trade.get("trade_id", "")),
                )
        
        return None
    
    def _parse_level2(self, data: Dict[str, Any]) -> Optional[MarketEvent]:
        """Parse level2 order book update."""
        events = data.get("events", [data])
        
        for event in events:
            product_id = event.get("product_id", "")
            if not product_id:
                continue
            
            updates = event.get("updates", [])
            bids = []
            asks = []
            
            for update in updates:
                side = update.get("side", "")
                price = Decimal(str(update.get("price_level", "0")))
                qty = Decimal(str(update.get("new_quantity", "0")))
                
                level = OrderBookLevel(price=price, quantity=qty)
                
                if side == "bid":
                    bids.append(level)
                elif side == "offer":
                    asks.append(level)
            
            if bids or asks:
                bids.sort(key=lambda x: x.price, reverse=True)
                asks.sort(key=lambda x: x.price)
                
                book = OrderBook(
                    bids=bids[:10],
                    asks=asks[:10],
                    timestamp=int(time.time() * 1_000_000),
                )
                
                return MarketEvent(
                    event_type="book_update",
                    exchange="coinbase",
                    symbol=self.normalize_symbol(product_id),
                    timestamp_exchange=int(time.time() * 1_000_000),
                    timestamp_received=int(time.time() * 1_000_000),
                    price=book.mid_price or Decimal("0"),
                    quantity=Decimal("0"),
                    side="buy",
                    book_snapshot=book,
                )
        
        return None
    
    async def _send_ping(self) -> None:
        """Send ping."""
        if self._ws:
            await self._ws.ping()
