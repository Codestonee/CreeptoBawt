"""
Binance WebSocket Connector - Production-grade Binance market data.

Features:
- Trade and order book streams
- Symbol normalization (BTCUSDT -> BTC-USDT)
- Exponential backoff reconnection
- Stale data detection
- Rate limit handling
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
class BinanceConfig(ConnectionConfig):
    """Binance-specific configuration."""
    # WebSocket URLs
    ws_base_url: str = "wss://stream.binance.com:9443"
    ws_testnet_url: str = "wss://testnet.binance.vision"
    
    # API URLs (for REST fallback)
    rest_base_url: str = "https://api.binance.com"
    rest_testnet_url: str = "https://testnet.binance.vision"
    
    # Use testnet
    use_testnet: bool = False
    
    # Stream settings
    depth_levels: int = 10  # 5, 10, or 20
    update_speed: str = "100ms"  # 100ms or 1000ms


class BinanceWebSocket(BaseConnector):
    """
    Binance WebSocket connector.
    
    Supports:
    - Trade streams (@trade)
    - Order book streams (@depth)
    - Combined streams (multiple symbols)
    """
    
    # Symbol mapping: Binance format -> Normalized format
    SYMBOL_MAP = {
        "BTCUSDT": "BTC-USDT",
        "ETHUSDT": "ETH-USDT",
        "BNBUSDT": "BNB-USDT",
        "SOLUSDT": "SOL-USDT",
        "XRPUSDT": "XRP-USDT",
        "ADAUSDT": "ADA-USDT",
        "DOGEUSDT": "DOGE-USDT",
        "DOTUSDT": "DOT-USDT",
        "MATICUSDT": "MATIC-USDT",
        "LINKUSDT": "LINK-USDT",
        "AVAXUSDT": "AVAX-USDT",
        "UNIUSDT": "UNI-USDT",
        "ATOMUSDT": "ATOM-USDT",
        "LTCUSDT": "LTC-USDT",
        "ETHBTC": "ETH-BTC",
        "BNBBTC": "BNB-BTC",
    }
    
    def __init__(
        self,
        config: Optional[BinanceConfig] = None,
    ) -> None:
        self._binance_config = config or BinanceConfig()
        super().__init__("binance", self._binance_config)
        
        self._ws = None
        self._stream_names: List[str] = []
        self._message_id = 0
    
    @property
    def ws_url(self) -> str:
        """Get WebSocket URL based on config."""
        if self._binance_config.use_testnet:
            return self._binance_config.ws_testnet_url
        return self._binance_config.ws_base_url
    
    def _get_stream_names(self, symbols: List[str]) -> List[str]:
        """Generate stream names for symbols."""
        streams = []
        for symbol in symbols:
            # Denormalize to Binance format
            binance_symbol = self.denormalize_symbol(symbol).lower()
            
            # Trade stream
            streams.append(f"{binance_symbol}@trade")
            
            # Order book stream
            depth = self._binance_config.depth_levels
            speed = self._binance_config.update_speed
            streams.append(f"{binance_symbol}@depth{depth}@{speed}")
        
        return streams
    
    async def _connect(self) -> None:
        """Establish WebSocket connection."""
        try:
            import websockets
            
            if self._stream_names:
                # Combined stream URL
                streams = "/".join(self._stream_names)
                url = f"{self.ws_url}/stream?streams={streams}"
            else:
                # Use generic connection, subscribe later
                url = f"{self.ws_url}/ws"
            
            self._ws = await websockets.connect(
                url,
                ping_interval=self.config.ping_interval_seconds,
                ping_timeout=self.config.pong_timeout_seconds,
                close_timeout=10,
                max_size=10 * 1024 * 1024,  # 10MB max message size
            )
            
            log.info("binance_connected", url=url)
            
        except ImportError:
            log.error("websockets_not_installed")
            raise
    
    async def _disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                log.warning("disconnect_error", error=str(e))
            self._ws = None
    
    async def _subscribe_symbols(self, symbols: List[str]) -> None:
        """Subscribe to market data streams."""
        if not self._ws:
            return
        
        streams = self._get_stream_names(symbols)
        self._stream_names.extend(streams)
        
        self._message_id += 1
        message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": self._message_id,
        }
        
        await self._ws.send(json.dumps(message))
        
        log.info("binance_subscribed", streams=streams)
    
    async def _unsubscribe_symbols(self, symbols: List[str]) -> None:
        """Unsubscribe from market data streams."""
        if not self._ws:
            return
        
        streams = self._get_stream_names(symbols)
        
        for stream in streams:
            if stream in self._stream_names:
                self._stream_names.remove(stream)
        
        self._message_id += 1
        message = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": self._message_id,
        }
        
        await self._ws.send(json.dumps(message))
        
        log.info("binance_unsubscribed", streams=streams)
    
    async def _receive_message(self) -> Optional[Dict[str, Any]]:
        """Receive and parse WebSocket message."""
        if not self._ws:
            return None
        
        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=30.0)
            return json.loads(raw)
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            log.warning("receive_error", error=str(e))
            raise
    
    def _parse_message(self, message: Dict[str, Any]) -> Optional[MarketEvent]:
        """Parse Binance message into MarketEvent."""
        # Handle combined stream format
        if "stream" in message:
            stream = message["stream"]
            data = message["data"]
        else:
            # Direct stream format
            data = message
            stream = ""
        
        # Skip subscription responses
        if "result" in message or "id" in message:
            return None
        
        # Determine message type
        event_type = data.get("e", "")
        
        if event_type == "trade":
            return self._parse_trade(data)
        elif event_type == "depthUpdate" or "@depth" in stream:
            return self._parse_depth(data)
        
        return None
    
    def _parse_trade(self, data: Dict[str, Any]) -> MarketEvent:
        """Parse trade message."""
        symbol = data.get("s", "")
        normalized_symbol = self.normalize_symbol(symbol)
        
        # Binance uses 'm' to indicate if buyer is market maker
        # m=True means buyer is maker, so it's a sell
        is_buyer_maker = data.get("m", False)
        side = "sell" if is_buyer_maker else "buy"
        
        return MarketEvent(
            event_type="trade",
            exchange="binance",
            symbol=normalized_symbol,
            timestamp_exchange=data.get("T", 0) * 1000,  # ms to us
            timestamp_received=int(time.time() * 1_000_000),
            price=Decimal(str(data.get("p", "0"))),
            quantity=Decimal(str(data.get("q", "0"))),
            side=side,
            trade_id=str(data.get("t", "")),
            metadata={
                "buyer_order_id": data.get("b"),
                "seller_order_id": data.get("a"),
            },
        )
    
    def _parse_depth(self, data: Dict[str, Any]) -> MarketEvent:
        """Parse order book depth message."""
        symbol = data.get("s", "")
        normalized_symbol = self.normalize_symbol(symbol)
        
        # Parse bids and asks
        bids = [
            OrderBookLevel(
                price=Decimal(str(level[0])),
                quantity=Decimal(str(level[1])),
            )
            for level in data.get("bids", data.get("b", []))
        ]
        
        asks = [
            OrderBookLevel(
                price=Decimal(str(level[0])),
                quantity=Decimal(str(level[1])),
            )
            for level in data.get("asks", data.get("a", []))
        ]
        
        # Sort: bids descending, asks ascending
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        
        book = OrderBook(
            bids=bids,
            asks=asks,
            timestamp=int(time.time() * 1_000_000),
        )
        
        # Use mid price as event price
        mid_price = book.mid_price or Decimal("0")
        
        return MarketEvent(
            event_type="book_update",
            exchange="binance",
            symbol=normalized_symbol,
            timestamp_exchange=data.get("E", 0) * 1000,  # ms to us
            timestamp_received=int(time.time() * 1_000_000),
            price=mid_price,
            quantity=Decimal("0"),  # N/A for book updates
            side="buy",  # N/A for book updates
            book_snapshot=book,
            metadata={
                "first_update_id": data.get("U"),
                "final_update_id": data.get("u"),
            },
        )
    
    async def _send_ping(self) -> None:
        """Send ping frame."""
        if self._ws:
            try:
                pong_waiter = await self._ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=self.config.pong_timeout_seconds)
            except Exception as e:
                log.warning("ping_failed", error=str(e))
                raise
    
    # =========================================================================
    # Binance-Specific Methods
    # =========================================================================
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Fetch exchange info via REST API."""
        import aiohttp
        
        url = f"{self._binance_config.rest_base_url}/api/v3/exchangeInfo"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
    
    async def get_ticker_24h(self, symbol: str) -> Dict[str, Any]:
        """Fetch 24h ticker stats."""
        import aiohttp
        
        binance_symbol = self.denormalize_symbol(symbol)
        url = f"{self._binance_config.rest_base_url}/api/v3/ticker/24hr?symbol={binance_symbol}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
