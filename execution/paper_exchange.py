"""
Paper Exchange - Simulated exchange for paper trading.

Features:
- Realistic fill simulation
- Slippage modeling
- Commission tracking
- Uses real orderbook data
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

import structlog

from core.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderBook,
    OrderState,
    OrderType,
    TimeInForce,
)

log = structlog.get_logger()


@dataclass
class PaperOrder:
    """Paper trading order."""
    client_order_id: str
    symbol: str
    side: str  # buy or sell
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal]
    time_in_force: TimeInForce = TimeInForce.GTC
    state: OrderState = OrderState.CREATED
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal = Decimal("0")
    created_at: float = field(default_factory=time.time)
    fill_events: List[FillEvent] = field(default_factory=list)
    
    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity


@dataclass
class PaperBalance:
    """Paper trading balance."""
    available: Decimal = Decimal("0")
    locked: Decimal = Decimal("0")
    
    @property
    def total(self) -> Decimal:
        return self.available + self.locked


class PaperExchange:
    """
    Simulated exchange for paper trading.
    
    Uses real market data for realistic fill simulation.
    """
    
    def __init__(
        self,
        initial_balances: Optional[Dict[str, Decimal]] = None,
        maker_fee: Decimal = Decimal("0.001"),  # 0.1%
        taker_fee: Decimal = Decimal("0.001"),  # 0.1%
        latency_ms: int = 50,
    ) -> None:
        # Fee structure
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.latency_ms = latency_ms
        
        # State
        self._balances: Dict[str, PaperBalance] = defaultdict(PaperBalance)
        self._orders: Dict[str, PaperOrder] = {}
        self._orderbooks: Dict[str, OrderBook] = {}
        
        # Callbacks
        self._fill_callbacks: List[Callable[[FillEvent], None]] = []
        self._order_callbacks: List[Callable[[OrderEvent], None]] = []
        
        # Initialize balances
        if initial_balances:
            for currency, amount in initial_balances.items():
                self._balances[currency].available = amount
        
        log.info(
            "paper_exchange_initialized",
            balances={k: str(v.total) for k, v in self._balances.items()},
        )
    
    def on_fill(self, callback: Callable[[FillEvent], None]) -> None:
        """Register fill event callback."""
        self._fill_callbacks.append(callback)
    
    def on_order_update(self, callback: Callable[[OrderEvent], None]) -> None:
        """Register order update callback."""
        self._order_callbacks.append(callback)
    
    def update_orderbook(self, symbol: str, orderbook: OrderBook) -> None:
        """Update orderbook from real market data."""
        self._orderbooks[symbol] = orderbook
        
        # Check if any limit orders should be filled
        asyncio.create_task(self._check_limit_fills(symbol))
    
    def process_market_event(self, event: MarketEvent) -> None:
        """Process market event (for orderbook updates)."""
        if event.book_snapshot:
            self.update_orderbook(event.symbol, event.book_snapshot)
    
    # =========================================================================
    # Order Operations
    # =========================================================================
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        client_order_id: Optional[str] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> PaperOrder:
        """Create a new paper order."""
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Generate order ID
        order_id = client_order_id or str(uuid.uuid4())
        
        # Validate
        base, quote = self._parse_symbol(symbol)
        
        if side.lower() == "buy":
            required = quantity * (price or self._get_market_price(symbol, "buy"))
            if self._balances[quote].available < required:
                raise ValueError(f"Insufficient {quote} balance")
        else:
            if self._balances[base].available < quantity:
                raise ValueError(f"Insufficient {base} balance")
        
        # Create order
        order = PaperOrder(
            client_order_id=order_id,
            symbol=symbol,
            side=side.lower(),
            order_type=order_type,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
        )
        
        self._orders[order_id] = order
        
        # Lock funds
        self._lock_funds(order)
        
        # Transition to pending
        order.state = OrderState.ACKNOWLEDGED
        
        log.info(
            "paper_order_created",
            order_id=order_id,
            symbol=symbol,
            side=side,
            type=order_type.value,
            quantity=str(quantity),
            price=str(price) if price else "market",
        )
        
        # Execute based on type
        if order_type == OrderType.MARKET:
            await self._execute_market_order(order)
        else:
            # Limit order - check if immediately fillable
            await self._check_limit_fills(symbol)
        
        return order
    
    async def cancel_order(self, client_order_id: str) -> bool:
        """Cancel an order."""
        order = self._orders.get(client_order_id)
        
        if not order:
            return False
        
        if order.state in (OrderState.FILLED, OrderState.CANCELED):
            return False
        
        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)
        
        # Cancel
        order.state = OrderState.CANCELED
        
        # Unlock remaining funds
        self._unlock_funds(order)
        
        log.info("paper_order_canceled", order_id=client_order_id)
        
        return True
    
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancel all open orders."""
        canceled = 0
        
        for order_id, order in list(self._orders.items()):
            if symbol and order.symbol != symbol:
                continue
            
            if order.state in (OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED):
                if await self.cancel_order(order_id):
                    canceled += 1
        
        return canceled
    
    # =========================================================================
    # Order Execution
    # =========================================================================
    
    async def _execute_market_order(self, order: PaperOrder) -> None:
        """Execute a market order."""
        orderbook = self._orderbooks.get(order.symbol)
        
        if not orderbook:
            log.warning("no_orderbook_for_symbol", symbol=order.symbol)
            return
        
        # Get fill price with slippage
        book_levels = orderbook.asks if order.side == "buy" else orderbook.bids
        
        if not book_levels:
            log.warning("empty_orderbook", symbol=order.symbol, side=order.side)
            return
        
        # Simulate walking the book
        remaining = order.remaining_quantity
        fills = []
        total_value = Decimal("0")
        
        for level in book_levels:
            if remaining <= 0:
                break
            
            fill_qty = min(remaining, level.quantity)
            fills.append((level.price, fill_qty))
            total_value += level.price * fill_qty
            remaining -= fill_qty
        
        if not fills:
            return
        
        # Execute fills
        for price, qty in fills:
            await self._record_fill(order, price, qty, is_maker=False)
    
    async def _check_limit_fills(self, symbol: str) -> None:
        """Check if any limit orders can be filled."""
        orderbook = self._orderbooks.get(symbol)
        
        if not orderbook:
            return
        
        for order in list(self._orders.values()):
            if order.symbol != symbol:
                continue
            
            if order.state not in (OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED):
                continue
            
            if order.order_type != OrderType.LIMIT:
                continue
            
            # Check if limit is crossed
            if order.side == "buy":
                if orderbook.best_ask and order.price >= orderbook.best_ask.price:
                    await self._fill_limit_order(order, orderbook.best_ask.price)
            else:
                if orderbook.best_bid and order.price <= orderbook.best_bid.price:
                    await self._fill_limit_order(order, orderbook.best_bid.price)
    
    async def _fill_limit_order(self, order: PaperOrder, fill_price: Decimal) -> None:
        """Fill a limit order."""
        # Fill at limit price (maker fill)
        await self._record_fill(
            order,
            order.price or fill_price,
            order.remaining_quantity,
            is_maker=True,
        )
    
    async def _record_fill(
        self,
        order: PaperOrder,
        price: Decimal,
        quantity: Decimal,
        is_maker: bool,
    ) -> None:
        """Record a fill and update state."""
        # Calculate fee
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        value = price * quantity
        fee = value * fee_rate
        
        base, quote = self._parse_symbol(order.symbol)
        
        # Create fill event
        fill = FillEvent(
            fill_id=str(uuid.uuid4()),
            client_order_id=order.client_order_id,
            exchange_order_id=order.client_order_id,
            exchange="paper",
            symbol=order.symbol,
            side=order.side,
            price=price,
            quantity=quantity,
            fee=fee,
            fee_currency=quote if order.side == "buy" else base,
            is_maker=is_maker,
        )
        
        # Update order state
        order.filled_quantity += quantity
        total_value = order.average_fill_price * (order.filled_quantity - quantity)
        total_value += price * quantity
        order.average_fill_price = total_value / order.filled_quantity
        order.fill_events.append(fill)
        
        if order.filled_quantity >= order.quantity:
            order.state = OrderState.FILLED
        else:
            order.state = OrderState.PARTIALLY_FILLED
        
        # Update balances
        if order.side == "buy":
            self._balances[base].available += quantity
            self._balances[quote].locked -= value
            self._balances[quote].available -= fee
        else:
            self._balances[quote].available += value - fee
            self._balances[base].locked -= quantity
        
        log.info(
            "paper_fill",
            order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            price=str(price),
            quantity=str(quantity),
            fee=str(fee),
            is_maker=is_maker,
        )
        
        # Notify callbacks
        for callback in self._fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                log.error("fill_callback_error", error=str(e))
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _parse_symbol(self, symbol: str) -> tuple:
        """Parse symbol into base and quote currencies."""
        if "-" in symbol:
            parts = symbol.split("-")
            return parts[0], parts[1]
        
        # Assume standard format like BTCUSDT
        for quote in ["USDT", "USDC", "USD", "BTC", "ETH"]:
            if symbol.endswith(quote):
                return symbol[:-len(quote)], quote
        
        return symbol[:3], symbol[3:]
    
    def _get_market_price(self, symbol: str, side: str) -> Decimal:
        """Get current market price."""
        orderbook = self._orderbooks.get(symbol)
        
        if not orderbook:
            raise ValueError(f"No orderbook for {symbol}")
        
        if side == "buy":
            if not orderbook.best_ask:
                raise ValueError(f"No asks for {symbol}")
            return orderbook.best_ask.price
        else:
            if not orderbook.best_bid:
                raise ValueError(f"No bids for {symbol}")
            return orderbook.best_bid.price
    
    def _lock_funds(self, order: PaperOrder) -> None:
        """Lock funds for an order."""
        base, quote = self._parse_symbol(order.symbol)
        
        if order.side == "buy":
            amount = order.quantity * (order.price or Decimal("0"))
            self._balances[quote].available -= amount
            self._balances[quote].locked += amount
        else:
            self._balances[base].available -= order.quantity
            self._balances[base].locked += order.quantity
    
    def _unlock_funds(self, order: PaperOrder) -> None:
        """Unlock funds for a canceled order."""
        base, quote = self._parse_symbol(order.symbol)
        remaining = order.remaining_quantity
        
        if order.side == "buy":
            amount = remaining * (order.price or Decimal("0"))
            self._balances[quote].locked -= amount
            self._balances[quote].available += amount
        else:
            self._balances[base].locked -= remaining
            self._balances[base].available += remaining
    
    # =========================================================================
    # Account Info
    # =========================================================================
    
    def get_balance(self, currency: str) -> PaperBalance:
        """Get balance for a currency."""
        return self._balances[currency]
    
    def get_all_balances(self) -> Dict[str, Dict[str, str]]:
        """Get all balances."""
        return {
            currency: {
                "available": str(balance.available),
                "locked": str(balance.locked),
                "total": str(balance.total),
            }
            for currency, balance in self._balances.items()
            if balance.total > 0
        }
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[PaperOrder]:
        """Get open orders."""
        orders = []
        
        for order in self._orders.values():
            if symbol and order.symbol != symbol:
                continue
            
            if order.state in (OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED):
                orders.append(order)
        
        return orders
    
    def get_order(self, client_order_id: str) -> Optional[PaperOrder]:
        """Get order by ID."""
        return self._orders.get(client_order_id)
