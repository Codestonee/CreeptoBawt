"""
Order Manager with Finite State Machine.

Proper order lifecycle tracking with state transitions, retry logic,
and network failure handling.

Features:
- FSM with valid state transitions
- client_order_id for idempotency
- Network failure handling with order existence checks
- Partial fill handling with cancel-and-replace
- Order timeout monitoring and stale order cleanup
- Exponential backoff retry (2^n * 100ms, max 30s)
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import structlog

from core.events import OrderState, OrderType, TimeInForce

log = structlog.get_logger()


class InvalidStateTransition(Exception):
    """Raised when attempting an invalid state transition."""
    pass


@dataclass
class Order:
    """
    Order with state machine tracking.
    
    Tracks full lifecycle from creation to completion with
    proper state transitions and history.
    """
    client_order_id: str
    exchange: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    state: OrderState = OrderState.CREATED
    exchange_order_id: Optional[str] = None
    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    timeout_seconds: float = 300.0  # 5 minutes default
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # State transition tracking
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    last_error: Optional[str] = None
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        OrderState.CREATED: [OrderState.PENDING, OrderState.FAILED],
        OrderState.PENDING: [OrderState.ACKNOWLEDGED, OrderState.REJECTED, OrderState.FAILED],
        OrderState.ACKNOWLEDGED: [
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELING,
        ],
        OrderState.PARTIALLY_FILLED: [
            OrderState.FILLED,
            OrderState.CANCELING,
            OrderState.CANCELED,
        ],
        OrderState.CANCELING: [OrderState.CANCELED, OrderState.FILLED],
        OrderState.FILLED: [],  # Terminal state
        OrderState.CANCELED: [],  # Terminal state
        OrderState.REJECTED: [],  # Terminal state
        OrderState.EXPIRED: [],  # Terminal state
        OrderState.FAILED: [],  # Terminal state
    }
    
    def transition(self, new_state: OrderState, reason: str = "") -> None:
        """
        Transition to a new state.
        
        Args:
            new_state: Target state
            reason: Reason for transition
            
        Raises:
            InvalidStateTransition: If transition is not valid
        """
        valid_next_states = self.VALID_TRANSITIONS.get(self.state, [])
        
        if new_state not in valid_next_states:
            raise InvalidStateTransition(
                f"Cannot transition from {self.state.value} to {new_state.value}. "
                f"Valid transitions: {[s.value for s in valid_next_states]}"
            )
        
        # Record in history
        self.state_history.append({
            "from_state": self.state.value,
            "to_state": new_state.value,
            "reason": reason,
            "timestamp": time.time(),
        })
        
        old_state = self.state
        self.state = new_state
        self.updated_at = time.time()
        
        log.info(
            "order_state_transition",
            client_order_id=self.client_order_id,
            from_state=old_state.value,
            to_state=new_state.value,
            reason=reason,
        )
    
    @property
    def is_complete(self) -> bool:
        """Check if order is in a terminal state."""
        return self.state in {
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.FAILED,
        }
    
    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.state in {
            OrderState.CREATED,
            OrderState.PENDING,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
        }
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_timed_out(self) -> bool:
        """Check if order has timed out."""
        age = time.time() - self.created_at
        return age > self.timeout_seconds and self.is_open
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "client_order_id": self.client_order_id,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type.value,
            "quantity": str(self.quantity),
            "price": str(self.price) if self.price else None,
            "state": self.state.value,
            "exchange_order_id": self.exchange_order_id,
            "filled_quantity": str(self.filled_quantity),
            "average_fill_price": str(self.average_fill_price) if self.average_fill_price else None,
            "time_in_force": self.time_in_force.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "retry_count": self.retry_count,
            "strategy_id": self.strategy_id,
            "metadata": self.metadata,
        }


class OrderManager:
    """
    Manages order lifecycle with state machine and retry logic.
    
    Features:
    - Order state tracking with FSM
    - Retry with exponential backoff
    - Network failure handling
    - Partial fill management
    - Order timeout monitoring
    """
    
    def __init__(
        self,
        max_retries: int = 5,
        base_retry_delay_ms: int = 100,
        max_retry_delay_seconds: int = 30,
    ) -> None:
        """
        Initialize order manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_retry_delay_ms: Base delay for exponential backoff (ms)
            max_retry_delay_seconds: Maximum retry delay (seconds)
        """
        self.max_retries = max_retries
        self.base_retry_delay_ms = base_retry_delay_ms
        self.max_retry_delay_seconds = max_retry_delay_seconds
        
        self._orders: Dict[str, Order] = {}
        self._running = False
        self._timeout_check_task: Optional[asyncio.Task] = None
        
        # Dependencies (to be injected)
        self.exchange_connector = None
        self.event_bus = None
        
        log.info(
            "order_manager_initialized",
            max_retries=max_retries,
            base_retry_delay_ms=base_retry_delay_ms,
        )
    
    def set_dependencies(self, exchange_connector: Any, event_bus: Any) -> None:
        """Inject dependencies."""
        self.exchange_connector = exchange_connector
        self.event_bus = event_bus
    
    async def start(self) -> None:
        """Start order manager."""
        if self._running:
            log.warning("order_manager_already_running")
            return
        
        self._running = True
        
        # Start timeout monitoring
        self._timeout_check_task = asyncio.create_task(self._timeout_check_loop())
        
        log.info("order_manager_started")
    
    async def stop(self) -> None:
        """Stop order manager."""
        if not self._running:
            return
        
        self._running = False
        
        if self._timeout_check_task:
            self._timeout_check_task.cancel()
            try:
                await self._timeout_check_task
            except asyncio.CancelledError:
                pass
        
        log.info("order_manager_stopped")
    
    def create_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        timeout_seconds: float = 300.0,
        **metadata: Any,
    ) -> Order:
        """
        Create a new order.
        
        Args:
            exchange: Exchange identifier
            symbol: Trading symbol
            side: "buy" or "sell"
            order_type: Order type
            quantity: Order quantity
            price: Order price (None for market orders)
            time_in_force: Time in force
            client_order_id: Optional client order ID (generated if not provided)
            strategy_id: Optional strategy identifier
            timeout_seconds: Order timeout in seconds
            **metadata: Additional metadata
            
        Returns:
            Created order
        """
        if client_order_id is None:
            client_order_id = self._generate_order_id()
        
        order = Order(
            client_order_id=client_order_id,
            exchange=exchange,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            strategy_id=strategy_id,
            timeout_seconds=timeout_seconds,
            metadata=metadata,
        )
        
        self._orders[client_order_id] = order
        
        log.info(
            "order_created",
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            quantity=str(quantity),
        )
        
        return order
    
    async def submit_order(self, order: Order) -> bool:
        """
        Submit an order to the exchange with retry logic.
        
        Args:
            order: Order to submit
            
        Returns:
            True if order was successfully submitted
        """
        if order.state != OrderState.CREATED:
            log.warning(
                "order_already_submitted",
                client_order_id=order.client_order_id,
                state=order.state.value,
            )
            return False
        
        # Transition to PENDING
        order.transition(OrderState.PENDING, "submitting_to_exchange")
        
        # Attempt submission with retry
        for attempt in range(self.max_retries + 1):
            try:
                # Submit to exchange
                result = await self._submit_to_exchange(order)
                
                # Update order with exchange ID
                order.exchange_order_id = result.get("order_id")
                
                # Transition to ACKNOWLEDGED
                order.transition(OrderState.ACKNOWLEDGED, "acknowledged_by_exchange")
                
                # Emit event
                await self._emit_order_event(order, "order_acknowledged")
                
                log.info(
                    "order_submitted_successfully",
                    client_order_id=order.client_order_id,
                    exchange_order_id=order.exchange_order_id,
                )
                
                return True
            
            except Exception as e:
                order.retry_count = attempt + 1
                order.last_error = str(e)
                
                log.warning(
                    "order_submission_failed",
                    client_order_id=order.client_order_id,
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )
                
                # Check if order was actually placed despite error
                if await self._check_order_exists(order):
                    order.transition(OrderState.ACKNOWLEDGED, "found_after_network_error")
                    await self._emit_order_event(order, "order_acknowledged")
                    return True
                
                # If max retries reached, mark as failed
                if attempt >= self.max_retries:
                    order.transition(OrderState.FAILED, f"max_retries_exceeded: {str(e)}")
                    await self._emit_order_event(order, "order_failed")
                    return False
                
                # Exponential backoff: 2^n * base_delay
                delay_ms = min(
                    (2 ** attempt) * self.base_retry_delay_ms,
                    self.max_retry_delay_seconds * 1000,
                )
                await asyncio.sleep(delay_ms / 1000.0)
        
        return False
    
    async def cancel_order(self, client_order_id: str, reason: str = "") -> bool:
        """
        Cancel an order.
        
        Args:
            client_order_id: Order to cancel
            reason: Cancellation reason
            
        Returns:
            True if cancellation was successful
        """
        order = self._orders.get(client_order_id)
        if not order:
            log.warning("cancel_order_not_found", client_order_id=client_order_id)
            return False
        
        if order.is_complete:
            log.warning(
                "cancel_order_already_complete",
                client_order_id=client_order_id,
                state=order.state.value,
            )
            return False
        
        try:
            # Transition to CANCELING
            order.transition(OrderState.CANCELING, reason or "user_requested")
            
            # Cancel on exchange
            await self._cancel_on_exchange(order)
            
            # Transition to CANCELED
            order.transition(OrderState.CANCELED, "canceled_by_exchange")
            
            # Emit event
            await self._emit_order_event(order, "order_canceled")
            
            log.info("order_canceled", client_order_id=client_order_id)
            return True
        
        except Exception as e:
            log.error(
                "cancel_order_failed",
                client_order_id=client_order_id,
                error=str(e),
            )
            return False
    
    async def update_order_fill(
        self,
        client_order_id: str,
        filled_quantity: Decimal,
        fill_price: Decimal,
    ) -> None:
        """
        Update order with fill information.
        
        Args:
            client_order_id: Order to update
            filled_quantity: Total filled quantity
            fill_price: Average fill price
        """
        order = self._orders.get(client_order_id)
        if not order:
            log.warning("update_fill_order_not_found", client_order_id=client_order_id)
            return
        
        order.filled_quantity = filled_quantity
        order.average_fill_price = fill_price
        order.updated_at = time.time()
        
        # Check if fully filled
        if order.filled_quantity >= order.quantity:
            if order.state != OrderState.FILLED:
                order.transition(OrderState.FILLED, "fully_filled")
                await self._emit_order_event(order, "order_filled")
        else:
            # Partially filled
            if order.state == OrderState.ACKNOWLEDGED:
                order.transition(OrderState.PARTIALLY_FILLED, "partial_fill")
                await self._emit_order_event(order, "order_partially_filled")
    
    def get_order(self, client_order_id: str) -> Optional[Order]:
        """Get order by client order ID."""
        return self._orders.get(client_order_id)
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """Get all open orders."""
        return [
            order.to_dict()
            for order in self._orders.values()
            if order.is_open
        ]
    
    def get_completed_orders(self) -> List[Dict[str, Any]]:
        """Get all completed orders."""
        return [
            order.to_dict()
            for order in self._orders.values()
            if order.is_complete
        ]
    
    async def handle_partial_fill_cancel_replace(
        self,
        client_order_id: str,
        new_price: Decimal,
    ) -> Optional[Order]:
        """
        Cancel and replace a partially filled order.
        
        Args:
            client_order_id: Original order to cancel
            new_price: New price for replacement order
            
        Returns:
            New replacement order if successful
        """
        original_order = self._orders.get(client_order_id)
        if not original_order:
            return None
        
        # Cancel original
        await self.cancel_order(client_order_id, "cancel_replace")
        
        # Create replacement for remaining quantity
        new_order = self.create_order(
            exchange=original_order.exchange,
            symbol=original_order.symbol,
            side=original_order.side,
            order_type=original_order.order_type,
            quantity=original_order.remaining_quantity,
            price=new_price,
            time_in_force=original_order.time_in_force,
            strategy_id=original_order.strategy_id,
            metadata={
                **original_order.metadata,
                "replaced_order": client_order_id,
            },
        )
        
        # Submit new order
        await self.submit_order(new_order)
        
        return new_order
    
    async def _timeout_check_loop(self) -> None:
        """Background task to check for timed out orders."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Find timed out orders
                timed_out = [
                    order for order in self._orders.values()
                    if order.is_timed_out
                ]
                
                for order in timed_out:
                    log.warning(
                        "order_timeout_detected",
                        client_order_id=order.client_order_id,
                        age_seconds=time.time() - order.created_at,
                    )
                    
                    # Cancel timed out order
                    await self.cancel_order(order.client_order_id, "timeout")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("timeout_check_error", error=str(e))
    
    def _generate_order_id(self) -> str:
        """Generate a unique client order ID."""
        return f"order_{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
    
    async def _submit_to_exchange(self, order: Order) -> Dict[str, Any]:
        """Submit order to exchange."""
        if not self.exchange_connector:
            raise RuntimeError("Exchange connector not set")
        
        # Placeholder - actual implementation depends on exchange connector API
        if hasattr(self.exchange_connector, "create_order"):
            return await self.exchange_connector.create_order(
                symbol=order.symbol,
                order_type=order.order_type.value,
                side=order.side,
                quantity=float(order.quantity),
                price=float(order.price) if order.price else None,
                client_order_id=order.client_order_id,
            )
        
        raise NotImplementedError("Exchange connector does not support create_order")
    
    async def _cancel_on_exchange(self, order: Order) -> None:
        """Cancel order on exchange."""
        if not self.exchange_connector:
            raise RuntimeError("Exchange connector not set")
        
        if hasattr(self.exchange_connector, "cancel_order"):
            await self.exchange_connector.cancel_order(
                order_id=order.exchange_order_id or order.client_order_id,
                symbol=order.symbol,
            )
    
    async def _check_order_exists(self, order: Order) -> bool:
        """Check if order exists on exchange (for network failure recovery)."""
        if not self.exchange_connector:
            return False
        
        try:
            if hasattr(self.exchange_connector, "fetch_order"):
                result = await self.exchange_connector.fetch_order(
                    order_id=order.client_order_id,
                    symbol=order.symbol,
                )
                return result is not None
        except Exception:
            pass
        
        return False
    
    async def _emit_order_event(self, order: Order, event_type: str) -> None:
        """Emit order event to event bus."""
        if not self.event_bus:
            return
        
        try:
            await self.event_bus.publish(
                f"orders.{event_type}",
                {
                    "client_order_id": order.client_order_id,
                    "exchange_order_id": order.exchange_order_id,
                    "symbol": order.symbol,
                    "state": order.state.value,
                    "timestamp": int(time.time() * 1_000_000),
                },
            )
        except Exception as e:
            log.error("emit_order_event_failed", event_type=event_type, error=str(e))
    
    # Methods for reconciliation support
    
    async def mark_order_closed(self, client_order_id: str, reason: str) -> None:
        """Mark a ghost order as closed (called by reconciliation)."""
        order = self._orders.get(client_order_id)
        if not order:
            return
        
        if order.is_complete:
            return
        
        try:
            order.transition(OrderState.CANCELED, f"reconciliation: {reason}")
            await self._emit_order_event(order, "order_canceled")
        except InvalidStateTransition:
            # Force state if transition not valid
            order.state = OrderState.CANCELED
            order.updated_at = time.time()
    
    async def add_orphan_order(self, order_data: Dict[str, Any]) -> None:
        """Add an orphan order found during reconciliation."""
        client_order_id = order_data.get("client_order_id")
        if not client_order_id or client_order_id in self._orders:
            return
        
        # Create order from exchange data
        order = Order(
            client_order_id=client_order_id,
            exchange=order_data.get("exchange", "unknown"),
            symbol=order_data.get("symbol", ""),
            side=order_data.get("side", "buy"),
            order_type=OrderType.LIMIT,  # Assume limit
            quantity=Decimal(str(order_data.get("quantity", 0))),
            price=Decimal(str(order_data.get("price", 0))),
            state=OrderState.ACKNOWLEDGED,
            exchange_order_id=order_data.get("order_id"),
        )
        
        self._orders[client_order_id] = order
        
        log.info("orphan_order_added", client_order_id=client_order_id)
    
    async def sync_order_state(self, client_order_id: str, exchange_state: str) -> None:
        """Sync order state with exchange (called by reconciliation)."""
        order = self._orders.get(client_order_id)
        if not order:
            return
        
        # Map exchange state to internal state
        state_mapping = {
            "open": OrderState.ACKNOWLEDGED,
            "closed": OrderState.FILLED,
            "canceled": OrderState.CANCELED,
            "cancelled": OrderState.CANCELED,
            "rejected": OrderState.REJECTED,
            "expired": OrderState.EXPIRED,
        }
        
        target_state = state_mapping.get(exchange_state.lower())
        if not target_state or target_state == order.state:
            return
        
        try:
            order.transition(target_state, f"reconciliation: synced from exchange")
            await self._emit_order_event(order, f"order_{target_state.value}")
        except InvalidStateTransition:
            # Force state if needed
            order.state = target_state
            order.updated_at = time.time()
