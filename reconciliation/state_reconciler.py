"""
State Reconciliation Engine - The "Silent Killer Solution".

Prevents state drift between local cache and exchange reality through
periodic REST API polling and verification.

Features:
- Balance verification (every 30s by default)
- Order status synchronization (every 10s)
- Position reconciliation for derivatives
- Ghost order detection (local thinks open, exchange says closed)
- Orphan order detection (exchange has orders we don't know about)
- Automatic discrepancy correction
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

import structlog

log = structlog.get_logger()


@dataclass
class BalanceCorrectionEvent:
    """Event emitted when balance discrepancy detected."""
    exchange: str
    currency: str
    local_balance: Decimal
    exchange_balance: Decimal
    discrepancy: Decimal
    timestamp: int = field(default_factory=lambda: int(time.time() * 1_000_000))
    corrected: bool = False


@dataclass
class OrderDiscrepancy:
    """Order state mismatch between local and exchange."""
    client_order_id: str
    exchange_order_id: Optional[str]
    local_state: str
    exchange_state: str
    discrepancy_type: str  # "ghost", "orphan", "status_mismatch"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1_000_000))


class StateReconciler:
    """
    Reconciles local state with exchange reality.
    
    Three-phase reconciliation:
    1. Balances - verify account balances match
    2. Orders - check order status sync
    3. Positions - verify derivatives positions (futures/perps)
    """
    
    def __init__(
        self,
        sync_interval_seconds: int = 30,
        order_sync_interval_seconds: int = 10,
    ) -> None:
        """
        Initialize state reconciler.
        
        Args:
            sync_interval_seconds: Interval for balance/position sync
            order_sync_interval_seconds: Interval for order status sync
        """
        self.sync_interval = sync_interval_seconds
        self.order_sync_interval = order_sync_interval_seconds
        self.discrepancy_count = 0
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Dependencies (to be injected)
        self.exchange_connector = None  # Set externally
        self.order_manager = None  # Set externally
        self.event_bus = None  # Set externally
        
        log.info(
            "reconciler_initialized",
            sync_interval=sync_interval_seconds,
            order_sync_interval=order_sync_interval_seconds,
        )
    
    def set_dependencies(
        self,
        exchange_connector: Any,
        order_manager: Any,
        event_bus: Any,
    ) -> None:
        """Inject dependencies."""
        self.exchange_connector = exchange_connector
        self.order_manager = order_manager
        self.event_bus = event_bus
    
    async def start(self) -> None:
        """Start reconciliation loops."""
        if self._running:
            log.warning("reconciler_already_running")
            return
        
        self._running = True
        
        # Start balance/position reconciliation loop
        self._tasks.append(
            asyncio.create_task(self._balance_reconciliation_loop())
        )
        
        # Start order reconciliation loop
        self._tasks.append(
            asyncio.create_task(self._order_reconciliation_loop())
        )
        
        log.info("reconciler_started")
    
    async def stop(self) -> None:
        """Stop reconciliation loops."""
        if not self._running:
            return
        
        self._running = False
        
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        log.info("reconciler_stopped")
    
    async def _balance_reconciliation_loop(self) -> None:
        """Background task for balance reconciliation."""
        while self._running:
            try:
                await asyncio.sleep(self.sync_interval)
                await self.reconcile_balances()
                await self.reconcile_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("balance_reconciliation_error", error=str(e))
    
    async def _order_reconciliation_loop(self) -> None:
        """Background task for order reconciliation."""
        while self._running:
            try:
                await asyncio.sleep(self.order_sync_interval)
                await self.reconcile_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("order_reconciliation_error", error=str(e))
    
    async def reconcile_state(self) -> None:
        """
        Run full three-phase reconciliation.
        
        This can be called manually for on-demand reconciliation.
        """
        await self.reconcile_balances()
        await self.reconcile_orders()
        await self.reconcile_positions()
    
    async def reconcile_balances(self) -> List[BalanceCorrectionEvent]:
        """
        Verify local balances match exchange balances.
        
        Returns:
            List of correction events for any discrepancies found
        """
        if not self.exchange_connector:
            log.warning("reconcile_balances_skipped", reason="no_exchange_connector")
            return []
        
        corrections: List[BalanceCorrectionEvent] = []
        
        try:
            # Fetch balances from exchange via REST
            exchange_balances = await self._fetch_exchange_balances()
            
            # Get local cached balances
            local_balances = self._get_local_balances()
            
            # Compare each currency
            all_currencies = set(exchange_balances.keys()) | set(local_balances.keys())
            
            for currency in all_currencies:
                local_bal = local_balances.get(currency, Decimal("0"))
                exchange_bal = exchange_balances.get(currency, Decimal("0"))
                
                # Check for significant discrepancy (> 0.0001 or 0.01%)
                diff = abs(exchange_bal - local_bal)
                threshold = max(Decimal("0.0001"), exchange_bal * Decimal("0.0001"))
                
                if diff > threshold:
                    correction = BalanceCorrectionEvent(
                        exchange=self.exchange_connector.exchange_name,
                        currency=currency,
                        local_balance=local_bal,
                        exchange_balance=exchange_bal,
                        discrepancy=exchange_bal - local_bal,
                    )
                    
                    corrections.append(correction)
                    self.discrepancy_count += 1
                    
                    log.warning(
                        "balance_discrepancy_detected",
                        currency=currency,
                        local=str(local_bal),
                        exchange=str(exchange_bal),
                        discrepancy=str(diff),
                    )
                    
                    # Emit event
                    if self.event_bus:
                        await self.event_bus.publish(
                            "reconciliation.balance_correction",
                            {
                                "exchange": correction.exchange,
                                "currency": correction.currency,
                                "local_balance": str(correction.local_balance),
                                "exchange_balance": str(correction.exchange_balance),
                                "discrepancy": str(correction.discrepancy),
                                "timestamp": correction.timestamp,
                            },
                        )
                    
                    # Correct local state
                    await self._correct_local_balance(currency, exchange_bal)
                    correction.corrected = True
        
        except Exception as e:
            log.error("balance_reconciliation_failed", error=str(e))
        
        if corrections:
            log.info("balance_reconciliation_complete", corrections=len(corrections))
        
        return corrections
    
    async def reconcile_orders(self) -> List[OrderDiscrepancy]:
        """
        Verify local order states match exchange.
        
        Detects:
        - Ghost orders (local thinks open, exchange says closed)
        - Orphan orders (exchange has order, local doesn't know)
        - Status mismatches (different states)
        
        Returns:
            List of discrepancies found
        """
        if not self.exchange_connector or not self.order_manager:
            log.warning("reconcile_orders_skipped", reason="missing_dependencies")
            return []
        
        discrepancies: List[OrderDiscrepancy] = []
        
        try:
            # Fetch open orders from exchange
            exchange_orders = await self._fetch_exchange_orders()
            
            # Get local open orders
            local_orders = self._get_local_open_orders()
            
            # Build maps for comparison
            local_by_id = {o["client_order_id"]: o for o in local_orders}
            exchange_by_id = {o["client_order_id"]: o for o in exchange_orders if "client_order_id" in o}
            
            # Check for ghost orders (local has, exchange doesn't)
            for order_id, local_order in local_by_id.items():
                if order_id not in exchange_by_id:
                    discrepancy = OrderDiscrepancy(
                        client_order_id=order_id,
                        exchange_order_id=local_order.get("exchange_order_id"),
                        local_state=local_order["state"],
                        exchange_state="not_found",
                        discrepancy_type="ghost",
                    )
                    discrepancies.append(discrepancy)
                    self.discrepancy_count += 1
                    
                    log.warning(
                        "ghost_order_detected",
                        client_order_id=order_id,
                        local_state=local_order["state"],
                    )
                    
                    # Mark as closed locally
                    await self._mark_order_closed(order_id, "ghost_order_detected")
            
            # Check for orphan orders (exchange has, local doesn't)
            for order_id, exchange_order in exchange_by_id.items():
                if order_id not in local_by_id:
                    discrepancy = OrderDiscrepancy(
                        client_order_id=order_id,
                        exchange_order_id=exchange_order.get("exchange_order_id"),
                        local_state="not_found",
                        exchange_state=exchange_order.get("status", "unknown"),
                        discrepancy_type="orphan",
                    )
                    discrepancies.append(discrepancy)
                    self.discrepancy_count += 1
                    
                    log.warning(
                        "orphan_order_detected",
                        client_order_id=order_id,
                        exchange_state=exchange_order.get("status"),
                    )
                    
                    # Add to local tracking
                    await self._add_orphan_order(exchange_order)
            
            # Check for status mismatches
            for order_id in set(local_by_id.keys()) & set(exchange_by_id.keys()):
                local_state = local_by_id[order_id]["state"]
                exchange_state = exchange_by_id[order_id].get("status", "unknown")
                
                if not self._states_match(local_state, exchange_state):
                    discrepancy = OrderDiscrepancy(
                        client_order_id=order_id,
                        exchange_order_id=local_by_id[order_id].get("exchange_order_id"),
                        local_state=local_state,
                        exchange_state=exchange_state,
                        discrepancy_type="status_mismatch",
                    )
                    discrepancies.append(discrepancy)
                    self.discrepancy_count += 1
                    
                    log.warning(
                        "order_status_mismatch",
                        client_order_id=order_id,
                        local_state=local_state,
                        exchange_state=exchange_state,
                    )
                    
                    # Update local state
                    await self._sync_order_state(order_id, exchange_state)
        
        except Exception as e:
            log.error("order_reconciliation_failed", error=str(e))
        
        if discrepancies:
            log.info("order_reconciliation_complete", discrepancies=len(discrepancies))
        
        return discrepancies
    
    async def reconcile_positions(self) -> List[Dict[str, Any]]:
        """
        Verify derivatives positions match exchange.
        
        Returns:
            List of position corrections made
        """
        if not self.exchange_connector:
            log.warning("reconcile_positions_skipped", reason="no_exchange_connector")
            return []
        
        corrections: List[Dict[str, Any]] = []
        
        try:
            # Check if exchange supports positions (derivatives)
            if not hasattr(self.exchange_connector, "fetch_positions"):
                return corrections
            
            # Fetch positions from exchange
            exchange_positions = await self._fetch_exchange_positions()
            
            # Get local positions
            local_positions = self._get_local_positions()
            
            # Compare positions
            all_symbols = set(exchange_positions.keys()) | set(local_positions.keys())
            
            for symbol in all_symbols:
                local_pos = local_positions.get(symbol, Decimal("0"))
                exchange_pos = exchange_positions.get(symbol, Decimal("0"))
                
                if local_pos != exchange_pos:
                    correction = {
                        "symbol": symbol,
                        "local_position": str(local_pos),
                        "exchange_position": str(exchange_pos),
                        "discrepancy": str(exchange_pos - local_pos),
                    }
                    corrections.append(correction)
                    self.discrepancy_count += 1
                    
                    log.warning(
                        "position_discrepancy",
                        symbol=symbol,
                        local=str(local_pos),
                        exchange=str(exchange_pos),
                    )
                    
                    # Correct local position
                    await self._correct_local_position(symbol, exchange_pos)
        
        except Exception as e:
            log.error("position_reconciliation_failed", error=str(e))
        
        if corrections:
            log.info("position_reconciliation_complete", corrections=len(corrections))
        
        return corrections
    
    # Helper methods (to be implemented based on actual connector/manager APIs)
    
    async def _fetch_exchange_balances(self) -> Dict[str, Decimal]:
        """Fetch balances from exchange REST API."""
        # Placeholder - actual implementation depends on exchange connector
        if hasattr(self.exchange_connector, "fetch_balance"):
            balances = await self.exchange_connector.fetch_balance()
            return {k: Decimal(str(v)) for k, v in balances.items()}
        return {}
    
    def _get_local_balances(self) -> Dict[str, Decimal]:
        """Get local cached balances."""
        # Placeholder - actual implementation depends on state storage
        return {}
    
    async def _correct_local_balance(self, currency: str, correct_balance: Decimal) -> None:
        """Update local balance to match exchange."""
        # Placeholder
        pass
    
    async def _fetch_exchange_orders(self) -> List[Dict[str, Any]]:
        """Fetch open orders from exchange REST API."""
        if hasattr(self.exchange_connector, "fetch_open_orders"):
            return await self.exchange_connector.fetch_open_orders()
        return []
    
    def _get_local_open_orders(self) -> List[Dict[str, Any]]:
        """Get local open orders."""
        if self.order_manager and hasattr(self.order_manager, "get_open_orders"):
            return self.order_manager.get_open_orders()
        return []
    
    async def _mark_order_closed(self, client_order_id: str, reason: str) -> None:
        """Mark a ghost order as closed."""
        if self.order_manager and hasattr(self.order_manager, "mark_order_closed"):
            await self.order_manager.mark_order_closed(client_order_id, reason)
    
    async def _add_orphan_order(self, order: Dict[str, Any]) -> None:
        """Add an orphan order to local tracking."""
        if self.order_manager and hasattr(self.order_manager, "add_orphan_order"):
            await self.order_manager.add_orphan_order(order)
    
    def _states_match(self, local_state: str, exchange_state: str) -> bool:
        """Check if local and exchange states are equivalent."""
        # Map exchange states to our internal states
        state_mapping = {
            "open": "acknowledged",
            "closed": "filled",
            "canceled": "canceled",
            "cancelled": "canceled",
            "rejected": "rejected",
            "expired": "expired",
        }
        
        normalized_exchange = state_mapping.get(exchange_state.lower(), exchange_state.lower())
        return local_state.lower() == normalized_exchange
    
    async def _sync_order_state(self, client_order_id: str, exchange_state: str) -> None:
        """Update local order state to match exchange."""
        if self.order_manager and hasattr(self.order_manager, "sync_order_state"):
            await self.order_manager.sync_order_state(client_order_id, exchange_state)
    
    async def _fetch_exchange_positions(self) -> Dict[str, Decimal]:
        """Fetch positions from exchange."""
        if hasattr(self.exchange_connector, "fetch_positions"):
            positions = await self.exchange_connector.fetch_positions()
            return {p["symbol"]: Decimal(str(p.get("contracts", 0))) for p in positions}
        return {}
    
    def _get_local_positions(self) -> Dict[str, Decimal]:
        """Get local positions."""
        # Placeholder
        return {}
    
    async def _correct_local_position(self, symbol: str, correct_position: Decimal) -> None:
        """Update local position to match exchange."""
        # Placeholder
        pass
