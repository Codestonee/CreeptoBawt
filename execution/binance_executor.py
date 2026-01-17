import logging
import os
import asyncio

import math
from typing import Optional, Dict

# Binance Imports
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceAPIException

# System Imports
from config.settings import settings
from core.events import SignalEvent, FillEvent
from core.event_store import get_event_store, EventType
from execution.order_manager import OrderManager, OrderState  # Class import, not singleton
from execution.reconciliation import ReconciliationService
from execution.smart_router import DeterministicOrderRouter
from utils.nonce_service import get_nonce_service
from utils.time_sync import get_time_sync_service
from database.db_manager import DatabaseManager
from execution.risk_gatekeeper import RiskGatekeeper
from execution.position_tracker import PositionTracker

logger = logging.getLogger("Execution.Binance")


class BinanceExecutionHandler:
    """
    Production-grade Binance Futures execution handler.
    
    Features:
    - OrderManager integration for state tracking
    - EventStore logging for crash recovery
    - NonceService for idempotent orders
    - TimeSync validation before trading
    - Proper recvWindow from time sync (not hardcoded)
    """
    
    def __init__(self, event_queue, risk_manager, testnet: bool = True):
        self.queue = event_queue
        self.risk_manager = risk_manager
        self.testnet = testnet
        self.spot_mode = settings.SPOT_MODE  # NEW: Spot trading mode for EU users
        self.client: Optional[AsyncClient] = None
        self.bsm: Optional[BinanceSocketManager] = None
        
        # Production services
        # Production services
        self.db = DatabaseManager()  # Shared DB Manager
        
        # Components to be initialized in connect()
        self.order_manager: Optional[OrderManager] = None
        self.gatekeeper: Optional[RiskGatekeeper] = None
        self.position_tracker: Optional[PositionTracker] = None
        
        self.event_store = get_event_store()
        self.nonce_service = get_nonce_service()
        self.time_sync = get_time_sync_service(testnet=testnet)
        
        # Track client_order_id -> trace_id mapping for fills
        self._order_trace_map: Dict[str, str] = {}
        
        # Symbol filters cache: symbol -> (tick_size, step_size)
        self.exchange_filters: Dict[str, tuple] = {}
        
        # Reconciliation service (initialized in connect() with API keys)
        self.reconciliation: Optional[ReconciliationService] = None
        
        # Deterministic Order Router for limit chasing
        # Deterministic Order Router for limit chasing
        self.order_router = DeterministicOrderRouter()
        
        # API credentials - USE SETTINGS MODULE (loads from .env via dotenv)
        if self.testnet:
            self.api_key = settings.BINANCE_TESTNET_API_KEY
            self.api_secret = settings.BINANCE_TESTNET_SECRET_KEY
        else:
            self.api_key = settings.BINANCE_API_KEY
            self.api_secret = settings.BINANCE_API_SECRET
            
        if not self.api_key or not self.api_secret:
            logger.critical(f"❌ MISSING API KEYS in .env file! (Testnet={self.testnet})")
    
    async def connect(self):
        """Connect to Binance REST API and start user data stream."""
        try:
            # 1. Sync time FIRST
            logger.info("⏱️ Syncing time with exchange...")
            await self.time_sync.start_periodic_sync()
            
            if not self.time_sync.is_synced:
                logger.critical("❌ Time sync failed - trading disabled")
                return
            
            # 2. Connect REST API
            self.client = await AsyncClient.create(
                self.api_key,
                self.api_secret,
                testnet=self.testnet
            )
            
            # Apply calculated offset to client to prevent -1021 errors
            self.client.timestamp_offset = self.time_sync.offset_ms
            logger.info(f"✅ Connected to Binance REST API (Offset: {self.time_sync.offset_ms}ms)")
            
            # --- COMPONENT INITIALIZATION (CRITICAL ORDER) ---
            
            # 1. Position Tracker (Single Source of Truth)
            self.position_tracker = PositionTracker(self.client, self.db)
            
            # 2. Risk Gatekeeper (Needs Tracker + Client)
            self.gatekeeper = RiskGatekeeper(
                position_tracker=self.position_tracker,
                exchange_client=self.client
            )
            
            # 3. Order Manager (The Orchestrator)
            self.order_manager = OrderManager(
                exchange_client=self.client,
                db_manager=self.db,
                position_tracker=self.position_tracker,
                risk_gatekeeper=self.gatekeeper
            )
            
            # --- STARTUP SYNCHRONIZATION ---
            # This is the "Fix Priority 1" - Sync Positions before trading
            logger.info("🔒 performing CRITICAL startup position sync...")
            sync_success = await self.position_tracker.initialize()
            
            if not sync_success:
                error_msg = "❌ STARTUP FAILED: Could not sync positions with exchange. Trading aborted."
                logger.critical(error_msg)
                raise RuntimeError(error_msg)
                
            await self.order_manager.initialize()
            
            # --- L4 MARGIN PROTECTION (CRITICAL) ---
            # Skip for spot mode (no margin in spot trading)
            if not self.spot_mode:
                try:
                    account_info = await self.client.futures_account()
                    total_margin_balance = float(account_info.get('totalMarginBalance', 0))
                    total_wallet_balance = float(account_info.get('totalWalletBalance', 0))
                    
                    if total_wallet_balance > 0:
                        margin_utilization = 1 - (total_margin_balance / total_wallet_balance)
                        self.risk_manager.circuit_breaker.update_margin_utilization(margin_utilization)
                        self.risk_manager.circuit_breaker.update_equity(total_wallet_balance)
                        logger.info(f"🛡️ Margin utilization: {margin_utilization*100:.1f}%")
                except Exception as e:
                    logger.warning(f"Could not check margin utilization: {e}")
            else:
                logger.info(f"💱 SPOT MODE enabled - no margin tracking")
            # -------------------------------
            
            # 2.1 Fetch Exchange Info (Dynamic Precision)
            await self._fetch_exchange_info()
            
            # 3. Start User Data Stream
            self.bsm = BinanceSocketManager(self.client)
            self._user_stream_task = asyncio.create_task(self._user_data_stream())
            
            # 4. Log system start event
            await self.event_store.append(
                EventType.SYSTEM_START,
                payload={
                    "testnet": self.testnet,
                    "time_offset_ms": self.time_sync.offset_ms
                },
                trace_id=self.event_store.generate_trace_id()
            )
            
            # 5. Start Reconciliation Service (Legacy/Redundant but kept for safety)
            # functionality mostly moved to PositionTracker, but keeping for orders?
            self.reconciliation = ReconciliationService(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet,
                spot_mode=self.spot_mode
            )
            # Link new order manager to reconciliation if needed, or disable it
            # For now, we rely on PositionTracker for positions.
            
            logger.info("✅ Binance Execution Handler READY")
            
        except Exception as e:
            logger.error(f"Failed to connect to Binance: {e}")
    
    async def _user_data_stream(self):
        """Listen for order updates from Binance user stream."""
        logger.info("🎧 Starting User Data Stream (Waiting for fills)...")
        
        while self.client: # Check if client exists (proxy for running)
            try:
                # 1. Get Listen Key explicitly (Raw Request to bypass Lib 404 bug)
                if self.spot_mode:
                    import aiohttp
                    url = "https://api.binance.com/api/v3/userDataStream"
                    headers = {"X-MBX-APIKEY": self.api_key}
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                listen_key = data['listenKey']
                            else:
                                text = await resp.text()
                                logger.error(f"Raw Listen Key Failed: {resp.status} {text}")
                                raise Exception(f"Could not get listen key: {resp.status}")
                else:
                    listen_key = await self.client.futures_stream_get_listen_key()
                
                logger.info(f"🔑 Listen Key obtained: {listen_key[:6]}...")
                
                # 2. Create Socket
                ts = self.bsm.multiplex_socket([listen_key])
                
                # 3. Keep-alive task (30 mins)
                # self.client.stream_keepalive(listen_key) - managed by lib or loop?
                # We'll rely on reconnection for now or add a keepalive
                
                async with ts as tscm:
                    while True:
                        res = await tscm.recv()
                        
                        if res is None:
                            continue
                        
                        # Handle Spot event
                        if self.spot_mode:
                            if res.get('e') == 'executionReport':
                                await self._handle_order_update(res)
                            elif res.get('e') == 'outboundAccountPosition':
                                # Optional: Update positions from stream
                                pass
                        
                        # Handle Futures event
                        else:
                            if res.get('e') == 'ORDER_TRADE_UPDATE':
                                await self._handle_order_update(res['o'])
                            
                        # Forward to reconciliation
                        if self.reconciliation:
                            # Adapt event for reconciliation
                            await self.reconciliation.on_user_stream_event(res)

            except asyncio.CancelledError:
                logger.info("User Stream cancelled.")
                break
            except Exception as e:
                logger.error(f"User Stream Error: {e}")
                await asyncio.sleep(5)
    
    # ... (skipping _handle_order_update) ...


    
    async def _handle_order_update(self, data: dict):
        """Process order update from exchange with OrderManager."""
        status = data['X']       # Order Status (NEW, FILLED, CANCELED...)
        exec_type = data['x']    # Execution Type (TRADE = Fill)
        client_order_id = data.get('c', '')  # Client Order ID
        exchange_order_id = str(data.get('i', ''))
        
        symbol = data['s'].lower()
        side = data['S'].upper()
        
        # Handle NEW orders
        if status == 'NEW' and exec_type == 'NEW':
            try:
                await self.order_manager.mark_submitted(
                    client_order_id=client_order_id,
                    exchange_order_id=exchange_order_id
                )
                logger.debug(f"✓ Confirmed: {client_order_id[:8]}...")
            except Exception as e:
                logger.warning(f"Could not update order state for {client_order_id}: {e}")
        
        # Handle FILLS
        elif exec_type == 'TRADE' and status in ['FILLED', 'PARTIALLY_FILLED']:
            last_filled_qty = float(data['l'])
            last_filled_price = float(data['L'])
            commission = float(data.get('n', 0))
            commission_asset = data.get('N', 'USDT')
            is_maker = data.get('m', False) # m=True means Maker side
            
            logger.info(f"⚡ FILL: {side} {last_filled_qty} {symbol} @ {last_filled_price}")
            
            try:
                # Get position BEFORE fill to calculate PnL
                pos_before = await self.order_manager.get_position(symbol)
                
                # Update OrderManager (handles position atomically)
                order = await self.order_manager.process_fill(
                    client_order_id=client_order_id,
                    filled_qty=last_filled_qty,
                    fill_price=last_filled_price,
                    commission=commission
                )
                
                # Calculate realized PnL
                # PnL is realized when closing/reducing a position
                realized_pnl = 0.0
                if pos_before.quantity != 0 and pos_before.avg_entry_price > 0:
                    # Check if this trade reduces the position
                    is_reducing = (
                        (pos_before.quantity > 0 and side == 'SELL') or
                        (pos_before.quantity < 0 and side == 'BUY')
                    )
                    if is_reducing:
                        # PnL = (exit_price - entry_price) * qty for longs
                        # PnL = (entry_price - exit_price) * qty for shorts
                        if pos_before.quantity > 0:  # Long position being reduced
                            realized_pnl = (last_filled_price - pos_before.avg_entry_price) * last_filled_qty
                        else:  # Short position being reduced
                            realized_pnl = (pos_before.avg_entry_price - last_filled_price) * last_filled_qty
                        realized_pnl -= commission  # Subtract commission
                
                # Create FillEvent for strategies
                fill_event = FillEvent(
                    symbol=symbol,
                    side=side,
                    quantity=last_filled_qty,
                    price=last_filled_price,
                    commission=commission,
                    commission_asset=commission_asset,
                    is_maker=is_maker,
                    pnl=realized_pnl
                )
                
                # Log to legacy DB for dashboard compatibility
                await self.db.log_trade(fill_event, strategy_id="binance_live")
                
                # Notify strategies
                await self.queue.put(fill_event)
                
                if realized_pnl != 0:
                    logger.info(f"💰 Realized PnL: ${realized_pnl:.4f}")
                
                # Update Risk Gatekeeper (Daily Loss Tracking)
                self.gatekeeper.update_pnl(realized_pnl)
                
                logger.info(f"💾 Fill processed. State: {order.state}")
                
                 # Notify router of fill (if router is active)
                if hasattr(self, 'order_router') and self.order_router:
                    self.order_router.on_fill_confirmed(
                        client_order_id=client_order_id,
                        filled_qty=last_filled_qty,
                        avg_price=last_filled_price
                    )
                
            except Exception as e:
                # With pre-registration, orphans should be RARE
                # If they still happen, it's a real bug that needs investigation
                logger.error(
                    f"❌ Fill processing failed for {client_order_id}: {e}\n"
                    f"   This should NOT happen with pre-registration enabled!\n"
                    f"   Raw data: data",
                    exc_info=True
                )
                
                # Log to event store for debugging
                if hasattr(self, 'event_store'):
                    await self.event_store.append(
                        "ORDER_REJECTED", # Use string constant if Enum not avail in scope, or EventType.ORDER_REJECTED
                        payload={
                            "client_order_id": client_order_id,
                            "error": str(e),
                            "error_type": "FILL_PROCESSING_ERROR",
                            "raw_data": str(data)
                        },
                        trace_id=self._order_trace_map.get(client_order_id, "unknown")
                    )
                else:
                    logger.error(f"❌ Fill processing failed: {e}")
                    # Log to event store for investigation
                    await self.event_store.append(
                        EventType.ORDER_REJECTED,
                        payload={
                            "client_order_id": client_order_id,
                            "error": str(e),
                            "raw_data": data
                        },
                        trace_id=self._order_trace_map.get(client_order_id, "unknown")
                    )
        
        # Handle CANCELED
        elif status == 'CANCELED':
            try:
                await self.order_manager.cancel_order(client_order_id)
                logger.info(f"🚫 Order canceled: {client_order_id}")
            except Exception as e:
                logger.debug(f"Could not mark order canceled (may be orphan): {e}")
    
    async def execute(self, signal: SignalEvent) -> Optional[str]:
        """
        Execute order with full state tracking.
        
        Returns:
            client_order_id if order created, None on failure
        """
        if not self.client:
            logger.error("Cannot execute: Client not connected.")
            return None
        
        # Check time sync
        if not self.time_sync.is_synced:
            logger.error("Cannot execute: Time not synced with exchange.")
            return None
        
        symbol = signal.symbol.upper()
        side = signal.side.upper()
        
        # HANDLE CANCELLATION SIGNAL
        if side == 'CANCEL':
            await self.order_manager.cancel_all_orders(symbol.lower())
            return None

        is_spot = self.spot_mode  # Use global setting, not signal attribute
        
        tick_size, step_size = self._get_filters(symbol)
        
        # Validate price for LIMIT orders
        if signal.price is None or signal.price <= 0:
            order_type = getattr(signal, 'order_type', 'LIMIT')
            if order_type != 'MARKET':
                logger.error(f"Invalid price for LIMIT order: {signal.price}")
                return None
        
        # Round correctly using step sizes
        quantity = self._round_step_size(signal.quantity, step_size)
        price = self._round_step_size(signal.price or 0, tick_size) if signal.price else 0
        
        # --- RISK GATEKEEPER (THE HARD STOP) ---
        # Note: If price is 0 (Market Order), the value check might be skipped/inaccurate.
        # Future improvement: Fetch mark price for Market orders.
        risk_check = await self.gatekeeper.validate_order(symbol, quantity, price, side)
        
        if not risk_check.is_allowed:
            if getattr(risk_check, 'severity', 'INFO') == 'CRITICAL':
                logger.critical(f"🛑 CRITICAL RISK REJECT: {risk_check.reason}")
            else:
                logger.error(f"🛑 RISK REJECT: {risk_check.reason}")
            return None
        # ---------------------------------------

        # 1. Create order in OrderManager (logs to EventStore)
        try:
            order = await self.order_manager.create_order(
                symbol=symbol.lower(),
                side=side,
                quantity=quantity,
                price=price,
                order_type="LIMIT",
                time_in_force="GTC"
            )
        except Exception as e:
            logger.error(f"Failed to create order record: {e}")
            return None
        
        # Track trace_id for this order
        self._order_trace_map[order.client_order_id] = order.trace_id
        
        logger.debug(f"Submitting: {side} {quantity} {symbol} @ {price}")
        
        # 2. Submit to exchange with retry logic for timeouts
        # Check if this is a MARKET order (for emergency closes)
        order_type = getattr(signal, 'order_type', 'LIMIT')
        
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second
        
        for attempt in range(max_retries):
            try:
                # --- SPOT EXECUTION ---
                if is_spot:
                    await self._execute_spot_order(
                        symbol, side, quantity, price, order_type, order.client_order_id
                    )
                    return order.client_order_id

                if order_type == 'MARKET':
                    # MARKET order - immediate fill at best price (emergency closes only)
                    logger.warning(f"⚡ MARKET ORDER: {side} {quantity} {symbol}")
                    result = await self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type='MARKET',
                        quantity=quantity,
                        newClientOrderId=order.client_order_id,
                        recvWindow=self.nonce_service.get_recv_window()
                    )
                    
                    exchange_order_id = str(result.get('orderId', ''))
                    fill_price = float(result.get('avgPrice', 0))
                    logger.info(f"✅ MARKET FILLED: {side} {quantity} {symbol} @ ${fill_price:,.2f}")
                    
                    # Log success to event store
                    await self.event_store.log_order_submitted(
                        trace_id=order.trace_id,
                        client_order_id=order.client_order_id,
                        exchange_order_id=exchange_order_id,
                        symbol=symbol.lower(),
                        side=side,
                        quantity=quantity,
                        price=fill_price
                    )
                    
                    return order.client_order_id
                    
                else:
                    # CHECK STRATEGY TYPE: 
                    # Market Making strategies (Avellaneda) need specific pricing (POST ONLY)
                    # Execution strategies (Arbitrage) need Limit Chase (Aggressive)
                    is_market_maker = hasattr(signal, 'strategy_id') and signal.strategy_id == 'avellaneda_stoikov'
                    
                    if is_market_maker:
                        # STANDARD MAKER ORDER (Post Only)
                        # Do NOT use Router (which chases BBO). Use Strategy Price.
                        logger.info(f"🛡️ MAKER ORDER: {side} {quantity} {symbol} @ ${price:,.2f}")
                        
                        try:
                            result = await self.client.futures_create_order(
                                symbol=symbol.upper(),
                                side=side.upper(),
                                type='LIMIT',
                                timeInForce='GTX',  # Post-Only (Maker or Cancel)
                                quantity=quantity,
                                price=str(price),
                                newClientOrderId=order.client_order_id,
                                recvWindow=self.nonce_service.get_recv_window()
                            )
                            
                            exchange_order_id = str(result.get('orderId', ''))
                            # Maker orders are NEW, not FILLED immediately
                            logger.info(f"✅ PLACED MAKER: {side} {quantity} {symbol} @ ${price:,.2f}")
                            
                            await self.event_store.log_order_submitted(
                                trace_id=order.trace_id,
                                client_order_id=order.client_order_id,
                                exchange_order_id=exchange_order_id,
                                symbol=symbol.lower(),
                                side=side,
                                quantity=quantity,
                                price=price
                            )
                            return order.client_order_id
                            
                        except BinanceAPIException as e:
                            # GTX Cancellation (Post Only would execute as taker)
                            if e.code == -2010 or "Order would immediately match" in e.message:
                                logger.debug(f"⚠️ GTX Reject (Price crosses book): {price}")
                                await self.order_manager.mark_rejected(
                                    order.client_order_id,
                                    error_message="GTX Reject: Would cross spread"
                                )
                                return None
                            raise e

                    else:
                        # LIMIT CHASE (Aggressive Execution)
                        # Used for Arbitrage, Sniping, or Manual Close
                        logger.info(f"🎯 LIMIT CHASE: {side} {quantity} {symbol}")
                        
                        router_result = await self.order_router.fill_order(
                            side=side,
                            quantity=quantity,
                            symbol=symbol,
                            get_best_bid_ask_fn=self._get_best_bid_ask,
                            place_order_fn=self._router_place_order,
                            cancel_order_fn=lambda oid: self._router_cancel_order(symbol, oid),
                            max_wait_seconds=3.0,
                            client_order_id=order.client_order_id  # PASS THE ID!
                        )
                    
                    # Log router stats
                    stats = self.order_router.get_stats()
                    maker_pct = stats.get('maker_fill_pct', 0)
                    logger.info(
                        f"✅ FILLED: {side} {router_result.filled_qty:.6f} {symbol} @ "
                        f"${router_result.avg_price:,.2f} | Maker: {maker_pct:.0f}% | "
                        f"Cost: {router_result.total_cost_bps:.1f}bps"
                    )
                    
                    # Log success to event store
                    await self.event_store.log_order_submitted(
                        trace_id=order.trace_id,
                        client_order_id=order.client_order_id,
                        exchange_order_id="ROUTER",  # Router handles multiple orders
                        symbol=symbol.lower(),
                        side=side,
                        quantity=router_result.filled_qty,
                        price=router_result.avg_price
                    )
                    
                    return order.client_order_id
                
            except BinanceAPIException as e:
                # MIN_NOTIONAL error - don't retry, it's a parameter issue
                if e.code == -4164:
                    logger.error(f"❌ MIN_NOTIONAL ERROR: {quantity} {symbol} = ${quantity * price:.2f} < Minimum Notional")
                    await self.order_manager.mark_rejected(
                        order.client_order_id,
                        error_message=f"MIN_NOTIONAL: ${quantity * price:.2f} too low"
                    )
                    return None
                
                # Timeout error (-1007) - retry with exponential backoff
                if e.code == -1007 and attempt < max_retries - 1:
                    logger.warning(f"⏳ Timeout (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                
                logger.error(f"❌ BINANCE API ERROR: {e.message} (Code: {e.code})")
                await self.order_manager.mark_rejected(
                    order.client_order_id,
                    error_message=f"{e.code}: {e.message}"
                )
                return None
                
            except Exception as e:
                logger.error(f"❌ EXECUTION ERROR: {e}")
                await self.order_manager.mark_rejected(
                    order.client_order_id,
                    error_message=str(e)
                )
                return None
        
        # All retries exhausted
        logger.error(f"❌ All {max_retries} retries exhausted for {side} {symbol}")
        await self.order_manager.mark_rejected(
            order.client_order_id,
            error_message="Max retries exhausted"
        )
        return None
    
    async def _fetch_exchange_info(self):
        """Fetch exchange filters (tickSize, stepSize) from API."""
        try:
            if self.spot_mode:
                info = await self.client.get_exchange_info()
            else:
                info = await self.client.futures_exchange_info()
            for symbol_data in info['symbols']:
                s = symbol_data['symbol']
                
                tick_size = 0.01
                step_size = 0.001
                
                # Parse filters
                for f in symbol_data['filters']:
                    if f['filterType'] == 'PRICE_FILTER':
                        tick_size = float(f['tickSize'])
                    elif f['filterType'] == 'LOT_SIZE':
                        step_size = float(f['stepSize'])
                        
                self.exchange_filters[s] = (tick_size, step_size)
            
            logger.info(f"Loaded filters for {len(self.exchange_filters)} symbols")
            
        except Exception as e:
            logger.error(f"Failed to fetch exchange info: {e}")
            # Fallback defaults
            self.exchange_filters['BTCUSDT'] = (0.1, 0.001)
            self.exchange_filters['ETHUSDT'] = (0.01, 0.001)
            self.exchange_filters['SOLUSDT'] = (0.01, 1.0) 

    def _get_filters(self, symbol: str):
        """Get tick_size and step_size for a symbol."""
        symbol = symbol.upper()
        if symbol in self.exchange_filters:
            return self.exchange_filters[symbol]
            
        # Fallbacks
        if symbol.startswith("BTC"): return 0.1, 0.001
        if symbol.startswith("ETH"): return 0.01, 0.001
        if symbol.startswith("SOL"): return 0.01, 1.0
        return 0.01, 0.001
        
    def _round_step_size(self, value: float, step_size: float) -> float:
        """Round value to the nearest step_size."""
        if step_size == 0: return value
        precision = int(round(-math.log(step_size, 10), 0))
        return round(value, precision)
    
    async def get_position(self, symbol: str) -> float:
        """Get current position quantity from OrderManager."""
        pos = await self.order_manager.get_position(symbol.lower())
        return pos.quantity
    
    async def get_total_equity(self, current_balance: float) -> float:
        """Get total equity (balance + unrealized PnL)."""
        # TODO: Fetch unrealized PnL from exchange
        return current_balance
    
    # =========================================================================
    # HELPER METHODS FOR DETERMINISTIC ORDER ROUTER
    # =========================================================================
    
    async def _get_best_bid_ask(self, symbol: str) -> tuple:
        """Get current best bid and ask prices from exchange."""
        try:
            # Use orderbook_ticker - faster than full order book
            if self.spot_mode:
                ticker = await self.client.get_orderbook_ticker(symbol=symbol.upper())
            else:
                ticker = await self.client.futures_orderbook_ticker(symbol=symbol.upper())
            best_bid = float(ticker['bidPrice'])
            best_ask = float(ticker['askPrice'])
            return (best_bid, best_ask)
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to get ticker: {e}")
            return (0, 0)
    
    async def _router_place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float = None,
        order_type: str = "LIMIT",
        client_order_id: Optional[str] = None
    ) -> Dict:
        """Place order for router - returns dict with order_id, filled_qty, avg_price."""
        try:
            tick_size, step_size = self._get_filters(symbol)
            quantity = self._round_step_size(quantity, step_size)
            
            if order_type == "MARKET":
                result = await self.client.futures_create_order(
                    symbol=symbol.upper(),
                    side=side.upper(),
                    type='MARKET',
                    quantity=quantity,
                    recvWindow=self.nonce_service.get_recv_window()
                )
            else:  # LIMIT order
                price = self._round_step_size(price, tick_size)
                
                # ==========================================
                # CRITICAL FIX: Pre-register before submission
                # ==========================================
                if client_order_id:
                    existing_order = await self.order_manager.get_order(client_order_id)
                    
                    if not existing_order:
                        # Create placeholder order record
                        # This prevents "orphan fill" errors when router repricings fill
                        logger.debug(f"📝 Pre-registering router order: {client_order_id}")
                        
                        # We don't have exchange_order_id yet, so use placeholder
                        await self.order_manager.register_existing_order(
                            client_order_id=client_order_id,
                            exchange_order_id="PENDING",  # Will be updated on submit
                            symbol=symbol.lower(),
                            side=side.upper(),
                            quantity=quantity,
                            price=price,
                            order_type=order_type,
                            filled_quantity=0.0
                        )
                
                # Submit to exchange
                kwargs = {'recvWindow': self.nonce_service.get_recv_window()}
                if client_order_id:
                    kwargs['newClientOrderId'] = client_order_id
                
                result = await self.client.futures_create_order(
                    symbol=symbol.upper(),
                    side=side.upper(),
                    type='LIMIT',
                    timeInForce='GTX',  # Post-only
                    quantity=quantity,
                    price=str(price),
                    **kwargs
                )
                
                exchange_order_id = str(result.get('orderId', ''))
                
                # Update the placeholder with real exchange ID
                if client_order_id:
                    order = await self.order_manager.get_order(client_order_id)
                    if order and order.exchange_order_id == "PENDING":
                        await self.order_manager.mark_submitted(
                            client_order_id=client_order_id,
                            exchange_order_id=exchange_order_id
                        )
            
            return {
                'order_id': str(result.get('orderId', '')),
                'filled_qty': float(result.get('executedQty', 0)),
                'avg_price': float(result.get('avgPrice', price or 0)),
                'status': result.get('status', 'NEW')
            }
        except BinanceAPIException as e:
            # GTX rejection is expected - not an error
            if e.code == -5022:
                return {'order_id': None, 'filled_qty': 0, 'avg_price': 0, 'status': 'REJECTED_GTX'}
            raise
    
    async def _router_cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel order for router."""
        try:
            # Use origClientOrderId for clientOrderId-based orders
            await self.client.futures_cancel_order(
                symbol=symbol.upper(),
                origClientOrderId=order_id
            )
            return True
        except Exception as e:
            logger.debug(f"[{symbol}] Cancel failed: {e}")
            return False


    
    async def close(self):
        """Graceful shutdown."""
        logger.info("Closing execution handler...")
        
        # Log shutdown event (skip if DB was deleted)
        try:
            await self.event_store.append(
                EventType.SYSTEM_SHUTDOWN,
                payload={"reason": "graceful"},
                trace_id=self.event_store.generate_trace_id()
            )
        except Exception as e:
            logger.debug(f"Could not log shutdown event (DB may be missing): {e}")
        
        # Stop reconciliation service
        if self.reconciliation:
            await self.reconciliation.stop()
        
        # Stop time sync
        await self.time_sync.stop()
        
        # Stop User Stream
        if hasattr(self, '_user_stream_task') and self._user_stream_task:
            self._user_stream_task.cancel()
            try:
                await self._user_stream_task
            except asyncio.CancelledError:
                pass
        
        # Close Binance connection
        if self.client:
            await self.client.close_connection()
            self.client = None # Clear client to stop loop
        
        logger.info("Execution handler closed.")
        
        # Close Binance connection
        if self.client:
            await self.client.close_connection()
        
        logger.info("Execution handler closed.")
    
    async def _execute_spot_order(self, symbol, side, quantity, price, order_type, client_order_id):
        """Execute a REAL Spot order."""
        # DEBUG: Log which key is being used
        key_prefix = self.api_key[:6] if self.api_key else "NONE"
        logger.info(f"🔑 Executing Spot Order check: KeyPrefix={key_prefix} Symbol={symbol}")

        if order_type == 'MARKET':
            return await self.client.create_order(
                symbol=symbol.upper(),
                side=side,
                type='MARKET',
                quantity=quantity,
                newClientOrderId=client_order_id
            )
        else:
            return await self.client.create_order(
                symbol=symbol.upper(),
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price,
                newClientOrderId=client_order_id
            )


    async def _execute_futures_order(self, symbol, side, quantity, price, order_type, client_order_id, stop_price: float = None):
        """
        Execute a REAL Futures order.
        
        Supports:
        - MARKET: Immediate fill at best price
        - LIMIT: Standard limit order
        - STOP_MARKET: Stop-loss market order (triggers when stop_price is reached)
        """
        if order_type == 'MARKET':
            return await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                newClientOrderId=client_order_id,
                recvWindow=self.nonce_service.get_recv_window()
            )
        elif order_type == 'STOP_MARKET':
            # Stop-loss market order - triggers when price crosses stop_price
            if not stop_price:
                raise ValueError("STOP_MARKET requires stop_price parameter")
            return await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='STOP_MARKET',
                quantity=quantity,
                stopPrice=str(stop_price),
                newClientOrderId=client_order_id,
                recvWindow=self.nonce_service.get_recv_window()
            )
        else:
            return await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=quantity,
                price=price,
                newClientOrderId=client_order_id,
                recvWindow=self.nonce_service.get_recv_window()
            )