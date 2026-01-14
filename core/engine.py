import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import Optional

# Events
from core.events import MarketEvent, SignalEvent, RegimeEvent, FillEvent, FundingRateEvent

# Modules
from analysis.regime_supervisor import RegimeSupervisor
from risk_engine.risk_manager import RiskManager, PortfolioPosition, RiskState
from execution.simulated import MockExecutionHandler
from execution.binance_executor import BinanceExecutionHandler
# from execution.okx_executor import OkxExecutionHandler # Deprecated
from execution.ccxt_executor import CCXTExecutor
from data.candle_provider import CandleProvider
import time

# Load environment variables
load_dotenv()

logger = logging.getLogger("Core.Engine")

class TradingEngine:
    """Core trading engine with event-driven architecture."""
    
    # Maximum consecutive errors before forcing shutdown
    MAX_CONSECUTIVE_ERRORS = 10
    # Event queue size limit to prevent memory exhaustion
    EVENT_QUEUE_MAXSIZE = 10000
    
    def __init__(self, settings):
        self.settings = settings
        self.running = False
        self.event_queue = asyncio.Queue(maxsize=self.EVENT_QUEUE_MAXSIZE)
        self._consecutive_errors = 0
        
        # Get trading symbols
        self.symbols = getattr(settings, 'TRADING_SYMBOLS', ['btcusdt'])
        self.testnet = getattr(settings, 'TESTNET', True)
        self.exchange = getattr(settings, 'EXCHANGE', 'binance').lower()
        
        # 1. Initialize CandleProvider (streaming candles)
        self.candle_provider: Optional[CandleProvider] = None
        
        # 2. Initialize RegimeSupervisor (will connect to CandleProvider in start())
        self.regime_supervisor = RegimeSupervisor(self.event_queue)
        
        # 3. Initialize Risk Manager
        initial_capital = getattr(settings, 'INITIAL_CAPITAL', 1000.0)
        self.risk_manager = RiskManager(account_balance=initial_capital)
        
        # 4. Initialize Execution Handlers
        self.executors = {}
        
        if getattr(self.settings, 'PAPER_TRADING', True):
            logger.info("Using PAPER TRADING execution handler.")
            self.executors['paper'] = MockExecutionHandler(self.event_queue, self.risk_manager)
        else:
            active_exchanges = getattr(settings, 'ACTIVE_EXCHANGES', ['binance'])
            logger.info(f"Initializing executors for: {active_exchanges}")
            
            for ex in active_exchanges:
                if ex == 'binance':
                    self.executors['binance'] = BinanceExecutionHandler(self.event_queue, self.risk_manager)
            
            # Initialize MultiExchangeManager for all other exchanges (OKX, MEXC, Coinbase, etc)
            from execution.exchange_manager import MultiExchangeManager
            self.multi_manager = MultiExchangeManager(testnet=self.testnet)
            # initialization happens in start() async
            
            # Note: We will merge multi_manager.executors into self.executors after init

        # Legacy support (primary executor)
        self.execution_handler = next(iter(self.executors.values())) if self.executors else None
        
        self.connectors = []
        self.strategies = []

    def add_connector(self, connector):
        self.connectors.append(connector)

    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    async def start(self):
        """Start all async processes."""
        self.running = True
        logger.info("Starting Trading Engine...")

        # Check for old emergency stop file
        if os.path.exists("EMERGENCY_STOP.flag"):
            logger.warning("⚠️ FOUND OLD EMERGENCY STOP FILE. Please remove 'EMERGENCY_STOP.flag' to run properly.")

        # 1. Start CandleProvider (streaming candles)
        self.candle_provider = CandleProvider(
            symbols=self.symbols,
            interval="1m",
            testnet=self.testnet
        )
        await self.candle_provider.start()
        
        # Wire CandleProvider to RegimeSupervisor
        self.regime_supervisor.set_candle_provider(self.candle_provider)
        logger.info(f"📊 CandleProvider started for {len(self.symbols)} symbols")
        
        # 2. Connect execution handlers
        # Sync MultiExchangeManager first
        if hasattr(self, 'multi_manager'):
            await self.multi_manager.initialize()
            # Merge unified executors into main map
            for name, exc in self.multi_manager.executors.items():
                self.executors[name] = exc
                
        for name, executor in self.executors.items():
            logger.info(f"Connecting executor: {name}...")
            if hasattr(executor, 'connect'):
                await executor.connect()
            
            # Start polling for CCXT executors
            if hasattr(executor, 'start_polling'):
                # Note: start_polling is async but fire-and-forget in CCXTExecutor? 
                # No, we defined it as async. We should await it?
                # Actually, in CCXTExecutor it uses create_task internally but is async def.
                await executor.start_polling(self.symbols)

        # =====================================================================
        # CRITICAL: Bootstrap exchange state BEFORE reconciliation starts
        # This prevents "orphan storm" where existing orders get canceled
        # =====================================================================
        await self._bootstrap_exchange_state()
        
        # 3. Start connectors (WebSockets)
        tasks = [asyncio.create_task(c.connect()) for c in self.connectors]
        
        # 4. Start Event Loop
        loop_task = asyncio.create_task(self._run_event_loop())
        
        # 5. Start Watchdog
        watchdog_task = asyncio.create_task(self._watchdog_loop())
        
        await asyncio.gather(*tasks, loop_task, watchdog_task)

    async def _bootstrap_exchange_state(self):
        """
        Bootstrap exchange state before strategy/reconciliation starts.
        
        Pre-registers existing orders and positions from the exchange so they
        aren't flagged as orphans or cause position mismatch shocks.
        """
        logger.info("📥 Bootstrapping exchange state...")
        
        for name, executor in self.executors.items():
            if not hasattr(executor, 'client'):
                continue
                
            try:
                # 1. Fetch and register open orders
                if hasattr(executor, 'reconciliation') and executor.reconciliation:
                    existing_orders = await executor.reconciliation._fetch_open_orders()
                    order_count = 0
                    
                    for order_data in existing_orders:
                        client_order_id = order_data.get('clientOrderId', '')
                        
                        # Skip if not our order format  
                        if not client_order_id.startswith('c_'):
                            continue
                            
                        await executor.order_manager.register_existing_order(
                            client_order_id=client_order_id,
                            exchange_order_id=str(order_data.get('orderId', '')),
                            symbol=order_data.get('symbol', ''),
                            side=order_data.get('side', ''),
                            quantity=float(order_data.get('origQty', 0)),
                            price=float(order_data.get('price', 0)),
                            order_type=order_data.get('type', 'LIMIT'),
                            filled_quantity=float(order_data.get('executedQty', 0))
                        )
                        order_count += 1
                    
                    if order_count > 0:
                        logger.info(f"✅ Pre-registered {order_count} existing orders from {name}")
                    
                    # 2. Fetch and sync positions
                    existing_positions = await executor.reconciliation._fetch_positions()
                    pos_count = 0
                    
                    for symbol, pos_data in existing_positions.items():
                        qty = float(pos_data.get('positionAmt', 0))
                        if qty == 0:
                            continue
                            
                        entry_price = float(pos_data.get('entryPrice', 0))
                        await executor.order_manager.set_position_from_exchange(
                            symbol=symbol,
                            quantity=qty,
                            entry_price=entry_price,
                            exchange_snapshot=str(pos_data)
                        )
                        pos_count += 1
                    
                    if pos_count > 0:
                        logger.info(f"✅ Pre-synced {pos_count} existing positions from {name}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Bootstrap failed for {name}: {e}")
        
        logger.info("📥 Bootstrap complete - safe to start strategies")

    async def _run_event_loop(self):
        """Main event loop that processes events."""
        logger.info("Event Loop started.")
        while self.running:
            # Emergency stop file check (legacy)
            if os.path.exists("EMERGENCY_STOP.flag"):
                logger.critical("EMERGENCY STOP FILE DETECTED! Shutting down engine immediately.")
                self.running = False
                break
            
            # Dashboard signal checks
            if os.path.exists("data/STOP_SIGNAL"):
                logger.critical("DASHBOARD STOP SIGNAL! Canceling all orders...")
                await self._emergency_flatten()
                self.running = False
                break
            
            # PAUSE_SIGNAL: Block new orders, let existing settle
            paused = os.path.exists("data/PAUSE_SIGNAL")
            if paused and not getattr(self, '_pause_logged', False):
                logger.warning("DASHBOARD PAUSE SIGNAL - blocking new orders")
                self._pause_logged = True
            elif not paused and getattr(self, '_pause_logged', False):
                logger.info("PAUSE CLEARED - resuming trading")
                self._pause_logged = False

            try:
                # Remove timeout to avoid busy waiting on empty queue
                event = await self.event_queue.get()
                
                # Process with timeout wrapper to prevent hanging handlers
                try:
                    async with asyncio.timeout(5.0):
                        if isinstance(event, MarketEvent):
                            await self._handle_market_data(event)
                        elif isinstance(event, SignalEvent):
                            if paused:
                                logger.debug(f"Signal blocked (paused): {event.side} {event.symbol}")
                            else:
                                await self._handle_signal(event)
                        elif isinstance(event, RegimeEvent):
                            await self._handle_regime_change(event)
                        elif isinstance(event, FillEvent):
                            await self._handle_fill(event)
                        elif isinstance(event, FundingRateEvent):
                            await self._handle_funding_rate(event)
                except asyncio.TimeoutError:
                    logger.error(f"Event processing timeout: {type(event)}")
                
                self.event_queue.task_done()
                self._consecutive_errors = 0  # Reset on successful processing

            except Exception as e:
                self._consecutive_errors += 1
                logger.error(f"Event loop error ({self._consecutive_errors}/{self.MAX_CONSECUTIVE_ERRORS}): {e}", exc_info=True)
                
                if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    logger.critical(f"Too many consecutive errors ({self._consecutive_errors}). Shutting down for safety.")
                    self.running = False
                    break
                
                await asyncio.sleep(0.1)  # Brief backoff before retry

    async def _emergency_flatten(self):
        """Cancel all orders and close positions on emergency stop."""
        logger.critical("🚨 EMERGENCY FLATTEN: Initiating...")
        
        for name, executor in self.executors.items():
            try:
                # 1. Cancel orders
                if hasattr(executor, 'cancel_all_orders'):
                    await executor.cancel_all_orders()
                
                # 2. VERIFY cancellation
                await asyncio.sleep(0.5)  # Brief delay for exchange
                if hasattr(executor, 'reconciliation') and executor.reconciliation:
                    remaining = await executor.reconciliation._fetch_open_orders()
                    if remaining:
                        logger.error(f"⚠️ {len(remaining)} orders still open on {name} - retrying...")
                        # Force cancel with exchange API directly if accessible
                        if hasattr(executor, 'client') and executor.client:
                            for order in remaining:
                                try:
                                    await executor.client.cancel_order(
                                        symbol=order['symbol'],
                                        orderId=order['orderId']
                                    )
                                except Exception as inner_e:
                                    logger.error(f"Failed to force cancel: {inner_e}")
                
                # 3. Flatten positions (if method exists)
                if hasattr(executor, 'flatten_all_positions'):
                    await executor.flatten_all_positions()
                    
                    # 4. VERIFY positions closed
                    await asyncio.sleep(1.0)
                    if hasattr(executor, 'reconciliation') and executor.reconciliation:
                        pos_snapshot = await executor.reconciliation._fetch_positions()
                        non_zero = {k: v for k, v in pos_snapshot.items() 
                                   if abs(float(v.get('positionAmt', 0))) > 0.001}
                        if non_zero:
                            logger.critical(f"🔴 POSITIONS STILL OPEN: {non_zero}")
                        else:
                            logger.info(f"✅ All positions closed on {name}")
                else:
                    logger.info(f"✅ Canceled orders on {name} (No flatten method)")
                    
            except Exception as e:
                logger.error(f"❌ Emergency flatten failed for {name}: {e}")

    async def _handle_market_data(self, event: MarketEvent):
        """Handle incoming price data."""
        for executor in self.executors.values():
            if hasattr(executor, 'on_tick'):
                await executor.on_tick(event)

        # Global kill switch check and Risk Update
        if self.risk_manager:
            total_equity = self.risk_manager.current_balance
            
            # BUILD position list for risk calculation
            positions = []
            for executor in self.executors.values():
                # Try to get positions from OrderManager or PositionTracker
                if hasattr(executor, 'order_manager') and hasattr(executor.order_manager, 'positions'):
                    for symbol, pos in executor.order_manager.positions.items():
                        positions.append(PortfolioPosition(
                            symbol=symbol,
                            size=pos.quantity,
                            mark_price=event.price if event.symbol == symbol else pos.mark_price,
                            unrealized_pnl=pos.unrealized_pnl
                        ))
                elif hasattr(executor, 'position_tracker') and executor.position_tracker:
                   # Fallback to position tracker directly if exposed
                   for symbol, pos in executor.position_tracker.positions.items():
                        positions.append(PortfolioPosition(
                            symbol=symbol,
                            size=pos.quantity,
                            mark_price=event.price if event.symbol == symbol else pos.mark_price,
                            unrealized_pnl=pos.unrealized_pnl
                        ))

            # UPDATE risk state with full portfolio
            # Sum unrealized pnl from positions
            current_pnl = sum(p.unrealized_pnl for p in positions)
            
            # Using current balance + pnl as proxy for equity if not synced
            metrics = self.risk_manager.update(
                current_pnl=current_pnl,
                current_balance=total_equity,
                positions=positions  # ← CRITICAL: Pass positions for CVaR
            )
            
            if metrics.state == RiskState.STOP:
                logger.critical(f"RISK STATE: STOP | Drawdown: {metrics.current_drawdown:.2%}")
                logger.critical("ACCOUNT HEALTH CRITICAL. STOPPING ENGINE.")
                await self._emergency_flatten()
                self.running = False
                return

        # Update market regime analysis
        if self.regime_supervisor:
            await self.regime_supervisor.update(event)
            
        # Forward data to strategies
        for strategy in self.strategies:
            await strategy.on_tick(event)

    async def _handle_signal(self, event: SignalEvent):
        """Handle buy/sell signals from strategies."""
        symbol = event.symbol.upper()
        
        # Validate via Risk Manager
        if self.risk_manager:
            if not self.risk_manager.validate_signal(event):
                logger.warning(f"REJECTED: {event.side} {event.quantity} {symbol}")
                return
        
        # Route to correct execution handler
        target_exchange = getattr(event, 'exchange', 'binance').upper()
        
        if getattr(self.settings, 'PAPER_TRADING', False):
            target_exchange = 'PAPER'
            
        executor = self.executors.get(target_exchange.lower())
        
        if executor:
            logger.info(f"{event.side} {event.quantity} {symbol} @ ${event.price:,.2f} -> {target_exchange}")
            await executor.execute(event)
        else:
            logger.error(f"No executor for: {target_exchange}")

    async def _handle_fill(self, event: FillEvent):
        """Handle confirmed fills."""
        logger.info(f"FILL CONFIRMED: {event.side} {event.quantity} {event.symbol} @ {event.price}")
        
        # Notify strategies so they can update their positions
        for strategy in self.strategies:
            if hasattr(strategy, 'on_fill'):
                await strategy.on_fill(event)

    async def _handle_regime_change(self, event: RegimeEvent):
        """Handle market regime changes (Trend vs Range)."""
        logger.info(f"MARKET REGIME CHANGE: {event.symbol} -> {event.regime} (ADX: {event.adx:.2f})")
        for strategy in self.strategies:
            if hasattr(strategy, 'on_regime_change'):
                await strategy.on_regime_change(event)

    async def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping engine...")
        self.running = False
        
        # Stop CandleProvider
        if self.candle_provider:
            await self.candle_provider.stop()
        
        # Close all connectors
        for c in self.connectors:
            await c.close()
            
        # Close execution handlers
        for name, executor in self.executors.items():
            if hasattr(executor, 'close'):
                await executor.close()
        
        logger.info("Engine stopped.")

            if hasattr(strategy, 'on_funding_rate'):
                await strategy.on_funding_rate(event)

    async def _watchdog_loop(self):
        """Monitor for stuck states and alive heartbeat."""
        logger.info("Watchdog started.")
        last_event_time = time.time()
        
        while self.running:
            await asyncio.sleep(30)  # Check every 30s
            
            # 1. Check if we're still receiving events (optional, maybe too strict for low liquid assets)
            # if time.time() - last_event_time > 300: # 5 mins
            #    logger.warning("⚠️ No events received in 5 mins - WebSocket may be idel")
            
            # 2. Check for stuck orders
            for executor in self.executors.values():
                if hasattr(executor, 'order_manager'):
                    stuck = executor.order_manager.get_stuck_orders(timeout_seconds=300)
                    if stuck:
                        logger.warning(f"Found {len(stuck)} stuck orders - auto-canceling")
                        for order_id in stuck:
                            if hasattr(executor, 'cancel_order'):
                                await executor.cancel_order(order_id)

    def get_risk_snapshot(self) -> dict:
        """Expose risk state for dashboard."""
        if not self.risk_manager:
            return {}
        
        return {
            'state': self.risk_manager.current_state.value,
            'position_multiplier': self.risk_manager._get_position_multiplier(
                abs(self.risk_manager.cumulative_loss) / 
                (self.risk_manager.initial_balance * self.risk_manager.daily_cvar_limit) if self.risk_manager.daily_cvar_limit > 0 else 0
            ),
            'drawdown_pct': (self.risk_manager.peak_balance - self.risk_manager.current_balance) / 
                            self.risk_manager.peak_balance if self.risk_manager.peak_balance > 0 else 0,
            'circuit_breaker': self.risk_manager.circuit_breaker.get_status().__dict__ if hasattr(self.risk_manager.circuit_breaker, 'get_status') else {}
        }