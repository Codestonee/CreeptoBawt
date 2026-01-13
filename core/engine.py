import asyncio
import logging
import os
from dotenv import load_dotenv
from typing import Optional

# Events
from core.events import MarketEvent, SignalEvent, RegimeEvent, FillEvent

# Modules
from analysis.regime_supervisor import RegimeSupervisor
from risk_engine.risk_manager import RiskManager
from execution.simulated import MockExecutionHandler
from execution.binance_executor import BinanceExecutionHandler
from execution.okx_executor import OkxExecutionHandler
from data.candle_provider import CandleProvider

# Load environment variables
load_dotenv()

logger = logging.getLogger("Core.Engine")

class TradingEngine:
    def __init__(self, settings):
        self.settings = settings
        self.running = False
        self.event_queue = asyncio.Queue()
        
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
            self.executors['mock'] = MockExecutionHandler(self.event_queue, self.risk_manager)
        else:
            active_exchanges = getattr(settings, 'ACTIVE_EXCHANGES', ['binance'])
            logger.info(f"Initializing executors for: {active_exchanges}")
            
            for ex in active_exchanges:
                if ex == 'binance':
                    self.executors['binance'] = BinanceExecutionHandler(self.event_queue, self.risk_manager)
                elif ex == 'okx':
                    self.executors['okx'] = OkxExecutionHandler(self.event_queue, self.risk_manager, testnet=self.testnet)

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
        for name, executor in self.executors.items():
            logger.info(f"Connecting executor: {name}...")
            if hasattr(executor, 'connect'):
                await executor.connect()

        # =====================================================================
        # CRITICAL: Bootstrap exchange state BEFORE reconciliation starts
        # This prevents "orphan storm" where existing orders get canceled
        # =====================================================================
        await self._bootstrap_exchange_state()
        
        # 3. Start connectors (WebSockets)
        tasks = [asyncio.create_task(c.connect()) for c in self.connectors]
        
        # 4. Start Event Loop
        loop_task = asyncio.create_task(self._run_event_loop())
        
        await asyncio.gather(*tasks, loop_task)

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
            # --- EMERGENCY STOP CHECK (Legacy) ---
            if os.path.exists("EMERGENCY_STOP.flag"):
                logger.critical("🚨 EMERGENCY STOP FILE DETECTED! Shutting down engine immediately.")
                self.running = False
                break
            
            # --- DASHBOARD SIGNAL CHECKS (New) ---
            # STOP_SIGNAL: Flatten all and stop trading
            # --- DASHBOARD SIGNAL CHECKS (New) ---
            # STOP_SIGNAL: Flatten all and stop trading
            if os.path.exists("signals/STOP_ALL"):
                logger.critical("🚨 DASHBOARD STOP SIGNAL! Canceling all orders...")
                try:
                    os.remove("signals/STOP_ALL")
                except Exception:
                    pass
                await self._emergency_flatten()
                self.running = False
                break
            
            # PAUSE_SIGNAL: Block new orders, let existing settle
            paused = os.path.exists("PAUSE_SIGNAL")
            if paused and not getattr(self, '_pause_logged', False):
                logger.warning("⏸️ DASHBOARD PAUSE SIGNAL - blocking new orders")
                self._pause_logged = True
            elif not paused and getattr(self, '_pause_logged', False):
                logger.info("▶️ PAUSE CLEARED - resuming trading")
                self._pause_logged = False
            # ----------------------------

            try:
                # Wait for next event (timeout to check stop flags frequently)
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                if isinstance(event, MarketEvent):
                    await self._handle_market_data(event)
                elif isinstance(event, SignalEvent):
                    # Block new signals if paused
                    if paused:
                        logger.debug(f"Signal blocked (paused): {event.side} {event.symbol}")
                        continue
                    await self._handle_signal(event)
                elif isinstance(event, RegimeEvent):
                    await self._handle_regime_change(event)
                elif isinstance(event, FillEvent):
                    await self._handle_fill(event)
                
                self.event_queue.task_done()

            except asyncio.TimeoutError:
                # No data for 1 second, loop continues
                continue
            except Exception as e:
                logger.error(f"CRITICAL: Event loop error: {e}", exc_info=True)

    async def _emergency_flatten(self):
        """Cancel all orders and close positions on emergency stop."""
        logger.critical("🚨 EMERGENCY FLATTEN: Canceling all orders...")
        
        for name, executor in self.executors.items():
            try:
                # Cancel all open orders
                if hasattr(executor, 'cancel_all_orders'):
                    await executor.cancel_all_orders()
                    logger.info(f"✅ Canceled all orders on {name}")
                
                # Flatten positions (if method exists)
                if hasattr(executor, 'flatten_all_positions'):
                    await executor.flatten_all_positions()
                    logger.info(f"✅ Flattened positions on {name}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to flatten {name}: {e}")

    async def _handle_market_data(self, event: MarketEvent):
        """Hanterar inkommande prisdata."""
        # 1. Update Execution Handlers (for simulator ticking)
        for executor in self.executors.values():
            if hasattr(executor, 'on_tick'):
                await executor.on_tick(event)

        # 2. SÄKERHETSKONTROLL: Global Kill Switch
        if self.risk_manager:
            # Calculate total equity across all executors
            total_equity = self.risk_manager.current_balance # Base balance
            # Add PnL from all executors if possible
            # TODO: Aggregate equity correctly from multi-exchange
            
            # Om kontot blöder för mycket -> NÖDSTOPP
            if not self.risk_manager.check_account_health(total_equity):
                logger.critical("⛔ ACCOUNT HEALTH CRITICAL. STOPPING ENGINE.")
                self.running = False
                return

        # 3. Uppdatera Marknadsanalys (Regime)
        if self.regime_supervisor:
            await self.regime_supervisor.update(event)
            
        # 4. Skicka data till strategier
        for strategy in self.strategies:
            await strategy.on_tick(event)

    async def _handle_signal(self, event: SignalEvent):
        """Hanterar köp/sälj-signaler från strategier."""
        symbol = event.symbol.upper()
        
        # 1. Validera via Risk Manager
        if self.risk_manager:
            if not self.risk_manager.validate_signal(event):
                logger.warning(f"❌ REJECTED: {event.side} {event.quantity} {symbol}")
                return
        
        # 2. Route to correct Execution Handler
        target_exchange = getattr(event, 'exchange', 'binance').upper()
        
        # Fallback for paper trading or legacy signals
        if getattr(self.settings, 'PAPER_TRADING', False):
            target_exchange = 'PAPER'
            
        executor = self.executors.get(target_exchange.lower())
        
        if executor:
            logger.info(f"📤 {event.side} {event.quantity} {symbol} @ ${event.price:,.2f} → {target_exchange}")
            await executor.execute(event)
        else:
            logger.error(f"No executor for: {target_exchange}")

    async def _handle_fill(self, event: FillEvent):
        """Hanterar bekräftade avslut (Fills)."""
        logger.info(f"FILL CONFIRMED: {event.side} {event.quantity} {event.symbol} @ {event.price}")
        
        # Meddela strategierna så de kan uppdatera sina positioner
        for strategy in self.strategies:
            if hasattr(strategy, 'on_fill'):
                await strategy.on_fill(event)

    async def _handle_regime_change(self, event: RegimeEvent):
        """Hanterar förändringar i marknadsläget (Trend vs Range)."""
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