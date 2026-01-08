import asyncio
import logging
import os
from dotenv import load_dotenv

# Importera Events
from core.events import MarketEvent, SignalEvent, RegimeEvent, FillEvent

# Importera Moduler
from analysis.regime_supervisor import RegimeSupervisor
from risk_engine.risk_manager import RiskManager
from execution.simulated import MockExecutionHandler
from execution.binance_executor import BinanceExecutionHandler

# Ladda miljövariabler från .env (för API-nycklar)
load_dotenv()

logger = logging.getLogger("Core.Engine")

class TradingEngine:
    def __init__(self, settings):
        self.settings = settings
        self.running = False
        self.event_queue = asyncio.Queue()
        
        # 1. Initiera Marknadsanalys
        self.regime_supervisor = RegimeSupervisor(self.event_queue)
        
        # 2. Initiera Risk Manager
        initial_capital = getattr(settings, 'INITIAL_CAPITAL', 1000.0)
        self.risk_manager = RiskManager(account_balance=initial_capital)
        
        # 3. Välj Execution Handler baserat på settings
        if getattr(self.settings, 'PAPER_TRADING', True):
            logger.info("Using PAPER TRADING execution handler.")
            self.execution_handler = MockExecutionHandler(self.event_queue, self.risk_manager)
        else:
            logger.warning("⚠️ USING LIVE BINANCE TESTNET EXECUTOR!")
            logger.warning("Ensure your API keys in .env are for the TESTNET, not real Binance.")
            self.execution_handler = BinanceExecutionHandler(self.event_queue, self.risk_manager)
        
        self.connectors = []
        self.strategies = []

    def add_connector(self, connector):
        self.connectors.append(connector)

    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    async def start(self):
        """Startar alla asynkrona processer."""
        self.running = True
        logger.info("Starting Trading Engine...")

        # Kolla om en gammal nödstopp-fil ligger kvar
        if os.path.exists("EMERGENCY_STOP.flag"):
            logger.warning("⚠️ FOUND OLD EMERGENCY STOP FILE. Please remove 'EMERGENCY_STOP.flag' to run properly.")

        # Om vi kör mot Binance (ej mock), måste vi ansluta klienten
        if isinstance(self.execution_handler, BinanceExecutionHandler):
            await self.execution_handler.connect()

        # Starta connectors (WebSockets)
        tasks = [asyncio.create_task(c.connect()) for c in self.connectors]
        
        # Starta Event Loop
        loop_task = asyncio.create_task(self._run_event_loop())
        
        await asyncio.gather(*tasks, loop_task)

    async def _run_event_loop(self):
        """Huvudloopen som bearbetar events."""
        logger.info("Event Loop started.")
        while self.running:
            # --- EMERGENCY STOP CHECK ---
            if os.path.exists("EMERGENCY_STOP.flag"):
                logger.critical("🚨 EMERGENCY STOP FILE DETECTED! Shutting down engine immediately.")
                self.running = False
                break
            # ----------------------------

            try:
                # Vänta på nästa event (timeout för att kunna kolla stop-flaggan ofta)
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                if isinstance(event, MarketEvent):
                    await self._handle_market_data(event)
                elif isinstance(event, SignalEvent):
                    await self._handle_signal(event)
                elif isinstance(event, RegimeEvent):
                    await self._handle_regime_change(event)
                elif isinstance(event, FillEvent):
                    await self._handle_fill(event)
                
                self.event_queue.task_done()

            except asyncio.TimeoutError:
                # Ingen data kom på 1 sekund, loopen går runt
                continue
            except Exception as e:
                logger.error(f"CRITICAL: Event loop error: {e}", exc_info=True)

    async def _handle_market_data(self, event: MarketEvent):
        """Hanterar inkommande prisdata."""
        # 1. Uppdatera Execution Handler (för simulatorns prislista)
        if self.execution_handler:
            await self.execution_handler.on_tick(event)

        # 2. SÄKERHETSKONTROLL: Global Kill Switch
        if self.risk_manager and self.execution_handler:
            # Hämta total equity (Saldo + Orealiserad vinst/förlust)
            # Notera: För Binance Testnet är detta förenklat än så länge
            current_equity = self.execution_handler.get_total_equity(self.risk_manager.current_balance)
            
            # Om kontot blöder för mycket -> NÖDSTOPP
            if not self.risk_manager.check_account_health(current_equity):
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
        logger.info(f"Signal received: {event}")
        
        # 1. Validera via Risk Manager
        if self.risk_manager:
            if not self.risk_manager.validate_signal(event):
                logger.warning(f"Signal REJECTED by Risk Manager: {event.symbol} {event.side}")
                return
            else:
                logger.info(f"Signal APPROVED by Risk Manager. Qty: {event.quantity}")

        # 2. Skicka till Execution Handler (Binance eller Mock)
        if self.execution_handler:
            await self.execution_handler.execute(event)

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
        """Stänger ner systemet snyggt."""
        logger.info("Stopping engine...")
        self.running = False
        
        # Stäng alla connectors
        for c in self.connectors:
            await c.close()
            
        # Stäng execution handler (viktigt för Binance connection)
        if hasattr(self.execution_handler, 'close'):
            await self.execution_handler.close()