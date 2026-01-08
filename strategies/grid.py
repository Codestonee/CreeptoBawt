import logging
import asyncio
from typing import List, Dict
from core.events import MarketEvent, SignalEvent, RegimeEvent, FillEvent

logger = logging.getLogger("Strategy.DynamicGrid")

class DynamicGridStrategy:
    def __init__(self, event_queue, symbol="btcusdt", max_positions=5, base_quantity=0.002):
        self.queue = event_queue
        self.symbol = symbol.lower()
        self.active = True
        
        # Konfiguration
        self.max_positions = max_positions
        self.base_quantity = base_quantity
        
        # Marknadsdata
        self.last_price = 0.0
        self.current_volatility = 0.0
        self.current_regime = "UNCERTAIN"
        
        # State Tracking
        self.current_position = 0.0
        self.open_buy_orders = 0
        self.open_sell_orders = 0
        
        self.last_rebalance = 0

    async def on_tick(self, event: MarketEvent):
        if event.symbol.lower() != self.symbol:
            return

        self.last_price = event.price
        
        # --- FIX: Lossa på säkerhetsspärren ---
        # Om vi inte har någon volatilitet än, gissa på 0.5% av priset
        if self.current_volatility == 0 and self.last_price > 0:
             self.current_volatility = self.last_price * 0.005
             logger.info(f"Using Fallback Volatility: {self.current_volatility:.2f}")

        # Om vi fortfarande har 0 i volatilitet (t.ex. priset är 0), då kan vi inte handla
        if self.current_volatility == 0:
            return
        # --------------------------------------

        if not self.active:
            return

        now = asyncio.get_event_loop().time()
        # Rebalansera var 10:e sekund
        if now - self.last_rebalance > 10:
            await self._evaluate_grid()
            self.last_rebalance = now

    async def on_regime_change(self, event: RegimeEvent):
        if event.symbol.lower() != self.symbol:
            return
            
        logger.info(f"GRID Regime Update: {event.regime} (ADX: {event.adx:.1f})")
        self.current_regime = event.regime
        self.current_volatility = event.volatility

        if event.regime == "TRENDING":
            if self.active:
                logger.warning("TREND DETECTED! Pausing Grid Strategy.")
                self.active = False
        elif event.regime == "RANGING":
            if not self.active:
                logger.info("MARKET RANGING. Resuming Grid Strategy.")
                self.active = True

    async def on_fill(self, event: FillEvent):
        if event.symbol.lower() != self.symbol:
            return

        if event.side == 'BUY':
            self.current_position += event.quantity
        elif event.side == 'SELL':
            self.current_position -= event.quantity
            
        logger.info(f"Strategy Position Update: Holding {self.current_position:.4f} {self.symbol}")

    async def _evaluate_grid(self):
        if self.last_price == 0:
            return

        # Grid Spacing = 2x Volatilitet (eller 1% fallback)
        grid_spacing = self.current_volatility * 2.0
        if grid_spacing == 0:
            grid_spacing = self.last_price * 0.01 
        
        # 1. KÖP-LOGIK (Bygg lager)
        if self.current_position < (self.max_positions * self.base_quantity):
            buy_price = self.last_price - grid_spacing
            
            # Logga bara var 10:e sekund för att inte spamma om vi inte lägger order
            # logger.info(f"Checking BUY: Target {buy_price:.2f} < Current {self.last_price:.2f}")
            
            await self.queue.put(SignalEvent(
                strategy_id="dynamic_grid_v2",
                symbol=self.symbol,
                side="BUY",
                quantity=self.base_quantity,
                price=buy_price
            ))
        else:
            # Vi har fullt lager
            pass

        # 2. SÄLJ-LOGIK (Ta vinst)
        if self.current_position > 0:
            sell_price = self.last_price + grid_spacing
            
            await self.queue.put(SignalEvent(
                strategy_id="dynamic_grid_v2",
                symbol=self.symbol,
                side="SELL",
                quantity=self.base_quantity,
                price=sell_price
            ))