import logging
import uuid
import time
from collections import defaultdict
from core.events import SignalEvent, OrderEvent, FillEvent, MarketEvent
from database.db_manager import DatabaseManager

logger = logging.getLogger("Execution.Simulated")

class MockExecutionHandler:
    def __init__(self, event_queue, risk_manager):
        self.queue = event_queue
        self.risk_manager = risk_manager
        
        # Initiera databasen
        self.db = DatabaseManager()
        
        # Priser och Innehav
        self.last_prices = {} 
        self.positions = defaultdict(float)
        self.avg_entry_price = defaultdict(float)

    async def on_tick(self, event: MarketEvent):
        """Uppdaterar prisuppfattning."""
        self.last_prices[event.symbol.lower()] = event.price

    def get_total_equity(self, current_balance):
        """
        Beräknar totalt kontovärde:
        Equity = Cash Balance + Unrealized PnL of open positions
        """
        unrealized_pnl = 0.0
        
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
                
            current_price = self.last_prices.get(symbol)
            if not current_price:
                continue
            
            entry_price = self.avg_entry_price[symbol]
            
            # Long PnL
            if qty > 0:
                unrealized_pnl += (current_price - entry_price) * qty
            # Short PnL
            elif qty < 0:
                unrealized_pnl += (entry_price - current_price) * abs(qty)
                
        return current_balance + unrealized_pnl

    async def execute(self, signal: SignalEvent):
        """Utför en simulerad order."""
        symbol_key = signal.symbol.lower()
        current_price = self.last_prices.get(symbol_key)
        
        if not current_price:
            logger.warning(f"Cannot execute {symbol_key}: No price data available.")
            return

        # Logga och simulera
        logger.info(f"PAPER TRADE: {signal.side} {signal.quantity} {signal.symbol} @ {current_price}")
        
        commission = (signal.quantity * current_price) * 0.0004
        realized_pnl = self._calculate_pnl(signal, current_price)
        
        # Uppdatera positioner
        self._update_position(symbol_key, signal.side, signal.quantity, current_price)

        # Skapa FillEvent
        fill = FillEvent(
            symbol=signal.symbol,
            side=signal.side,
            quantity=signal.quantity,
            price=current_price,
            commission=commission,
            pnl=realized_pnl
        )
        
        # Logga till DB och kö
        await self.db.log_trade(fill, strategy_id=signal.strategy_id)
        await self.queue.put(fill)
        
        if realized_pnl != 0:
            self.risk_manager.record_trade_result(realized_pnl)

    def _calculate_pnl(self, signal, exit_price):
        symbol = signal.symbol.lower()
        current_pos = self.positions[symbol]
        entry_price = self.avg_entry_price[symbol]
        pnl = 0.0
        
        if signal.side == 'SELL' and current_pos > 0:
            qty_closed = min(current_pos, signal.quantity)
            pnl = (exit_price - entry_price) * qty_closed
        elif signal.side == 'BUY' and current_pos < 0:
            qty_closed = min(abs(current_pos), signal.quantity)
            pnl = (entry_price - exit_price) * qty_closed
            
        return pnl

    def _update_position(self, symbol, side, qty, price):
        symbol = symbol.lower()
        prev_pos = self.positions[symbol]
        
        if side == 'BUY':
            if prev_pos >= 0:
                total_cost = (prev_pos * self.avg_entry_price[symbol]) + (qty * price)
                new_qty = prev_pos + qty
                if new_qty > 0: self.avg_entry_price[symbol] = total_cost / new_qty
            self.positions[symbol] += qty
            
        elif side == 'SELL':
            if prev_pos <= 0:
                total_cost = (abs(prev_pos) * self.avg_entry_price[symbol]) + (qty * price)
                new_qty = abs(prev_pos) + qty
                if new_qty > 0: self.avg_entry_price[symbol] = total_cost / new_qty
            self.positions[symbol] -= qty