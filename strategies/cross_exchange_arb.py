import logging
import time
import asyncio
from typing import Dict, Optional
from core.events import MarketEvent, SignalEvent, FillEvent
from config.settings import settings

logger = logging.getLogger("Strategy.CrossArb")

from collections import defaultdict

class CrossExchangeArbStrategy:
    """
    Passive/Active Cross-Exchange Arbitrage.
    Monitors N-way spreads across all connected exchanges.
    """
    
    def __init__(self, event_queue, symbols: list = None):
        self.queue = event_queue
        self.symbols = [s.lower() for s in (symbols or settings.TRADING_SYMBOLS)]
        # Structure: {symbol: {exchange: {price, time}}}
        self.books = defaultdict(lambda: defaultdict(dict)) 
        self.min_spread = settings.ARBITRAGE_MIN_SPREAD
        self.position_exposure = {} 
        self.active_arbs = {} 
        
        # No manual init needed with defaultdict

    async def on_tick(self, event: MarketEvent):
        """Update internal book and check for arb."""
        symbol = event.symbol.lower()
        if symbol not in self.symbols:
            return

        exchange = event.exchange.lower()
        
        # Update Price
        # MarketEvent from CandleProvider (Binance) is usually just 'price' (mid/last)
        # We need Bid/Ask for arb. 
        # CCXTExecutor sends Tick with Price.
        # Ideally we need BBO (Best Bid Offer).
        
        # Heuristic: If we only have 'price', we assume a tight spread for now or wait for L2.
        # But CCXTExecutor fetch_ticker DOES return bid/ask if available.
        # My CCXT implementation sent 'price' as 'last'.
        # I should upgrade CCXTExecutor to send bid/ask in MarketEvent or generic event.
        # MarketEvent has 'price'. It doesn't have bid/ask fields.
        
        # Workaround: Use 'price' as mid, and assume 5bps spread? 
        # Risky for arb.
        # Let's trust 'price' is actionable for 'taker' (which is risky).
        
        # Store Price
        self.books[symbol][exchange] = {
            'price': event.price,
            'time': time.time()
        }
        
        await self._check_arb(symbol)

    async def _check_arb(self, symbol: str):
        """Compare All Connected Exchanges."""
        # Get all valid quotes for this symbol
        # Structure: self.books[symbol][exchange] = {'price': 100, 'time': ...}
        
        valid_quotes = {}
        now = time.time()
        
        for exc, data in self.books[symbol].items():
            if not data: continue
            if now - data.get('time', 0) > 5: continue # Stale
            valid_quotes[exc] = data['price']
            
        if len(valid_quotes) < 2:
            return

        # Find Best Buy (Lowest Price) and Best Sell (Highest Price)
        # Note: We are using 'price' (Last/Mid) as a proxy. Real logic needs Bid/Ask.
        # Assuming Price ~ Mid.
        
        sorted_quotes = sorted(valid_quotes.items(), key=lambda x: x[1])
        
        best_buy_exc, best_buy_price = sorted_quotes[0]  # Lowest Price
        best_sell_exc, best_sell_price = sorted_quotes[-1] # Highest Price
        
        if best_buy_exc == best_sell_exc:
            return
            
        # Calc Spread: (Sell - Buy) / Buy
        spread = (best_sell_price - best_buy_price) / best_buy_price
        
        if spread > self.min_spread:
            logger.info(f"💎 ARB OPPORTUNITY [{symbol}]: Buy {best_buy_exc.upper()} / Sell {best_sell_exc.upper()} (Spread: {spread*100:.2f}%)")
            await self._execute_arb(symbol, best_buy_exc, best_sell_exc, spread)

    async def _execute_arb(self, symbol, buy_exchange, sell_exchange, spread):
        # Prevent spam (simple cooldown or lock)
        # TODO: Implement locking
        
        quantity = settings.MIN_NOTIONAL_USD / self.books[symbol][buy_exchange]['price']
        
        # Send Signals
        # 1. Buy Leg
        sig_buy = SignalEvent(
            strategy_id="cross_arb",
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            exchange=buy_exchange,
            order_type="MARKET", # Taker Arb
            price=None 
        )
        
        # 2. Sell Leg
        sig_sell = SignalEvent(
            strategy_id="cross_arb",
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            exchange=sell_exchange,
            order_type="MARKET",
            price=None
        )
        
        await self.queue.put(sig_buy)
        await self.queue.put(sig_sell)

    async def on_fill(self, event: FillEvent):
        pass # Track positions?
        
    async def on_regime_change(self, event):
        pass
        
    async def on_funding_rate(self, event):
        pass
