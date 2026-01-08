import logging
import asyncio
import random
from core.events import MarketEvent, SignalEvent

logger = logging.getLogger("Strategy.Dummy")

class DummyStrategy:
    def __init__(self, event_queue):
        self.queue = event_queue
        self.symbol = "btcusdt"
        self.last_trade = 0

    async def on_tick(self, event: MarketEvent):
        # Handla bara var 10:e sekund för test
        now = asyncio.get_event_loop().time()
        if now - self.last_trade < 10:
            return

        # Slumpmässig signal
        if random.random() > 0.5:
            side = "BUY"
            price = event.price * 0.99 # Limit order lite under marknad
        else:
            side = "SELL"
            price = event.price * 1.01

        logger.info(f"DUMMY STRATEGY: Generating {side} signal!")
        
        signal = SignalEvent(
            strategy_id="dummy_v1",
            symbol=self.symbol,
            side=side,
            quantity=0.001, # Liten mängd
            price=price
        )
        await self.queue.put(signal)
        self.last_trade = now