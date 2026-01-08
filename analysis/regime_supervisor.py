import logging
import numpy as np
from collections import defaultdict
from core.events import MarketEvent, RegimeEvent
from analysis.indicators import calculate_adx, calculate_atr

logger = logging.getLogger("Analysis.Regime")

class RegimeSupervisor:
    def __init__(self, event_queue):
        self.queue = event_queue
        # Lagra candles per symbol: { 'btcusdt': { 'high': [], 'low': [], 'close': [] } }
        self.history = defaultdict(lambda: {'high': [], 'low': [], 'close': [], 'current_candle': None})
        self.last_regime = defaultdict(lambda: None)

    async def update(self, event: MarketEvent):
        """Tar emot en tick och uppdaterar candle-historik."""
        symbol = event.symbol
        price = event.price
        timestamp = event.timestamp
        
        # Enkel tidsbaserad candle-konstruktion (1-minut)
        current_minute = int(timestamp // 60)
        
        state = self.history[symbol]
        
        # Initiera candle om ingen finns
        if state['current_candle'] is None:
            state['current_candle'] = {
                'minute': current_minute,
                'open': price,
                'high': price,
                'low': price,
                'close': price
            }
            return

        candle = state['current_candle']
        
        # Om vi är kvar i samma minut -> Uppdatera High/Low/Close
        if current_minute == candle['minute']:
            candle['high'] = max(candle['high'], price)
            candle['low'] = min(candle['low'], price)
            candle['close'] = price
            
        # Ny minut -> Stäng föregående candle och analysera
        else:
            # Spara stängd candle till historik
            state['high'].append(candle['high'])
            state['low'].append(candle['low'])
            state['close'].append(candle['close'])
            
            # Begränsa historik till 100 candles för minnesoptimering
            if len(state['close']) > 100:
                state['high'].pop(0)
                state['low'].pop(0)
                state['close'].pop(0)
            
            logger.info(f"Candle closed for {symbol}: Close={candle['close']}")
            
            # Starta ny candle
            state['current_candle'] = {
                'minute': current_minute,
                'open': price,
                'high': price,
                'low': price,
                'close': price
            }
            
            # Kör analys
            await self._analyze_regime(symbol)

    async def _analyze_regime(self, symbol):
        """Beräknar ADX/ATR och publicerar RegimeEvent."""
        state = self.history[symbol]
        highs = np.array(state['high'])
        lows = np.array(state['low'])
        closes = np.array(state['close'])
        
        # Behöver minst 30 candles för vettig ADX
        if len(closes) < 30:
            return

        try:
            adx = calculate_adx(highs, lows, closes)
            volatility = calculate_atr(highs, lows, closes)[-1]
            
            # Klassificering enligt din strategi
            if adx > 25:
                regime = "TRENDING"
            elif adx < 20:
                regime = "RANGING"
            else:
                regime = "UNCERTAIN"
                
            # Publicera bara om regim ändras eller var 5:e minut (för keep-alive)
            if regime != self.last_regime[symbol]:
                self.last_regime[symbol] = regime
                
                logger.info(f"REGIME DETECTED [{symbol}]: {regime} (ADX={adx:.2f})")
                
                event = RegimeEvent(
                    symbol=symbol,
                    regime=regime,
                    adx=adx,
                    volatility=volatility
                )
                await self.queue.put(event)
                
        except Exception as e:
            logger.error(f"Error analyzing regime for {symbol}: {e}")