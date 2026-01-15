import logging
import time
import asyncio
from typing import Dict, Optional
from collections import defaultdict
from core.events import MarketEvent, SignalEvent, FillEvent, RegimeEvent, FundingRateEvent
from strategies.base import BaseStrategy
from config.settings import settings

logger = logging.getLogger("Strategy.CrossArb")


class CrossExchangeArbStrategy(BaseStrategy):
    """
    Cross-Exchange Arbitrage Strategy.
    
    Monitors spreads across connected exchanges using proper bid/ask prices.
    Executes when spread exceeds minimum threshold after accounting for fees.
    """
    
    def __init__(self, event_queue, symbols: list = None):
        super().__init__(event_queue, symbols or settings.TRADING_SYMBOLS)
        
        # Structure: {symbol: {exchange: {bid, ask, price, time}}}
        self.books = defaultdict(lambda: defaultdict(dict))
        
        self.min_spread = settings.ARBITRAGE_MIN_SPREAD
        self.position_exposure: Dict[str, float] = {}
        self.active_arbs: Dict[str, dict] = {}
        
        # Cooldown to prevent spam (per symbol)
        self._last_arb_time: Dict[str, float] = {}
        self._arb_cooldown_sec = 5.0
        
        # Fee estimates for profitability check
        self._taker_fee = 0.001  # 0.1% per side (conservative estimate)
        
        # Max concurrent arb legs
        self._max_active_arbs = 3

    async def on_tick(self, event: MarketEvent):
        """Update internal book and check for arb opportunities."""
        symbol = event.symbol.lower()
        if symbol not in self.symbols:
            return

        exchange = event.exchange.lower()
        
        # Store quote data - prefer bid/ask if available, fallback to price
        quote_data = {
            'time': time.time(),
            'price': event.price,
        }
        
        # Use actual bid/ask if available (proper arbitrage)
        if event.bid is not None and event.ask is not None:
            quote_data['bid'] = event.bid
            quote_data['ask'] = event.ask
            quote_data['has_bbo'] = True
        else:
            # Fallback: estimate bid/ask from price with conservative spread
            # This is RISKY for arbitrage - log warning
            estimated_spread = 0.001  # 10 bps estimated spread
            quote_data['bid'] = event.price * (1 - estimated_spread / 2)
            quote_data['ask'] = event.price * (1 + estimated_spread / 2)
            quote_data['has_bbo'] = False
        
        self.books[symbol][exchange] = quote_data
        
        await self._check_arb(symbol)

    async def _check_arb(self, symbol: str):
        """Find and validate arbitrage opportunities across exchanges."""
        now = time.time()
        
        # Cooldown check
        if now - self._last_arb_time.get(symbol, 0) < self._arb_cooldown_sec:
            return
            
        # Max active arbs check
        if len(self.active_arbs) >= self._max_active_arbs:
            return
        
        # Collect valid (non-stale) quotes
        valid_quotes = {}
        stale_threshold = 5.0  # 5 seconds
        
        for exc, data in self.books[symbol].items():
            if not data:
                continue
            if now - data.get('time', 0) > stale_threshold:
                continue
            valid_quotes[exc] = data
        
        if len(valid_quotes) < 2:
            return
        
        # Find best buy opportunity (lowest ask) and best sell (highest bid)
        # For arb: we BUY at the ASK and SELL at the BID
        best_buy_exc = None
        best_buy_ask = float('inf')
        best_sell_exc = None
        best_sell_bid = 0.0
        has_reliable_data = True
        
        for exc, data in valid_quotes.items():
            if not data.get('has_bbo', False):
                has_reliable_data = False
            
            if data['ask'] < best_buy_ask:
                best_buy_ask = data['ask']
                best_buy_exc = exc
            
            if data['bid'] > best_sell_bid:
                best_sell_bid = data['bid']
                best_sell_exc = exc
        
        # Can't arb on same exchange
        if best_buy_exc == best_sell_exc:
            return
        
        # Calculate net spread after fees
        # Profit = sell_bid - buy_ask - fees_on_both_sides
        gross_spread = (best_sell_bid - best_buy_ask) / best_buy_ask
        fee_cost = self._taker_fee * 2  # Both legs are taker
        net_spread = gross_spread - fee_cost
        
        if net_spread > self.min_spread:
            # Extra caution if we don't have real bid/ask
            if not has_reliable_data:
                logger.warning(
                    f"[{symbol}] Arb signal with estimated bid/ask - high risk! "
                    f"Spread: {net_spread*100:.3f}%"
                )
                # Require higher threshold for unreliable data
                if net_spread < self.min_spread * 2:
                    return
            
            logger.info(
                f"ARB OPPORTUNITY [{symbol}]: "
                f"Buy {best_buy_exc.upper()} @ {best_buy_ask:.4f} / "
                f"Sell {best_sell_exc.upper()} @ {best_sell_bid:.4f} "
                f"(Net Spread: {net_spread*100:.3f}%)"
            )
            
            await self._execute_arb(
                symbol, best_buy_exc, best_sell_exc, 
                best_buy_ask, best_sell_bid, net_spread
            )

    async def _execute_arb(
        self, 
        symbol: str, 
        buy_exchange: str, 
        sell_exchange: str,
        buy_price: float,
        sell_price: float,
        spread: float
    ):
        """Execute arbitrage by sending simultaneous buy and sell signals."""
        # Update cooldown
        self._last_arb_time[symbol] = time.time()
        
        # Calculate quantity based on min notional
        quantity = settings.MIN_NOTIONAL_USD / buy_price
        
        # Generate unique arb ID for tracking
        arb_id = f"arb_{symbol}_{int(time.time()*1000)}"
        
        # Track active arb
        self.active_arbs[arb_id] = {
            'symbol': symbol,
            'buy_exchange': buy_exchange,
            'sell_exchange': sell_exchange,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'quantity': quantity,
            'spread': spread,
            'status': 'pending',
            'created_at': time.time(),
            'buy_filled': False,
            'sell_filled': False,
        }
        
        # Send buy signal
        sig_buy = SignalEvent(
            strategy_id="cross_arb",
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            exchange=buy_exchange,
            order_type="MARKET",
            price=None,
            arb_id=arb_id
        )
        
        # Send sell signal
        sig_sell = SignalEvent(
            strategy_id="cross_arb",
            symbol=symbol,
            side="SELL",
            quantity=quantity,
            exchange=sell_exchange,
            order_type="MARKET",
            price=None,
            arb_id=arb_id
        )
        
        # Send both legs (ideally simultaneously)
        await asyncio.gather(
            self.queue.put(sig_buy),
            self.queue.put(sig_sell)
        )
        
        logger.info(f"[{arb_id}] Arb signals sent: BUY {buy_exchange} / SELL {sell_exchange}")

    async def on_fill(self, event: FillEvent):
        """Track fill events to update arb status."""
        # Find matching arb by checking active arbs
        for arb_id, arb_data in list(self.active_arbs.items()):
            if arb_data['symbol'] != event.symbol.lower():
                continue
            
            # Update fill status
            if event.side == 'BUY' and not arb_data['buy_filled']:
                arb_data['buy_filled'] = True
                arb_data['buy_fill_price'] = event.price
                logger.info(f"[{arb_id}] Buy leg filled @ {event.price}")
            elif event.side == 'SELL' and not arb_data['sell_filled']:
                arb_data['sell_filled'] = True
                arb_data['sell_fill_price'] = event.price
                logger.info(f"[{arb_id}] Sell leg filled @ {event.price}")
            
            # Check if arb is complete
            if arb_data['buy_filled'] and arb_data['sell_filled']:
                realized_pnl = (
                    arb_data['sell_fill_price'] - arb_data['buy_fill_price']
                ) * arb_data['quantity']
                arb_data['status'] = 'completed'
                arb_data['realized_pnl'] = realized_pnl
                logger.info(
                    f"[{arb_id}] Arb COMPLETED. PnL: ${realized_pnl:.4f}"
                )
                # Remove completed arb after short delay
                del self.active_arbs[arb_id]
                break
        
    async def on_regime_change(self, event):
        """Handle regime changes (could pause arb in high volatility)."""
        pass
        
    async def on_funding_rate(self, event):
        """Handle funding rate updates."""
        pass
    
    def get_active_arbs(self) -> Dict[str, dict]:
        """Return currently active arbitrage trades."""
        return dict(self.active_arbs)
    
    def cleanup_stale_arbs(self, max_age_sec: float = 60.0):
        """Remove arbs that have been pending too long."""
        now = time.time()
        stale = [
            arb_id for arb_id, data in self.active_arbs.items()
            if now - data['created_at'] > max_age_sec and data['status'] == 'pending'
        ]
        for arb_id in stale:
            logger.warning(f"[{arb_id}] Arb timed out - removing")
            del self.active_arbs[arb_id]
