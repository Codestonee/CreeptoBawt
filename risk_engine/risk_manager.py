import logging
from core.events import SignalEvent
from risk_engine.circuit_breaker import CircuitBreaker
from risk_engine.position_sizer import PositionSizer

logger = logging.getLogger("Risk.Manager")

class RiskManager:
    def __init__(self, account_balance=1000.0, max_drawdown_pct=0.10):
        self.initial_balance = account_balance
        self.current_balance = account_balance
        self.max_drawdown_pct = max_drawdown_pct  # 10% max total förlust
        
        self.circuit_breaker = CircuitBreaker()
        self.sizer = PositionSizer()
        
        self.kill_switch_triggered = False

    def validate_signal(self, signal: SignalEvent) -> bool:
        """Godkänner eller avvisar nya signaler."""
        
        # 1. Om Kill Switch är aktiv -> Inga nya affärer!
        if self.kill_switch_triggered:
            logger.warning("Signal REJECTED: Kill Switch is ACTIVE.")
            return False

        # 2. Kolla Circuit Breaker (Daglig förlust)
        if not self.circuit_breaker.can_trade():
            logger.warning("Signal REJECTED: Circuit Breaker is tripped.")
            return False

        # 3. Beräkna storlek
        if signal.quantity <= 0:
            stop_loss = signal.price * 0.99 if signal.side == 'BUY' else signal.price * 1.01
            signal.quantity = self.sizer.calculate_size(
                self.current_balance, 
                signal.price, 
                stop_loss
            )
            # Logga bara om det faktiskt blev en kvantitet
            if signal.quantity > 0:
                logger.info(f"Risk Manager calculated quantity: {signal.quantity:.4f}")

        # 4. Sanity Check
        if signal.quantity * signal.price > self.current_balance * 2:
             logger.warning(f"Signal REJECTED: Order value > 2x Balance.")
             return False

        return True

    def check_account_health(self, total_equity):
        """
        Kollar om kontot mår bra. Anropas av Engine vid varje tick.
        total_equity = Saldo + Orealiserad PnL
        """
        if self.kill_switch_triggered:
            return False

        # Räkna ut total drawdown från start
        drawdown = (self.initial_balance - total_equity) / self.initial_balance
        
        if drawdown >= self.max_drawdown_pct:
            logger.critical(f"🚨 KILL SWITCH TRIGGERED! Drawdown: {drawdown*100:.2f}% (Limit: {self.max_drawdown_pct*100}%)")
            self.kill_switch_triggered = True
            return False
            
        return True

    def record_trade_result(self, pnl):
        """Uppdatera saldo och Circuit Breaker."""
        self.current_balance += pnl
        self.circuit_breaker.update_trade_result(pnl, self.current_balance)