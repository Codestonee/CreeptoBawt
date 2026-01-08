import time
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("Risk.CircuitBreaker")

class CircuitBreaker:
    def __init__(self, max_daily_loss_pct=0.03, max_consecutive_losses=5):
        self.max_daily_loss_pct = max_daily_loss_pct  # Stoppa vid 3% daglig förlust
        self.max_consecutive_losses = max_consecutive_losses
        
        # State
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self.is_tripped = False
        self.last_reset = datetime.now().date()

    def update_trade_result(self, pnl_amount, account_balance):
        """Uppdaterar status efter en avslutad trade."""
        self._check_daily_reset()

        self.daily_pnl += pnl_amount
        
        if pnl_amount < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0  # Nollställ vid vinst

        # Kontroll 1: Daglig förlustgräns
        # Ex: Om saldo 1000 och vi backat -35 -> 3.5% förlust -> STOPP
        current_drawdown_pct = abs(min(0, self.daily_pnl)) / account_balance
        
        if current_drawdown_pct >= self.max_daily_loss_pct:
            logger.critical(f"CIRCUIT BREAKER TRIPPED: Daily Loss {current_drawdown_pct*100:.2f}% exceeds limit.")
            self.is_tripped = True
            return False

        # Kontroll 2: För många förluster i rad
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(f"CIRCUIT BREAKER WARNING: {self.consecutive_losses} consecutive losses. Pausing recommended.")
            # Här kan man välja att vara "soft" (pausa 1h) eller "hard" (stoppa helt)
            # För nu: Logga varning men stoppa inte helt, om inte strategin kräver det
            
        return True

    def can_trade(self):
        """Returnerar True om säkringen är hel."""
        self._check_daily_reset()
        return not self.is_tripped

    def _check_daily_reset(self):
        """Nollställer PnL vid midnatt (UTC)."""
        current_date = datetime.utcnow().date()
        if current_date > self.last_reset:
            logger.info("New day detected. Resetting Circuit Breaker PnL.")
            self.daily_pnl = 0.0
            self.last_reset = current_date
            # Valfritt: Återställ is_tripped om man vill tillåta handel nästa dag automatiskt
            # self.is_tripped = False