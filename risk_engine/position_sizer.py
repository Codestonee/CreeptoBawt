import logging

logger = logging.getLogger("Risk.Sizer")

class PositionSizer:
    def __init__(self, risk_per_trade_pct=0.01):
        self.risk_per_trade_pct = risk_per_trade_pct  # Riskera 1% per trade

    def calculate_size(self, account_balance, entry_price, stop_loss_price):
        """
        Beräknar positionsstorlek baserat på risk.
        Formel: (Saldo * Risk%) / (Entry - StopLoss)
        """
        if stop_loss_price is None or entry_price is None:
            # Fallback om strategin inte ger stop loss: Använd fast % av saldo (farligare)
            # Ex: Köp för 10% av saldot
            safe_size_usd = account_balance * 0.10 
            return safe_size_usd / entry_price

        risk_amount = account_balance * self.risk_per_trade_pct
        price_diff = abs(entry_price - stop_loss_price)
        
        if price_diff == 0:
            return 0.0

        position_size_contracts = risk_amount / price_diff
        
        # Validering: Inte större än t.ex. 50% av kontot (för att undvika extrem hävstång)
        max_position_value = account_balance * 0.50
        position_value = position_size_contracts * entry_price
        
        if position_value > max_position_value:
            logger.info(f"Position size capped by max exposure limit.")
            position_size_contracts = max_position_value / entry_price

        return position_size_contracts