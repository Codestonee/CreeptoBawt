"""PnL schemas."""
from pydantic import BaseModel


class PnLResponse(BaseModel):
    """PnL response."""
    total_pnl: str
    realized_pnl: str
    unrealized_pnl: str
    today_pnl: str
    today_pnl_pct: str
    fees_paid: str
    sharpe_ratio: str
    max_drawdown: str
