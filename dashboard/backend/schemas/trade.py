"""Trade schemas."""
from pydantic import BaseModel
from typing import List


class Trade(BaseModel):
    """Single trade."""
    id: str
    symbol: str
    side: str
    price: str
    quantity: str
    fee: str
    realized_pnl: str
    executed_at: str


class TradesResponse(BaseModel):
    """Trades response."""
    trades: List[Trade]
