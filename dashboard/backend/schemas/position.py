"""Position schemas."""
from pydantic import BaseModel
from typing import List


class Position(BaseModel):
    """Single position."""
    symbol: str
    side: str
    quantity: str
    entry_price: str
    current_price: str
    unrealized_pnl: str
    unrealized_pnl_pct: str


class PositionsResponse(BaseModel):
    """Positions response."""
    positions: List[Position]
