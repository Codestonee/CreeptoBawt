"""Balance schemas."""
from pydantic import BaseModel
from typing import List


class Balance(BaseModel):
    """Single currency balance."""
    currency: str
    available: str
    locked: str
    total: str
    usd_value: str


class BalancesResponse(BaseModel):
    """Balances response."""
    balances: List[Balance]
    total_usd_value: str
