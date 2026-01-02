"""Order schemas."""
from pydantic import BaseModel
from typing import List


class Order(BaseModel):
    """Single order."""
    id: str
    symbol: str
    side: str
    type: str
    price: str
    quantity: str
    filled_quantity: str
    status: str
    created_at: str


class OrdersResponse(BaseModel):
    """Orders response."""
    orders: List[Order]
