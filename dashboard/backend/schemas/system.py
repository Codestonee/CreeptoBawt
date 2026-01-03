"""System schemas."""
from pydantic import BaseModel
from typing import List


class SystemStatus(BaseModel):
    """System status response."""
    status: str
    mode: str
    uptime_seconds: int
    kill_switch_active: bool
    connected_exchanges: List[str]
    active_strategies: List[str]


class KillSwitchRequest(BaseModel):
    """Kill switch activation request."""
    activate: bool


class KillSwitchResponse(BaseModel):
    """Kill switch response."""
    success: bool
    kill_switch_active: bool
    orders_cancelled: int
