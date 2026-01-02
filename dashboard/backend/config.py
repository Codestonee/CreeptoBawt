"""Dashboard configuration."""
from pydantic import BaseModel


class DashboardConfig(BaseModel):
    """Dashboard configuration."""
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS settings
    allow_origins: list[str] = ["*"]
    
    # API settings
    api_prefix: str = "/api"
    
    # Mock data mode (until trading bridge is connected)
    use_mock_data: bool = True
