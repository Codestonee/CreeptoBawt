from enum import Enum, auto

class AlertLevel(Enum):
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()

class AlertManager:
    async def send_alert(self, level, title, message, context=None):
        pass

_alert_manager = AlertManager()

def init_alert_manager(*args, **kwargs):
    return _alert_manager

def get_alert_manager():
    return _alert_manager
