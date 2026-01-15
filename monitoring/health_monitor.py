class HealthMonitor:
    async def start(self):
        pass
    async def stop(self):
        pass

_health_mon = HealthMonitor()

def get_health_monitor():
    return _health_mon
