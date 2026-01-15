class PerformanceTracker:
    pass

_tracker = PerformanceTracker()

def get_performance_tracker(*args, **kwargs):
    return _tracker
