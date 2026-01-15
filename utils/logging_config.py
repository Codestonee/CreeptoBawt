import logging
import sys
import os
from datetime import datetime

class CleanFormatter(logging.Formatter):
    """Clean, readable log format for console."""
    
    # Emoji indicators for quick visual scanning
    LEVEL_ICONS = {
        'DEBUG': '🔍',
        'INFO': '📋',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }
    
    def format(self, record):
        # Shorten module names: "Strategy.AvellanedaStoikov" -> "MM"
        name_map = {
            'Strategy.AvellanedaStoikov': 'MM',
            'Execution.Binance': 'BIN',
            'Execution.OKX': 'OKX',
            'Execution.OrderManager': 'ORD',
            'Execution.Reconciliation': 'SYNC',
            'Core.Engine': 'ENG',
            'Data.ShadowBook': 'BOOK',
            'Data.CandleProvider': 'CANDLE',
            'Analysis.Regime': 'REGIME',
            'Main': 'MAIN',
            'Utils.TelegramAlerts': 'TELEGR',
            'Strategy.FundingArb': 'ARB',
            'PositionTracker': 'POS',
            'Database': 'DB'
        }
        
        # Smart abbreviation if not in map
        if record.name in name_map:
            short_name = name_map[record.name]
        else:
            parts = record.name.split('.')
            short_name = parts[-1][:6].upper()
        
        # Time only (HH:MM:SS)
        time_str = self.formatTime(record, '%H:%M:%S')
        
        # Icon for level
        icon = self.LEVEL_ICONS.get(record.levelname, '')
        
        return f"{time_str} [{short_name:6}] {icon} {record.getMessage()}"

class NoiseFilter(logging.Filter):
    """Filter out repetitive or noisy logs."""
    
    def filter(self, record):
        msg = record.getMessage()
        
        # Suppress repetitive regime updates if they are UNCERTAIN (too frequent)
        if "REGIME" in record.name and "UNCERTAIN" in msg:
            # Only show 1 in 10 or suppress entirely? 
            # For now, suppress distinct 'UNCERTAIN' logs if they don't add value
            # But keep transition logs "Regime changed: ... -> UNCERTAIN"
            if "MARKET REGIME CHANGE" not in msg and "Regime:" not in msg:
                return False

        return True

def setup_logging(log_level: str = "INFO"):
    """
    Setup centralized logging configuration.
    
    Args:
        log_level: Default log level (DEBUG, INFO, etc)
    """
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Root logger config
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all, handlers will filter by level
    
    # Remove existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 1. Console Handler (Clean, INFO+)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(CleanFormatter())
    console_handler.addFilter(NoiseFilter())
    root_logger.addHandler(console_handler)

    # 2. File Handler (Detailed, DEBUG)
    # Using simple FileHandler to avoid Windows permissions issues with rotation
    file_handler = logging.FileHandler(
        "logs/bot_execution.log", 
        encoding='utf-8', 
        mode='a'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    root_logger.addHandler(file_handler)

    # 3. Dashboard Handler (Condensed, INFO)
    dashboard_handler = logging.FileHandler(
        "logs/dashboard_log.txt", 
        encoding='utf-8',
        mode='w' # Overwrite on restart for clean dash
    )
    dashboard_handler.setLevel(logging.INFO)
    dashboard_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'
    ))
    root_logger.addHandler(dashboard_handler)

    # 4. Quiet down noisy libraries
    noisy_modules = [
        'asyncio', 
        'aiosqlite', 
        'urllib3', 
        'websockets',
        'Execution.Reconciliation', 
        'Data.ShadowBook', 
        'Data.CandleProvider', 
        'Utils.TimeSync', 
        'Utils.NonceService', 
        'Core.EventStore',
        'Analysis.VPIN' # VPIN warmup logs
    ]
    
    for module in noisy_modules:
        logging.getLogger(module).setLevel(logging.WARNING)
        
    logging.getLogger("Main").info("✅ Logging system initialized")
