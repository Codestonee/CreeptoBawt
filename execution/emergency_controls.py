"""
Emergency Controls - Panic Buttons & Manual Overrides

Use this module to manually interact with the running bot in case of emergency.
Run these commands from a separate python script or REPL.
"""

import os
import sys
import time
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmergencyControls")

SIGNAL_DIR = "signals"
STOP_FILE = "STOP_ALL"
PAUSE_FILE = "PAUSE_SIGNAL"

def _ensure_signal_dir():
    if not os.path.exists(SIGNAL_DIR):
        os.makedirs(SIGNAL_DIR)

def emergency_stop():
    """
    CRITICAL: Immediately stops the bot and flattens all positions.
    Writes a STOP_ALL signal that the engine watches for.
    """
    _ensure_signal_dir()
    path = os.path.join(SIGNAL_DIR, STOP_FILE)
    
    with open(path, "w") as f:
        f.write(f"MANUAL_STOP_TRIGGERED_AT_{time.time()}")
        
    logger.critical(f"🚨 EMERGENCY STOP TRIGGERED! Signal written to {path}")
    logger.critical("The bot should pick up this signal within 1 second, cancel orders, and flatten positions.")

def emergency_pause():
    """
    Pauses all NEW trading. Existing orders manage themselves.
    Useful if you see high volatility but aren't sure if you should kill it.
    """
    _ensure_signal_dir()
    path = os.path.join(SIGNAL_DIR, PAUSE_FILE)
    
    with open(path, "w") as f:
        f.write(f"MANUAL_PAUSE_TRIGGERED_AT_{time.time()}")
        
    logger.warning(f"⏸️ PAUSE SIGNAL TRIGGERED! Signal written to {path}")
    logger.warning("Bot will NOT place new orders until resume() is called.")

def resume_trading():
    """
    Resumes trading from a PAUSED state.
    Does NOT recover from a STOP state (you must restart the bot for that).
    """
    path = os.path.join(SIGNAL_DIR, PAUSE_FILE)
    
    if os.path.exists(path):
        os.remove(path)
        logger.info("▶️ RESUME SIGNAL SENT. Pause file removed.")
    else:
        logger.info("Bot was not paused.")

def force_kill_process():
    """
    OS-level kill. Use only if bot is frozen and ignoring signals.
    WARNING: Does NOT close positions!
    """
    import psutil
    
    current_pid = os.getpid()
    
    # Try to find bot process (heuristic)
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Check for python process running main.py
            cmdline = proc.info['cmdline']
            if cmdline and 'python' in proc.info['name'] and any('main.py' in arg for arg in cmdline):
                if proc.info['pid'] != current_pid:
                    logger.warning(f"Killing bot process PID {proc.info['pid']}...")
                    proc.kill()
                    logger.info("Process killed.")
                    return
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    logger.error("Could not find running bot process to kill.")

if __name__ == "__main__":
    # simple CLI
    if len(sys.argv) < 2:
        print("Usage: python emergency_controls.py [stop|pause|resume|kill]")
        sys.exit(1)
        
    cmd = sys.argv[1].lower()
    
    if cmd == "stop":
        confirm = input("Are you sure you want to STOP and FLATTEN? (yes/no): ")
        if confirm.lower() == "yes":
            emergency_stop()
    elif cmd == "pause":
        emergency_pause()
    elif cmd == "resume":
        resume_trading()
    elif cmd == "kill":
        confirm = input("Are you sure you want to FORCE KILL (Positions stay open)? (yes/no): ")
        if confirm.lower() == "yes":
            force_kill_process()
    else:
        print(f"Unknown command: {cmd}")
