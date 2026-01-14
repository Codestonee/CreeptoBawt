import os
import shutil

LOG_FILE = "logs/bot_execution.log"
KEEP_LINES = 20000

def truncate_log():
    try:
        if not os.path.exists(LOG_FILE):
            print(f"Log file {LOG_FILE} not found.")
            return

        size_mb = os.path.getsize(LOG_FILE) / (1024 * 1024)
        print(f"Current size: {size_mb:.2f} MB")
        
        if size_mb < 10:
            print("Log file is small enough. Skipping truncation.")
            return

        print(f"Reading last {KEEP_LINES} lines...")
        
        # Read lines
        lines = []
        with open(LOG_FILE, 'rb') as f:
            # Efficient tailing for large files would be better, but for 300MB reading into memory 
            # might be okay for a script, or we seek. 
            # Let's use a simple approach: seek to near end and read
            f.seek(0, os.SEEK_END)
            end_pos = f.tell()
            # Estimate bytes per line (avg 150 bytes?) -> 20k lines ~ 3MB
            # Let's read last 10MB to be safe
            read_len = min(end_pos, 10 * 1024 * 1024) 
            f.seek(max(0, end_pos - read_len))
            lines = f.readlines()
            
            # Keep only requested count
            if len(lines) > KEEP_LINES:
                lines = lines[-KEEP_LINES:]

        print(f"Truncating to {len(lines)} lines...")
        
        # Write back (Overwrite)
        try:
            with open(LOG_FILE, 'wb') as f:
                f.writelines(lines)
            print("✅ Truncation complete.")
            new_size = os.path.getsize(LOG_FILE) / (1024 * 1024)
            print(f"New size: {new_size:.2f} MB")
        except PermissionError:
            print("❌ ERROR: File is locked (Bot likely running). Cannot truncate safely.")
            print("Action: Please stop the bot first, or rely on rotation implementation (next step).")
            
    except Exception as e:
        print(f"❌ Error during truncation: {e}")

if __name__ == "__main__":
    truncate_log()
