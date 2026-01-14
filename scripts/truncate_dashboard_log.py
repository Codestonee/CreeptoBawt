import os

LOG_FILE = "logs/dashboard_log.txt"
KEEP_LINES = 1000

def truncate_log():
    try:
        if not os.path.exists(LOG_FILE):
            print(f"Log file {LOG_FILE} not found.")
            return

        size_mb = os.path.getsize(LOG_FILE) / (1024 * 1024)
        print(f"Current size: {size_mb:.2f} MB")
        
        if size_mb < 1:
            print("Log file is small enough. Skipping truncation.")
            return

        print(f"Reading last {KEEP_LINES} lines...")
        
        lines = []
        with open(LOG_FILE, 'rb') as f:
            f.seek(0, os.SEEK_END)
            end_pos = f.tell()
            # Read last 1MB to be safe
            read_len = min(end_pos, 1 * 1024 * 1024) 
            f.seek(max(0, end_pos - read_len))
            lines = f.readlines()
            
            if len(lines) > KEEP_LINES:
                lines = lines[-KEEP_LINES:]

        print(f"Truncating to {len(lines)} lines...")
        
        try:
            with open(LOG_FILE, 'wb') as f:
                f.writelines(lines)
            print("✅ Truncation complete.")
            new_size = os.path.getsize(LOG_FILE) / (1024 * 1024)
            print(f"New size: {new_size:.2f} MB")
        except PermissionError:
            print("❌ ERROR: File is locked. Please stop the bot first.")
            
    except Exception as e:
        print(f"❌ Error during truncation: {e}")

if __name__ == "__main__":
    truncate_log()
