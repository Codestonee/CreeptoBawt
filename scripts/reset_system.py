import os
import glob
import time

def reset_system():
    print("🧹 STARTING CREEPTBAWS SYSTEM RESET...")
    print("---------------------------------------")
    
    files_to_delete = [
        "trading_data.db",
        "event_store.db",
        "bot_execution.log",
        "dashboard_log.txt",
        "strategy_state.json",
        "health_status.json",
        "shadow_book_snapshot.json",
        "STOP_SIGNAL",
        "PAUSE_SIGNAL"
    ]
    
    # Add globs if needed
    for f in files_to_delete:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"✅ Deleted: {f}")
            except Exception as e:
                print(f"❌ Failed to delete {f}: {e}")
        else:
            print(f"⚪ Skipped (not found): {f}")
            
    # Also clear pycache to be super clean
    if os.path.exists("dashboard/__pycache__"):
        import shutil
        try:
            shutil.rmtree("dashboard/__pycache__")
            print("✅ Cleared dashboard cache")
        except:
            pass
            
    print("---------------------------------------")
    print("✨ RESET COMPLETE!")
    print("   Initial Capital set to: $500.00")
    print("   Ready for fresh run.")

if __name__ == "__main__":
    confirm = input("⚠️  This will DELETE all trading history and logs. Type 'yes' to confirm: ")
    if confirm.lower() == "yes":
        reset_system()
    else:
        print("❌ Cancelled.")
