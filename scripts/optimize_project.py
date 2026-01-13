import os
import shutil
import glob
import sqlite3

def optimize_project():
    print("🧹 STARTING PROJECT OPTIMIZATION...")
    print("---------------------------------------")
    
    root_dir = os.getcwd()
    
    # 1. DELETE PYCACHE (Recursive)
    print("1️⃣  Cleaning __pycache__...")
    count = 0
    for root, dirs, files in os.walk(root_dir):
        if "__pycache__" in dirs:
            path = os.path.join(root, "__pycache__")
            try:
                shutil.rmtree(path)
                print(f"   Deleted: {path}")
                count += 1
            except Exception as e:
                print(f"   ❌ Error deleting {path}: {e}")
    print(f"   ✅ Removed {count} __pycache__ directories.")
    
    # 2. DELETE TEMP FILES
    print("\n2️⃣  Removing temporary files...")
    temp_files = [
        "startup_pydantic.log",
        "bot_pid.txt",
        "dash_pid.txt",
        "test_results.txt",
        "okx_auth_result.txt",
        "bot_execution.log", # Double check
        "dashboard_log.txt",
        "*.tmp",
        "*.bak"
    ]
    
    for pattern in temp_files:
        for f in glob.glob(pattern):
            try:
                os.remove(f)
                print(f"   Deleted: {f}")
            except Exception as e:
                print(f"   ❌ Error deleting {f}: {e}")

    # 3. VACUUM DATABASE (If exists)
    print("\n3️⃣  Optimizing Databases (VACUUM)...")
    dbs = ["trading_data.db", "event_store.db"]
    for db in dbs:
        if os.path.exists(db):
            try:
                conn = sqlite3.connect(db)
                conn.execute("VACUUM")
                conn.close()
                print(f"   ✅ Vacuumed {db}")
            except Exception as e:
                print(f"   ⚠️ Could not vacuum {db} (might be locked): {e}")

    # 4. ORGANIZE UTILITY SCRIPTS
    print("\n4️⃣  Organizing Utility Scripts...")
    scripts_dir = os.path.join(root_dir, "scripts")
    if not os.path.exists(scripts_dir):
        os.makedirs(scripts_dir)
        print(f"   Created {scripts_dir}")
        
    utils_to_move = [
        "SystemDryRun.py",
        "check_positions.py",
        "close_positions.py",
        "close_all_positions.py",
        "reset_system.py",
        "test_okx_auth.py", 
        "test_telegram.py",
        "validate_components.py"
    ]
    
    for script in utils_to_move:
        if os.path.exists(script):
            try:
                shutil.move(script, os.path.join(scripts_dir, script))
                print(f"   Moved {script} -> scripts/")
            except Exception as e:
                print(f"   ❌ Error moving {script}: {e}")

    print("---------------------------------------")
    print("✨ OPTIMIZATION COMPLETE!")

if __name__ == "__main__":
    optimize_project()
