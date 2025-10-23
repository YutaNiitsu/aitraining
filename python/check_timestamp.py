import os
import time
import subprocess

CONFIG_PATH = "/mnt/c/Users/yniit/Documents/aitraining/config/learn.yaml"
STAMP_PATH = ".last_config_timestamp"

def get_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0

def load_last_timestamp():
    if os.path.exists(STAMP_PATH):
        with open(STAMP_PATH, "r") as f:
            return float(f.read().strip())
    return 0

def save_timestamp(ts):
    with open(STAMP_PATH, "w") as f:
        f.write(str(ts))

def check_timestamp_update():
    current_ts = get_mtime(CONFIG_PATH)
    last_ts = load_last_timestamp()

    if current_ts > last_ts:
        print("設定ファイルが更新されました。")
        save_timestamp(current_ts)
        return True
    else:
        print("設定ファイルは更新されていません。")
        return False