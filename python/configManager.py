import yaml
import os
import sys
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.last_timestamp = None

    def load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"❌ YAMLの構文エラー: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 設定ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
            sys.exit(1)

    def is_updated(self):
        current_ts = self.config_path.stat().st_mtime
        if self.last_timestamp is None:
            return True  # 初回は常に更新されたとみなす
        return current_ts > self.last_timestamp

    def update_timestamp(self):
        self.last_timestamp = self.config_path.stat().st_mtime