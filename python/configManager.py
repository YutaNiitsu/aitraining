import yaml
import os
import sys
from pathlib import Path
import logging.config

class ConfigManager:
    def __init__(self, learn_conf_path, log_conf_path, labels_path):
        self.learn_conf_path = Path(learn_conf_path)
        self.log_conf_path = Path(log_conf_path)
        self.labels_path = Path(labels_path)
        self.last_timestamp = None

    def load_config(self):
        try:
            with open(self.learn_conf_path, 'r', encoding='utf-8') as f:
                self.learn_config = yaml.safe_load(f)
            with open(self.log_conf_path, 'r', encoding='utf-8') as f:
                log_config = yaml.safe_load(f)
                logging.config.dictConfig(log_config)
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self.labels = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"❌ YAMLの構文エラー: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 設定ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
            sys.exit(1)

    def is_updated(self):
        current_ts = self.learn_conf_path.stat().st_mtime
        if self.last_timestamp is None:
            return True  # 初回は常に更新されたとみなす
        return current_ts > self.last_timestamp

    def update_timestamp(self):
        self.last_timestamp = self.learn_conf_path.stat().st_mtime