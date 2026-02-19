import yaml
import os
import sys
from pathlib import Path
import logging.config

class ConfigManager:
    def __init__(self, learn_conf_path, log_conf_path, labels_path):
        # learn_conf_path のチェック
        try:
            if learn_conf_path is None:
                raise ValueError("learn_conf_path が None です")
            self.learn_conf_path = Path(learn_conf_path)
        except Exception as e:
            raise RuntimeError(f"learn_conf_path の初期化に失敗しました: {e}")

        # log_conf_path のチェック
        try:
            if log_conf_path is None:
                raise ValueError("log_conf_path が None です")
            self.log_conf_path = Path(log_conf_path)
        except Exception as e:
            raise RuntimeError(f"log_conf_path の初期化に失敗しました: {e}")

        # labels_path のチェック
        try:
            if labels_path is None:
                raise ValueError("labels_path が None です")
            self.labels_path = Path(labels_path)
        except Exception as e:
            raise RuntimeError(f"labels_path の初期化に失敗しました: {e}")

        self.last_timestamp = None


    def load_config(self):
        # learn_conf_path 読み込み
        try:
            with open(self.learn_conf_path, 'r', encoding='utf-8') as f:
                self.learn_config = yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError(f"学習設定ファイルが存在しません: {self.learn_conf_path}")
        except PermissionError:
            raise RuntimeError(f"学習設定ファイルにアクセスできません: {self.learn_conf_path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"学習設定ファイルの YAML 構文エラー: {e}")
        except Exception as e:
            raise RuntimeError(f"学習設定ファイルの読み込み中に予期せぬエラー: {e}")

        # log_conf_path 読み込み
        try:
            with open(self.log_conf_path, 'r', encoding='utf-8') as f:
                self.log_config = yaml.safe_load(f)
                logging.config.dictConfig(self.log_config)
        except FileNotFoundError:
            raise RuntimeError(f"ログ設定ファイルが存在しません: {self.log_conf_path}")
        except PermissionError:
            raise RuntimeError(f"ログ設定ファイルにアクセスできません: {self.log_conf_path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"ログ設定ファイルの YAML 構文エラー: {e}")
        except Exception as e:
            raise RuntimeError(f"ログ設定ファイルの読み込み中に予期せぬエラー: {e}")

        # labels_path 読み込み
        try:
            with open(self.labels_path, 'r', encoding='utf-8') as f:
                self.labels = yaml.safe_load(f)
        except FileNotFoundError:
            raise RuntimeError(f"ラベルファイルが存在しません: {self.labels_path}")
        except PermissionError:
            raise RuntimeError(f"ラベルファイルにアクセスできません: {self.labels_path}")
        except yaml.YAMLError as e:
            raise RuntimeError(f"ラベルファイルの YAML 構文エラー: {e}")
        except Exception as e:
            raise RuntimeError(f"ラベルファイルの読み込み中に予期せぬエラー: {e}")


    def is_updated(self):
        current_ts = self.learn_conf_path.stat().st_mtime
        if self.last_timestamp is None:
            return True  # 初回は常に更新されたとみなす
        return current_ts > self.last_timestamp

    def update_timestamp(self):
        self.last_timestamp = self.learn_conf_path.stat().st_mtime