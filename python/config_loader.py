import yaml
import os
import sys
from pathlib import Path

def load_config(path):
    if not os.path.exists(path):
        print(f"❌ 設定ファイルが見つかりません: {path}")
        sys.exit(1)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"❌ YAMLの構文エラー: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 設定ファイルの読み込み中に予期せぬエラーが発生しました: {e}")
        sys.exit(1)

    return config

config_learn = load_config(Path(__file__).resolve().parent.parent / 'config' / 'learn.yaml')
config_labels = load_config(Path(__file__).resolve().parent.parent / 'config' / 'labels.yaml')