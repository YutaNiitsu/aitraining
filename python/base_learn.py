import torch
from pathlib import Path
from config_loader import load_config

class BaseLearn:
    def __init__(self):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        # 設定ファイル読み込み
        self.config_learn = load_config(Path(__file__).resolve().parent.parent / 'config' / 'learn.yaml')
        self.config_labels = load_config(Path(__file__).resolve().parent.parent / 'config' / 'labels.yaml')
        