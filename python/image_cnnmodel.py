import torch.nn as nn
import torch

# モデル定義
class CNNModel(nn.Module):
    # ctgy_num:出力クラス数
    def __init__(self, ctgy_num):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 入力3ch → 出力32ch
            nn.ReLU(),                                   # 活性化関数
            nn.MaxPool2d(2),                             # 空間サイズを半分に
            nn.AdaptiveAvgPool2d((64, 64))               # 常に 64×64 に変換
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 64 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, ctgy_num)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x