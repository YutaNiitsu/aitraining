import torch.nn as nn

# モデル定義
class CNNModel(nn.Module):

    def __init__(self, ctgy_num):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 入力3ch → 出力32ch
            nn.ReLU(),                                   # 活性化関数
            nn.MaxPool2d(2)                              # 空間サイズを半分に
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                              # 32×32×32 → 1次元ベクトル
            nn.Linear(32 * 32 * 32, 128),              # 中間層（128ユニット）
            nn.ReLU(),                                 # 活性化
            nn.Linear(128, ctgy_num)                   # 出力層（ctgy_numは分類する数)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x