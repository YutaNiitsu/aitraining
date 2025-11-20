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
            nn.MaxPool2d(2)                              # 空間サイズを半分に
        )
        # fcは後でin_featuresを決める
        self.fc1 = nn.Linear(0, 128)  # ダミー、後で置き換える
        self.fc2 = nn.Linear(128, ctgy_num)

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        if self.fc1.in_features == 0:  # 初回だけin_featuresを決定
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x