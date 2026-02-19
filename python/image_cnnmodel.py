import torch.nn as nn
import torch

# モデル定義
class CNNModel(nn.Module):
    def __init__(self, ctgy_num):
        try:
            if not isinstance(ctgy_num, int):
                raise TypeError("ctgy_num は int である必要があります")
            if ctgy_num <= 0:
                raise ValueError("ctgy_num は 1 以上である必要があります")
        except Exception as e:
            raise RuntimeError(f"ctgy_num の検証に失敗しました: {e}")

        super().__init__()

        # Conv ブロック
        try:
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((64, 64))
            )
        except Exception as e:
            raise RuntimeError(f"畳み込み層(conv) の初期化に失敗しました: {e}")

        # 全結合層
        try:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32 * 64 * 64, 128),
                nn.ReLU(),
                nn.Linear(128, ctgy_num)
            )
        except Exception as e:
            raise RuntimeError(f"全結合層(fc) の初期化に失敗しました: {e}")

    def forward(self, x):
        try:
            x = self.conv[0](x)  # Conv2d
        except Exception as e:
            raise RuntimeError(f"Conv2d でエラーが発生しました: {e}")

        try:
            x = self.conv[1](x)  # ReLU
        except Exception as e:
            raise RuntimeError(f"ReLU でエラーが発生しました: {e}")

        try:
            x = self.conv[2](x)  # MaxPool2d
        except Exception as e:
            raise RuntimeError(f"MaxPool2d でエラーが発生しました: {e}")

        try:
            x = self.conv[3](x)  # AdaptiveAvgPool2d
        except Exception as e:
            raise RuntimeError(f"AdaptiveAvgPool2d でエラーが発生しました: {e}")

        try:
            x = self.fc(x)  # Flatten → Linear → ReLU → Linear
        except Exception as e:
            raise RuntimeError(f"全結合層 (fc) でエラーが発生しました: {e}")

        return x
