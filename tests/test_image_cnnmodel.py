import pytest
import torch
import torch.nn as nn
from python.image_cnnmodel import CNNModel


# ============================================================
# __init__() のテスト
# ============================================================

# 2-1-1 正常：初期化
def test_init_success():
    model = CNNModel(3)
    assert isinstance(model.conv, nn.Sequential)
    assert isinstance(model.fc, nn.Sequential)


# 2-1-2 異常：ctgy_num が None
def test_init_ctgy_num_none():
    with pytest.raises(RuntimeError, match="ctgy_num の検証に失敗しました.*ctgy_num は int"):
        CNNModel(None)


# 2-1-3 異常：ctgy_num が負数
def test_init_ctgy_num_negative():
    with pytest.raises(RuntimeError, match="ctgy_num の検証に失敗しました.*ctgy_num は 1 以上"):
        CNNModel(-1)


# 2-1-4 異常：ctgy_num が 0
def test_init_ctgy_num_zero():
    with pytest.raises(RuntimeError, match="ctgy_num の検証に失敗しました.*ctgy_num は 1 以上"):
        CNNModel(0)


# 2-1-5 異常：ctgy_num が文字列
def test_init_ctgy_num_string():
    with pytest.raises(RuntimeError, match="ctgy_num の検証に失敗しました.*ctgy_num は int"):
        CNNModel("abc")


# 2-1-6 異常：畳み込み層(conv)  初期化失敗
def test_init_conv_error(monkeypatch):
    def fake_conv(*args, **kwargs):
        raise Exception("conv error")

    monkeypatch.setattr("torch.nn.Conv2d", fake_conv)

    with pytest.raises(RuntimeError, match="畳み込み層\\(conv\\) の初期化に失敗しました"):
        CNNModel(3)


# 2-1-7 異常：全結合層(fc) 初期化失敗
def test_init_fc_error(monkeypatch):
    def fake_linear(*args, **kwargs):
        raise Exception("linear error")

    monkeypatch.setattr("torch.nn.Linear", fake_linear)

    with pytest.raises(RuntimeError, match="全結合層\\(fc\\) の初期化に失敗しました"):
        CNNModel(3)


# ============================================================
# 2. forward() のテスト
# ============================================================

# 2-2-1 正常：推論
def test_forward_success():
    model = CNNModel(3)
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 3)


# 2-2-2 異常：Conv2d 失敗（入力チャンネル不一致）
def test_forward_conv_error():
    model = CNNModel(3)
    x = torch.randn(1, 1, 128, 128)  # 1ch → Conv2d(3ch) mismatch

    with pytest.raises(RuntimeError, match="Conv2d でエラーが発生しました"):
        model(x)


# 2-2-3 異常：ReLU 失敗
def test_forward_relu_error(monkeypatch):
    def fake_relu(*args, **kwargs):
        raise Exception("relu error")

    monkeypatch.setattr("torch.nn.ReLU.__call__", fake_relu)

    model = CNNModel(3)
    x = torch.randn(1, 3, 128, 128)

    with pytest.raises(RuntimeError, match="ReLU でエラーが発生しました"):
        model(x)


# 2-2-4 異常：MaxPool2d 失敗
def test_forward_maxpool_error(monkeypatch):
    def fake_pool(*args, **kwargs):
        raise Exception("pool error")

    monkeypatch.setattr("torch.nn.MaxPool2d.__call__", fake_pool)

    model = CNNModel(3)
    x = torch.randn(1, 3, 128, 128)

    with pytest.raises(RuntimeError, match="MaxPool2d でエラーが発生しました"):
        model(x)


# 2-2-5 異常：AdaptiveAvgPool2d 失敗 
def test_forward_avgpool_error(monkeypatch):
    def fake_avgpool(*args, **kwargs):
        raise Exception("avgpool error")

    monkeypatch.setattr("torch.nn.AdaptiveAvgPool2d.__call__", fake_avgpool)

    model = CNNModel(3)
    x = torch.randn(1, 3, 128, 128)

    with pytest.raises(RuntimeError, match="AdaptiveAvgPool2d でエラーが発生しました"):
        model(x)


# 2-2-6 異常：fc 失敗
def test_forward_fc_error(monkeypatch):
    model = CNNModel(3)

    def fake_fc(*args, **kwargs):
        raise Exception("fc error")

    # nn.Linear.__call__ を例外化
    monkeypatch.setattr("torch.nn.Linear.__call__", fake_fc)

    x = torch.randn(1, 3, 128, 128)

    with pytest.raises(RuntimeError, match="全結合層 \\(fc\\) でエラーが発生しました"):
        model(x)
