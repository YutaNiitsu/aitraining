import pytest
import numpy as np
from python.image_dataset import ImageDataset


# ============================================================
# __getitem__() のテスト
# ============================================================

# 3-1-1 正常：正常追加・取得
# 3-1-1 正常：複数データの正常追加・取得
def test_getitem_success_multiple():
    dataset = ImageDataset(transform=None)

    # 複数のダミー画像とラベル
    images = [
        np.zeros((128, 128, 3), dtype=np.uint8),
        np.ones((128, 128, 3), dtype=np.uint8) * 255,
        np.full((128, 128, 3), 127, dtype=np.uint8),
    ]
    labels = [1, 2, 3]

    # データセットに追加
    for img, lbl in zip(images, labels):
        dataset.addData(img, lbl)

    # すべてのデータを取得して比較
    for idx in range(len(images)):
        out_image, out_label = dataset[idx]

        # 画像が完全一致すること
        assert (out_image == images[idx]).all() # all() で NumPy 配列の全要素を比較

        # ラベルが一致すること
        assert out_label == labels[idx]



# 3-1-2 異常：idx 範囲外
def test_getitem_index_out_of_range():
    dataset = ImageDataset(transform=None)

    with pytest.raises(RuntimeError, match="指定された idx が範囲外です"):
        dataset[0]


# 3-1-3 異常：transform 失敗
def test_getitem_transform_error(monkeypatch):
    # transform を例外化
    def fake_transform(x):
        raise Exception("transform error")

    dataset = ImageDataset(transform=fake_transform)

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    label = 1
    dataset.addData(image, label)

    with pytest.raises(RuntimeError, match="画像変換\\(transform\\) に失敗しました"):
        dataset[0]


# ============================================================
# addData() のテスト
# ============================================================

# 3-2-1 正常：正常追加
def test_addData_success():
    dataset = ImageDataset(transform=None)

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    label = 1

    dataset.addData(image, label)

    assert dataset.images[0] is image
    assert dataset.labels[0] == label


# 3-2-2 異常：addData 失敗（images が list でない）
def test_addData_failure():
    dataset = ImageDataset(transform=None)

    # images が None
    dataset.images = None

    image = np.zeros((128, 128, 3), dtype=np.uint8)
    label = 1

    with pytest.raises(RuntimeError, match="データ追加に失敗しました"):
        dataset.addData(image, label)
