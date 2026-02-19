import pytest
import logging
from unittest.mock import patch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from python.preprocessor import Preprocessor
from python.image_dataset import ImageDataset


# ============================================================
# _load_train_config()
# ============================================================

# 5-1-1 正常：正常読み込み
def test_load_train_config_success():
    p = Preprocessor()

    train_config = {
        "augmentation": {"rotation": 30},
        "batch_size": 32,
        "shuffle": True
    }

    aug, batch, shuffle = p._load_train_config(train_config)

    assert aug == {"rotation": 30}
    assert batch == 32
    assert shuffle is True


# 5-1-2 異常：augmentation が dict でない
def test_load_train_config_invalid_augmentation():
    p = Preprocessor()

    train_config = {
        "augmentation": "abc"
    }

    with pytest.raises(TypeError, match="augmentation は dict である必要があります"):
        p._load_train_config(train_config)


# 5-1-3 異常：batch_size が int でない
def test_load_train_config_invalid_batch_size():

    p = Preprocessor()

    train_config = {
        "batch_size": "abc"
    }

    with pytest.raises(ValueError, match="batch_size は正の整数である必要があります"):
        p._load_train_config(train_config)


# 5-1-4 異常：shuffle が bool でない
def test_load_train_config_invalid_shuffle():
    p = Preprocessor()

    train_config = {
        "shuffle": "yes"
    }

    with pytest.raises(TypeError, match="shuffle は bool である必要があります"):
        p._load_train_config(train_config)


# ============================================================
# _build_transform()
# ============================================================

# 5-2-1 正常：評価用 transform
def test_build_transform_eval_normalize():
    p = Preprocessor()

    aug_config = {"normalize": True}

    transform = p._build_transform(aug_config, is_eval=True)

    assert isinstance(transform, transforms.Compose)
    # ToTensor + Normalize の2つが入る
    assert len(transform.transforms) == 2
    assert isinstance(transform.transforms[0], transforms.ToTensor)
    assert isinstance(transform.transforms[1], transforms.Normalize)


# 5-2-2 正常：学習用 transform
def test_build_transform_train_all_augments():
    p = Preprocessor()

    aug_config = {
        "normalize": True,
        "horizontal_flip": True,
        "vertical_flip": True,
        "rotation": 30,
        "color_jitter": {
            "brightness": 0.2,
            "contrast": 0.3,
            "saturation": 0.4,
            "hue": 0.1,
        },
        "random_crop": True,
    }

    transform = p._build_transform(aug_config, is_eval=False)

    assert isinstance(transform, transforms.Compose)
    # transform.transforms の中に各インスタンスが1つでもあることを確認
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in transform.transforms)
    assert any(isinstance(t, transforms.RandomVerticalFlip) for t in transform.transforms)
    assert any(isinstance(t, transforms.RandomRotation) for t in transform.transforms)
    assert any(isinstance(t, transforms.ColorJitter) for t in transform.transforms)
    assert any(isinstance(t, transforms.RandomResizedCrop) for t in transform.transforms)
    assert any(isinstance(t, transforms.Normalize) for t in transform.transforms)


# 5-2-3 異常：aug_config が dict でない
def test_build_transform_augconfig_not_dict():
    p = Preprocessor()

    with pytest.raises(TypeError, match="aug_config は dict である必要があります"):
        p._build_transform(None, is_eval=True)


# 5-2-4 異常：rotation が整数でない
def test_build_transform_rotation_not_int():
    p = Preprocessor()

    aug_config = {"rotation": "abc"}

    with pytest.raises(RuntimeError, match="rotation は整数である必要があります"):
        p._build_transform(aug_config, is_eval=False)


# 5-2-5 異常：color_jitter が dict でない
def test_build_transform_color_jitter_not_dict():
    p = Preprocessor()

    aug_config = {"color_jitter": 123}

    with pytest.raises(RuntimeError, match="color_jitter は dict である必要があります"):
        p._build_transform(aug_config, is_eval=False)


# 5-2-6 異常：transforms.Compose 例外発生
def test_build_transform_compose_error(monkeypatch):
    p = Preprocessor()

    aug_config = {"normalize": True}

    # Compose を例外化
    def fake_compose(*args, **kwargs):
        raise Exception("compose error")

    monkeypatch.setattr("torchvision.transforms.Compose", fake_compose)

    with pytest.raises(Exception, match="Transform 構築エラー"):
        p._build_transform(aug_config, is_eval=True)


# ============================================================
# _create_dataset()
# ============================================================

# 5-3-1 正常：正常作成
def test_create_dataset_success():
    p = Preprocessor()

    class DummyTransform:
        def __call__(self, x):
            return x

    transform = DummyTransform()

    dataset = p._create_dataset(transform)

    assert isinstance(dataset, ImageDataset)


# 5-3-2 異常：transform=None
def test_create_dataset_transform_none():
    p = Preprocessor()

    with pytest.raises(ValueError, match="transform が None です"):
        p._create_dataset(None)


# 5-3-3 異常：ImageDataset 例外
def test_create_dataset_internal_error(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    class DummyTransform:
        def __call__(self, x):
            return x

    transform = DummyTransform()

    # ImageDataset を例外化
    def fake_ImageDataset(*args, **kwargs):
        raise Exception("dataset error")

    monkeypatch.setattr("python.preprocessor.ImageDataset", fake_ImageDataset)

    caplog.set_level(logging.ERROR, logger="myapp")

    dataset = p._create_dataset(transform)

    # 戻り値は None
    assert dataset is None

    # ログ確認
    assert "Dataset 作成エラー" in caplog.text


# ============================================================
# _collect_files()
# ============================================================

# 5-4-1 正常：正常収集
def test_collect_files_success(tmp_path):
    p = Preprocessor()

    cats = tmp_path / "cats"
    dogs = tmp_path / "dogs"
    cats.mkdir()
    dogs.mkdir()

    # 画像ファイル作成
    (cats / "a.jpg").write_bytes(b"dummy")
    (cats / "b.png").write_bytes(b"dummy")
    (dogs / "c.jpg").write_bytes(b"dummy")

    data_dirs = [str(cats), str(dogs)]

    result = p._collect_files(data_dirs)

    # {0: [...], 1: [...]} となることを確認
    assert 0 in result
    assert 1 in result
    # result[0] == ["cats/a.jpg", "cats/b.png"]
    assert len(result[0]) == 2
    # result[1] == ["dogs/c.jpg"]
    assert len(result[1]) == 1


# 5-4-2 異常：data_dirs の要素が str でない
def test_collect_files_invalid_dirname(caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    caplog.set_level(logging.ERROR, logger="myapp")

    data_dirs = [123]  # 無効

    p._collect_files(data_dirs)

    assert "無効なディレクトリ名" in caplog.text


# 5-4-3 異常：ディレクトリが存在しない
def test_collect_files_dir_not_exist(caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    caplog.set_level(logging.ERROR, logger="myapp")

    data_dirs = ["not_exist"]

    p._collect_files(data_dirs)

    assert "ディレクトリが存在しません" in caplog.text


# 5-4-4 異常：ディレクトリ読み込みエラー
def test_collect_files_listdir_error(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    dir1 = tmp_path / "dir1"
    dir1.mkdir()

    def fake_listdir(*args, **kwargs):
        raise Exception("listdir error")

    monkeypatch.setattr("os.listdir", fake_listdir)

    p._collect_files([str(dir1)])

    assert "ディレクトリ読み込みエラー" in caplog.text


# 5-4-5 異常：画像が 0 件
def test_collect_files_no_images(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")
    
    p = Preprocessor()

    caplog.set_level(logging.ERROR, logger="myapp")

    empty = tmp_path / "empty"
    empty.mkdir()

    p._collect_files([str(empty)])

    assert "画像ファイルが見つかりません" in caplog.text


# ============================================================
# _load_and_resize_image()
# ============================================================

# 5-5-1 正常：読み込み＋リサイズ成功
def test_load_and_resize_image_success(tmp_path):
    p = Preprocessor()

    # 画像作成
    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)

    # safe_imread が正常に numpy array を返すように mock
    with patch("preprocessor.safe_imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        result = p._load_and_resize_image(str(img_path), (64, 64))

    assert result is not None
    assert result.shape == (64, 64, 3)


# 5-5-2 異常：safe_imread が例外
def test_load_and_resize_image_safe_imread_exception(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)

    caplog.set_level(logging.ERROR, logger="myapp")

    def fake_safe_imread(*args, **kwargs):
        raise Exception("safe error")

    with patch("python.preprocessor.safe_imread", fake_safe_imread):
        result = p._load_and_resize_image(str(img_path), (64, 64))

    assert result is None
    assert "safe_imread でエラー" in caplog.text


# 5-5-3 異常：safe_imread が None
def test_load_and_resize_image_safe_imread_none(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)

    caplog.set_level(logging.ERROR, logger="myapp")

    with patch("python.preprocessor.safe_imread", return_value=None):
        result = p._load_and_resize_image(str(img_path), (64, 64))

    assert result is None
    assert "読み込み不可の画像をスキップ" in caplog.text


# 5-5-4 異常：cv2.resize が例外
def test_load_and_resize_image_resize_exception(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    img_path = tmp_path / "test.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(img_path)

    caplog.set_level(logging.ERROR, logger="myapp")

    # safe_imread は正常
    with patch("preprocessor.safe_imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        # cv2.resize を例外化
        with patch("preprocessor.cv2.resize", side_effect=Exception("resize error")):
            result = p._load_and_resize_image(str(img_path), (64, 64))

    assert result is None
    assert "cv2.resize 失敗" in caplog.text


# ============================================================
# _add_images_to_dataset()
# ============================================================

# 5-6-1 正常：正常追加
def test_add_images_to_dataset_success():
    p = Preprocessor()

    # ダミー dataset
    class DummyDataset:
        def __init__(self):
            self.data = []

        def addData(self, img, label):
            self.data.append((img, label))

    dataset = DummyDataset()

    # 画像のパスリスト
    files_by_label = {
        0: ["img1.jpg", "img2.jpg"],
        1: ["img3.jpg"]
    }

    # _load_and_resize_image を mock（常に画像を返す）
    with patch("python.preprocessor.Preprocessor._load_and_resize_image", return_value=np.zeros((10, 10, 3))):
        p._add_images_to_dataset(dataset, files_by_label, (10, 10))

    assert len(dataset.data) == 3  # データ数
    # 0:画像データ、1:ラベル
    assert dataset.data[0][1] == 0 # ラベル比較
    assert dataset.data[2][1] == 1 # ラベル比較


# 5-6-2 異常：dataset が None
def test_add_images_to_dataset_dataset_none():
    p = Preprocessor()

    with pytest.raises(ValueError, match="dataset が None です"):
        p._add_images_to_dataset(None, {}, (10, 10))


# 5-6-3 異常：dataset.addData が例外
def test_add_images_to_dataset_addData_exception(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    caplog.set_level(logging.ERROR, logger="myapp")

    # ダミー dataset（addData が例外を投げる）
    class DummyDataset:
        def addData(self, img, label):
            raise Exception("add error")

    dataset = DummyDataset()

    # _load_and_resize_image は正常に画像を返す
    with patch("python.preprocessor.Preprocessor._load_and_resize_image", return_value=np.zeros((10, 10, 3))):
        p._add_images_to_dataset(dataset, {0: ["img1"]}, (10, 10))

    # ログ確認
    assert "dataset.addData 失敗" in caplog.text


# ============================================================
# _create_dataloader()
# ============================================================

# 5-7-1 正常：正常に DataLoader が作成される
def test_create_dataloader_success():
    p = Preprocessor()

    # ダミー dataset
    class DummyDataset:
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return idx

    dataset = DummyDataset()

    loader = p._create_dataloader(dataset, batch_size=32, shuffle=True)
    
    assert isinstance(loader, DataLoader)


# 5-7-2 異常：dataset が空
def test_create_dataloader_empty_dataset(caplog):
    p = Preprocessor()

    caplog.set_level(logging.ERROR, logger="myapp")

    class EmptyDataset:
        def __len__(self):
            return 0

    dataset = EmptyDataset()

    with pytest.raises(ValueError, match="Dataset が空のため DataLoader は作成されません"):
        p._create_dataloader(dataset, batch_size=32, shuffle=True)


# 5-7-3 異常：batch_size が整数でない
def test_create_dataloader_invalid_batch_size():
    p = Preprocessor()

    class DummyDataset:
        def __len__(self):
            return 3

    dataset = DummyDataset()

    with pytest.raises(ValueError, match="batch_size は正の整数である必要があります"):
        p._create_dataloader(dataset, batch_size="abc", shuffle=True)


# 5-7-4 異常：shuffle が bool でない
def test_create_dataloader_invalid_shuffle():
    p = Preprocessor()

    class DummyDataset:
        def __len__(self):
            return 3

    dataset = DummyDataset()

    with pytest.raises(TypeError, match="shuffle は bool である必要があります"):
        p._create_dataloader(dataset, batch_size=32, shuffle="yes")


# 5-7-5 異常：DataLoader 内部で例外発生
def test_create_dataloader_internal_error(monkeypatch):
    p = Preprocessor()

    class DummyDataset:
        def __len__(self):
            return 3

        def __getitem__(self, idx):
            return idx

    dataset = DummyDataset()

    # DataLoader を例外化
    def fake_dataloader(*args, **kwargs):
        raise Exception("loader error")

    monkeypatch.setattr("python.preprocessor.DataLoader", fake_dataloader)

    with pytest.raises(RuntimeError, match="DataLoader 初期化エラー"):
        p._create_dataloader(dataset, batch_size=32, shuffle=True)


# ============================================================
# preprocessor()
# ============================================================

# 5-8-1 正常：正常処理
def test_preprocessor_success(tmp_path):
    p = Preprocessor()

    # ディレクトリ作成
    cats = tmp_path / "cats"
    dogs = tmp_path / "dogs"
    cats.mkdir()
    dogs.mkdir()

    # 画像作成
    img = Image.new("RGB", (100, 100))
    img.save(cats / "a.jpg")
    img.save(dogs / "b.jpg")

    data_dirs = [str(cats), str(dogs)]
    train_config = {"augmentation": {}, "batch_size": 2, "shuffle": True}
    image_size = (64, 64)

    # safe_imread を正常化
    with patch("preprocessor.safe_imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)):
        p.preprocessor(data_dirs, train_config, image_size, is_eval=False)
    
    assert p.dataLoader is not None
    assert len(p.dataLoader.dataset) == 2


# 5-8-2 異常：data_dirs が list でない
def test_preprocessor_invalid_data_dirs():
    p = Preprocessor()

    train_config = {"augmentation": {}, "batch_size": 2, "shuffle": True}

    with pytest.raises(TypeError, match="data_dirs は list である必要があります"):
        p.preprocessor(None, train_config, (64, 64), is_eval=False)


# 5-8-3 異常：train_config が dict でない
def test_preprocessor_invalid_train_config():
    p = Preprocessor()

    with pytest.raises(TypeError, match="train_config は dict である必要があります"):
        p.preprocessor([], None, (64, 64), is_eval=False)


# 5-8-4 異常：image_size がタプルでない
def test_preprocessor_invalid_image_size():
    p = Preprocessor()

    train_config = {"augmentation": {}, "batch_size": 2, "shuffle": True}

    with pytest.raises(ValueError, match="image_size は"):
        p.preprocessor([], train_config, 100, is_eval=False)


# 5-8-5 異常：dataset 作成失敗（transform=None）
def test_preprocessor_create_dataset_internal_error(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")
    
    p = Preprocessor()

    class DummyTransform:
        def __call__(self, x):
            return x

    transform = DummyTransform()

    # ImageDataset を例外化
    def fake_ImageDataset(*args, **kwargs):
        raise Exception("dataset error")

    monkeypatch.setattr("python.preprocessor.ImageDataset", fake_ImageDataset)

    dataset = p._create_dataset(transform)

    # 戻り値は None
    assert dataset is None
    assert "Dataset 作成エラー" in caplog.text



# 5-8-6 異常：画像が 0 件
def test_preprocessor_no_images(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    empty = tmp_path / "empty"
    empty.mkdir()

    data_dirs = [str(empty)]
    train_config = {"augmentation": {}, "batch_size": 2, "shuffle": True}

    with patch("preprocessor.safe_imread", return_value=None):
        p.preprocessor(data_dirs, train_config, (64, 64), is_eval=False)

    assert "有効な画像ファイルが見つかりませんでした" in caplog.text


# 5-8-7 異常：dataset が空（safe_imread が常に None）
def test_preprocessor_dataset_empty(tmp_path, caplog):
    caplog.set_level(logging.ERROR, logger="myapp")

    p = Preprocessor()

    d = tmp_path / "d"
    d.mkdir()

    # 壊れた画像を作成（safe_imread が None を返す）
    (d / "a.jpg").write_bytes(b"dummy")

    data_dirs = [str(d)]
    train_config = {"augmentation": {}, "batch_size": 2, "shuffle": True}

    with patch("preprocessor.safe_imread", return_value=None):
        p.preprocessor(data_dirs, train_config, (64, 64), is_eval=False)

    assert p.dataLoader is None
    assert "Dataset が空です。DataLoader は None になります" in caplog.text
