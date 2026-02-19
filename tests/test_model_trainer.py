import pytest
import logging
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from python.modelTrainer import ModelTrainer
from python.image_cnnmodel import CNNModel


# ============================================================
# build_model()
# ============================================================

# 6-1-1 正常：CNN モデル構築
def test_build_model_cnn_success(tmp_path):
    trainer = ModelTrainer()

    trainer.build_model(num_classes=3, used_model="cnn", save_model_path=str(tmp_path / "none.pth"))

    assert isinstance(trainer.model, CNNModel)


# 6-1-2 正常：ResNet 構築（pretrained=True）
def test_build_model_resnet_success(tmp_path):
    trainer = ModelTrainer()

    trainer.build_model(num_classes=3, used_model="resnet", save_model_path=str(tmp_path / "none.pth"))

    assert isinstance(trainer.model, models.ResNet)
    assert trainer.model.fc.out_features == 3


# 6-1-3 異常：ResNet pretrained 読み込み失敗 → pretrained=False で再試行
def test_build_model_resnet_pretrained_fail(monkeypatch, caplog, tmp_path):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    # 元の resnet18 を退避
    real_resnet18 = models.resnet18

    # pretrained=True のときだけ例外を出す
    def fake_resnet18_pretrained(*args, **kwargs):
        if kwargs.get("pretrained", False):
            raise Exception("pretrained error")
        return real_resnet18(pretrained=False)

    monkeypatch.setattr("torchvision.models.resnet18", fake_resnet18_pretrained)

    trainer.build_model(
        num_classes=3,
        used_model="resnet",
        save_model_path=str(tmp_path / "none.pth")
    )

    assert "ResNet の読み込みに失敗しました" in caplog.text
    assert isinstance(trainer.model, models.ResNet)


# 6-1-4 異常：ResNet pretrained=True も False も例外
def test_build_model_resnet_both_fail(monkeypatch, tmp_path):
    trainer = ModelTrainer()

    def fake_resnet18(*args, **kwargs):
        raise Exception("both fail")

    monkeypatch.setattr("torchvision.models.resnet18", fake_resnet18)

    with pytest.raises(Exception, match="モデル構築中にエラーが発生しました"):
        trainer.build_model(num_classes=3, used_model="resnet", save_model_path=str(tmp_path / "none.pth"))


# 6-1-5 異常：不正なモデル名
def test_build_model_invalid_name(caplog, tmp_path):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    with pytest.raises(RuntimeError, match="不正なモデル名です"):
        trainer.build_model(num_classes=3, used_model="xxx", save_model_path=str(tmp_path / "none.pth"))


# 6-1-6 異常：学習済みモデル読み込み失敗
def test_build_model_load_fail(monkeypatch, tmp_path):
    trainer = ModelTrainer()

    # 保存ファイルを作成
    broken = tmp_path / "broken.pth"
    broken.write_bytes(b"dummy")

    # torch.load を例外化
    monkeypatch.setattr("torch.load", lambda *a, **k: (_ for _ in ()).throw(Exception("load error")))

    with pytest.raises(RuntimeError, match="モデルの読み込みに失敗しました"):
        trainer.build_model(num_classes=3, used_model="cnn", save_model_path=str(broken))


# 6-1-7 異常：デバイス移動失敗
def test_build_model_device_fail(monkeypatch, caplog, tmp_path):
    trainer = ModelTrainer()

    # 存在しないデバイスを指定
    trainer.device = "cuda:99"
    with pytest.raises(RuntimeError, match="モデルをデバイスへ移動できません"):
        trainer.build_model(
            num_classes=3,
            used_model="cnn",
            save_model_path=str(tmp_path / "none.pth")
        )


# ============================================================
# _validate_dataloader()
# ============================================================

# 6-2-1 正常：dataloader が有効
def test_validate_dataloader_success():
    trainer = ModelTrainer()

    # ダミー dataloader
    class DummyLoader:
        def __len__(self):
            return 5

        def __iter__(self):
            return iter([1, 2, 3])

    loader = DummyLoader()

    result = trainer._validate_dataloader(loader)

    assert result is True


# 6-2-2 異常：dataloader が None
def test_validate_dataloader_none(caplog):
    trainer = ModelTrainer()

    caplog.set_level(logging.ERROR, logger="myapp")

    result = trainer._validate_dataloader(None)

    assert result is False
    assert "dataloader が None です" in caplog.text


# ============================================================
# _load_train_config()
# ============================================================

# 6-3-1 正常：epochs / learning_rate 読み込み
def test_load_train_config_success():
    trainer = ModelTrainer()
    EPOCHS = 5
    LEARNING_RATE = 0.01

    train_config = {"epochs": EPOCHS, "learning_rate": LEARNING_RATE}

    trainer._load_train_config(train_config)

    assert trainer.epochs == EPOCHS
    assert trainer.learning_rate == LEARNING_RATE


# 6-3-2 正常：デフォルト値設定
def test_load_train_config_default():
    trainer = ModelTrainer()

    train_config = {}

    result = trainer._load_train_config(train_config)

    assert result is True
    assert trainer.epochs == 100
    assert trainer.learning_rate == 0.001


# 6-3-3 異常：train_config が None
def test_load_train_config_none(caplog):
    trainer = ModelTrainer()

    caplog.set_level(logging.ERROR, logger="myapp")

    result = trainer._load_train_config(None)

    assert result is False
    assert "学習設定でエラーが発生しました" in caplog.text


# ============================================================
# _init_optimizer()
# ============================================================

# 6-4-1 正常：optimizer が正しく初期化される
def test_init_optimizer_success_adam():
    trainer = ModelTrainer()

    # ダミー model
    class DummyModel:
        def parameters(self): 
            return [torch.nn.Parameter(torch.randn(2, 2))]

    trainer.model = DummyModel()
    trainer.learning_rate = 0.001

    train_config = {"optimizer": "adam"}

    result = trainer._init_optimizer(train_config)

    assert result is True
    assert isinstance(trainer.optimizer, optim.Adam)


# 6-4-2 異常：不正 optimizer
def test_init_optimizer_invalid_name(caplog):
    trainer = ModelTrainer()

    class DummyModel:
        def parameters(self):
            return [torch.nn.Parameter(torch.randn(2, 2))]

    trainer.model = DummyModel()
    trainer.learning_rate = 0.001

    caplog.set_level(logging.ERROR, logger="myapp")

    train_config = {"optimizer": "xxx"}

    result = trainer._init_optimizer(train_config)

    assert result is False
    assert "不正な optimizer" in caplog.text


# 6-4-3 異常：self.model.parameters() が例外
def test_init_optimizer_parameters_exception(monkeypatch, caplog):
    trainer = ModelTrainer()

    # parameters() が例外を投げる model
    class DummyModel:
        def parameters(self):
            raise Exception("param error")

    trainer.model = DummyModel()
    trainer.learning_rate = 0.001

    caplog.set_level(logging.ERROR, logger="myapp")

    train_config = {"optimizer": "adam"}

    result = trainer._init_optimizer(train_config)

    assert result is False
    assert "Optimizer 初期化エラー" in caplog.text


# ============================================================
# _init_loss_function()
# ============================================================

# 6-5-1 正常：CrossEntropyLoss
def test_init_loss_function_cross_entropy():
    trainer = ModelTrainer()

    train_config = {"loss_function": "cross_entropy"}

    result = trainer._init_loss_function(train_config)

    assert result is True
    assert isinstance(trainer.criterion, nn.CrossEntropyLoss)


# 6-5-1-2 正常：MSELoss
def test_init_loss_function_mse():
    trainer = ModelTrainer()

    train_config = {"loss_function": "mse"}

    result = trainer._init_loss_function(train_config)

    assert result is True
    assert isinstance(trainer.criterion, nn.MSELoss)


# 6-5-2 異常：不正 loss_function
def test_init_loss_function_invalid_name(caplog):
    trainer = ModelTrainer()

    caplog.set_level(logging.ERROR, logger="myapp")

    train_config = {"loss_function": "xxx"}

    result = trainer._init_loss_function(train_config)

    assert result is False
    assert "不正な loss_function" in caplog.text


# 6-5-3 異常：loss 関数生成時の例外
def test_init_loss_function_exception(monkeypatch, caplog):
    trainer = ModelTrainer()

    caplog.set_level(logging.ERROR, logger="myapp")

    # nn.CrossEntropyLoss を例外化
    def fake_loss(*args, **kwargs):
        raise Exception("loss error")

    monkeypatch.setattr("torch.nn.CrossEntropyLoss", fake_loss)

    train_config = {"loss_function": "cross_entropy"}

    result = trainer._init_loss_function(train_config)

    assert result is False
    assert "Loss 関数初期化エラー" in caplog.text


# ============================================================
# _train_one_epoch()
# ============================================================

# ダミー dataloader（3 バッチ分）
class DummyLoader:
    def __iter__(self):
        return iter([
            ("img1", "label1"),
            ("img2", "label2"),
            ("img3", "label3"),
        ])


# 6-6-1 正常：running_loss が加算される
def test_train_one_epoch_success(monkeypatch):
    trainer = ModelTrainer()
    trainer.epochs = 1

    # _train_one_batch が (0.5, True) を返す
    monkeypatch.setattr(
        "python.modelTrainer.ModelTrainer._train_one_batch",
        lambda self, img, lbl: (0.5, True)
    )

    loader = DummyLoader()

    result = trainer._train_one_epoch(loader, epoch=0)

    assert result > 0.0


# 6-6-2 異常：_train_one_batch が (None, True)
def test_train_one_epoch_skip_batch(monkeypatch):
    trainer = ModelTrainer()
    trainer.epochs = 1

    # (None, True) → running_loss に加算されない
    monkeypatch.setattr(
        "python.modelTrainer.ModelTrainer._train_one_batch",
        lambda self, img, lbl: (None, True)
    )

    loader = DummyLoader()

    result = trainer._train_one_epoch(loader, epoch=0)

    assert result == 0.0


# 6-6-3 異常：_train_one_batch が (None, False) → 致命的エラー
def test_train_one_epoch_fatal_error(monkeypatch, caplog):
    trainer = ModelTrainer()
    trainer.epochs = 1

    caplog.set_level(logging.ERROR, logger="myapp")

    # (None, False) → 致命的エラーで break
    monkeypatch.setattr(
        "python.modelTrainer.ModelTrainer._train_one_batch",
        lambda self, img, lbl: (None, False)
    )

    loader = DummyLoader()

    result = trainer._train_one_epoch(loader, epoch=0)

    assert result is False
    assert "致命的エラーのため学習を停止します" in caplog.text


# ============================================================
# _train_one_batch()
# ============================================================

# 正常な images / labels
def make_tensors():
    images = torch.zeros((2, 3, 64, 64))
    labels = torch.tensor([0, 1])
    return images, labels


# 6-7-1 正常：バッチ学習成功
def test_train_one_batch_success():
    trainer = ModelTrainer()
    trainer.device = torch.device("cpu")

    # ダミー model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x):
            return torch.randn(2, 2, requires_grad=True)

    trainer.model = DummyModel()
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.001)
    trainer.criterion = nn.CrossEntropyLoss()

    images, labels = make_tensors()
    images = images.cpu()
    labels = labels.cpu()

    loss_value, can_continue = trainer._train_one_batch(images, labels)

    assert isinstance(loss_value, float)
    assert can_continue is True



# 6-7-2 異常：images.to() が例外
def test_train_one_batch_images_to_exception(monkeypatch, caplog):
    trainer = ModelTrainer()

    caplog.set_level(logging.ERROR, logger="myapp")

    images, labels = make_tensors()

    # images.to を例外化
    def fake_to(*args, **kwargs):
        raise Exception("to error")

    monkeypatch.setattr(images, "to", fake_to)

    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.criterion = MagicMock()

    loss_value, can_continue = trainer._train_one_batch(images, labels)

    assert loss_value is None
    assert can_continue is False
    assert "学習中に予期せぬエラー" in caplog.text


# 6-7-3 異常：CUDA メモリ不足
def test_train_one_batch_cuda_oom(monkeypatch, caplog):
    trainer = ModelTrainer()
    
    caplog.set_level(logging.ERROR, logger="myapp")

    images, labels = make_tensors()

    def fake_to(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")
    
    monkeypatch.setattr(images, "to", fake_to)

    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.criterion = MagicMock()

    loss_value, can_continue = trainer._train_one_batch(images, labels)

    assert loss_value is None
    assert can_continue is True
    assert "CUDA out of memory" in caplog.text


# 6-7-4 異常：shape エラー（size mismatch）
def test_train_one_batch_shape_error(monkeypatch, caplog):
    trainer = ModelTrainer()

    caplog.set_level(logging.ERROR, logger="myapp")

    images, labels = make_tensors()

    def fake_to(*args, **kwargs):
        raise RuntimeError("size mismatch")

    monkeypatch.setattr(images, "to", fake_to)

    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.criterion = MagicMock()

    loss_value, can_continue = trainer._train_one_batch(images, labels)

    assert loss_value is None
    assert can_continue is True
    assert "バッチの shape エラー" in caplog.text


# 6-7-5 異常：GPU 致命的エラー
def test_train_one_batch_gpu_fatal(monkeypatch, caplog):
    trainer = ModelTrainer()

    caplog.set_level(logging.ERROR, logger="myapp")

    images, labels = make_tensors()

    def fake_to(*args, **kwargs):
        raise RuntimeError("device-side assert triggered")
    
    monkeypatch.setattr(images, "to", fake_to)

    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.criterion = MagicMock()

    loss_value, can_continue = trainer._train_one_batch(images, labels)

    assert loss_value is None
    assert can_continue is False
    assert "GPU の致命的エラー。学習を停止します" in caplog.text


# 6-7-6 異常：cuDNN 致命的エラー
def test_train_one_batch_cudnn_error(monkeypatch, caplog):
    trainer = ModelTrainer()

    caplog.set_level(logging.ERROR, logger="myapp")

    images, labels = make_tensors()

    def fake_to(*args, **kwargs):
        raise RuntimeError("cuDNN error")
    
    monkeypatch.setattr(images, "to", fake_to)

    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.criterion = MagicMock()

    loss_value, can_continue = trainer._train_one_batch(images, labels)

    assert loss_value is None
    assert can_continue is False
    assert "cuDNN の致命的エラー。学習を停止します" in caplog.text


# ============================================================
# train()
# ============================================================

# 6-ダミー dataloader
class DummyLoader:
    def __iter__(self):
        return iter([("img", "label")])


# 6-8-1 正常：全処理成功
def test_train_success(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    dataloader = DummyLoader()
    train_config = {}

    # 各内部メソッドを正常化
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._load_train_config", lambda self, cfg: None)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_optimizer", lambda self, cfg: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_loss_function", lambda self, cfg: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._train_one_epoch", lambda self, dl, ep: True)

    trainer.epochs = 2

    result = trainer.train(dataloader, train_config)

    assert result is True


# 6-8-2 異常：dataloader が不正
def test_train_invalid_dataloader(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: False)

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "dataloader が不正" in caplog.text


# 6-8-3 異常：dataloader 検証中に例外
def test_train_validate_exception(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    def raise_error(self, dl):
        raise Exception("validate error")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", raise_error)

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "dataloader 検証中に例外発生" in caplog.text


# 6-8-4 異常：train_config の読み込みで例外
def test_train_load_config_exception(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._load_train_config", lambda self, cfg: (_ for _ in ()).throw(Exception("config error")))

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "train_config の読み込みに失敗しました" in caplog.text


# 6-8-5 異常：optimizer 初期化が False
def test_train_optimizer_init_false(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._load_train_config", lambda self, cfg: None)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_optimizer", lambda self, cfg: False)

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "optimizer の初期化に失敗しました" in caplog.text


# 6-8-6 異常：optimizer 初期化中に例外
def test_train_optimizer_exception(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._load_train_config", lambda self, cfg: None)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_optimizer", lambda self, cfg: (_ for _ in ()).throw(Exception("opt error")))

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "optimizer 初期化中に例外発生" in caplog.text


# 6-8-7 異常：loss 関数初期化が False
def test_train_loss_init_false(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._load_train_config", lambda self, cfg: None)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_optimizer", lambda self, cfg: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_loss_function", lambda self, cfg: False)

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "loss 関数の初期化に失敗しました" in caplog.text


# 6-8-8 異常：loss 関数初期化中に例外
def test_train_loss_exception(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._load_train_config", lambda self, cfg: None)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_optimizer", lambda self, cfg: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_loss_function", lambda self, cfg: (_ for _ in ()).throw(Exception("loss error")))

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "loss 関数初期化中に例外発生" in caplog.text


# 6-8-9 異常：_train_one_epoch が False を返す（致命的エラー）
def test_train_epoch_fatal(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._load_train_config", lambda self, cfg: None)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_optimizer", lambda self, cfg: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_loss_function", lambda self, cfg: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._train_one_epoch", lambda self, dl, ep: False)

    trainer.epochs = 1

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "致命的エラーが発生したため学習を停止します" in caplog.text


# 6-8-10 異常：_train_one_epoch が例外
def test_train_epoch_exception(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    monkeypatch.setattr("python.modelTrainer.ModelTrainer._validate_dataloader", lambda self, dl: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._load_train_config", lambda self, cfg: None)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_optimizer", lambda self, cfg: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._init_loss_function", lambda self, cfg: True)
    monkeypatch.setattr("python.modelTrainer.ModelTrainer._train_one_epoch", lambda self, dl, ep: (_ for _ in ()).throw(Exception("epoch error")))

    trainer.epochs = 1

    result = trainer.train(DummyLoader(), {})

    assert result is False
    assert "エポック 0 実行中に例外発生" in caplog.text


# ============================================================
# save_model()
# ============================================================

# 6-9-1 正常：モデル保存成功
def test_save_model_success(tmp_path):
    trainer = ModelTrainer()

    # ダミー model
    class DummyModel:
        def state_dict(self):
            return {"w": 1}

    trainer.model = DummyModel()

    save_path = tmp_path / "model" / "model.pth"

    result = trainer.save_model(str(save_path))

    assert result is True
    assert save_path.exists()


# 6-9-2 異常：path が空文字
def test_save_model_empty_path(caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    result = trainer.save_model("")

    assert result is False
    assert "path が不正です" in caplog.text


# 6-9-3 異常：ディレクトリ作成 PermissionError
def test_save_model_makedirs_permission_error(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {}

    def fake_makedirs(*args, **kwargs):
        raise PermissionError("no permission")

    monkeypatch.setattr("python.modelTrainer.os.makedirs", fake_makedirs)

    result = trainer.save_model("abc/model.pth")

    assert result is False
    assert "ディレクトリ作成権限がありません" in caplog.text


# 6-9-4 異常：ディレクトリ作成 FileNotFoundError
def test_save_model_makedirs_filenotfound(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {}

    def fake_makedirs(*args, **kwargs):
        raise FileNotFoundError("bad path")

    monkeypatch.setattr("python.modelTrainer.os.makedirs", fake_makedirs)

    result = trainer.save_model("abc/model.pth")

    assert result is False
    assert "ディレクトリパスが不正です" in caplog.text


# 6-9-5 異常：ディレクトリ作成 その他の例外
def test_save_model_makedirs_other_exception(monkeypatch, caplog):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {}

    def fake_makedirs(*args, **kwargs):
        raise Exception("other error")

    monkeypatch.setattr("python.modelTrainer.os.makedirs", fake_makedirs)

    result = trainer.save_model("abc/model.pth")

    assert result is False
    assert "ディレクトリ作成中に予期せぬエラー" in caplog.text


# 6-9-6 異常：state_dict の取得失敗
def test_save_model_state_dict_fail(monkeypatch, caplog, tmp_path):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    class DummyModel:
        def state_dict(self):
            raise Exception("state error")

    trainer.model = DummyModel()

    result = trainer.save_model(str(tmp_path / "model.pth"))

    assert result is False
    assert "state_dict の取得に失敗しました" in caplog.text


# 6-9-7 異常：torch.save PermissionError
def test_save_model_torchsave_permission_error(monkeypatch, caplog, tmp_path):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {}

    def fake_save(*args, **kwargs):
        raise PermissionError("no permission")

    monkeypatch.setattr("python.modelTrainer.torch.save", fake_save)

    result = trainer.save_model(str(tmp_path / "model.pth"))

    assert result is False
    assert "ファイル保存権限がありません" in caplog.text


# 6-9-8 異常：torch.save FileNotFoundError
def test_save_model_torchsave_filenotfound(monkeypatch, caplog, tmp_path):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {}

    def fake_save(*args, **kwargs):
        raise FileNotFoundError("bad path")

    monkeypatch.setattr("python.modelTrainer.torch.save", fake_save)

    result = trainer.save_model(str(tmp_path / "model.pth"))

    assert result is False
    assert "保存先パスが不正です" in caplog.text


# 6-9-9 異常：torch.save OSError
def test_save_model_torchsave_oserror(monkeypatch, caplog, tmp_path):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {}

    def fake_save(*args, **kwargs):
        raise OSError("fs error")

    monkeypatch.setattr("python.modelTrainer.torch.save", fake_save)

    result = trainer.save_model(str(tmp_path / "model.pth"))

    assert result is False
    assert "ファイルシステムエラー" in caplog.text


# 6-9-10 異常：torch.save その他の例外
def test_save_model_torchsave_other_exception(monkeypatch, caplog, tmp_path):
    trainer = ModelTrainer()
    caplog.set_level(logging.ERROR, logger="myapp")

    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {}

    def fake_save(*args, **kwargs):
        raise Exception("other error")

    monkeypatch.setattr("python.modelTrainer.torch.save", fake_save)

    result = trainer.save_model(str(tmp_path / "model.pth"))

    assert result is False
    assert "モデル保存中に予期せぬエラー" in caplog.text
