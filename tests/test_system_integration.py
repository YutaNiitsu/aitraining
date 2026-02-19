import logging
import os
import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
from python.configManager import ConfigManager
from python.imageCollector import ImageCollector
from python.preprocessor import Preprocessor
from python.modelTrainer import ModelTrainer
from python.modelEvaluator import ModelEvaluator


# ============================================================
# ダミーモデル（Trainer / Evaluator 用）
# ============================================================

class DummyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        return torch.randn((x.size(0), 2))


# ============================================================
# 統合テスト：Config → Collect → Preprocess → Train → Evaluate
# ============================================================

def test_full_pipeline(tmp_path, monkeypatch, caplog):
    # --------------------------------------------------------
    # ConfigManager
    # --------------------------------------------------------
    learn_yaml = tmp_path / "learn.yaml"
    log_yaml = tmp_path / "log.yaml"
    labels_yaml = tmp_path / "labels.yaml"

    learn_yaml.write_text("batch_size: 1\nimage_size: 32", encoding="utf-8")
    log_yaml.write_text("version: 1\nhandlers: {}", encoding="utf-8")
    labels_yaml.write_text("label_map:\n  cat: 0\n  dog: 1", encoding="utf-8")

    cm = ConfigManager(learn_yaml, log_yaml, labels_yaml)
    cm.load_config()

    # --------------------------------------------------------
    # ImageCollector
    # --------------------------------------------------------
    collector = ImageCollector()

    IMAGE_NUM = 10
    VAL_RATIO = 0.2
    VAL_IMAGE_NUM = int(IMAGE_NUM * VAL_RATIO)
    IMAGE_SIZE = (32, 32)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # ダミー画像を作成
    for i in range(IMAGE_NUM):
        img = Image.fromarray(np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8))
        img.save(raw_dir / f"{i}.jpg")

    output_root = tmp_path / "dataset"
    collector.split_images(raw_dir, output_root, "cat", val_ratio=VAL_RATIO)

    # ファイルが正しく分割されていることを確認
    train_dir = output_root / "cat" / "train"
    eval_dir = output_root / "cat" / "eval"

    assert len(os.listdir(train_dir)) == IMAGE_NUM - VAL_IMAGE_NUM
    assert len(os.listdir(eval_dir)) == VAL_IMAGE_NUM

    # --------------------------------------------------------
    # Preprocessor
    # --------------------------------------------------------
    pre = Preprocessor()

    # Preprocessor.preprocessor() に渡すパラメータ
    data_dirs = [str(train_dir)]
    train_config = {"batch_size": 1, "aug": {"flip": False}}
    image_size = IMAGE_SIZE
    is_eval = False

    pre.preprocessor(
        data_dirs=data_dirs,
        train_config=train_config,
        image_size=image_size,
        is_eval=is_eval
    )

    # DataLoader が作成されていること
    assert pre.dataLoader is not None
    assert len(pre.dataLoader.dataset) == IMAGE_NUM - VAL_IMAGE_NUM


    # --------------------------------------------------------
    # ModelTrainer
    # --------------------------------------------------------
    trainer = ModelTrainer()

    trainer.model = DummyModel()

    # モデルをダミーに差し替え
    monkeypatch.setattr(trainer, "build_model", lambda num_classes: DummyModel())

    # optimizer / loss / train_one_epoch をすべて成功扱いにする
    monkeypatch.setattr(trainer, "_init_optimizer", lambda cfg: True)
    monkeypatch.setattr(trainer, "_init_loss_function", lambda cfg: True)
    monkeypatch.setattr(trainer, "_train_one_epoch", lambda dl, ep: True)

    # dataloader は Preprocessor で作ったものを使用
    train_result = trainer.train(pre.dataLoader, train_config)

    assert train_result is True
    assert isinstance(trainer.model, DummyModel)

    model_path = tmp_path / "model" / "dummy.pth"
    save_result = trainer.save_model(str(model_path))

    assert save_result is True
    assert model_path.exists()


    # --------------------------------------------------------
    # ModelEvaluator
    # --------------------------------------------------------
    evaluator = ModelEvaluator()

    # --- eval_conf_mat に必要なもの ---
    label_map = {"cat": 0, "dog": 1} 
    model = trainer.model 
    dataloader = pre.dataLoader 

    # --- 内部処理をすべて成功扱いにするため monkeypatch ---
    monkeypatch.setattr(evaluator, "_prepare_output_dir", lambda model: tmp_path / "misclassified")
    monkeypatch.setattr(evaluator, "_extract_label_names", lambda lm: list(lm.keys()))
    monkeypatch.setattr(
        evaluator,
        "_run_inference_loop",
        lambda model, dl, labels, outdir: ([0, 1], [0, 1])  # 正解ラベルと予測ラベル
    )
    monkeypatch.setattr(
        evaluator,
        "_compute_confusion_matrix",
        lambda labels, preds: [[1, 0], [0, 1]]  # ダミー混同行列
    )
    monkeypatch.setattr(
        evaluator,
        "_generate_classification_report",
        lambda labels, preds, names, model: "report OK"
    )
    monkeypatch.setattr(
        evaluator,
        "_plot_confusion_matrix",
        lambda cm, names, outdir, model: True
    )

    evaluator.eval_conf_mat(label_map, model, dataloader)

    # ログレベルを ERROR から INFO に変更
    caplog.set_level(logging.INFO, logger="myapp")

    # _generate_classification_report が返した文字列がログに含まれることを確認
    assert "report OK" in caplog.text
