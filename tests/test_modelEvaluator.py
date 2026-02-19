import pytest
import logging
from unittest.mock import patch, MagicMock
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
import torchvision.utils as vutils
import torch.optim as optim
import os
from python.modelEvaluator import ModelEvaluator

# ============================================================
# _prepare_output_dir()
# ============================================================

# 7-1-1 正常系: 出力ディレクトリ作成成功
def test_prepare_output_dir_success(tmp_path, caplog):
    class DummyModel:
        __class__ = type("ResNet18", (), {})

    evaluator = ModelEvaluator()
    dir_name = evaluator._prepare_output_dir(DummyModel())
    model_name = DummyModel().__class__.__name__

    # misclassified_モデル名 というディレクトリが作成されていることを確認
    assert dir_name == f"misclassified_{model_name}"
    assert os.path.exists(dir_name)


# 7-1-2 異常系: 出力ディレクトリ作成失敗
def test_prepare_output_dir_fail(monkeypatch):
    evaluator = ModelEvaluator()

    def fake_makedirs(*args, **kwargs):
        raise OSError("cannot create dir")

    monkeypatch.setattr(os, "makedirs", fake_makedirs)

    class DummyModel:
        __class__ = type("ResNet18", (), {})

    with pytest.raises(Exception, match="誤分類保存ディレクトリ作成エラー"):
        evaluator._prepare_output_dir(DummyModel())


# ============================================================
# _extract_label_names()
# ============================================================

# 7-2-1 正常系: ラベル名抽出成功
def test_extract_label_names_success():
    evaluator = ModelEvaluator()
    label_map = {"cat": 0, "dog": 1}

    result = evaluator._extract_label_names(label_map)

    # label_map のキーのリストが返ることを確認
    assert result == list(label_map.keys())


# 7-2-2 異常系: ラベル名抽出失敗
def test_extract_label_names_fail():
    evaluator = ModelEvaluator()

    with pytest.raises(Exception, match="ラベル名抽出エラー"):
        evaluator._extract_label_names(None)


# ============================================================
# _run_inference_loop()
# ============================================================
# ダミーモデル(常にクラス 1 を返す)
class DummyModel(nn.Module): 
        def forward(self, x): 
            # クラス 1 のスコア 0.9
            return torch.tensor([[0.1, 0.9]], device="cuda")

# 7-3-1 正常系: 推論ループ成功
def test_run_inference_loop_success():
    evaluator = ModelEvaluator()

    model = DummyModel()

    # dataloader ダミー
    image1 = torch.randn(1, 3, 32, 32)
    label1 = torch.tensor([1])

    image2 = torch.randn(1, 3, 32, 32)
    label2 = torch.tensor([1])

    image3 = torch.randn(1, 3, 32, 32)
    label3 = torch.tensor([0])

    dataloader = [(image1, label1), (image2, label2), (image3, label3)]

    labels_name = ["cat", "dog"]

    all_labels, all_preds = evaluator._run_inference_loop(
        model, dataloader, labels_name, "dummy"
    )

    assert all_labels == [1, 1, 0]
    assert all_preds == [1, 1, 1]  # 常にクラス 1 を予測

# 7-3-2 異常系: モデル出力 shape エラー
def test_run_inference_loop_shape_mismatch():
    evaluator = ModelEvaluator()

    model = DummyModel()

    def fake_forward(x): 
        raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")
    
    model.forward = fake_forward

    image = torch.randn(1, 3, 32, 32)
    label = torch.tensor([0])
    dataloader = [(image, label)]

    with pytest.raises(RuntimeError, match="モデル出力 shape エラー"):
        evaluator._run_inference_loop(model, dataloader, ["a", "b"], "dummy")

# 7-3-3 異常系: サイズ不一致エラー
def test_run_inference_loop_size_mismatch():
    evaluator = ModelEvaluator()

    model = DummyModel()

    def fake_forward(x): 
        raise RuntimeError("size mismatch")
    
    model.forward = fake_forward
    
    image = torch.randn(1, 3, 32, 32)
    label = torch.tensor([0])
    dataloader = [(image, label)]

    with pytest.raises(RuntimeError, match="サイズ不一致"):
        evaluator._run_inference_loop(model, dataloader, ["a", "b"], "dummy")

# 7-3-4 異常系: その他の RuntimeError は継続
def test_run_inference_loop_continue_on_minor_error(caplog):
    evaluator = ModelEvaluator()

    model = MagicMock()
    model.eval = MagicMock()
    model.forward.side_effect = [RuntimeError("random error"), torch.tensor([[0.1, 0.9]])]

    image = torch.randn(1, 3, 32, 32)
    label = torch.tensor([1])
    dataloader = [(image, label), (image, label)]

    evaluator._run_inference_loop(model, dataloader, ["cat", "dog"], "dummy")

    assert "推論ループ中の RuntimeError（継続）" in caplog.text


# ============================================================
# _save_misclassified_images()
# ============================================================

# 7-4-1 正常系: 誤分類画像保存成功
def test_save_misclassified_images_success(tmp_path):
    evaluator = ModelEvaluator()

    image = torch.randn(1, 3, 32, 32)
    label = torch.tensor([0])
    predicted = torch.tensor([1])  # 誤分類

    evaluator._save_misclassified_images(
        image, label, predicted, 0, ["cat", "dog"], str(tmp_path)
    )

    files = list(tmp_path.iterdir())
    assert len(files) == 1

# 7-4-2 異常系: 誤分類画像保存失敗
def test_save_misclassified_images_fail(monkeypatch, caplog):
    evaluator = ModelEvaluator()

    def fake_save(*args, **kwargs):
        raise IOError("save failed")

    monkeypatch.setattr(vutils, "save_image", fake_save)

    images = torch.randn(1, 3, 32, 32)
    labels = torch.tensor([0])
    predicted = torch.tensor([1])

    evaluator._save_misclassified_images(
        images, labels, predicted, 0, ["cat", "dog"], "dummy"
    )

    assert "誤分類画像保存エラー" in caplog.text


# ============================================================
# _compute_confusion_matrix()
# ============================================================

# 7-5-1 正常系: 混同行列計算成功
def test_compute_confusion_matrix_success():
    evaluator = ModelEvaluator()
    cm = evaluator._compute_confusion_matrix([0, 1], [1, 0])
    assert cm.shape == (2, 2) # 2 x 2 の混同行列

# 7-5-2 異常系: 混同行列計算失敗
def test_compute_confusion_matrix_fail():
    evaluator = ModelEvaluator()

    with pytest.raises(RuntimeError, match="混同行列計算エラー"):
        evaluator._compute_confusion_matrix([0], [0, 1])


# ============================================================
# _generate_classification_report()
# ============================================================

# 7-6-1 正常系: レポート生成成功
def test_generate_classification_report_success():
    evaluator = ModelEvaluator()
    report = evaluator._generate_classification_report(
        [0, 1], [1, 0], ["cat", "dog"], MagicMock()
    )
    
    # precision recall f1-score support が含まれることを確認
    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report
    assert "support" in report

# 7-6-2 異常系: レポート生成失敗
def test_generate_classification_report_fail():
    evaluator = ModelEvaluator()

    with pytest.raises(RuntimeError, match="レポート生成エラー"):
        evaluator._generate_classification_report(
            [0, 1], [1, 2], ["cat"], MagicMock()
        )


# ============================================================
# _plot_confusion_matrix()
# ============================================================

# 7-7-1 正常系: 混同行列保存成功
def test_plot_confusion_matrix_success(tmp_path):
    evaluator = ModelEvaluator()

    cm = np.array([[1, 0], [0, 1]])
    labels = ["cat", "dog"]

    class DummyModel:
        __class__ = type("ResNet18", (), {})

    evaluator._plot_confusion_matrix(cm, labels, str(tmp_path), DummyModel())

    # PNG ファイルが生成されていること
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0].suffix == ".png"


# 7-7-2 異常系: 保存ディレクトリ作成失敗
def test_plot_confusion_matrix_dir_fail(monkeypatch):
    evaluator = ModelEvaluator()

    def fake_makedirs(*args, **kwargs):
        raise OSError("cannot create")

    monkeypatch.setattr(os, "makedirs", fake_makedirs)

    cm = np.array([[1, 0], [0, 1]])

    with pytest.raises(RuntimeError, match="混同行列保存ディレクトリ作成エラー"):
        evaluator._plot_confusion_matrix(cm, ["a", "b"], "dummy", MagicMock())


# 7-7-3 異常系: ファイル名生成失敗
def test_plot_confusion_matrix_filename_fail(monkeypatch):
    evaluator = ModelEvaluator()

    cm = np.array([[1, 0], [0, 1]])
    labels = ["cat", "dog"]

    class BadModel:
        # __class__.__name__ にアクセスすると例外を投げる
        @property
        def __class__(self):
            raise ValueError("broken class")

    with pytest.raises(RuntimeError, match="ファイル名生成エラー"):
        evaluator._plot_confusion_matrix(cm, labels, "dummy", BadModel())


# 7-7-4 異常系: 描画失敗
def test_plot_confusion_matrix_heatmap_fail():
    evaluator = ModelEvaluator()

    cm = "abc" # 不正な形状
    labels = ["a"]

    class DummyModel:
        __class__ = type("ResNet18", (), {})

    with pytest.raises(RuntimeError, match="混同行列描画エラー"):
        evaluator._plot_confusion_matrix(cm, labels, "dummy", DummyModel())


# 7-7-5 異常系: 保存失敗
def test_plot_confusion_matrix_save_fail(monkeypatch):
    evaluator = ModelEvaluator()

    def fake_savefig(*args, **kwargs):
        raise IOError("save failed")

    monkeypatch.setattr(plt, "savefig", fake_savefig)

    cm = np.array([[1, 0], [0, 1]])
    labels = ["cat", "dog"]

    class DummyModel:
        __class__ = type("ResNet18", (), {})

    with pytest.raises(RuntimeError, match="混同行列保存エラー"):
        evaluator._plot_confusion_matrix(cm, labels, "dummy", DummyModel())


# ============================================================
# eval_conf_mat()
# ============================================================

# 7-8-1 正常系: eval_conf_mat 全体成功
def test_eval_conf_mat_success(monkeypatch, caplog):
    # ログレベルを ERROR から INFO に変更
    caplog.set_level(logging.INFO, logger="myapp")

    evaluator = ModelEvaluator()

    label_map={"cat":0,"dog":1}

    # 各メソッドを正常動作にモック
    monkeypatch.setattr(evaluator, "_prepare_output_dir", lambda m: "dummy_dir")
    monkeypatch.setattr(evaluator, "_extract_label_names", lambda lm: ["cat", "dog"])
    monkeypatch.setattr(evaluator, "_run_inference_loop", lambda *args: ([0, 1], [1, 0]))
    monkeypatch.setattr(evaluator, "_compute_confusion_matrix", lambda a, b: np.array([[1, 1], [1, 1]]))
    # _generate_classification_report が report OK を返すようにモック
    monkeypatch.setattr(evaluator, "_generate_classification_report", lambda *args: "report OK")
    monkeypatch.setattr(evaluator, "_plot_confusion_matrix", lambda *args: None)

    evaluator.eval_conf_mat(label_map, MagicMock(), [])
    
    # _generate_classification_report が返した文字列がログに含まれることを確認
    assert "report OK" in caplog.text


# 7-8-2 異常系: 出力ディレクトリ作成失敗
def test_eval_conf_mat_dir_fail(monkeypatch, caplog):
    evaluator = ModelEvaluator()

    def fake_prepare(*args, **kwargs):
        raise RuntimeError("mkdir failed")

    monkeypatch.setattr(evaluator, "_prepare_output_dir", fake_prepare)

    with pytest.raises(RuntimeError):
        evaluator.eval_conf_mat({"cat": 0}, MagicMock(), [])

    assert "出力ディレクトリ作成失敗" in caplog.text


# 7-8-3 異常系: ラベル名抽出失敗
def test_eval_conf_mat_label_extract_fail(monkeypatch, caplog):
    evaluator = ModelEvaluator()

    monkeypatch.setattr(evaluator, "_prepare_output_dir", lambda m: "dummy")
    monkeypatch.setattr(evaluator, "_extract_label_names", lambda lm: (_ for _ in ()).throw(RuntimeError("label error")))

    with pytest.raises(RuntimeError):
        evaluator.eval_conf_mat(None, MagicMock(), [])

    assert "ラベル名抽出失敗" in caplog.text


# 7-8-4 異常系: 推論ループ致命的エラー
def test_eval_conf_mat_inference_fail(monkeypatch, caplog):
    evaluator = ModelEvaluator()

    monkeypatch.setattr(evaluator, "_prepare_output_dir", lambda m: "dummy")
    monkeypatch.setattr(evaluator, "_extract_label_names", lambda lm: ["cat", "dog"])
    monkeypatch.setattr(evaluator, "_run_inference_loop", lambda *args: (_ for _ in ()).throw(RuntimeError("size mismatch")))

    with pytest.raises(RuntimeError):
        evaluator.eval_conf_mat({"cat": 0}, MagicMock(), [])

    assert "推論ループ全体エラー" in caplog.text


# 7-8-5 異常系: 混同行列計算失敗
def test_eval_conf_mat_confusion_fail(monkeypatch, caplog):
    evaluator = ModelEvaluator()

    monkeypatch.setattr(evaluator, "_prepare_output_dir", lambda m: "dummy")
    monkeypatch.setattr(evaluator, "_extract_label_names", lambda lm: ["cat", "dog"])
    monkeypatch.setattr(evaluator, "_run_inference_loop", lambda *args: ([], []))
    monkeypatch.setattr(evaluator, "_compute_confusion_matrix", lambda a, b: (_ for _ in ()).throw(RuntimeError("cm error")))

    with pytest.raises(RuntimeError):
        evaluator.eval_conf_mat({"cat": 0}, MagicMock(), [])

    assert "混同行列計算失敗" in caplog.text


# 7-8-6 異常系: レポート生成失敗
def test_eval_conf_mat_report_fail(monkeypatch, caplog):
    evaluator = ModelEvaluator()

    monkeypatch.setattr(evaluator, "_prepare_output_dir", lambda m: "dummy")
    monkeypatch.setattr(evaluator, "_extract_label_names", lambda lm: ["cat", "dog"])
    monkeypatch.setattr(evaluator, "_run_inference_loop", lambda *args: ([0, 1], [1, 2]))
    monkeypatch.setattr(evaluator, "_compute_confusion_matrix", lambda a, b: np.array([[1, 1], [1, 1]]))
    monkeypatch.setattr(evaluator, "_generate_classification_report", lambda *args: (_ for _ in ()).throw(RuntimeError("report error")))

    with pytest.raises(RuntimeError):
        evaluator.eval_conf_mat({"cat": 0}, MagicMock(), [])

    assert "レポート生成失敗" in caplog.text


# 7-8-7 異常系: 混同行列保存失敗
def test_eval_conf_mat_save_fail(monkeypatch, caplog):
    evaluator = ModelEvaluator()

    monkeypatch.setattr(evaluator, "_prepare_output_dir", lambda m: "dummy")
    monkeypatch.setattr(evaluator, "_extract_label_names", lambda lm: ["cat", "dog"])
    monkeypatch.setattr(evaluator, "_run_inference_loop", lambda *args: ([0, 1], [1, 0]))
    monkeypatch.setattr(evaluator, "_compute_confusion_matrix", lambda a, b: np.array([[1, 1], [1, 1]]))
    monkeypatch.setattr(evaluator, "_generate_classification_report", lambda *args: "OK")
    monkeypatch.setattr(evaluator, "_plot_confusion_matrix", lambda *args: (_ for _ in ()).throw(RuntimeError("save error")))

    with pytest.raises(RuntimeError):
        evaluator.eval_conf_mat({"cat": 0}, MagicMock(), [])

    assert "混同行列保存失敗" in caplog.text
