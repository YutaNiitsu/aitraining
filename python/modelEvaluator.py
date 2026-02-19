import datetime
import os
import torch
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from logging import getLogger

class ModelEvaluator:
    def __init__(self):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = getLogger("myapp")

    # 出力ディレクトリ作成
    def _prepare_output_dir(self, model):
        try:
            dir_name = f"misclassified_{model.__class__.__name__}"
            os.makedirs(dir_name, exist_ok=True)
            return dir_name
        except Exception as e:
            raise RuntimeError(f"誤分類保存ディレクトリ作成エラー: {e}")

    # ラベル名抽出
    def _extract_label_names(self, label_map):
        try:
            return list(label_map.keys())
        except Exception as e:
            raise RuntimeError(f"ラベル名抽出エラー: {e}")

    # 推論ループ
    def _run_inference_loop(self, model, dataloader, labels_name, misclassified_dir):
        all_labels = []
        all_preds = []

        model.eval()

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                try:
                    images = images.to(self.device)
                    labels = labels.to(self.device).long()

                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    self._save_misclassified_images(
                        images, labels, predicted, batch_idx, labels_name, misclassified_dir
                    )

                except Exception as e:
                    # shape mismatch は致命的
                    msg = str(e).lower()
                        
                    if "mat1 and mat2 shapes cannot be multiplied" in msg:
                        raise RuntimeError(f"モデル出力 shape エラー: {e}")
                        
                    if "size mismatch" in msg:
                        raise RuntimeError(f"サイズ不一致: {e}")

                    # それ以外は継続
                    self.logger.error(f"推論ループ中の RuntimeError（継続）: {e}")
                    continue

        return all_labels, all_preds


    # 誤分類画像保存
    def _save_misclassified_images(self, images, labels, predicted, batch_idx, labels_name, out_dir):
        for i in range(images.size(0)):
            if predicted[i] != labels[i]:
                try:
                    img = images[i].cpu()
                    true_label = labels_name[labels[i].item()]
                    pred_label = labels_name[predicted[i].item()]

                    filename = f"{out_dir}/img_{batch_idx}_{i}_true-{true_label}_pred-{pred_label}.png"
                    vutils.save_image(img, filename, normalize=True)

                except Exception as e:
                    self.logger.error(f"誤分類画像保存エラー: {e}")

    # 混同行列計算
    def _compute_confusion_matrix(self, all_labels, all_preds):
        try:
            return confusion_matrix(all_labels, all_preds)
        except Exception as e:
            raise RuntimeError(f"混同行列計算エラー: {e}")

    # レポート生成
    def _generate_classification_report(self, all_labels, all_preds, labels_name, model):
        try:
            report = classification_report(
                all_labels, all_preds, target_names=labels_name, digits=4
            )
            return f"{model.__class__.__name__}\n{report}"
        except Exception as e:
            raise RuntimeError(f"レポート生成エラー: {e}")

    # 混同行列の保存（日時＋モデル名）
    def _plot_confusion_matrix(self, cm, labels_name, save_dir, model):
        # 保存ディレクトリ作成
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"混同行列保存ディレクトリ作成エラー: {e}")

        # ファイル名生成
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = model.__class__.__name__
            filename = f"confmat_{model_name}_{timestamp}.png"
            save_path = os.path.join(save_dir, filename)
        except Exception as e:
            raise RuntimeError(f"ファイル名生成エラー: {e}")

        # 描画処理
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels_name,
                yticklabels=labels_name
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title(f"Confusion Matrix ({model_name})")
        except Exception as e:
            raise RuntimeError(f"混同行列描画エラー: {e}")

        # 保存処理
        try:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"混同行列を保存しました: {save_path}")
        except Exception as e:
            raise RuntimeError(f"混同行列保存エラー: {e}")


    def eval_conf_mat(self, label_map, model, dataloader):
        """
        label_map: ラベルマップ
        model: 学習済みモデル
        dataloader: 評価用データローダー
        """

        # 出力ディレクトリ
        try:
            misclassified_dir = self._prepare_output_dir(model)
            confusion_matrix_dir = "confusion_matrices"
        except Exception as e:
            self.logger.error(f"出力ディレクトリ作成失敗: {e}")
            raise

        # ラベル名抽出
        try:
            labels_name = self._extract_label_names(label_map)
        except Exception as e:
            self.logger.error(f"ラベル名抽出失敗: {e}")
            raise

        # 推論ループ
        try:
            all_labels, all_preds = self._run_inference_loop(
                model, dataloader, labels_name, misclassified_dir
            )
        except Exception as e:
            self.logger.error(f"推論ループ全体エラー: {e}")
            raise

        # 混同行列計算
        try:
            cm = self._compute_confusion_matrix(all_labels, all_preds)
        except Exception as e:
            self.logger.error(f"混同行列計算失敗: {e}")
            raise

        # レポート生成
        try:
            report = self._generate_classification_report(
                all_labels, all_preds, labels_name, model
            )
            self.logger.info(report)
        except Exception as e:
            self.logger.error(f"レポート生成失敗: {e}")
            raise

        # 混同行列保存
        try:
            self._plot_confusion_matrix(cm, labels_name, confusion_matrix_dir, model)
        except Exception as e:
            self.logger.error(f"混同行列保存失敗: {e}")
            raise

