import os
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import glob
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet18
import torchvision.utils as vutils
import cv2
from logging import getLogger
from safe_imread import safe_imread

class ModelEvaluator:
    def __init__(self):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = getLogger("myapp")
        
    def evaluate(self, num_classes, label_map, model, dataloader):
        criterion = nn.CrossEntropyLoss(reduction='none') # 個別サンプルごとの損失を返すように設定
        correct_per_class = [0] * num_classes
        total_per_class = [0] * num_classes
        loss_per_class = [0.0] * num_classes

        # 辞書のキーと値を反転
        reverse_label_map = {v: k for k, v in label_map.items()}

        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                outputs = model(images)

                # 各サンプルごとの損失を計算
                losses = criterion(outputs, labels)

                # 予測クラスを取得
                _, predicted = torch.max(outputs, 1)

                for i in range(labels.size(0)):
                    label = labels[i].item()
                    total_per_class[label] += 1
                    loss_per_class[label] += losses[i].item()
                    if predicted[i].item() == label:
                        correct_per_class[label] += 1

        # クラスごとの精度と平均損失を計算
        for c in range(num_classes):
            if total_per_class[c] > 0:
                acc = correct_per_class[c] / total_per_class[c]
                avg_loss = loss_per_class[c] / total_per_class[c]
                log = f"Class {reverse_label_map[c]}: Accuracy = {acc:.4f}, Avg Loss = {avg_loss:.4f}"
                print(log)
                if hasattr(self, 'logger'):
                    self.logger.info(log)
            else:
                print(f"Class {c}: No samples")
    
    def eval_conf_mat(self, label_map, model, dataloader):
        misclassified_dir = "misclassified_" + f"{model.__class__.__name__}"
        os.makedirs(misclassified_dir, exist_ok=True)
        labels_name = []
        for k, _ in label_map.items():
            labels_name.append(k)
        
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                # CPUに戻してリストに追加
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 誤分類画像を保存
                for i in range(images.size(0)):
                    if predicted[i] != labels[i]:
                        img = images[i].cpu()
                        true_label = labels[i].item()
                        pred_label = predicted[i].item()
                        filename = f"{misclassified_dir}/img_{batch_idx}_{i}_true-{labels_name[true_label]}_pred-{labels_name[pred_label]}.png"
                        vutils.save_image(img, filename, normalize=True)

        # 混同行列を計算
        cm = confusion_matrix(all_labels, all_preds)

        # F1スコアなどを出力
        report = classification_report(all_labels, all_preds, target_names=labels_name, digits=4)
        report = f"{model.__class__.__name__}\n" + report
        print(report)
        self.logger.info(report)

        # 表示
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_name, yticklabels=labels_name)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        #images = glob.glob("misclassified/*.png")
        #for path in images[:10]:  # 最初の10枚だけ表示
        #    img = Image.open(path)
        #    plt.imshow(img)
        #    plt.title(path.split("/")[-1])
        #    plt.axis("off")
        #    plt.show()
    
    