import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torchvision.models import resnet18
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
        labels_name = []
        for k, _ in label_map.items():
            labels_name.append(k)
        
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)

                # CPUに戻してリストに追加
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 混同行列を計算
        cm = confusion_matrix(all_labels, all_preds)

        # F1スコアなどを出力
        report = classification_report(all_labels, all_preds, target_names=labels_name, digits=4)
        print(report)

        # 表示
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_name, yticklabels=labels_name)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()
    
    def grad_cam(self, model, image_path):
        model.eval()
        # Grad-CAM対象の層（最後のConv層）を指定
        target_layer = model.conv[2]
        # Grad-CAMインスタンス作成
        cam = GradCAM(model=model, target_layers=[target_layer])

        # 入力画像の読み込みと前処理
        rgb_img = safe_imread(image_path)
        rgb_img = cv2.resize(rgb_img, (32, 32))  # モデルの入力サイズに合わせる
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # Grad-CAMの実行
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]

        # ヒートマップの重ね合わせ
        visualization = show_cam_on_image(rgb_img / 255., grayscale_cam, use_rgb=True)
        cv2.imwrite("gradcam_result.png", visualization)