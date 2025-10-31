from image_cnnmodel import CNNModel
import torch
import torch.nn as nn
import torch.optim as optim
import logging

class ModelEvaluator:
    def __init__(self):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(
            filename='evaluate.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def evaluate(self, model, dataloader):
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        # 精度と損失を計算
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                outputs = model(images)                        # モデルが予測を出力
                loss = criterion(outputs, labels)              # 損失を計算（予測と正解のズレを数値化）
                total_loss += loss.item()                      # 全体の損失を加算

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)                        # サンプル数を加算
                correct += (predicted == labels).sum().item()  # 正解した数を加算

        # 精度（正答率）  多クラス分類（10クラス以上）の場合0.70以上で良好、0.50以上で改善余地あり
        accuracy = correct / total

        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Loss: {total_loss:.4f}")
        
        logging.info(f"Test: Loss = {total_loss:.4f}, Accuracy = {accuracy:.4f}")