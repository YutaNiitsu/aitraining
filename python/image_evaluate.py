import os, glob
import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from image_dataset import ImageDataset
from image_cnnmodel import CNNModel
from config_loader import config_learn
import logging

class ImageEvaluate:
    def __init__(self, config_path='config.yaml'):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ログ設定（ファイル出力とコンソール出力）
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler("training.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        # 設定ファイル読み込み
        self.keywords = config_learn['keywords']
        output_root = config_learn['output_root']
        self.model_path = config_learn['save_model_path']

        # データのディレクトリ
        self.data_dirs = []
        for keyword in self.keywords:
            self.data_dirs.append(os.path.join(output_root, keyword, 'train'))
        
        # 変換（正規化）
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # データセットとローダー
        dataset = ImageDataset(self.data_dirs, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # 表示（ランダムに25枚）
        plt.figure(figsize=(10,10))
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            index = random.randrange(len(dataset))
            img, label = dataset[index]
            img_np = img.permute(1, 2, 0).numpy()
            plt.imshow(img_np)
            plt.xlabel(str(label))
        plt.show()

        
    
    def readModel(self):
        # モデル読み込み
        self.model = CNNModel(len(self.keywords)).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def evaluate(self):
        # 評価
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_acc = correct / total

        # ログ出力
        logging.info(f"Accuracy: {test_acc:.4f}, Loss: {total_loss:.4f}")

        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {total_loss:.4f}")
