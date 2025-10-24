import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from image_dataset import ImageDataset
from image_cnnmodel import CNNModel
from config_loader import config_learn



class ImageLearn:

    def __init__(self):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # 設定ファイル読み込み
        data_config = config_learn.get('data', {})
        self.keywords = data_config.get('keywords', {})
        output_root = data_config.get('output_root')
        self.train_config = config_learn.get('train', {})
        output = config_learn.get('output')
        self.model_path = output.get('save_model_path')

        # 学習パラメータ
        self.batch_size = self.train_config.get('batch_size', 32)
        self.epochs = self.train_config.get('epochs', 100)
        self.learning_rate = self.train_config.get('learning_rate', 0.001)
        self.shuffle = self.train_config.get('shuffle', True)
        aug_config = self.train_config.get('augmentation', {})

        # データのディレクトリ
        self.data_dirs = []
        for keyword in self.keywords:
            self.data_dirs.append(os.path.join(output_root, keyword, 'train'))
        
        # データ拡張と正規化
        transform_list = [transforms.ToPILImage()]
        if aug_config.get('horizontal_flip', False):
            transform_list.append(transforms.RandomHorizontalFlip())
        if aug_config.get('vertical_flip', False):
            transform_list.append(transforms.RandomVerticalFlip())
        if aug_config.get('rotation', 0) > 0:
            transform_list.append(transforms.RandomRotation(aug_config['rotation']))
        transform_list.append(transforms.ToTensor())
        if aug_config.get('normalize', False):
            transform_list.append(transforms.Normalize((0.5,), (0.5,)))  # RGBなら3タプルに変更

        transform = transforms.Compose(transform_list)
        
        # データセットとデータローダー
        dataset = ImageDataset(self.data_dirs, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

    def createModel(self):
        # モデル定義
        self.model = CNNModel(len(self.keywords)).to(self.device)
        # 損失関数と最適化
        if self.train_config.get('optimizer', 'adam') == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif self.train_config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if self.train_config.get('loss_function', 'cross_entropy') == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif self.train_config['loss_function'] == 'mse':
            self.criterion = nn.MSELoss()

    def readModel(self, path):
        model = CNNModel(len(self.keywords))
        model.load_state_dict(torch.load(path))
        model.to(self.device)
        model.train()  # 学習モード

    # モデルの学習
    def learn(self):
        # 学習ループ
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in self.dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                print(f"Epoch {epoch+1}/100, Loss: {running_loss:.4f}")

    # モデル保存
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        
