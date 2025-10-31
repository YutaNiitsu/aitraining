import os
from image_cnnmodel import CNNModel
import torch
import torch.nn as nn
import torch.optim as optim

class ModelTrainer:
    def __init__(self):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("使用デバイス:", self.device)

    def build_model(self, num_classes, save_model_path):
        # モデルのインスタンスを生成
        self.model = CNNModel(num_classes)
        if os.path.exists(save_model_path):
            print("学習済みモデルを読み込みます")
            self.model.load_state_dict(torch.load(save_model_path))
            
        self.model.to(self.device)

    def train(self, dataloader, train_config):
        # 学習パラメータ
        self.epochs = train_config.get('epochs', 100)
        self.learning_rate = train_config.get('learning_rate', 0.001)
        # 最適化手法 #self.model.parameters()の参照を渡す（モデルのパラメータを登録）
        if train_config.get('optimizer', 'adam') == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        elif train_config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # 損失関数
        if train_config.get('loss_function', 'cross_entropy') == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif train_config['loss_function'] == 'mse':
            self.criterion = nn.MSELoss()
        # 学習ループ
        for epoch in range(self.epochs):
            running_loss = 0.0    # 累積損失
            for images, labels in dataloader:
                # デバイスにデータを移動
                images = images.to(self.device)
                labels = labels.to(self.device)
                # 前回の勾配情報をリセット
                self.optimizer.zero_grad()
                # 順伝播（Forward）と損失計算
                outputs = self.model(images)            # モデルが予測を出力
                loss = self.criterion(outputs, labels)  # 損失を計算（予測と正解のズレを数値化）
                # 逆伝播（Backward）とパラメータ更新
                loss.backward()                         # 損失を計算し、各パラメータに対する勾配（∂Loss/∂W）を求める
                self.optimizer.step()                   # 保持しているモデルのパラメータを更新してモデルを改善（学習）

                running_loss += loss.item()
                print(f"Epoch {epoch+1}/100, Loss: {running_loss:.4f}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)