import torch
import torchvision.transforms as transforms
import cv2
from image_cnnmodel import CNNModel
from config_loader import config_labels

class ImagePredict:
    def __init__(self):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        self.label_map = config_labels['label_map']

    def readModel(self, path):
        # モデル読み込み
        self.model = CNNModel(3).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()

    def readImage(self, path):
        # 画像読み込みと前処理
        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, (64, 64))

    def predict(self):
        # Tensor変換 + 正規化
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(self.image).unsqueeze(0).to(self.device)  # shape: [1, 3, 64, 64]

        # 推論
        with torch.no_grad():
            outputs = self.model(image_tensor)
            predictions = torch.softmax(outputs, dim=1)
            answer = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0, answer].item()
        
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        predicted_class = reverse_label_map[answer]

        print(f"予測: {predicted_class}, 信頼度: {confidence:.4f}")
        return predicted_class, confidence

