from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, transform):
        self.images = []
        self.labels = []
        self.transform = transform

    def __len__(self):
        try:
            return len(self.images)
        except Exception as e:
            raise RuntimeError(f"データセット長の取得に失敗しました: {e}")

    def __getitem__(self, idx):
        try:
            image = self.images[idx]
        except IndexError:
            raise RuntimeError(f"指定された idx が範囲外です: {idx}")
        except Exception as e:
            raise RuntimeError(f"画像データの取得に失敗しました: {e}")

        try:
            label = self.labels[idx]
        except IndexError:
            raise RuntimeError(f"ラベル idx が範囲外です: {idx}")
        except Exception as e:
            raise RuntimeError(f"ラベルデータの取得に失敗しました: {e}")

        try:
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            raise RuntimeError(f"画像変換(transform) に失敗しました: {e}")

        return image, label
    
    def addData(self, image, label):
        try:
            self.images.append(image)
            self.labels.append(label)
        except Exception as e:
            raise RuntimeError(f"データ追加に失敗しました: {e}")
