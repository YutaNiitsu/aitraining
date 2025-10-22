import os, glob
import cv2
from get_image_hash import get_image_hash
from torch.utils.data import Dataset

# データセット定義
class ImageDataset(Dataset):
    def __init__(self, dirs, transform=None):
        hash_set = set()
        self.images = []
        self.labels = []
        self.transform = transform

        for label_index, dir in enumerate(dirs):
            files = glob.glob(os.path.join(dir, '*.jpg'))
            for file in files:
                img = cv2.imread(file)
                if img is None:
                    continue
                img = cv2.resize(img, (64, 64))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img_hash = get_image_hash(img)
                if img_hash in hash_set:
                    continue  # 重複画像をスキップ
                hash_set.add(img_hash)

                self.images.append(img)
                self.labels.append(label_index)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label