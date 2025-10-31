from get_image_hash import get_image_hash
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# データセット定義
class ImageDataset(Dataset):
    def __init__(self, transform):
        self.images = []
        self.labels = []
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def addData(self, image, label):
        self.images.append(image)
        self.labels.append(label)