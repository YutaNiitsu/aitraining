import glob
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from image_dataset import ImageDataset
from safe_imread import safe_imread

class Preprocessor:
    def preprocessor(self, data_dirs, train_config, image_size, eval):
        aug_config = train_config.get('augmentation', {})
        batch_size = train_config.get('batch_size', 32)
        shuffle = train_config.get('shuffle', True)
        # データ拡張と正規化
        transform_list = None
        if not eval:
            transform_list = [transforms.ToPILImage()]
            if aug_config.get('horizontal_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug_config.get('vertical_flip', False):
                transform_list.append(transforms.RandomVerticalFlip())
            if aug_config.get('rotation', 0) > 0:
                transform_list.append(transforms.RandomRotation(aug_config['rotation']))
            if "color_jitter" in aug_config:
                cj = aug_config["color_jitter"]
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=cj.get("brightness", 0),
                        contrast=cj.get("contrast", 0),
                        saturation=cj.get("saturation", 0),
                        hue=cj.get("hue", 0),
                    )
                )
            if aug_config.get('random_crop', False):
                transform_list.append(transforms.RandomResizedCrop(224))

            transform_list.append(transforms.ToTensor())
            if aug_config.get('normalize', False):
                transform_list.append(
                    transforms.Normalize(mean=[0.5, ],
                                        std=[0.5, ])
                )
        else:
            transform_list = [transforms.ToTensor()]
        
        transform = transforms.Compose(transform_list)
        # データセットのインスタンス生成
        dataset = ImageDataset(transform)
        # 前処理
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        for label_index, dir in enumerate(data_dirs):
            files = [
                os.path.join(dir, f)
                for f in os.listdir(dir)
                if f.lower().endswith(valid_exts)    # f.lower()によって大文字拡張子
            ]
            for file in files:
                img = safe_imread(file)
                if img is None:
                    continue
                # リサイズと色空間の変換
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # データセットに追加
                dataset.addData(img, label_index)
        
        self.dataLoader = DataLoader(dataset, batch_size, shuffle)