import glob
from logging import getLogger
import os
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from python.image_dataset import ImageDataset
from safe_imread import safe_imread

class Preprocessor:
    def __init__(self):
        self.logger = getLogger("myapp") 


    def _load_train_config(self, train_config):
        aug_config = train_config.get("augmentation", {})
        if not isinstance(aug_config, dict):
            raise TypeError("augmentation は dict である必要があります")

        batch_size = train_config.get("batch_size", 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size は正の整数である必要があります")

        shuffle = train_config.get("shuffle", True)
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle は bool である必要があります")

        return aug_config, batch_size, shuffle


    def _build_transform(self, aug_config, is_eval):
        
        if not isinstance(aug_config, dict):
            raise TypeError("aug_config は dict である必要があります")

        try:
            if is_eval:
                transform_list = [transforms.ToTensor()]
                if aug_config.get("normalize", False):
                    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
                return transforms.Compose(transform_list)

            transform_list = [transforms.ToPILImage()]

            if aug_config.get("horizontal_flip", False):
                transform_list.append(transforms.RandomHorizontalFlip())
            if aug_config.get("vertical_flip", False):
                transform_list.append(transforms.RandomVerticalFlip())

            rotation = aug_config.get("rotation", 0)
            if not isinstance(rotation, int):
                raise ValueError("rotation は整数である必要があります")
            if rotation > 0:
                transform_list.append(transforms.RandomRotation(rotation))

            if "color_jitter" in aug_config:
                cj = aug_config["color_jitter"]
                if not isinstance(cj, dict):
                    raise TypeError("color_jitter は dict である必要があります")
                transform_list.append(
                    transforms.ColorJitter(
                        brightness=cj.get("brightness", 0),
                        contrast=cj.get("contrast", 0),
                        saturation=cj.get("saturation", 0),
                        hue=cj.get("hue", 0),
                    )
                )

            if aug_config.get("random_crop", False):
                transform_list.append(transforms.RandomResizedCrop(224))

            transform_list.append(transforms.ToTensor())

            if aug_config.get("normalize", False):
                transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

            return transforms.Compose(transform_list)

        except Exception as e:
            raise RuntimeError(f"Transform 構築エラー: {e}")


    def _create_dataset(self, transform):
        if transform is None:
            raise ValueError("transform が None です")
        try:
            return ImageDataset(transform)
        except Exception as e:
            self.logger.error(f"Dataset 作成エラー: {e}")
            return None


    def _collect_files(self, data_dirs):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files_by_label = {}

        for label_index, dir in enumerate(data_dirs):
            if not isinstance(dir, str):
                self.logger.error(f"無効なディレクトリ名: {dir}")
                continue

            if not os.path.exists(dir):
                self.logger.error(f"ディレクトリが存在しません: {dir}")
                continue

            try:
                files = [
                    os.path.join(dir, f)
                    for f in os.listdir(dir)
                    if f.lower().endswith(valid_exts)
                ]
            except Exception as e:
                self.logger.error(f"ディレクトリ読み込みエラー: {dir} → {e}")
                continue

            if len(files) == 0:
                self.logger.error(f"画像ファイルが見つかりません: {dir}")

            files_by_label[label_index] = files

        return files_by_label

    def _load_and_resize_image(self, file_path, image_size):
        """
        safe_imread → cv2.resize の一連処理
        """
        try:
            img = safe_imread(file_path)
        except Exception as e:
            self.logger.error(f"safe_imread でエラー: {file_path} → {e}")
            return None

        if img is None:
            self.logger.error(f"読み込み不可の画像をスキップ: {file_path}")
            return None

        try:
            return cv2.resize(img, image_size)
        except Exception as e:
            self.logger.error(f"cv2.resize 失敗: {file_path} → {e}")
            return None


    def _add_images_to_dataset(self, dataset, files_by_label, image_size):
        if dataset is None:
            raise ValueError("dataset が None です")

        for label_index, files in files_by_label.items():
            for file in files:
                img = self._load_and_resize_image(file, image_size)
                if img is None:
                    continue

                try:
                    dataset.addData(img, label_index)
                except Exception as e:
                    self.logger.error(f"dataset.addData 失敗: {file} → {e}")


    def _create_dataloader(self, dataset, batch_size, shuffle):
        if len(dataset) == 0:
            raise ValueError("Dataset が空のため DataLoader は作成されません")

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size は正の整数である必要があります")

        if not isinstance(shuffle, bool):
            raise TypeError("shuffle は bool である必要があります")

        try:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        except Exception as e:
            raise RuntimeError(f"DataLoader 初期化エラー: {e}")


    def preprocessor(self, data_dirs, train_config, image_size, is_eval):
        """
        data_dirs: データディレクトリのリスト
        train_config: 学習設定
        image_size: 画像リサイズサイズ (幅, 高さ)
        is_eval: 評価用データかどうかのフラグ
        """
        # --- 入力チェック ---
        if not isinstance(data_dirs, list):
            raise TypeError("data_dirs は list である必要があります")

        if not isinstance(train_config, dict):
            raise TypeError("train_config は dict である必要があります")

        if not (isinstance(image_size, tuple) and len(image_size) == 2):
            raise ValueError("image_size は (width, height) のタプルである必要があります")

        # 設定読み込み
        aug_config, batch_size, shuffle = self._load_train_config(train_config)

        # Transform構築
        transform = self._build_transform(aug_config, is_eval)

        # Dataset作成
        dataset = self._create_dataset(transform)
        if dataset is None:
            raise RuntimeError("Dataset の作成に失敗しました")
        
        # ファイル収集
        files_by_label = self._collect_files(data_dirs)
        if all(len(v) == 0 for v in files_by_label.values()):
            self.logger.error("有効な画像ファイルが見つかりませんでした")

        # Datasetに画像追加
        self._add_images_to_dataset(dataset, files_by_label, image_size)

        if len(dataset) == 0:
            self.logger.error("Dataset が空です。DataLoader は None になります")
            self.dataLoader = None
            return

        # DataLoader作成
        self.dataLoader = self._create_dataloader(dataset, batch_size, shuffle)
