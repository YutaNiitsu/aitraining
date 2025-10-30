import yaml
from icrawler.builtin import BingImageCrawler
from sklearn.model_selection import train_test_split
import os
import shutil
from config_loader import config_learn
from pathlib import Path

# ディレクトリ構成
# dataset/
# ├── keyword1/
# │   ├── train/
# │   └── val/
# ├── keyword2/
# │   ├── train/
# │   └── val/

class ImageCrawl:
    def __init__(self):
        # 設定ファイル読み込み
        data_config = config_learn.get('data', {})
        self.keywords = data_config.get('keywords', {})
        self.num_images = data_config.get('num_images')    # 各キーワードで収集する画像数
        self.val_ratio = data_config.get('val_ratio')      # 検証用の割合
        self.output_root = Path(__file__).resolve().parent.parent / data_config.get('output_root')
        
    def crawl(self):
        for keyword in self.keywords:
            # 一時保存フォルダ
            temp_dir = f'temp_{keyword}'
            os.makedirs(temp_dir, exist_ok=True)

            # 画像収集
            crawler = BingImageCrawler(storage={'root_dir': temp_dir})
            crawler.crawl(keyword=keyword, max_num=self.num_images)

            # 画像ファイル一覧
            images = [f for f in os.listdir(temp_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            train_imgs, val_imgs = train_test_split(images, test_size=self.val_ratio, random_state=42)

            # 保存先フォルダ作成
            train_dir = os.path.join(self.output_root, keyword, 'train')
            val_dir = os.path.join(self.output_root, keyword, 'val')
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # ファイル移動
            for img in train_imgs:
                shutil.move(os.path.join(temp_dir, img), os.path.join(train_dir, img))
            for img in val_imgs:
                shutil.move(os.path.join(temp_dir, img), os.path.join(val_dir, img))

            # 一時フォルダ削除
            os.rmdir(temp_dir)

        print("✅ 画像収集と分割が完了しました。")
