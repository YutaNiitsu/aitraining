import time
from icrawler.builtin import BingImageCrawler
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path
import hashlib
from PIL import Image
from logging import getLogger

class ImageCollector:
    def __init__(self):
        self.logger = getLogger(__name__)
    
    def collect(self, data_config):
        num_images = data_config.get('num_images')    # 各キーワードで収集する画像数
        val_ratio = data_config.get('val_ratio')      # 検証用の割合
        categorys = data_config['categorys']
        output_root = Path(__file__).resolve().parent.parent / data_config['output_root']

        for category, info in categorys.items():
            # 一時保存フォルダ作成
            self.temp_dir = f'temp_{category}'
            os.makedirs(self.temp_dir, exist_ok=True)
            # 画像収集して一時保存フォルダに保存
            print(f"カテゴリー: {category} 収集する画像数: {num_images} 検証用の割合: {val_ratio}" )
            self.imageCrawl(keywords=info['keywords'], target_num=num_images, save_dir=self.temp_dir)
            self.split_images(output_root, category, val_ratio)
            # 一時保存フォルダ削除
            os.rmdir(self.temp_dir)

    # 指定枚数ダウンロードする
    def imageCrawl(self, keywords, target_num, save_dir, batch_size=200, max_retries=3):
        total_saved = 0   # 総収集枚数
        # 一時保存フォルダ
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        # キーワードごとに検索
        for keyword in keywords:
            print(f"キーワード: {keyword}")
            try_times = 0          # 総試行回数
            retries = 0            # リトライ回数
            
            # 指定枚数に達したら終了
            # 最大リトライ回数に達したら終了
            # 総試行回数が1000超えたら終了
            while total_saved < target_num and retries < max_retries:
                remaining = target_num - total_saved
                max_num=min(batch_size, remaining)
                print(f"収集中: 残り {remaining} 枚")
                crawler = BingImageCrawler(downloader_threads=4, storage={'root_dir': temp_dir}, log_level='ERROR')   # GoogleImageCrawlerだと失敗（botをブロック？）
                # 画像収集
                crawler.crawl(keyword=keyword, max_num=max_num, offset=try_times, file_idx_offset=total_saved)
                # 重複画像を削除
                self.remove_duplicate(temp_dir)
                # ファイル移動
                if os.path.exists(temp_dir):
                    images = [f for f in os.listdir(temp_dir)]
                    try:
                        for img in images:
                            shutil.move(os.path.join(temp_dir, img), os.path.join(save_dir, img))
                        for f in os.listdir(temp_dir):
                            os.remove(os.path.join(temp_dir, f))
                    except Exception as e:
                        print(f"❌ ファイル移動中に予期せぬエラーが発生しました: {e}")

                # 総試行回数を更新
                try_times += max_num
                new_total = len(os.listdir(save_dir))
                if new_total == total_saved:
                    # 収集が進まない場合はリトライ
                    retries += 1
                    print(f"収集枚数が増えていません。リトライ {retries}/{max_retries}")
                    time.sleep(2)  # 少し待ってから再試行
                else:
                    retries = 0  # 成功したらリトライカウントをリセット
                total_saved = new_total
                
            # 指定枚数に達したら終了
            if total_saved >= target_num:
                break

        os.rmdir(temp_dir)
        msg = f"収集完了: {total_saved} 枚保存されました（目標: {target_num} 枚）"
        print(msg)
        if hasattr(self, 'logger'):
            self.logger.info(msg)

    # ハッシュ値を取得
    def get_image_hash(self, file_path):
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            return hashlib.md5(img.tobytes()).hexdigest()
        
    # 重複画像を削除
    def remove_duplicate(self, target_dir):
        hash_set = set()
        unique_files = []
        for file in os.listdir(target_dir):
            path = os.path.join(target_dir, file)
            try:
                h = self.get_image_hash(path)
                if h not in hash_set:
                    hash_set.add(h)
                    unique_files.append(file)
                else:
                    os.remove(path)
            except Exception as e:
                print(f"❌ ハッシュ化中に予期せぬエラーが発生しました: {e}")
                continue
    
    # 収集画像を振り分ける
    def split_images(self, target_dir, output_root, category, val_ratio):
        # 画像ファイルの取得
        images = [f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(images) == 0:
            print(f"{category}: 画像が収集できませんでした。")
            return
        
        # 学習用と検証用に分ける
        train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=42)
        # 保存先フォルダ作成
        train_dir = os.path.join(output_root, category, 'train')
        eval_dir = os.path.join(output_root, category, 'eval')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        # ファイル移動
        for img in train_imgs:
            shutil.move(os.path.join(target_dir, img), os.path.join(train_dir, img))
        for img in val_imgs:
            shutil.move(os.path.join(target_dir, img), os.path.join(eval_dir, img))
        print(f"学習用画像枚数: {len(train_imgs)} 検証用画像枚数: {len(val_imgs)}")
    
    def termination(self):
        # 一時保存フォルダ削除
        os.rmdir(self.temp_dir)