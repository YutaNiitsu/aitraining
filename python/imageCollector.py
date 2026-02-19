import time
from icrawler.builtin import BingImageCrawler
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path
import hashlib
from PIL import Image, UnidentifiedImageError
from logging import getLogger

class ImageCollector:
    def __init__(self):
        self.logger = getLogger("myapp")
    
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
            self.split_images(self.temp_dir, output_root, category, val_ratio)
            # 一時保存フォルダ削除
            os.rmdir(self.temp_dir)

    # 指定枚数ダウンロードする
    def imageCrawl(self, keywords, target_num, save_dir, batch_size=200, max_retries=3):
        """
        keywords: 検索キーワードのリスト
        target_num: 収集する画像の総数
        save_dir: 画像保存先ディレクトリ
        batch_size: 1回のクロールで収集する画像数
        max_retries: 収集が進まない場合の最大リトライ回
        """
        hash_set = set()
        unique_files = []
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
                self.remove_duplicate(temp_dir, hash_set, unique_files)
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
        """
        file_path: 画像ファイルパス
        """
        try:
            with Image.open(file_path) as img:
                img = img.convert('RGB')
                return hashlib.md5(img.tobytes()).hexdigest()
        except FileNotFoundError:
            self.logger.error(f"ファイルが見つかりません: {file_path}", exc_info=True)
        except UnidentifiedImageError:
            self.logger.error(f"画像として読み込めません: {file_path}", exc_info=True)
        except PermissionError:
            self.logger.error(f"ファイルの読み込み権限がありません: {file_path}", exc_info=True)
        except Exception as e:
            self.logger.error(f"ハッシュ取得中に予期せぬエラー: {file_path} → {e}", exc_info=True)
        return None


    # 重複画像を削除
    def remove_duplicate(self, target_dir, hash_set, unique_files):
        """
        target_dir: 画像保存先ディレクトリ
        hash_set: 既存のハッシュ値セット
        unique_files: 重複しない画像ファイルリスト
        """
        try:
            files = os.listdir(target_dir)
        except Exception as e:
            self.logger.error(f"ディレクトリの読み込みに失敗しました: {target_dir} → {e}", exc_info=True)
            return

        for file in files:
            path = os.path.join(target_dir, file)
            h = self.get_image_hash(path)

            if h is None:
                # ハッシュ取得失敗
                continue

            if h not in hash_set:
                hash_set.add(h)
                unique_files.append(file)
            else:
                try:
                    os.remove(path)
                except Exception as e:
                    self.logger.error(f"重複画像の削除に失敗: {path} → {e}", exc_info=True)


    # 収集画像を振り分ける
    def split_images(self, target_dir, output_root, category, val_ratio):
        """
        target_dir: 画像保存先ディレクトリ
        output_root: 出力先ルートディレクトリ
        category: カテゴリー名
        val_ratio: 評価用データの割合
        """
        try:
            images = [f for f in os.listdir(target_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        except Exception as e:
            self.logger.error(f"ディレクトリの読み込みに失敗しました: {target_dir} → {e}", exc_info=True)
            return

        if len(images) == 0:
            self.logger.error(f"{category}: 画像が収集できませんでした。", exc_info=True)
            return

        if len(images) < 2:
            self.logger.error(f"{category}: 画像が少なすぎるため split できません（{len(images)} 枚）", exc_info=True)
            return

        try:
            train_imgs, val_imgs = train_test_split(
                images, test_size=val_ratio, random_state=42
            )
        except ValueError as e:
            self.logger.error(f"train_test_split エラー: {e}", exc_info=True)
            return

        train_dir = os.path.join(output_root, category, 'train')
        eval_dir = os.path.join(output_root, category, 'eval')

        try:
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(eval_dir, exist_ok=True)
        except Exception as e:
            self.logger.error(f"出力フォルダ作成エラー: {e}", exc_info=True)
            return

        # ファイル移動
        for img in train_imgs:
            try:
                shutil.move(os.path.join(target_dir, img), os.path.join(train_dir, img))
            except Exception as e:
                self.logger.error(f"ファイル移動エラー（train）: {img} → {e}", exc_info=True)

        for img in val_imgs:
            try:
                shutil.move(os.path.join(target_dir, img), os.path.join(eval_dir, img))
            except Exception as e:
                self.logger.error(f"ファイル移動エラー（eval）: {img} → {e}", exc_info=True)

        print(f"学習用画像枚数: {len(train_imgs)} 検証用画像枚数: {len(val_imgs)}")


    def termination(self):
        # 一時保存フォルダ削除
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"一時フォルダ削除エラー: {self.temp_dir} → {e}", exc_info=True)
