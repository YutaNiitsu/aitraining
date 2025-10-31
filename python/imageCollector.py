import time
from icrawler.builtin import BingImageCrawler
from sklearn.model_selection import train_test_split
import os
import shutil

class ImageCollector:
    def collect(self, keyword, output_root, data_config):
        num_images = data_config.get('num_images')    # 各キーワードで収集する画像数
        val_ratio = data_config.get('val_ratio')      # 検証用の割合
        # 一時保存フォルダ作成
        temp_dir = f'temp_{keyword}'
        os.makedirs(temp_dir, exist_ok=True)
        # 画像収集して一時保存フォルダに保存
        print(f"キーワード: {keyword} 収集する画像数: {num_images} 検証用の割合: {val_ratio}" )
        self.imageCrawl(keyword=keyword, target_num=num_images, save_dir=temp_dir)
        # 画像ファイルの取得
        images = [f for f in os.listdir(temp_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(images) == 0:
            print(f"{keyword}: 画像が収集できませんでした。")
            return
        # 学習用と検証用に分ける
        train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=42)
        # 保存先フォルダ作成
        train_dir = os.path.join(output_root, keyword, 'train')
        eval_dir = os.path.join(output_root, keyword, 'eval')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        # ファイル移動
        for img in train_imgs:
            shutil.move(os.path.join(temp_dir, img), os.path.join(train_dir, img))
        for img in val_imgs:
            shutil.move(os.path.join(temp_dir, img), os.path.join(eval_dir, img))
        print(f"学習用画像枚数: {len(train_imgs)} 検証用画像枚数: {len(val_imgs)}")
        # 一時保存フォルダ削除
        os.rmdir(temp_dir)

    # 指定枚数ダウンロードする
    def imageCrawl(self, keyword, target_num, save_dir, batch_size=100, max_retries=5):
        os.makedirs(save_dir, exist_ok=True)
        total_saved = len(os.listdir(save_dir))
        try_times = 0          # 総試行回数
        retries = 0            # リトライ回数
        # 最大リトライ回数に達したら終了
        while total_saved < target_num and retries < max_retries:
            remaining = target_num - total_saved
            print(f"収集中: 残り {remaining} 枚")
            # Bing Image Search APIの1回の呼び出しで取得できる画像の最大数は150枚
            crawler = BingImageCrawler(storage={'root_dir': save_dir}, log_level='ERROR')   # GoogleImageCrawlerだと失敗（botをブロック？）
            max_num=min(batch_size, remaining)
            crawler.crawl(keyword=keyword, max_num=max_num, offset=try_times)
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

        print(f"収集完了: {total_saved} 枚保存されました（目標: {target_num} 枚）")