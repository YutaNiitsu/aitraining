import os
from configManager import ConfigManager
from imageCollector import ImageCollector
from modelEvaluator import ModelEvaluator
from modelTrainer import ModelTrainer
from preprocessor import Preprocessor
import schedule
import time
from pathlib import Path

class ModelTrainerApp:
    def __init__(self):
        self.config_mgr = ConfigManager(Path(__file__).resolve().parent.parent / 'config' / 'learn.yaml')
        self.collector = ImageCollector()
        self.preprocessor = Preprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

    def run_daily(self):
        # 設定ファイル更新の確認
        if self.config_mgr.is_updated():
            print("設定ファイルの更新を確認しました。")
            self.config_mgr.load_config()
            self.config_mgr.update_timestamp()
        else:
            print("設定ファイルは未更新です。")
            return

        keywords = self.config_mgr.config['data']['keywords']
        output_root = Path(__file__).resolve().parent.parent / self.config_mgr.config['data']['output_root']
        save_model_path = Path(__file__).resolve().parent.parent / self.config_mgr.config['output']['save_model_path']
        data_config = self.config_mgr.config['data']
        train_config = self.config_mgr.config['train']

        # スクレイピングと分割
        for keyword in keywords:
            self.collector.collect(keyword, output_root, data_config)

        # 前処理
        # データのディレクトリ
        train_data_dirs = []
        eval_data_dirs = []
        for keyword in keywords:
            train_data_dirs.append(os.path.join(output_root, keyword, 'train'))
            eval_data_dirs.append(os.path.join(output_root, keyword, 'eval'))
        # 前処理
        self.preprocessor.preprocessor(train_data_dirs, train_config)
        # 学習用データセット
        train_dataLoader = self.preprocessor.dataLoader
        self.preprocessor.preprocessor(eval_data_dirs, train_config)
        # 検証用データセット
        eval_dataLoader = self.preprocessor.dataLoader
        # 学習
        self.trainer.build_model(num_classes=len(keywords), save_model_path=save_model_path)
        self.trainer.train(train_dataLoader, train_config)
        self.trainer.save_model(save_model_path)
        # 評価
        self.evaluator.evaluate(self.trainer.model, eval_dataLoader)

    def run(self):
        schedule.every().day.at("06:00").do(self.run_daily)
        while True:
            schedule.run_pending()
            time.sleep(60)