import os
import signal
import platform
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
        self.ppath = None
        os_name = platform.system()
        if os_name == 'Windows':
            self.ppath = Path("C:/Users/yniit/Documents/aitraining").resolve()
        elif os_name == 'Linux':
            self.ppath = Path(__file__).resolve().parent.parent
        learn_conf_path = self.ppath / 'config' / 'learn.yaml'
        log_conf_path = self.ppath / 'config' / 'logging.yaml'
        labels_path = self.ppath / 'config' / 'labels.yaml'
        self.config_mgr = ConfigManager(learn_conf_path, log_conf_path, labels_path)
        self.config_mgr.load_config()
        self.collector = ImageCollector()
        self.preprocessor = Preprocessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def run_daily(self):
        # 設定ファイル更新の確認
        if self.config_mgr.is_updated():
            print("設定ファイルの更新を確認しました。")
            self.config_mgr.load_config()
            self.config_mgr.update_timestamp()
        else:
            print("設定ファイルは未更新です。")
            return
        
        data_config = self.config_mgr.learn_config['data']
        train_config = self.config_mgr.learn_config['train']
        categorys = data_config['categorys']
        image_size = data_config['image_size']
        output_root = Path(__file__).resolve().parent.parent / data_config['output_root']
        label_map = self.config_mgr.labels['label_map']
        using_model = train_config['using_model']
        save_model_path = ''
        if using_model == 'cnn':
            save_model_path = self.ppath / self.config_mgr.learn_config['output']['cnn_model']['save_model_path']
        elif using_model == 'resnet':
            save_model_path = self.ppath / self.config_mgr.learn_config['output']['resnet_model']['save_model_path']
        else:
            print("using_modelが正しくありません。")
            return
        
        # スクレイピングと分割
        #self.collector.collect(data_config)
        val_ratio = data_config.get('val_ratio') 
        for category, _ in categorys.items():
            tar_dir = f'python/temp_{category}'
            self.collector.remove_duplicate(tar_dir)
            #self.collector.split_images(tar_dir, output_root, category, val_ratio)
        return
        # 前処理
        # データのディレクトリ
        train_data_dirs = []
        eval_data_dirs = []
        for category in categorys:
            train_data_dirs.append(os.path.join(output_root, category, 'train'))
            eval_data_dirs.append(os.path.join(output_root, category, 'eval'))

        # 前処理
        # 学習用データセット
        self.preprocessor.preprocessor(train_data_dirs, train_config, image_size, False)
        train_dataLoader = self.preprocessor.dataLoader
        
        # 検証用データセット
        self.preprocessor.preprocessor(eval_data_dirs, train_config, image_size, True)
        eval_dataLoader = self.preprocessor.dataLoader
        # 学習
        self.trainer.build_model(len(categorys), using_model, save_model_path)
        #self.trainer.train(train_dataLoader, train_config)
        #self.trainer.save_model(save_model_path)
        # 評価
        #self.evaluator.evaluate(len(categorys), self.label_map, self.trainer.model, eval_dataLoader)   
        self.evaluator.eval_conf_mat(label_map, self.trainer.model, eval_dataLoader)
        #self.evaluator.grad_cam(self.trainer.model, os.path.join(output_root, 'dog', 'eval', '000013.jpg'))

    def run(self):
        schedule.every().day.at("06:00").do(self.run_daily)
        while True:
            schedule.run_pending()
            time.sleep(60)

    def handle_signal(self):
        print("\n中断されました。終了処理を実行します…")
        self.collector.termination()