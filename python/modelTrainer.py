from logging import getLogger
import os
from python.image_cnnmodel import CNNModel
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class ModelTrainer:
    def __init__(self):
        # GPU使用可能なら使う
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("使用デバイス:", self.device)
        self.logger = getLogger("myapp")

    def build_model(self, num_classes, used_model, save_model_path):
        """
        num_classes: クラス数
        used_model: 使用するモデル名（'cnn'または'resnet'）
        save_model_path: 学習済みモデルの保存パス
        """
        try:
            if used_model == 'cnn':
                self.model = CNNModel(num_classes)

            elif used_model == 'resnet':
                try:
                    self.model = models.resnet18(pretrained=True) # 事前に訓練されたモデルを使用
                except Exception as e:
                    self.logger.error(f"ResNet の読み込みに失敗しました（pretrained=False で再試行）: {e}", exc_info=True)
                    self.model = models.resnet18(pretrained=False)

                self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

            else:
                raise ValueError(f"不正なモデル名です: {used_model}")

            # 学習済みモデルの読み込み
            if os.path.exists(save_model_path):
                try:
                    print("学習済みモデルを読み込みます")
                    state = torch.load(save_model_path, map_location=self.device)
                    self.model.load_state_dict(state)
                except Exception as e:
                    raise RuntimeError(f"モデルの読み込みに失敗しました: {e}")
        except Exception as e:
            raise RuntimeError(f"モデル構築中にエラーが発生しました: {e}")
        
        # デバイスへ移動
        try:
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"モデルをデバイスへ移動できません: {e}")



    def _validate_dataloader(self, dataloader):
        if dataloader is None:
            self.logger.error("dataloader が None です", exc_info=True)
            return False
        return True

    def _load_train_config(self, train_config):
        try:
            self.epochs = train_config.get('epochs', 100)
            self.learning_rate = train_config.get('learning_rate', 0.001)
        except Exception as e:
            self.logger.error(f"学習設定でエラーが発生しました: {e}", exc_info=True)
            return False

        return True
            

    def _init_optimizer(self, train_config):
        try:
            opt = train_config.get('optimizer', 'adam')
            if opt == 'adam':
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            elif opt == 'sgd':
                self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
            else:
                self.logger.error(f"不正な optimizer: {opt}", exc_info=True)
                return False
        except Exception as e:
            self.logger.error(f"Optimizer 初期化エラー: {e}", exc_info=True)
            return False
        return True

    def _init_loss_function(self, train_config):
        try:
            loss_fn = train_config.get('loss_function', 'cross_entropy')
            if loss_fn == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss()
            elif loss_fn == 'mse':
                self.criterion = nn.MSELoss()
            else:
                self.logger.error(f"不正な loss_function: {loss_fn}", exc_info=True)
                return False
        except Exception as e:
            self.logger.error(f"Loss 関数初期化エラー: {e}", exc_info=True)
            return False
        return True

    def _train_one_epoch(self, dataloader, epoch):
        running_loss = 0.0

        for _, (images, labels) in enumerate(dataloader):
            loss_value, can_continue = self._train_one_batch(images, labels)
            if not can_continue: 
                self.logger.error("致命的エラーのため学習を停止します", exc_info=True) 
                return False
            if loss_value is not None:
                running_loss += loss_value

        print(f"Epoch {epoch+1}/{self.epochs}, Loss: {running_loss:.4f}")
        return running_loss
    
    def _train_one_batch(self, images, labels):
        'returns: (loss_value, 継続可能かどうか)'
        try:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            return loss.item(), True  # True = 継続可能

        except RuntimeError as e:
            msg = str(e)

            # 継続可能なエラー
            if "CUDA out of memory" in msg:
                self.logger.error("CUDA メモリ不足。バッチをスキップします", exc_info=True)
                torch.cuda.empty_cache()
                return None, True

            if "size mismatch" in msg or "shape" in msg:
                self.logger.error(f"バッチの shape エラー: {e} → スキップ", exc_info=True)
                return None, True

            # 継続不可能なエラー
            if "device-side assert triggered" in msg:
                self.logger.error("GPU の致命的エラー。学習を停止します", exc_info=True)
                torch.cuda.empty_cache()
                return None, False

            if "cuDNN error" in msg:
                self.logger.error("cuDNN の致命的エラー。学習を停止します", exc_info=True)
                return None, False

            # その他の RuntimeError
            self.logger.error(f"学習中にエラーが発生: {e}", exc_info=True)
            return None, True

        except Exception as e:
            self.logger.error(f"学習中に予期せぬエラー: {e}", exc_info=True)
            return None, False

    def train(self, dataloader, train_config):
        # dataloader の検証
        try:
            if not self._validate_dataloader(dataloader):
                self.logger.error("dataloader が不正のため学習を中止します", exc_info=True)
                return False
        except Exception as e:
            self.logger.error(f"dataloader 検証中に例外発生: {e}", exc_info=True)
            return False

        # 設定ロード
        try:
            self._load_train_config(train_config)
        except Exception as e:
            self.logger.error(f"train_config の読み込みに失敗しました: {e}", exc_info=True)
            return False

        # optimizer 初期化
        try:
            if not self._init_optimizer(train_config):
                self.logger.error("optimizer の初期化に失敗しました", exc_info=True)
                return False
        except Exception as e:
            self.logger.error(f"optimizer 初期化中に例外発生: {e}", exc_info=True)
            return False

        # loss 関数初期化
        try:
            if not self._init_loss_function(train_config):
                self.logger.error("loss 関数の初期化に失敗しました", exc_info=True)
                return False
        except Exception as e:
            self.logger.error(f"loss 関数初期化中に例外発生: {e}", exc_info=True)
            return False

        # エポックループ
        for epoch in range(self.epochs):
            try:
                result = self._train_one_epoch(dataloader, epoch)
            except Exception as e:
                self.logger.error(f"エポック {epoch} 実行中に例外発生: {e}", exc_info=True)
                return False

            # _train_one_epoch が False を返したら致命的エラー
            if result is False:
                self.logger.error(f"エポック {epoch} で致命的エラーが発生したため学習を停止します", exc_info=True)
                return False

        self.logger.error("全エポックが正常に完了しました")
        return True
    

    def save_model(self, path):
        # path のチェック
        if not isinstance(path, str) or path.strip() == "":
            self.logger.error("path が不正です（空文字または非文字列）", exc_info=True)
            return False

        dir_path = os.path.dirname(path)

        # ディレクトリ作成
        try:
            if dir_path:  # 空文字の場合はスキップ
                os.makedirs(dir_path, exist_ok=True)
        except PermissionError:
            self.logger.error(f"ディレクトリ作成権限がありません: {dir_path}", exc_info=True)
            return False
        except FileNotFoundError:
            self.logger.error(f"ディレクトリパスが不正です: {dir_path}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"ディレクトリ作成中に予期せぬエラー: {e}", exc_info=True)
            return False

        # モデル保存
        try:
            state = self.model.state_dict()
        except Exception as e:
            self.logger.error(f"state_dict の取得に失敗しました: {e}", exc_info=True)
            return False

        try:
            torch.save(state, path)
        except PermissionError:
            self.logger.error(f"ファイル保存権限がありません: {path}", exc_info=True)
            return False
        except FileNotFoundError:
            self.logger.error(f"保存先パスが不正です: {path}", exc_info=True)
            return False
        except OSError as e:
            self.logger.error(f"ファイルシステムエラー: {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"モデル保存中に予期せぬエラー: {e}", exc_info=True)
            return False

        print(f"モデルを保存しました: {path}")
        return True
