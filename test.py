import yaml
from omegaconf import OmegaConf
from yolo.module.model_module import ModelModule
from yolo.module.data_module import DataModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import os
import datetime

def main():
    base_save_dir = './result'
    
    # 実行時のタイムスタンプを付与して、一意のディレクトリ名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(base_save_dir, timestamp)

    # configファイルを読み込み
    yaml_file = "./config/param.yaml"
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    config = OmegaConf.create(config)

    # データモジュールとモデルモジュールのインスタンスを作成
    data = DataModule(config)
    model = ModelModule(config)

    
    # TensorBoard Loggerもsave_dirに対応させる
    logger = pl_loggers.TensorBoardLogger(
        save_dir=base_save_dir,  # ベースディレクトリ（ここでlightning_logsが作成される）
        name='',  # デフォルトの`lightning_logs`ディレクトリに保存
        version=timestamp  # タイムスタンプをバージョン名に使用
    )

    # トレーナーを設定
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        logger=logger,  # Loggerに対応させる
        callbacks=None,
        accelerator='gpu',
        devices=[0],  # 使用するGPUのIDのリスト
        benchmark=True,  # cudnn.benchmarkを使用して高速化
    )

    # モデルの学習を実行
    trainer.test(model, datamodule=data)

if __name__ == '__main__':
    main()
