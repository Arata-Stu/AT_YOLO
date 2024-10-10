import yaml
from omegaconf import OmegaConf
from yolo.module.model_module import ModelModule
from yolo.module.data_module import DataModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

def main():
    save_dir = './result'

    yaml_file = "./config/param.yaml"
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    config = OmegaConf.create(config)

    data = DataModule(config)
    model = ModelModule(config)

    callbacks = [
        ModelCheckpoint(
            dirpath=save_dir,
            filename='{epoch:02d}-{AP:.2f}',
            monitor='epoch_train_loss',                                 # 基準とする量
            mode="min",                                         
        ),  
    ]
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        max_steps= config.training.max_steps,
        logger=[pl_loggers.TensorBoardLogger(save_dir=save_dir)],
        callbacks=callbacks,
        accelerator='gpu',
        devices=[0],                            # 使用するGPUのIDのリスト
        # auto_lr_find=True,                      # learning rateを自動で設定するか
        # accumulate_grad_batches=1,              # 勾配を累積して一度に更新することでバッチサイズを仮想的にN倍にする際のN
        # gradient_clip_val=1,                    # 勾配クリッピングの値
        # fast_dev_run=True,                      # デバッグ時にonにすると、1回だけtrain,validを実行する
        # overfit_batches=1.0,                    # デバッグ時にonにすると、train = validで学習が進み、過学習できているかを確認できる
        # deterministic=True,                     # 再現性のために乱数シードを固定するか
        # resume_from_checkpoint='bbb/aaa.ckpt',  # チェックポイントから再開する場合に利用
        # precision=16,                           # 小数を何ビットで表現するか
        # amp_backend="apex",                     # 少数の混合方式を使用するかどうか。nvidiaのapexがインストールされている必要あり。
        benchmark=True,                         # cudnn.benchmarkを使用して高速化するか（determisticがTrueの場合はFalseに上書きされる）
    )
    # trainer.tune(model, datamodule=data_module)   # 「auto_lr_find=True」を指定した場合に実行する
    
    trainer.fit(model, datamodule=data)
    # trainer.test(model, datamodule=data)

if __name__ == '__main__':
    main()