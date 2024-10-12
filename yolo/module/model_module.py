import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig
import json
import os

from .model.yolox.yolox import YOLOX
from .model.yolox.utils.boxes import postprocess
from ..utils.eval.evaluation import to_coco_format, evaluation


class ModelModule(pl.LightningModule):

    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config
        self.train_config = self.full_config.training
        self.validation_scores = []  # バリデーションスコアを保存するリスト
        self.test_scores = []
        self.model = YOLOX(self.full_config.model)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
                    
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        # 重みファイルをロード（バックボーンのみ）
        if self.full_config.model.weight_file_path:
            ckpt = torch.load(self.full_config.model.weight_file_path)
            # バックボーンのみをロードするために、headの部分を無視
            state_dict = ckpt['model']
            backbone_state_dict = {k: v for k, v in state_dict.items() if 'head' not in k}
            self.model.load_state_dict(backbone_state_dict, strict=False)  # strict=Falseでヘッドを無視



        #凍結設定
        if self.full_config.model.backbone.is_cold:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            
        elif self.full_config.model.head.is_cold:
            for param in self.model.head.parameters():
                param.requires_grad = False


    def forward(self, x, targets=None):
        return self.model(x, targets)
    
    def training_step(self, batch, batch_idx):
        self.model.train()
        imgs, targets, _, _ = batch
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.float32)
        targets.requires_grad = False
        
        outputs = self(imgs, targets)
        loss = outputs["total_loss"]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        lr = self.optimizers().param_groups[0]['lr']
        self.log('lr', lr, prog_bar=True, logger=True)

        return loss
    
    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['train_loss'].mean()
        self.log('epoch_train_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        
        self.model.eval()
        model_to_eval = self.model

        model_to_eval.to(self.device)
        imgs, targets, img_info, _ = batch
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.float32)
        targets.requires_grad = False
        
        predictions = model_to_eval(imgs, _)
        # xyxy
        processed_pred = postprocess(prediction=predictions,
                                     num_classes=self.full_config.model.head.num_classes,
                                     conf_thre=self.full_config.model.postprocess.conf_thre,
                                     nms_thre=self.full_config.model.postprocess.nms_thre)
        
        height, width = img_info

        if self.full_config.dataset.name == "coco":
            from .data.dataset.coco.coco_classes import COCO_CLASSES as CLASSES
        elif self.full_config.dataset.name == "dsec":
            from .data.dataset.dsec.label import CLASSES
            
        classes = CLASSES
        categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(classes)]
        num_data = len(targets)
        gt, pred = to_coco_format(gts=targets, detections=processed_pred, categories=categories, height=height, width=width)
        
        # COCO evaluationでスコアを取得
        scores = evaluation(Gt=gt, Dt=pred, num_data=num_data)
        
        # スコアをリストに追加
        self.validation_scores.append(scores)
        
        return scores

    def on_validation_epoch_end(self):
        # スコアの集計処理
        avg_scores = {
            'AP': torch.tensor([x['AP'] for x in self.validation_scores]).mean(),
            'AP_50': torch.tensor([x['AP_50'] for x in self.validation_scores]).mean(),
            'AP_75': torch.tensor([x['AP_75'] for x in self.validation_scores]).mean(),
            'AP_S': torch.tensor([x['AP_S'] for x in self.validation_scores]).mean(),
            'AP_M': torch.tensor([x['AP_M'] for x in self.validation_scores]).mean(),
            'AP_L': torch.tensor([x['AP_L'] for x in self.validation_scores]).mean(),
        }

        # 各スコアをログに記録する
        self.log('AP', avg_scores['AP'], prog_bar=True, logger=True)
        self.log('AP_50', avg_scores['AP_50'], prog_bar=True, logger=True)
        self.log('AP_75', avg_scores['AP_75'], prog_bar=True, logger=True)
        self.log('AP_S', avg_scores['AP_S'], prog_bar=False, logger=True)
        self.log('AP_M', avg_scores['AP_M'], prog_bar=False, logger=True)
        self.log('AP_L', avg_scores['AP_L'], prog_bar=False, logger=True)

        # バリデーションスコアのリセット
        self.validation_scores.clear()

    def test_step(self, batch, batch_idx):
        self.model.eval()
        model_to_eval = self.model

        model_to_eval.to(self.device)
        imgs, targets, img_info, _ = batch
        imgs = imgs.to(torch.float32)
        targets = targets.to(torch.float32)
        targets.requires_grad = False
        
        predictions = model_to_eval(imgs, _)
        # xyxy
        processed_pred = postprocess(prediction=predictions,
                                     num_classes=self.full_config.model.head.num_classes,
                                     conf_thre=self.full_config.model.postprocess.conf_thre,
                                     nms_thre=self.full_config.model.postprocess.nms_thre)
        
        height, width = img_info

        if self.full_config.dataset.name == "coco":
            from .data.dataset.coco.coco_classes import COCO_CLASSES as CLASSES
        elif self.full_config.dataset.name == "dsec":
            from .data.dataset.dsec.label import CLASSES
            
        classes = CLASSES
        categories = [{"id": idx + 1, "name": name} for idx, name in enumerate(classes)]
        num_data = len(targets)
        gt, pred = to_coco_format(gts=targets, detections=processed_pred, categories=categories, height=height, width=width)
        
        # COCO evaluationでスコアを取得
        scores = evaluation(Gt=gt, Dt=pred, num_data=num_data)
        
        # スコアをリストに追加
        self.test_scores.append(scores)
        
        return scores

    def on_test_epoch_end(self):
        # スコアの集計処理
        avg_scores = {
            'AP': torch.tensor([x['AP'] for x in self.test_scores]).mean(),
            'AP_50': torch.tensor([x['AP_50'] for x in self.test_scores]).mean(),
            'AP_75': torch.tensor([x['AP_75'] for x in self.test_scores]).mean(),
            'AP_S': torch.tensor([x['AP_S'] for x in self.test_scores]).mean(),
            'AP_M': torch.tensor([x['AP_M'] for x in self.test_scores]).mean(),
            'AP_L': torch.tensor([x['AP_L'] for x in self.test_scores]).mean(),
        }

        # 各スコアをログに記録する
        self.log('AP', avg_scores['AP'], prog_bar=True, logger=True)
        self.log('AP_50', avg_scores['AP_50'], prog_bar=True, logger=True)
        self.log('AP_75', avg_scores['AP_75'], prog_bar=True, logger=True)
        self.log('AP_S', avg_scores['AP_S'], prog_bar=False, logger=True)
        self.log('AP_M', avg_scores['AP_M'], prog_bar=False, logger=True)
        self.log('AP_L', avg_scores['AP_L'], prog_bar=False, logger=True)

        # バリデーションスコアのリセット
        self.test_scores.clear()
        
    def configure_optimizers(self):
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}



