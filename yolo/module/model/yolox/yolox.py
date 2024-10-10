import torch
import torch.nn as nn

from omegaconf import DictConfig

from .network.backbone.pafpn import YOLOPAFPN
from .network.head.yolo_head import YOLOXHead

def build_backbone(config: DictConfig):
    
    return YOLOPAFPN(
        depth=config.depth,  
        width=config.width,  
        in_features=config.in_features,  
        in_channels=config.in_channels,  
        depthwise=config.depthwise,  
        act=config.act  
    )

def build_head(config: DictConfig):
    return YOLOXHead(
        num_classes=config.num_classes,  
        width=config.width,  
        strides=config.strides,  
        in_channels=config.in_channels,  
        act=config.act, 
        depthwise=config.depthwise  
    )



class YOLOX(nn.Module):
    def __init__(self, config:DictConfig):
        super().__init__()

        bb_config = config.backbone
        backbone = build_backbone(bb_config)
        head_config = config.head
        head = build_head(head_config)
            
        self.backbone = backbone
        self.head=head

    def forward(self, x, targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs

