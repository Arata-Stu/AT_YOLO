from omegaconf import DictConfig
from .coco_dataset import COCODataset
from ...data_augment import TrainTransform, ValTransform


def build_coco_dataset(full_config: DictConfig, mode="train"):

    if mode == "train":
        json_file = full_config.dataset.train.json_file
        name = "train2017"
        transform = TrainTransform()

    elif mode == "val":
        json_file = full_config.dataset.val.json_file
        name = "val2017"
        transform = ValTransform()

    elif mode=="test":
        json_file = full_config.dataset.test.json_file
        name = "test2017"
        transform = ValTransform()

    return COCODataset(
        data_dir=full_config.dataset.data_dir,
        json_file=json_file,
        name=name,
        img_size=full_config.dataset.img_size,
        transform=transform)