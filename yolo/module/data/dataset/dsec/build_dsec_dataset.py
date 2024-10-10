import yaml
from omegaconf import DictConfig
from .dataset import DsecDetDataset
from .transform import TrainTransform, ValTransform


def build_dsec_dataset(full_config: DictConfig, mode="train"):

    config_path = full_config.dataset.split_config
    with open(config_path, 'r') as f:
        split = yaml.safe_load(f)
    
    if mode == "train":
        split = split['train']
        transform = TrainTransform()

    elif mode == "val":
        split = split['val']
        transform = ValTransform()

    elif mode=="test":
        split = split['test']
        transform = ValTransform()

    return DsecDetDataset(
        data_dir=full_config.dataset.data_dir,
        split_config=split,
        img_size=full_config.dataset.img_size,
        transform=transform,
        use_imgs=full_config.dataset.use_imgs,
        use_events=full_config.dataset.use_events)