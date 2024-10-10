import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
from .directory import DSECDirectory
from .preprocessing import compute_img_idx_to_track_idx
from .io import extract_from_h5_by_timewindow

class DsecDetDataset(Dataset):

    def __init__(self, data_dir, split_config, img_size=(256, 256), transform=None, sync="back", use_events=True, use_imgs=True):
        self.data_dir = Path(data_dir)
        self.split_config = split_config
        self.img_size = img_size
        self.transform = transform
        self.sync = sync
        self.use_events = use_events
        self.use_imgs = use_imgs

        available_dirs = list(self.data_dir.glob("*/"))
        self.subsequence_directories = [self.data_dir / s for s in split_config if self.data_dir / s in available_dirs]

        self.subsequence_directories = sorted(self.subsequence_directories, key=self.first_time_from_subsequence)

        self.directories = dict()
        self.img_idx_track_idxs = dict()
        for f in self.subsequence_directories:
            directory = DSECDirectory(f)
            self.directories[f.name] = directory
            self.img_idx_track_idxs[f.name] = compute_img_idx_to_track_idx(
                directory.tracks.tracks['t'],
                directory.images.timestamps
            )

    def __len__(self):
        return sum(len(v) - 1 for v in self.img_idx_track_idxs.values())

    def __getitem__(self, item):
        output = {}
        
        if self.use_imgs:
            output['image'] = self.get_image(item)
        
        if self.use_events:
            output['events'] = self.get_events(item)
        
        output['tracks'] = self.get_tracks(item)
        output['img_info'] = (480, 640)
        output['img_id'] = 0

        if self.transform is not None:
            img, target, info, id = self.transform(output) 
        else:
            img, target, info, id = output['image'], output['tracks'], output['img_info'], output['img_id']

        return img, target, info, id
    
    def load_sequence(self, sequence_name):
        """
        指定されたシーケンスの全データ（画像、イベント、トラック）をロードします。
        sequence_name はサブシーケンスディレクトリの名前に対応します。
        """
        if sequence_name not in self.directories:
            raise ValueError(f"指定されたシーケンス '{sequence_name}' は存在しません。")

        directory = self.directories[sequence_name]
        img_idx_to_track_idx = self.img_idx_track_idxs[sequence_name]

        # シーケンスの全てのデータを格納するリスト
        images = []
        events = []
        tracks = []

        # 各インデックスについてデータを取得
        for i in range(len(img_idx_to_track_idx) - 1):
            images.append(self.get_image(i, directory_name=sequence_name))
            events.append(self.get_events(i, directory_name=sequence_name))
            tracks.append(self.get_tracks(i, directory_name=sequence_name))

        # 必要に応じて変換処理
        if self.transform is not None:
            images, tracks, _, _ = self.transform({
                'image': images,
                'tracks': tracks,
                'img_info': (480, 640),
                'img_id': 0
            })

        return {
            'images': images,
            'events': events,
            'tracks': tracks
        }
    
    def first_time_from_subsequence(self, subsequence):
        return np.genfromtxt(subsequence / "images/timestamps.txt", dtype="int64")[0]
    
    def get_image(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        image_files = directory.images.image_files_distorted
        image = cv2.imread(str(image_files[index]))
        return image
    
    def get_tracks(self, index, mask=None, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        idx0, idx1 = img_idx_to_track_idx[i_1]
        tracks = directory.tracks.tracks[idx0:idx1]

        if mask is not None:
            tracks = tracks[mask[idx0:idx1]]

        return tracks

    def get_events(self, index, directory_name=None):
        index, img_idx_to_track_idx, directory = self.rel_index(index, directory_name)
        i_0, i_1 = self.get_index_window(index, len(img_idx_to_track_idx), sync=self.sync)
        t_0, t_1 = directory.images.timestamps[[i_0, i_1]]
        events = extract_from_h5_by_timewindow(directory.events.event_file, t_0, t_1)
        return events
    
    def get_index_window(self, index, num_idx, sync="back"):
        if sync == "front":
            assert 0 < index < num_idx
            i_0 = index - 1
            i_1 = index
        else:
            assert 0 <= index < num_idx - 1
            i_0 = index
            i_1 = index + 1

        return i_0, i_1

    def rel_index(self, index, directory_name=None):
        if directory_name is not None:
            img_idx_to_track_idx = self.img_idx_track_idxs[directory_name]
            directory = self.directories[directory_name]
            return index, img_idx_to_track_idx, directory

        for f in self.subsequence_directories:
            img_idx_to_track_idx = self.img_idx_track_idxs[f.name]
            if len(img_idx_to_track_idx) - 1 <= index:
                index -= (len(img_idx_to_track_idx) - 1)
                continue
            else:
                return index, img_idx_to_track_idx, self.directories[f.name]
        else:
            raise ValueError
