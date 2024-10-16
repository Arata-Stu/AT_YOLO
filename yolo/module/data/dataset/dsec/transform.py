import numpy as np
import numba
from ....model.yolox.utils.boxes import xywh2xyxy, xywh2cxcywh

@numba.jit(nopython=True)
def render_events_on_empty_image(height, width, x, y, p):
    
    image = np.ones((height, width, 3), dtype=np.uint8) * 114
    
    # イベントに基づいて画像を更新
    for x_, y_, p_ in zip(x, y, p):
        if p_ == 0:
            image[y_, x_] = np.array([0, 0, 255])  # OFFイベントは青
        else:
            image[y_, x_] = np.array([255, 0, 0])  # ONイベントは赤
    return image

@numba.jit(nopython=True)
def render_events_on_image(image, x, y, p):
    # 画像にイベントを描画
    for x_, y_, p_ in zip(x, y, p):
        if p_ == 0:
            image[y_, x_] = np.array([0, 0, 255])  # OFFイベントは青
        else:
            image[y_, x_] = np.array([255, 0, 0])  # ONイベントは赤
    return image


class TrainTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, img_size=(640,640)):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.img_size = img_size

    def __call__(self, inputs):
        events = inputs['events']
        tracks = inputs['tracks']
        height, width = self.img_size

        # imagesがある場合は、その上にイベントを描画
        if 'image' in inputs and inputs['image'] is not None:
            image = inputs['image']
            event_frame = render_events_on_image(image, events['x'], events['y'], events['p'])
        else:
            # イベントフレームの作成 (例)
            event_frame = render_events_on_empty_image(height, width, events['x'], events['y'], events['p'])

        event_frame = np.transpose(event_frame, (2, 0, 1))  # (channels, height, width)

        # 複数のボックスに対応するために、各トラックデータから x, y, w, h を抽出
        bboxes = np.stack((tracks['x'], tracks['y'], tracks['w'], tracks['h']), axis=-1)

        # (x, y, w, h) -> (cx, cy, w, h) への変換
        bboxes_cxcywh = xywh2cxcywh(bboxes.copy())

        # ターゲットを作成
        targets_t = np.hstack((tracks['class_id'].reshape(-1, 1), bboxes_cxcywh))

        # padded_labels の作成
        padded_labels = np.zeros((self.max_labels, 5))  # 5 = class_id + cx, cy, w, h
        padded_labels[:min(len(targets_t), self.max_labels)] = targets_t[:self.max_labels]

        # 必要に応じてデータ型を調整
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)

        return event_frame, padded_labels, inputs['img_info'], inputs['img_id']
    

class TrainImageTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, img_size=(640,640)):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.img_size = img_size

    def __call__(self, inputs):
        image = inputs['image']
        tracks = inputs['tracks']
        height, width = self.img_size

        
        frame = np.transpose(image, (2, 0, 1))  # (channels, height, width)

        # 複数のボックスに対応するために、各トラックデータから x, y, w, h を抽出
        bboxes = np.stack((tracks['x'], tracks['y'], tracks['w'], tracks['h']), axis=-1)

        # (x, y, w, h) -> (cx, cy, w, h) への変換
        bboxes_cxcywh = xywh2cxcywh(bboxes.copy())

        # ターゲットを作成
        targets_t = np.hstack((tracks['class_id'].reshape(-1, 1), bboxes_cxcywh))

        # padded_labels の作成
        padded_labels = np.zeros((self.max_labels, 5))  # 5 = class_id + cx, cy, w, h
        padded_labels[:min(len(targets_t), self.max_labels)] = targets_t[:self.max_labels]

        # 必要に応じてデータ型を調整
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)

        return frame, padded_labels, inputs['img_info'], inputs['img_id']
    
class ValTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, img_size=(640,640)):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.img_size = img_size

    def __call__(self, inputs):
        events = inputs['events']
        tracks = inputs['tracks']
        height, width = self.img_size

        # imagesがある場合は、その上にイベントを描画
        if 'image' in inputs and inputs['image'] is not None:
            image = inputs['image']
            event_frame = render_events_on_image(image, events['x'], events['y'], events['p'])
        else:
            # イベントフレームの作成 (例)
            event_frame = render_events_on_empty_image(height, width, events['x'], events['y'], events['p'])

        event_frame = np.transpose(event_frame, (2, 0, 1))  # (channels, height, width)

        # 複数のボックスに対応するために、各トラックデータから x, y, w, h を抽出
        bboxes = np.stack((tracks['x'], tracks['y'], tracks['w'], tracks['h']), axis=-1)

        # (x, y, w, h) -> (x1, y1, x2, y2) への変換
        bboxes_xyxy = xywh2xyxy(bboxes.copy())

        # ターゲットを作成 [x1, y1, x2, y2, clsss])
        targets_t = np.hstack((bboxes_xyxy, tracks['class_id'].reshape(-1, 1)))

        # padded_labels の作成
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[:min(len(targets_t), self.max_labels)] = targets_t[:self.max_labels]

        # 必要に応じてデータ型を調整
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)

        return event_frame, padded_labels, inputs['img_info'], inputs['img_id']

class ValImageTransform:
    def __init__(self, max_labels=50, flip_prob=0.5, hsv_prob=1.0, img_size=(640,640)):
        self.max_labels = max_labels
        self.flip_prob = flip_prob
        self.hsv_prob = hsv_prob
        self.img_size = img_size

    def __call__(self, inputs):
        image = inputs['image']
        tracks = inputs['tracks']
        height, width = self.img_size

    
        frame = np.transpose(image, (2, 0, 1))  # (channels, height, width)

        # 複数のボックスに対応するために、各トラックデータから x, y, w, h を抽出
        bboxes = np.stack((tracks['x'], tracks['y'], tracks['w'], tracks['h']), axis=-1)

        # (x, y, w, h) -> (x1, y1, x2, y2) への変換
        bboxes_xyxy = xywh2xyxy(bboxes.copy())

        # ターゲットを作成 [x1, y1, x2, y2, clsss])
        targets_t = np.hstack((bboxes_xyxy, tracks['class_id'].reshape(-1, 1)))

        # padded_labels の作成
        padded_labels = np.zeros((self.max_labels, 5))
        padded_labels[:min(len(targets_t), self.max_labels)] = targets_t[:self.max_labels]

        # 必要に応じてデータ型を調整
        padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)

        return frame, padded_labels, inputs['img_info'], inputs['img_id']
