import torch
from dataclasses import dataclass, field
from typing import Type, Dict, Tuple

from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.cameras.cameras import Cameras

class TemporalFullImageDatamanager(FullImageDatamanager):
    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """拦截默认的数据流，将前后相邻帧的数据塞入 batch 字典中"""
        # 先拿到默认的单帧数据
        camera, batch = super().next_train(step)
        
        # 【修复点】：从 batch 字典中安全获取 image_idx，兼容 Tensor 和 int 格式
        image_idx_raw = batch.get("image_idx", 0)
        image_idx = image_idx_raw.item() if hasattr(image_idx_raw, "item") else int(image_idx_raw)
        
        num_images = len(self.train_dataset)
        
        # 计算前后帧的索引 (加上 max 和 min 防止第一帧和最后一帧越界报错)
        prev_idx = max(0, image_idx - 1)
        next_idx = min(num_images - 1, image_idx + 1)
        
        # 提取前一帧 (如果有真实运动)
        if prev_idx != image_idx:
            # 优先从内存缓存中拿图，保证训练速度
            prev_batch = self.cached_train[prev_idx] if self.config.cache_images else self.train_dataset[prev_idx]
            batch["image_prev"] = prev_batch["image"]
            # 提取前一帧相机的真实位姿 c2w [3, 4]
            batch["c2w_prev"] = self.train_dataset.cameras.camera_to_worlds[prev_idx]
            
        # 提取后一帧
        if next_idx != image_idx:
            next_batch = self.cached_train[next_idx] if self.config.cache_images else self.train_dataset[next_idx]
            batch["image_next"] = next_batch["image"]
            batch["c2w_next"] = self.train_dataset.cameras.camera_to_worlds[next_idx]

        # 把当前相机对象显式传入 batch，提供给后续的时序重投影计算内参
        batch["camera"] = camera
        
        return camera, batch

@dataclass
class TemporalFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: TemporalFullImageDatamanager)