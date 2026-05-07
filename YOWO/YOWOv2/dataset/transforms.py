import torch
import numpy as np
import random
from PIL import Image

class Augmentation(object):
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def random_flip(self, video_clip, target):
        if random.random() < 0.5:
            video_clip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in video_clip]
            if len(target) > 0:
                target[:, [0, 2]] = 1.0 - target[:, [2, 0]]
        return video_clip, target

    def random_crop(self, video_clip, target, ow, oh):
        if random.random() < 0.5:
            swidth = random.uniform(0.6 * ow, ow)
            sheight = random.uniform(0.6 * oh, oh)
            pleft = random.uniform(0, ow - swidth)
            ptop = random.uniform(0, oh - sheight)
            video_clip = [img.crop((pleft, ptop, pleft + swidth, ptop + sheight)) for img in video_clip]
            if len(target) > 0:
                target[:, [0, 2]] = (target[:, [0, 2]] * ow - pleft) / swidth
                target[:, [1, 3]] = (target[:, [1, 3]] * oh - ptop) / sheight
                target = np.clip(target, 0, 1)
        return video_clip, target

    def __call__(self, video_clip, target):
        # 影像增強處理
        ow, oh = video_clip[0].size
        video_clip, target = self.random_flip(video_clip, target)
        video_clip, target = self.random_crop(video_clip, target, ow, oh)
        video_clip = [img.resize((self.img_size, self.img_size)) for img in video_clip]

        # 轉為 Tensor 格式 [3, T, H, W]
        video_clip = np.array([np.array(img) for img in video_clip]) # [T, H, W, 3]
        video_clip = torch.from_numpy(video_clip).float() / 255.0
        video_clip = video_clip.permute(3, 0, 1, 2).contiguous()

        target = torch.from_numpy(target).float()
        return video_clip, target

class BaseTransform(object):
    def __init__(self, img_size=224):
        self.img_size = img_size

    def __call__(self, video_clip, target):
        video_clip = [img.resize((self.img_size, self.img_size)) for img in video_clip]
        video_clip = np.array([np.array(img) for img in video_clip])
        video_clip = torch.from_numpy(video_clip).float() / 255.0
        video_clip = video_clip.permute(3, 0, 1, 2).contiguous()
        target = torch.from_numpy(target).float()
        return video_clip, target