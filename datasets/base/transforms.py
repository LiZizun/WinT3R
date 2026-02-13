import cv2
from PIL import Image
import numpy as np
import PIL
import torchvision.transforms as tvf
import torch
import torchvision.transforms.v2.functional as TF
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
    bilinear = PIL.Image.Resampling.BILINEAR
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC
    bilinear = PIL.Image.BILINEAR

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])            # to [-1, 1]
ImgToTensor = tvf.ToTensor()
# ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])

CustomNorm = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def inverse_ImgNorm(img_norm):
    mean = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1).to(img_norm.device)
    std = torch.tensor([0.5, 0.5, 0.5]).reshape(3, 1, 1).to(img_norm.device)
    return img_norm * std + mean

def inverse_CustomNorm(custom_norm_img):
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1).to(custom_norm_img.device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1).to(custom_norm_img.device)
    return custom_norm_img * std + mean

class JpegLoss:
    def __init__(self, seed=2024, prob=0.5, quality_range=(20, 100)):
        self.prob = prob
        self.quality_range = quality_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        if self.rng.uniform() < self.prob:
            img_cv = np.array(img)[:, :, ::-1] 
            quality = self.rng.integers(*self.quality_range)
            _, encoded = cv2.imencode('.jpg', img_cv, [cv2.IMWRITE_JPEG_QUALITY, quality])
            img_cv = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            img = Image.fromarray(img_cv[:, :, ::-1])  # BGR to RGB
        return img

class Blurring:
    def __init__(self, seed=2025, prob=0.5, resize_ratio_range=(0.25, 1), interpolation_methods=None):
        self.prob = prob
        self.resize_ratio_range = resize_ratio_range
        # self.interpolation_methods = interpolation_methods or [
        #     cv2.INTER_LINEAR_EXACT, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        # ]
        self.interpolation_methods = interpolation_methods or [
            lanczos, bicubic, bilinear
        ]
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        if self.rng.uniform() < self.prob:

            w, h = img.size
            ratio = self.rng.uniform(*self.resize_ratio_range)
            interpolation = self.rng.choice(self.interpolation_methods)
            resized_small = img.resize((int(w * ratio), int(h * ratio)), resample=lanczos)
            img = resized_small.resize((w, h), resample=interpolation)

        return img

class ColorJitter:
    def __init__(self, seed=2026, brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.7, 1.3), hue=(-0.1, 0.1), gamma=(0.7, 1.3)):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma
        self.rng = np.random.default_rng(seed) 

        self.to_tensor = tvf.v2.Compose([tvf.v2.ToImage(), tvf.v2.ToDtype(torch.float32, scale=True)])

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            # img = TF.to_tensor(img)
            img = self.to_tensor(img)
        
        img = TF.adjust_brightness(img, self.rng.uniform(*self.brightness))
        img = TF.adjust_contrast(img, self.rng.uniform(*self.contrast))
        img = TF.adjust_saturation(img, self.rng.uniform(*self.saturation))
        img = TF.adjust_hue(img, self.rng.uniform(*self.hue))
        img = TF.adjust_gamma(img, self.rng.uniform(*self.gamma))
        
        img = TF.to_pil_image(img)
        return img

CustomNormJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), CustomNorm])

CustomNormJpegLoss = tvf.Compose([
    JpegLoss(prob=0.5, quality_range=(20, 100)),  
    CustomNorm 
])

CustomNormBlurring = tvf.Compose([
    Blurring(prob=0.5, resize_ratio_range=(0.25, 1)), 
    CustomNorm 
])

CustomNormJpegLossBlurring = tvf.Compose([
    JpegLoss(prob=0.5, quality_range=(20, 100)), 
    Blurring(prob=0.5, resize_ratio_range=(0.25, 1)), 
    CustomNorm 
])

CustomNormJitterJpegLossBlurring = tvf.Compose([
    ColorJitter(
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3), 
        saturation=(0.7, 1.3), 
        hue=(-0.1, 0.1), 
        gamma=(0.7, 1.3) 
    ),
    JpegLoss(prob=0.5, quality_range=(20, 100)),
    Blurring(prob=0.5, resize_ratio_range=(0.25, 1)), 
    CustomNorm 
])

JitterJpegLossBlurring = tvf.Compose([
    ColorJitter(
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3), 
        saturation=(0.7, 1.3), 
        hue=(-0.1, 0.1), 
        gamma=(0.7, 1.3) 
    ),
    JpegLoss(prob=0.5, quality_range=(20, 100)),
    Blurring(prob=0.5, resize_ratio_range=(0.25, 1)), 
    CustomNorm 
])