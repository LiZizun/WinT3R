from datasets.base.easy_dataset import EasyDataset
from dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, get_pixel, inv, geotrf
import numpy as np
import os
import PIL
import dust3r.utils.cropping as cropping
import torchvision.transforms as tvf
from omegaconf import OmegaConf
from .transforms import *
import pandas as pd
from .utils import *
from utils.pylogger import RankedLogger
from enum import Enum, auto

class DataloaderMode(Enum):
    train = auto()
    test = auto()
    inference = auto()


def tensor2pil(img):
    if -1 <= img.min() <= 0:
        img_ = inverse_ImgNorm(img) 
    elif img.min() < -1:
        img_ = inverse_CustomNorm(img)
    else:
        img_ = img

    return PIL.Image.fromarray((img_.detach().cpu().numpy()*255).astype(np.uint8).transpose(1, 2, 0))

def tensor2numpy(img):
    if -1 <= img.min() <= 0:
        img_ = inverse_ImgNorm(img) 
    elif img.min() < -1:
        img_ = inverse_CustomNorm(img)
    else:
        img_ = img
    return (img_.detach().cpu().numpy()*255).astype(np.uint8).transpose(1, 2, 0)

class BaseDataset(EasyDataset):
    def __init__(
        self,
        seed=2024,
        resolution=None,            # (width, height) or list of (width, height) or list of int
        aug_crop=False,             # False or int, slightly scale the image a bit larger than the target resolution
        aug_focal=False,            # False or float in [0, 1]
        z_far=0,
        frame_num=2,
        transform=tvf.ToTensor(),
        cache_file=None,
        save_cache=False,
        mode='train',
        cache_name=None,
        max_refetch=3,
    ):
        super().__init__()
        self.frame_num = frame_num

        self.transform = transform

        self._rng = np.random.default_rng(seed)
        self._set_resolutions(resolution)

        self.logger = RankedLogger(__name__, rank_zero_only=True)

        self.aug_crop = aug_crop
        self.aug_focal = aug_focal

        self.z_far = z_far

        self.dataset_label = 'BaseDataset'

        self.save_cache = save_cache
        self.cache_loaded = False
        self.cache_name = cache_name
        if cache_file is not None:
            print(f'[BaseDataset] Loading cache from {cache_file}..')
            res = self.load_cache(cache_file)
            if res:
                self.cache_loaded = True
                print(f'[BaseDataset] Cache is loaded.')

        self.mode = mode
        self.max_refetch = max_refetch

    def convert_attributes(self):
        """
        Avoid memory leak caused by python list or python dict
        https://github.com/pytorch/pytorch/issues/13246
        """

        def _is_equivalent(original, converted):
            """
            Check if the converted data structure is equivalent to the original.
            """
            try:
                return original == converted
            except Exception:
                return False

        for attr_name in dir(self):
            # 排除私有属性和方法
            if attr_name.startswith("__") or callable(getattr(self, attr_name)):
                continue
            
            # 获取属性值
            attr_value = getattr(self, attr_name)
            
            # 如果是 list，转换为 numpy array
            if isinstance(attr_value, list):
                try:
                    # 使用 dtype=object 确保保留原始数据类型
                    converted_value = np.array(attr_value)
                    
                    # 确保转换后与原始列表等效
                    if _is_equivalent(attr_value, converted_value.tolist()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)
            
            # 如果是 dict，转换为 pandas DataFrame
            elif isinstance(attr_value, dict):
                try:
                    # 确保键和值不丢失，保持使用等效性
                    converted_value = pd.Series(attr_value)
                    if _is_equivalent(attr_value, converted_value.to_dict()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'
        if OmegaConf.is_config(resolutions):
            resolutions = OmegaConf.to_object(resolutions)

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            # assert width >= height
            # self._resolutions.append((width, height))
            self._resolutions.append([width, height])

        self.num_resoluions = len(self._resolutions)

    @classmethod
    def proj_view(cls, views, with_color=False, is_batchified=False, batch_id=0):
        points3d = []

        for view in views:
            if is_batchified:
                depth = view['depthmap'][batch_id]
                pose = view['camera_pose'][batch_id]
                intrinsic = view['camera_intrinsics'][batch_id]
            else:
                depth = view['depthmap']
                pose = view['camera_pose']
                intrinsic = view['camera_intrinsics']

            mask = depth.reshape(-1) > 0

            H, W = depth.shape
            pixel = get_pixel(H, W).astype(np.float32)
            points = (np.linalg.inv(intrinsic) @ pixel) * depth.reshape(-1)
            points = pose[:3, :4] @ np.concatenate([points, np.ones((1, points.shape[1]))], axis=0)

            if with_color:
                if is_batchified:
                    img = np.array(view['img'][batch_id])/255.
                else:
                    img = np.array(view['img'])/255.
                points3d.append([points.T[mask], img.reshape(-1, 3)[mask]])
            else:
                points3d.append(points.T[mask])

        return points3d


    def get_stride_distribution(self, strides, dist_type='uniform'):

        # input strides sorted by descreasing order by default
        
        if dist_type == 'uniform':
            dist = np.ones(len(strides)) / len(strides)
        elif dist_type == 'exponential':
            lambda_param = 1.0
            dist = np.exp(-lambda_param * np.arange(len(strides)))
        elif dist_type.startswith('linear'): # e.g., linear_1_2
            try:
                start, end = map(float, dist_type.split('_')[1:])
                dist = np.linspace(start, end, len(strides))
            except ValueError:
                raise ValueError(f'Invalid linear distribution format: {dist_type}')
        else:
            raise ValueError('Unknown distribution type %s' % dist_type)

        # normalize to sum to 1
        return dist / np.sum(dist)

    def _sample_stride(self, strides, dists):
        # sampled_stride = np.random.choice(strides, p=dist)    # avoid slow random choice

        if not hasattr(self, 'stride_buffer'):
            max_buffer_size = 10000
            self.stride_buffer = []
            for stride, dist in zip(strides, dists):
                self.stride_buffer.extend(int(max_buffer_size*dist)*[stride])

        i = self._rng.integers(0, len(self.stride_buffer))
        return self.stride_buffer[i]

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal)

        # transpose the resolution if necessary
        W, H = image.size  # new size

        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, intrinsics, normal, far_mask = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        other = [x for x in [normal, far_mask] if x is not None]
        return image, depthmap, intrinsics2, *other
    
    def _get_resolution(self):
        i = self._rng.integers(0, self.num_resoluions)
        return self._resolutions[i]
    
    def check_overlap(self, views, thres=0.01):
        _, H, W = views[0]['img'].shape
        tolerance = thres * H * W * 2
        # pass
        for i in range(len(views)):
            in_camera = inv(views[i]['camera_pose'])
            intrinsic = np.stack([views[ii]['camera_intrinsics'] for ii in range(len(views)) if ii != i], axis=0)           # (N-1, 3, 3)
            pts = np.stack([views[ii]['pts3d'][views[ii]['depthmap'] > 0] for ii in range(len(views)) if ii != i], axis=0)        # (N-1, ?, 3)

            pts_cam = geotrf(in_camera, pts)
            pts_img = np.einsum('nij, nkj -> nki', intrinsic, pts_cam)
            uv = pts_img[..., :2] / (pts_img[..., -1:] + 1e-8)

            mask = (uv >= 0) * (uv[..., 0:1] < W-1) * (uv[..., 1:] < H-1) * (pts_img[..., -1:] > 0)

            if mask.sum() < tolerance:
                return False
            
        return True

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # over-loaded code
        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        # resolution = self._get_resolution()
        
        error = None
        for _ in range(10):              # default: 3
            try:
                views = self._get_views(idx, resolution, self._rng)

                assert len(views) == self.frame_num

                # check data-types
                for v, view in enumerate(views):
                    assert 'pts3d' not in view, f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
                    view['idx'] = (idx, ar_idx, v)
                    # view['idx'] = (idx, v)

                    # encode the image
                    width, height = view['img'].size
                    view['true_shape'] = np.int32((height, width))
                    view['img'] = self.transform(view['img'])

                    assert 'camera_intrinsics' in view
                    if 'camera_pose' not in view:
                        view['camera_pose'] = np.full((4, 4), np.nan, dtype=np.float32)
                    else:
                        assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for view {view_name(view)}'
                    assert 'pts3d' not in view
                    assert 'valid_mask' not in view
                    assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for view {view_name(view)}'
                    view['z_far'] = self.z_far
                    pts3d, pts3d_local, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

                    view['pts3d'] = pts3d
                    view['pts3d_local'] = pts3d_local
                    view['valid_mask'] = valid_mask & np.isfinite(pts3d).all(axis=-1)

                    if 'normal' not in view:
                        view['normal'] = None

                # last thing done!
                for view in views:

                    # this allows to check whether the RNG is is the same state each time
                    view['rng'] = int.from_bytes(self._rng.bytes(4), 'big')


            except Exception as e:
                views = None
                if hasattr(self, 'this_views_info'):
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} ({self.this_views_info}) for error {e}.", flush=True
                    )
                else:
                    print(
                        f"Failed to load data from {self.dataset_label}-{idx} for error {e}.", flush=True
                    )
                idx = np.random.randint(0, len(self))
                error = e
            
            if views is not None:
                error = None
                break

        if views is None:
            raise error
        
        return views
    
    def load_cache(self, cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True).item()
            # Step 2: 遍历字典中的每个键值对，并将其赋值为类实例的属性
            if isinstance(data, dict):
                for key, value in data.items():
                    setattr(self, key, value)
                return True
            else:
                print("Error: The npy file does not contain a dictionary.")
                return False
        except Exception as e:
            print(f"An error occurred while loading the cache: {e}")
            return False

    def _save_cache(self, keys, desc=None):
        if desc is None:
            save_path = f'data/dataset_cache/{self.dataset_label}_cache.npy'
        else:
            save_path = f'data/dataset_cache/{self.dataset_label}_{desc}_cache.npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {}
        for key in keys:
            save_dict[key] = getattr(self, key)
        
        np.save(save_path, save_dict)

        print(f'Saved cache to {save_path}.', flush=True)

def sample_resolutions(aspect_ratio_range=(0.5, 2.0), pixel_count_range=(250000, 500000), patch_size=1, num_resolutions=5, seed=None):
    """
    Sample a list of random resolutions based on aspect ratio, pixel count constraints, 
    and ensure the width and height are divisible by patch_size.

    Args:
        aspect_ratio_range (tuple): The range of aspect ratios (width / height), e.g., (0.5, 2.0).
        pixel_count_range (tuple): The range of total pixel counts (width * height), e.g., (250000, 500000).
        patch_size (int): Ensure the output width and height are divisible by this value.
        num_resolutions (int): The number of resolutions to sample.

    Returns:
        list of (int, int): A list of (width, height) tuples representing the sampled resolutions.
    """
    rng = np.random.default_rng(seed=seed)
    resolutions = set()  # Use a set to ensure uniqueness

    while len(resolutions) < num_resolutions:
        # Randomly sample an aspect ratio within the given range
        aspect_ratio = rng.uniform(*aspect_ratio_range)

        # Randomly sample a total pixel count within the given range
        pixel_count = rng.uniform(*pixel_count_range)

        # Compute height and width based on the sampled aspect ratio and pixel count
        height = math.sqrt(pixel_count / aspect_ratio)
        width = aspect_ratio * height

        # Round height and width to the nearest integers that are divisible by patch_size
        height = int(round(height / patch_size) * patch_size)
        width = int(round(width / patch_size) * patch_size)

        # Add the resolution to the set (duplicates are automatically ignored)
        resolutions.add((width, height))

    return list(resolutions)

def view_name(view, batch_index=None):
    def sel(x): return x[batch_index] if batch_index not in (None, slice(None)) else x
    db = sel(view['dataset'])
    label = sel(view['label'])
    instance = sel(view['instance'])
    return f"{db}/{label}/{instance}"

def is_good_type(key, v):
    """ returns (is_good, err_msg) 
    """
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (torch.bool, np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def transpose_to_landscape(view):
    height, width = view['true_shape']

    if width < height:
        # rectify portrait to landscape
        assert view['img'].shape == (3, height, width)
        view['img'] = view['img'].swapaxes(1, 2)

        assert view['valid_mask'].shape == (height, width)
        view['valid_mask'] = view['valid_mask'].swapaxes(0, 1)

        assert view['depthmap'].shape == (height, width)
        view['depthmap'] = view['depthmap'].swapaxes(0, 1)

        if 'normal' in view:
            assert view['normal'].shape == (height, width, 3)
            view['normal'] = view['normal'].swapaxes(0, 1)

        if 'far_mask' in view:
            assert view['far_mask'].shape == (height, width)
            view['far_mask'] = view['far_mask'].swapaxes(0, 1)

        assert view['pts3d'].shape == (height, width, 3)
        view['pts3d'] = view['pts3d'].swapaxes(0, 1)

        # transpose x and y pixels
        view['camera_intrinsics'] = view['camera_intrinsics'][[1, 0, 2]]

def unified_collate_fn(batch):
    views_num = len(batch[0])
    all_keys = batch[0][0].keys()

    batched_data = [{key: [] for key in all_keys} for _ in range(views_num)]
    for sample in batch:
        for i in range(views_num):
            for key in all_keys:
                batched_data[i][key].append(sample[i].get(key, None))

    for i in range(views_num):
        for key, data in batched_data[i].items():
            try:
                batched_data[i][key] = default_collate(data)
            except Exception:
                batched_data[i][key] = data

    return batched_data