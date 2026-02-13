import sys
sys.path.append('.')
import cv2
from PIL import Image
import imageio

from datasets.base.base_dataset import BaseDataset
from datasets.utils_ceph import PetrelBackend
import os
import numpy as np
import os.path as osp
import h5py
from utils.basic import seed_anything
from PIL import Image
from tqdm import tqdm
from datasets.base.transforms import *

def xyzqxqyqxqw_to_c2w(xyzqxqyqxqw):
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)
    #NOTE: we need to convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    c2w[:3, 3] = np.array([x, y, z])
    return c2w

class TarTanAirDataset(BaseDataset):
    def __init__(
        self,
        data_root='ssd:s3://TartanAir',
        verbose=False,
        dist_type='uniform',
        strides=[8],                # final frame num interval
        clip_step=2,                # interval for all frame
        seq_num=-1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.client = PetrelBackend('~/petreloss.conf')

        self.verbose = verbose
        self.dataset_label = 'TarTanAir'

        self.sequences = []
        for seq in self.client.listdir(data_root):
            names = self.client.listdir(os.path.join(data_root, seq, seq, 'Easy'))
            seq_ = [[seq, 'Easy', name] for name in names]
            self.sequences.extend(seq_)
            names = self.client.listdir(os.path.join(data_root, seq, seq, 'Hard'))
            seq_ = [[seq, 'Hard', name] for name in names]
            self.sequences.extend(seq_)

        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]

        # self.sequences = sorted(self.sequences)

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences of {self.dataset_label} dataset:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} unique videos in {data_root}', flush=True)

        if not self.cache_loaded:
            self.rgb_paths = []
            self.depth_paths = []
            self.annotations = []
            self.intrinsics = []
            self.full_idxs = []
            self.total_frame_num = 0

            self.stride_index = {}
            for seq in tqdm(self.sequences, desc=f'[{self.dataset_label}] Loading pair..'):
                rgb_path = os.path.join(data_root, seq[0], seq[0], seq[1], seq[2], 'image_left')
                depth_path = os.path.join(data_root, seq[0], seq[0], seq[1], seq[2], 'depth_left')
                cam_path = os.path.join(data_root, seq[0], seq[0], seq[1], seq[2], 'pose_left.txt')
                caminfo = self.client.get_numpy_txt(cam_path)
                # focals = self.client.get_numpy_txt(os.path.join(cam_path, 'focaldistance.txt'))

                num_image = len(self.client.listdir(rgb_path))
                self.total_frame_num += num_image

                for stride in strides:
                    if stride not in self.stride_index:
                        self.stride_index[stride] = []
                    for ii in range(0, num_image-(self.frame_num-1)*stride, clip_step):
                        full_idx = ii + np.arange(self.frame_num)*stride            # start with 0
                        self.rgb_paths.append([os.path.join(rgb_path, '%06d_left.png' % idx) for idx in full_idx])
                        self.depth_paths.append([os.path.join(depth_path, '%06d_left_depth.npy' % idx) for idx in full_idx])
                        self.annotations.append(caminfo[full_idx].tolist())
                        self.full_idxs.append(full_idx.tolist())
                        # self.sample_stride.append(stride)
                        self.stride_index[stride].append(len(self.rgb_paths) - 1)

            if self.save_cache:
                cahce_name = self.cache_name if self.cache_name is not None else self.mode
                self._save_cache(
                    ['rgb_paths', 'depth_paths', 'annotations', 'intrinsics', 'full_idxs', 'total_frame_num', 'stride_index'],
                    desc=cahce_name
                )

        tmp = [len(v)*[k] for k, v in self.stride_index.items()]
        self.sample_stride = []
        for lst in tmp:
            self.sample_stride.extend(lst)

        self.stride_counts = {}
        self.stride_idxs = {}
        for stride in strides:
            self.stride_counts[stride] = 0
            self.stride_idxs[stride] = []
        for i, stride in enumerate(self.sample_stride):
            self.stride_counts[stride] += 1
            self.stride_idxs[stride].append(i)
        print('stride counts:', self.stride_counts)

        if len(strides) > 1 and dist_type is not None:
            self._resample_clips(strides, dist_type)

        # print(self.stride_index, flush=True)
        if self.verbose:
            print(f'[TarTanAir dataset] There is {self.total_frame_num} frames totally.', flush=True)
        self.stride_dist = self.get_stride_distribution(strides, dist_type=dist_type)
        self.strides = strides

        fx = 320.0  # focal length x
        fy = 320.0  # focal length y
        cx = 320.0  # optical center x
        cy = 240.0  # optical center y

        width = 640
        height = 480

        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    def _resample_clips(self, strides, dist_type):

        # Get distribution of strides, and sample based on that
        dist = self.get_stride_distribution(strides, dist_type=dist_type)
        dist = dist / np.max(dist)
        max_num_clips = self.stride_counts[strides[int(np.argmax(dist))]]
        num_clips_each_stride = [min(self.stride_counts[stride], int(dist[i]*max_num_clips)) for i, stride in enumerate(strides)]
        print('resampled_num_clips_each_stride:', num_clips_each_stride)
        resampled_idxs = []
        for i, stride in enumerate(strides):
            resampled_idxs += np.random.choice(self.stride_idxs[stride], num_clips_each_stride[i], replace=False).tolist()

        self.rgb_paths = [self.rgb_paths[i] for i in resampled_idxs]
        self.depth_paths = [self.depth_paths[i] for i in resampled_idxs]
        self.annotations = [self.annotations[i] for i in resampled_idxs]
        # self.dynamic_mask_paths = [self.dynamic_mask_paths[i] for i in resampled_idxs]
        self.full_idxs = [self.full_idxs[i] for i in resampled_idxs]
        self.sample_stride = [self.sample_stride[i] for i in resampled_idxs]

    def __len__(self):
        return len(self.rgb_paths)
                    
    def _get_views(self, index, resolution, rng):

        # stride = self.sample_stride(self.strides, self.stride_dist)
        # ii = self._rng.integers(0, len(self.stride_index[stride]))
        # idx = self.stride_index[stride][ii]

        rgb_paths = self.rgb_paths[index]
        depth_path = self.depth_paths[index]
        annotations = self.annotations[index]

        self.this_views_info = dict(
            rgb_paths=rgb_paths,
            depth_paths=depth_path,
        )

        views = []
        for i in range(self.frame_num):
            impath = rgb_paths[i]
            depthpath = depth_path[i]

            # load camera params
            camera_pose = np.array(xyzqxqyqxqw_to_c2w(annotations[i]), dtype=np.float32)

            # load image and depth
            rgb_image = np.array(self.client.imageopen(impath))

            depthmap = self.client.get_npy(depthpath)
            depthmap[depthmap > 80] = -1

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, self.intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=rgb_paths[i].split('/')[-3],
                instance=osp.split(rgb_paths[i])[1],
            ))
        return views

class TarTanAirMVDataset(BaseDataset):
    def __init__(
        self,
        data_root='ssd:s3://TartanAir',
        verbose=False,
        max_distance=24,
        seq_num=-1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.client = PetrelBackend('~/petreloss.conf')

        self.verbose = verbose
        self.dataset_label = 'TarTanAir'
        self.max_distance = max_distance
        self.data_root = data_root

        self.sequences = []
        for seq in self.client.listdir(data_root):
            names = self.client.listdir(os.path.join(data_root, seq, seq, 'Easy'))
            seq_ = [(seq, 'Easy', name) for name in names]
            self.sequences.extend(seq_)
            names = self.client.listdir(os.path.join(data_root, seq, seq, 'Hard'))
            seq_ = [(seq, 'Hard', name) for name in names]
            self.sequences.extend(seq_)

        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]

        # self.sequences = sorted(self.sequences)

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences of {self.dataset_label} dataset:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} unique videos in {data_root}', flush=True)

        fx = 320.0  # focal length x
        fy = 320.0  # focal length y
        cx = 320.0  # optical center x
        cy = 240.0  # optical center y

        # width = 640
        # height = 480

        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        self.num_imgs = {}
        for seq in self.sequences:
            rgb_path = os.path.join(data_root, seq[0], seq[0], seq[1], seq[2], 'image_left')
            self.num_imgs[seq] = len(self.client.listdir(rgb_path))

    def __len__(self):
        return len(self.sequences)
                    
    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        num_imgs = self.num_imgs[scene]

        if self.frame_num > 999 and rng.random() < self.random_sample_thres:
            should_replace = num_imgs < self.frame_num
            idxs = list(rng.choice(num_imgs, size=self.frame_num, replace=should_replace))
        else:
            idxs = [rng.integers(0, num_imgs)]

            # max_distance = 12 if scene in self.special_scenes else self.max_distance
            max_distance = int(self.max_distance / 8 * self.frame_num)
            start_idx = max(0, idxs[-1] - max_distance)
            end_idx = min(num_imgs-1, start_idx + 2*max_distance)
            start_idx = max(0, end_idx - 2*max_distance)
            valid_indices = np.arange(start_idx, end_idx + 1)

            should_replace = len(valid_indices) < self.frame_num - 1
            idxs.extend(list(rng.choice(valid_indices, self.frame_num-1, replace=should_replace)))

        self.this_views_info = dict(
            scene=scene,
            pairs=idxs,
        )

        cam_path = os.path.join(self.data_root, scene[0], scene[0], scene[1], scene[2], 'pose_left.txt')
        caminfo = self.client.get_numpy_txt(cam_path)

        views = []
        for idx in idxs:
            impath = os.path.join(self.data_root, scene[0], scene[0], scene[1], scene[2], 'image_left', f'{idx:06d}_left.png')
            depthpath = os.path.join(self.data_root, scene[0], scene[0], scene[1], scene[2], 'depth_left', f'{idx:06d}_left_depth.npy')

            # load camera params
            camera_pose = np.array(xyzqxqyqxqw_to_c2w(caminfo[idx]), dtype=np.float32)

            # load image and depth
            rgb_image = np.array(self.client.imageopen(impath))

            depthmap = self.client.get_npy(depthpath)
            depthmap[depthmap > 80] = -1

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, self.intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=f'{scene[0]}_{scene[1]}_{scene[2]}',
                instance=str(idx),
            ))
        return views
    
if __name__ == "__main__":

    from utils.debug import setup_debug

    setup_debug(True)

    seed_anything(2024)

    # dataset = TarTanAirDataset(
    #     strides=[48],
    #     resolution=(960, 560),
    # )

    # dataset = TarTanAirDataset(z_far=80, frame_num=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9, save_cache=True, cache_name='train_stride_1-9')
    # dataset = TarTanAirDataset(z_far=80, frame_num=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[1,2,3,4,5,6,7,8,9], dist_type='linear_1_2', aug_focal=0.9, cache_file='data/dataset_cache/TarTanAir_train_stride_1-9_cache.npy')
    # dataset = TarTanAirDataset(z_far=80, frame_num=2, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=ColorJitter, strides=[12, 18, 24, 30, 36], dist_type='uniform', aug_focal=0.9, save_cache=True, cache_name='train_large_stride')
    dataset = TarTanAirMVDataset(z_far=80, frame_num=17, aug_crop=16, resolution=[(512, 288), (512, 384), (512, 336)], transform=CustomNormJitter, aug_focal=0.9)

    dataset.convert_attributes()

    for i in range(20):
        idx = np.random.randint(0, len(dataset))
        # views = dataset._get_views(idx, (256, 256), dataset._rng)
        # views = dataset.__getitem__((idx, 0))
        views = dataset._get_views(idx, (224, 224), dataset._rng)
        cat_img = np.concatenate([np.array(x['img']) for x in views], axis=1)
        Image.fromarray(cat_img).save(f"outputs/img_{views[0]['label']}-{views[0]['instance']}-{views[1]['instance']}.png")
        # (views[0]['img']).save('outputs/img1.png')
        # (views[1]['img']).save('outputs/img2.png')

        dataset.vis_views(views)

    