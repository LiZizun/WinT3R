from .base.transforms import *

from utils.misc import get_world_size, get_rank
from torch.utils.data import DataLoader
import hydra
from datasets.base.base_dataset import sample_resolutions, unified_collate_fn

__HIGH_QUALITY_DATASETS__ = ['BlinkVision', 'Game', 'GameNew', 'DynamicStereo', 'FlyingThings3D', 'GTA-sfm', 'Hypersim', 'MatrixCity', 'MidAir', 'Monkaa', 'PointOdyssey', 'Sintel', 'Spring', 'TarTanAir', 'Unreal4k', 'VirtualKitti', 'Habitat']
__MIDDLE_QUALITY_DATASETS__ = ['BlendedMVG', 'BlendedMVS', 'DTU', 'ETH3D', 'ScanNet', 'Scannetpp', 'Taskonomy']
__INDOOR_DATASETS__ = ['Hypersim', 'ScanNet', 'Scannetpp', 'Taskonomy', 'ARKitScenes', 'Habitat']

def create_dataloader(cfg, mode):
    data_loader = DataLoader

    # pytorch dataset
    if mode == 'train':
        cfg_dataset = cfg.train_dataset
        cfg_dataloader = cfg.train_dataloader
        batch_size = cfg.train.batch_size
        num_workers = cfg.train.num_workers
    else:
        cfg_dataset = cfg.test_dataset
        cfg_dataloader = cfg.test_dataloader
        batch_size = cfg.test.batch_size
        num_workers = cfg.test.num_workers

    if isinstance(cfg_dataset, str):
        dataset = eval(cfg_dataset) 
    elif 'weights' in cfg_dataset:
        weights = cfg_dataset.weights
        if 'length' in cfg_dataset:
            dataset_length = cfg_dataset.length
            weight_sum = sum([v for k, v in weights.items()])
            new_weights = {}
            for dataset_name, weight in weights.items():
                new_weights[dataset_name] = max(int(weight / weight_sum * dataset_length), 1)
            weights = new_weights
            print(f'New weights for dataset (adjusting to dataset length {dataset_length}): {new_weights}')

        datasets_all = []

        num_resolution = cfg.train.num_resolution if 'num_resolution' in cfg.train and mode == 'train' else 1
        if mode == 'train' and 'random_reslution' in cfg.train and cfg.train.random_reslution:
            seed = 777 + 0
            resolutions = sample_resolutions(aspect_ratio_range=cfg.train.aspect_ratio_range, pixel_count_range=cfg.train.pixel_count_range, patch_size=cfg.train.patch_size, num_resolutions=num_resolution, seed=seed)
            print('Initialized resolution', resolutions)
            num_resolution = len(resolutions)
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append(weight @ dataset_i)
        elif 'resolution' in cfg.train:
            resolutions = cfg.train.resolution
            print('Setting dataset resolution', resolutions)
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name], resolution=resolutions)
                dataset_i.convert_attributes()
                datasets_all.append(weight @ dataset_i)
        else:
            for dataset_name, weight in weights.items():
                dataset_i = hydra.utils.instantiate(cfg_dataset[dataset_name])
                dataset_i.convert_attributes()
                datasets_all.append(weight @ dataset_i)
        dataset = datasets_all[0]
        for dataset_ in datasets_all[1:]:
            dataset += dataset_
    else:
        dataset = hydra.utils.instantiate(cfg_dataset)
        dataset.convert_attributes()
    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=cfg_dataloader.shuffle, world_size=world_size,
                                       rank=rank, drop_last=cfg_dataloader.drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=cfg_dataloader.shuffle, drop_last=cfg_dataloader.drop_last
            )
        elif cfg_dataloader.shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    return data_loader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=unified_collate_fn
    )