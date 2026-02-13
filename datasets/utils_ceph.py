"""
Date: 2022-07-18 2:15:47 pm
Author: dihuangdh
Descriptions: 
-----
LastEditTime: 2022-09-14 3:44:19 pm
LastEditors: dihuangdh
"""

import json
import pickle
import warnings
import cv2
import numpy as np
from PIL import Image
import imageio
try:
    import torch
except:
    pass
from io import BytesIO, StringIO  # TODO:
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Tuple, Union
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"              # set before importing cv2 to enable loading .exr file
import os
from glob import glob



def has_method(obj: object, method: str) -> bool:
    """Check whether the object has a method.
    Args:
        method (str): The method name to check.
        obj (object): The object to check.
    Returns:
        bool: True if the object has the method else False.
    """
    return hasattr(obj, method) and callable(getattr(obj, method))


class PetrelBackend:
    """Petrel storage backend - simple version"""

    def __init__(self, use_ceph = True, enable_mc: bool = False, conf_path=None) -> None:
        # self._client = Client(enable_mc=enable_mc)
        if use_ceph:
            try:
                from petrel_client.client import Client
            except ImportError:
                raise ImportError(
                    "Please install petrel_client to enable " "PetrelBackend."
                )

            conf_path = "~/petreloss.conf" if conf_path is None else conf_path
            self._client = Client(conf_path)
        
        # self.use_ceph = use_ceph
    
    @classmethod
    def is_ceph_path(cls, path):
        colon_split = path.split(':')
        # return len(colon_split) >= 1 and (colon_split[0] == 's3' or colon_split[1] == 's1')
        return len(colon_split) > 1 and (colon_split[0] == 's3' or colon_split[1] == 's3')

    def get(self, filepath) -> memoryview:
        value = self._client.Get(filepath)
        value_buf = memoryview(value)
        return value_buf
    
    def get_raw_bytes(self, filepath):
        return self._client.Get(filepath)
    
    def get_Bytes(self, filepath):
        return BytesIO(self.get(filepath))
    
    def get_String(self, filepath):
        return StringIO(self.get(filepath))

    def get_text(self, filepath, warning=False) -> str:
        try:
            value = self._client.Get(filepath)
        except:
            if warning:
                warnings.warn("Failed to get text from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get text from {}".format(filepath))
        return str(value, encoding="utf-8")

    def get_uint16_png(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.frombuffer(self._client.get(filepath), np.uint16)
            value = cv2.imdecode(value, cv2.IMREAD_UNCHANGED)
        except:
            if warning:
                warnings.warn("Failed to get uint16_png from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get uint16_png from {}".format(filepath))
        return value

    def get_uint8_jpg(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.frombuffer(self._client.get(filepath), np.uint8)
            value = cv2.imdecode(value, cv2.IMREAD_UNCHANGED)
        except:
            if warning:
                warnings.warn("Failed to get uint8_jpg from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get uint8_jpg from {}".format(filepath))
        return value

    def get_uint8_png(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.frombuffer(self._client.get(filepath), np.uint8)
            value = cv2.imdecode(value, cv2.IMREAD_UNCHANGED)
        except:
            if warning:
                warnings.warn("Failed to get uint8_png from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get uint8_png from {}".format(filepath))
        return value

    def get_npz(self, filepath, warning=False) -> Any:
        try:
            value = self._client.get(filepath)
            value = np.load(BytesIO(value), allow_pickle=True)
        except Exception as e:
            if warning:
                warnings.warn("Failed to get npz from {}".format(filepath))
                value = None
            else:
                print(e)
                raise Exception("Failed to get npz from {}".format(filepath))
        return value
    
    def get_npy(self, filepath, warning=False) -> Any:
        try:
            value = self._client.get(filepath)
            value = np.load(BytesIO(value), allow_pickle=True)
        except Exception as e:
            if warning:
                warnings.warn("Failed to get npy from {}".format(filepath))
                value = None
            else:
                print(e)
                raise Exception("Failed to get npy from {}".format(filepath))
        return value


    def get_numpy_txt(self, filepath, warning=False) -> np.ndarray:
        try:
            value = np.loadtxt(StringIO(self.get_text(filepath)))
        except:
            if warning:
                warnings.warn("Failed to get numpy_txt from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get numpy_txt from {}".format(filepath))
        return value

    def get_json(self, filepath, warning=False) -> Any:
        try:
            value = self._client.get(filepath)
            value = json.loads(value)
        except:
            if warning:
                warnings.warn("Failed to get json from {}".format(filepath))
                value = None
            else:
                raise Exception("Failed to get json from {}".format(filepath))
        return value
    
    def put_bytes(self, filepath, bytes_value):
        self._client.put(filepath, bytes_value)

    def put_uint16_png(self, filepath, value) -> None:
        success, img_array = cv2.imencode(".png", value, params=[cv2.CV_16U])
        assert success
        img_bytes = img_array.tobytes()
        self._client.put(filepath, img_bytes)
        # self._client.put(filepath, img_bytes, update_cache=True)

    def put_uint8_jpg(self, filepath, value) -> None:
        success, img_array = cv2.imencode(".jpg", value)
        assert success
        img_bytes = img_array.tobytes()
        self._client.put(filepath, img_bytes)
        # self._client.put(filepath, img_bytes, update_cache=True)

    def put_uint8_png(self, filepath, value) -> None:
        success, img_array = cv2.imencode(".png", value)
        assert success
        img_bytes = img_array.tobytes()
        self._client.put(filepath, img_bytes)
        # self._client.put(filepath, img_bytes, update_cache=True)

    def put_npz(self, filepath, value) -> None:
        value = pickle.dumps(value)
        self._client.put(filepath, value)
        # self._client.put(filepath, value, update_cache=True)

    def put_json(self, filepath, value) -> None:
        value = json.dumps(value).encode()
        self._client.put(filepath, value)
        # self._client.put(filepath, value, update_cache=True)

    def put_text(self, filepath, value) -> None:
        self._client.put(filepath, bytes(value, encoding="utf-8"))
        # self._client.put(filepath, bytes(value, encoding='utf-8'), update_cache=True)

    def join_path(
        self, filepath: Union[str, Path], *filepaths: Union[str, Path]
    ) -> str:
        """Concatenate all file paths.
        Args:
            filepath (str or Path): Path to be concatenated.
        Returns:
            str: The result after concatenation.
        """
        # filepath = self._format_path(self._map_path(filepath))
        if filepath.endswith("/"):
            filepath = filepath[:-1]
        formatted_paths = [filepath]
        for path in filepaths:
            formatted_paths.append(path)
        return "/".join(formatted_paths)

    # from mmcv
    def list_dir_or_file(
        self,
        dir_path: Union[str, Path],
        list_dir: bool = True,
        list_file: bool = True,
        suffix: Optional[Union[str, Tuple[str]]] = None,
        recursive: bool = False,
    ) -> Iterator[str]:
        """Scan a directory to find the interested directories or files in
        arbitrary order.
        Note:
            Petrel has no concept of directories but it simulates the directory
            hierarchy in the filesystem through public prefixes. In addition,
            if the returned path ends with '/', it means the path is a public
            prefix which is a logical directory.
        Note:
            :meth:`list_dir_or_file` returns the path relative to ``dir_path``.
            In addition, the returned path of directory will not contains the
            suffix '/' which is consistent with other backends.
        Args:
            dir_path (str | Path): Path of the directory.
            list_dir (bool): List the directories. Default: True.
            list_file (bool): List the path of files. Default: True.
            suffix (str or tuple[str], optional):  File suffix
                that we are interested in. Default: None.
            recursive (bool): If set to True, recursively scan the
                directory. Default: False.
        Yields:
            Iterable[str]: A relative path to ``dir_path``.
        """
        # if not has_method(self._client, 'list'):
        #     raise NotImplementedError(
        #         'Current version of Petrel Python SDK has not supported '
        #         'the `list` method, please use a higher version or dev'
        #         ' branch instead.')

        # dir_path = self._map_path(dir_path)
        # dir_path = self._format_path(dir_path)
        # if list_dir and suffix is not None:
        #     raise TypeError(
        #         '`list_dir` should be False when `suffix` is not None')

        # if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        #     raise TypeError('`suffix` must be a string or tuple of strings')

        # Petrel's simulated directory hierarchy assumes that directory paths
        # should end with `/`
        if not PetrelBackend.is_ceph_path(dir_path):
            return os.listdir(dir_path)
        if not dir_path.endswith("/"):
            dir_path += "/"

        root = dir_path

        def _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive):
            for path in self._client.list(dir_path):
                # the `self.isdir` is not used here to determine whether path
                # is a directory, because `self.isdir` relies on
                # `self._client.list`
                if path.endswith("/"):  # a directory path
                    next_dir_path = self.join_path(dir_path, path)
                    if list_dir:
                        # get the relative path and exclude the last
                        # character '/'
                        rel_dir = next_dir_path[len(root) : -1]
                        yield rel_dir
                    if recursive:
                        yield from _list_dir_or_file(
                            next_dir_path, list_dir, list_file, suffix, recursive
                        )
                else:  # a file path
                    absolute_path = self.join_path(dir_path, path)
                    rel_path = absolute_path[len(root) :]
                    if (suffix is None or rel_path.endswith(suffix)) and list_file:
                        yield rel_path

        return _list_dir_or_file(dir_path, list_dir, list_file, suffix, recursive)
    
    def glob_imgs(
            self,
            path: Union[str, Path],
    ):
        imgs = []
        if PetrelBackend.is_ceph_path(path):
            for ext in ['.png', '.jpg', '.JPEG', '.JPG']:
                imgs.extend(self.list_dir_or_file(path, suffix=ext))
            imgs = [os.path.join(path, img) for img in imgs]
        else:
            for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:    
                imgs.extend(glob(os.path.join(path, ext)))
        return imgs
    
    def imageopen(self, image_path: str):
        if PetrelBackend.is_ceph_path(image_path):
            img = Image.fromarray(cv2.cvtColor(self.get_uint8_png(image_path), cv2.COLOR_BGR2RGB))
        else:
            img = Image.open(image_path) 
        return img
    
    def mask_or_depth_read(self, image_path: str):
        if PetrelBackend.is_ceph_path(image_path):
            img = self.get_uint8_png(image_path)
        else:
            img = imageio.v2.imread(image_path)
        return img
    
    def cv2_imread(self,  image_path: str, flag=1):
        if PetrelBackend.is_ceph_path(image_path):
            img = self.get_uint8_png(image_path)
        else:
            img = cv2.imread(image_path, flags=flag)
        return img
    
    def check_img_file(self, image_path):
        if PetrelBackend.is_ceph_path(image_path):
            value = np.frombuffer(self._client.get(image_path), np.uint8)
            img = cv2.imdecode(value, cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(image_path)
        return img
    
    def listdir(self, dir_path: str):
        if PetrelBackend.is_ceph_path(dir_path):
            res_list = self.list_dir_or_file(dir_path)
            return [i for i in res_list]
        else:
            return os.listdir(dir_path)
    
    # def load_image(self, image_path: str):
    #     transform = T.Compose(
    #     [
    #         T.RandomResize([800], max_size=1333),
    #         T.ToTensor(),
    #         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #     ]
    #     )
    #     # image_source = Image.open(image_path).convert("RGB")
    #     if PetrelBackend.is_ceph_path(image_path):
    #         image = cv2.cvtColor(self.get_uint8_png(image_path), cv2.COLOR_BGR2RGB)
    #         image_source = Image.fromarray(image)
    #     else:
    #         image_source = Image.open(image_path).convert("RGB")
    #         image = np.asarray(image_source)
    #     image_transformed, _ = transform(image_source, None)
    #     return image, image_transformed
    
    def json_load(self, json_path: str):
        if PetrelBackend.is_ceph_path(json_path):
            return self.get_json(json_path)
        else:
            with open(json_path) as user_file:
                res_file = json.load(user_file)
            return res_file
        
    def json_write(self, json_path: str, value):
        if PetrelBackend.is_ceph_path(json_path):
            value = json.dumps(value, indent=4, default=int).encode()
            self._client.put(json_path, value)
        else:
            json_info = json.dumps(value, indent=4, default=int)
            file_json = open(json_path, 'w')
            file_json.write(json_info)
            file_json.close()
    
    def load_img_as_tensor(self, img_path, image_size):
        if PetrelBackend.is_ceph_path(img_path):
            img_pil = Image.fromarray(cv2.cvtColor(self.get_uint8_png(img_path), cv2.COLOR_BGR2RGB))
        else:
            img_pil = Image.open(img_path)
        img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
        if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
            img_np = img_np / 255.0
        else:
            raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
        img = torch.from_numpy(img_np).permute(2, 0, 1)
        video_width, video_height = img_pil.size  # the original video size
        return img, video_height, video_width
    
    def isdir(self, dir_path):
        if PetrelBackend.is_ceph_path(dir_path):
            return len(self.listdir(dir_path)) > 0
        else:
            return os.path.isdir(dir_path)

    # from mmcv
    def exists(self, filepath: Union[str, Path]) -> bool:
        """Check whether a file path exists.
        Args:
            filepath (str or Path): Path to be checked whether exists.
        Returns:
            bool: Return ``True`` if ``filepath`` exists, ``False`` otherwise.
        """
        if not PetrelBackend.is_ceph_path(filepath):
            return os.path.exists(filepath)
        if not (
            has_method(self._client, "contains") and has_method(self._client, "isdir")
        ):
            raise NotImplementedError(
                "Current version of Petrel Python SDK has not supported "
                "the `contains` and `isdir` methods, please use a higher"
                "version or dev branch instead."
            )

        return self._client.contains(filepath) or self._client.isdir(filepath)
