from abc import ABC, abstractmethod
import albumentations as aug
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from typing import Tuple, Any, List
import numpy as np
import torch
from collections import OrderedDict

from Pipeline.results import LungsDXResult


class BaseModel(ABC):

    @abstractmethod
    def predictions(self, orig_image_data: LungsDXResult):
        pass

    @staticmethod
    def imagewise_normalize(img, eps=1e-6) -> np.array:
        """
        Method for image image-wize normalization.
        :parameter img: timage to be normalized.
        :parameter eps: parameter.
        :return normalized image
        """
        img = img.astype(np.float32)
        mean = img.mean(axis=(0, 1), dtype=np.float32)
        std = img.std(axis=(0, 1), dtype=np.float32)
        denominator = np.reciprocal(std + eps, dtype=np.float32)
        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img.astype(np.float32)

    @staticmethod
    def py3round(number):
        if abs(round(number) - number) == 0.5:
            return int(2.0 * round(number / 2.0))

        return int(round(number))

    def longest_maxsize(self, img, max_size, interpolation=cv.INTER_LINEAR) -> np.array:
        """
        Method for resizing image according its longest size.
        :parameter img: numpy array containing image.
        :parameter max_size: numpy array containing image.
        :parameter interpolation: type of opencv interpolation.
        :return resized image.
        """
        height, width = img.shape[:2]
        scale = max_size / float(max(width, height))
        if scale != 1.0:
            new_height, new_width = tuple(self.py3round(dim * scale) for dim in (height, width))
            img = cv.resize(img, dsize=(new_width, new_height), interpolation=interpolation)
        return img

    @staticmethod
    def to_3channels_shape(x) -> np.array:
        """
        Method for reshaping grayscale image into 3 channels.
        :parameter x: numpy array containing image.
        :return 3 channel image.
        """
        x = np.atleast_3d(np.squeeze(x))

        if x.shape[-1] > 3:
            x = x[:, :, x.shape[-1] // 2:x.shape[-1] // 2 + 1]

        if x.shape[-1] == 1:
            x = np.repeat(x, repeats=3, axis=-1)

        return x

    @staticmethod
    def pad_if_needed(
            image, min_height, min_width, border_mode=cv.BORDER_REFLECT_101, value=None, return_pad=False
    ) -> np.array or Tuple[np.array, Tuple[int, int, int, int]]:
        """
        Method for image padding.
        :parameter image: numpy array containing image.
        :parameter min_height: minimum image height.
        :parameter min_width: minimum image width.
        :parameter border_mode: border mode.
        :parameter value: padding value.
        :parameter return_pad: parameter indicating whether to return padding shape or not.
        :return padded image or padded image and padding shape.
        """
        rows, cols = image.shape[:2]

        if min_height > rows:
            h_pad_top = int((min_height - rows) / 2.0)
            h_pad_bottom = min_height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if min_width > cols:
            w_pad_left = int((min_width - cols) / 2.0)
            w_pad_right = min_width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        image = cv.copyMakeBorder(
            image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, borderType=border_mode, value=value
        )
        if return_pad:
            return image, (h_pad_top, h_pad_bottom, w_pad_left, w_pad_right)
        return image

    @staticmethod
    def get_state_dict(path) -> OrderedDict:
        """
        Method for getting state dict with trained weights.
        :parameter path: path to trained weights.
        :return weights state dict
        """
        state_dict = torch.load(path, map_location='cpu')
        if 'model' in state_dict:
            return state_dict['model']
        new_state_dict = OrderedDict()

        for key in state_dict:
            new_key = key

            if 'backbone' in key:
                new_key = new_key.replace('backbone', 'model')

            if key.startswith('module'):
                new_key = new_key.split('.', maxsplit=1)[1]

            new_state_dict[new_key] = state_dict[key]

        return new_state_dict


class PreprocessingModel(BaseModel, ABC):
    def __init__(self, target_size: Tuple[int, int], preprocessing_fn: Any) -> None:
        """
        Preprocessing model base class.
        :param target_size: size of image required for model input.
        :param preprocessing_fn: function for image preprocessing.
        """
        self.target_size = target_size
        self.preprocessing_fn = preprocessing_fn

    @abstractmethod
    def predictions(self, orig_image_data):
        pass

    def preprocess_image(self, image: np.array) -> torch.Tensor:
        """
        Method for image pre-processing.
        :param image: numpy array containing image.
        :return transformed image.
        """
        image = self.to_3channels_shape(image)
        transform = [aug.Resize(height=self.target_size[1], width=self.target_size[2],
                                interpolation=cv.INTER_AREA, always_apply=True),
                     aug.Lambda(image=self.preprocessing_fn),
                     ToTensorV2()]
        transformer = aug.Compose(transform)
        image = transformer(image=image)['image']
        return image.float().unsqueeze(0)
