import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import timm
from scipy.stats import norm
from typing import Tuple, List
import cv2 as cv
import torch

from Pipeline.results import LungsDXResult
from Pipeline.configs import *
from Pipeline.models.base import BaseModel


class BinaryTriageModel(nn.Module):
    def __init__(self):
        """
        Binary triage nn model class.
        """
        super().__init__()
        self.model = timm.create_model("tf_efficientnet_b5", num_classes=1, pretrained=False)

    def forward(self, x):
        """
        Method for getting predictions for single image.
        :param x: tensor containing image under study.
        :returns torch tensor containing predictions.
        """
        x = self.model(x)
        return torch.sigmoid(x)


class BinaryTriage(BaseModel):
    def __init__(self, use_projection: str, device: torch.device) -> None:
        """
        Binary triage class.
        :parameter use_projection: list of parameters indicating which nn weights to load.
        :param device: the device on which the models working in the pipeline will be launched.
        """
        super().__init__()
        self.target_size = TRIAGE_TARGET_SIZE
        self.triage = BinaryTriageModel()
        self.triage.load_state_dict(self.get_state_dict(PATH_TO_TRIAGE_MODEL), strict=True)
        self.triage = self.triage.to(device)
        self.triage.eval()
        self.device = device
        self.side_size = TRIAGE_TARGET_SIZE[0]
        self.threshold = TRIAGE_THRESHOLD
        self.batch_size = 1
        self.num_patches = 5

    @torch.no_grad()
    def predictions(self, image_data: LungsDXResult) -> LungsDXResult:
        """
        Method for making predictions for image.
        :param image_data: LungsDXResult class object containing image data.
        :returns updated image_data object
        """
        img = self.preprocess_image(image=image_data.triage_image.copy())
        img = img.to(self.device)
        prediction = self.triage(img)
        prediction = float(prediction.squeeze())
        is_pathological = prediction >= self.threshold
        if is_pathological:
            heatmap = self.__create_heatmap(
                confidence=prediction,
                img=image_data.pathology_image.copy(),
                lungs=image_data.lungs_segmentation_mask.copy()
            )
            mask = cv.flip(heatmap, 1) if image_data.flip else heatmap
            image_data.triage_result.update(confidence=prediction, mask=mask)
        else:
            image_data.triage_result.update(confidence=prediction)
        return image_data

    def preprocess_image(
            self, image: np.array, return_pad: bool = False
    ) -> torch.Tensor or Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Method for image pre-processing.
        :param image: numpy array containing image.
        :param return_pad: parameter indicating whether to return padding or not.
        :return transformed image.
        """
        image = self.to_3channels_shape(image)
        image = self.longest_maxsize(image, max_size=self.target_size[0])
        if return_pad:
            image, padding = self.pad_if_needed(
                image,
                min_height=self.target_size[0],
                min_width=self.target_size[1],
                border_mode=cv.BORDER_CONSTANT,
                value=0.0,
                return_pad=True
            )
            image = self.imagewise_normalize(image)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).unsqueeze(0)
            return image, padding
        else:
            image = self.pad_if_needed(image, min_height=self.target_size[0], min_width=self.target_size[1])
            image = self.imagewise_normalize(image)
            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).unsqueeze(0)
            return image

    def mask_preprocess(self, mask: np.array) -> np.array:
        """
        Method for pathology mask preprocessing.
        :param mask: pathology mask.
        :return processed pathology mask.
        """
        mask = self.longest_maxsize(mask, max_size=self.target_size[0])
        mask = self.pad_if_needed(
            mask,
            min_height=self.target_size[0],
            min_width=self.target_size[1],
            border_mode=cv.BORDER_CONSTANT,
            value=0
        )
        return mask

    def __create_heatmap(self, img: np.ndarray, lungs: np.ndarray, confidence: float) -> np.ndarray:
        """
        Method for making binary triage heatmap.
        :param img: image under study.
        :param lungs: lungs segmentation mask.
        :param confidence: predicted pathological confidence score.
        :returns numpy array containing binary triage heatmap
        """
        h, w = img.shape[:2]
        roi = self.__get_roi_coordinates(lungs_mask=lungs, preprocess_mask=False)
        width = roi[2] - roi[0]

        predictions, heatmap_shape = self.__sliding_window(image=img, lungs_mask=lungs)
        patch_size = width // self.num_patches
        heatmap_height, heatmap_width = heatmap_shape

        if (predictions < confidence).any():
            predictions = np.clip(predictions - confidence, a_min=-10, a_max=0)
        predictions = (predictions - predictions.mean()) / (predictions.std() + 1e-8)
        predictions = 1 - norm.cdf(predictions)
        threshold = 0.75

        while threshold > 0.0:
            if (predictions > threshold).any():
                break
            threshold -= 0.25

        prediction_index = 0
        heatmap_with_borders = np.zeros((heatmap_height + 2, heatmap_width + 2), dtype=np.float32)
        heatmap = heatmap_with_borders[1:-1, 1:-1]
        width_middle = (heatmap_width - 1) // 2

        for i in range(heatmap_height):
            for j in range(heatmap_width):
                pred = predictions[prediction_index]
                prediction_index += 1

                if (i == 0 and j in [0, heatmap_width - 1]) or \
                        (i in [0, heatmap_height - 1] and j == width_middle) or \
                        (i == heatmap_height - 1 and j in [width_middle - 1, width_middle + 1]):
                    continue

                if pred > threshold:
                    heatmap[i, j] = 1.0

        if heatmap[:, width_middle].sum() <= 1:
            heatmap[:, width_middle] = 0.0

        heatmap = cv.resize(heatmap_with_borders, None, fx=patch_size, fy=patch_size, interpolation=cv.INTER_CUBIC)
        heatmap = (heatmap > 0.5).astype(np.float32)
        imagesize_heatmap = np.zeros((h, w), dtype=np.float32)
        x1, x2 = roi[0], min(roi[0] + heatmap_width * patch_size, w)
        y1, y2 = roi[1], min(roi[1] + heatmap_height * patch_size, h)
        imagesize_heatmap[y1:y2, x1:x2] = heatmap[patch_size:patch_size + y2 - y1, patch_size:patch_size + x2 - x1]
        return imagesize_heatmap

    @torch.no_grad()
    def __sliding_window(self, image: np.ndarray, lungs_mask: np.array) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Method for finding areas of the greatest interest for the model.
        :param image: image under study.
        :param lungs_mask: lungs segmentation mask.
        :returns numpy array containing predictions and initial size.
        """
        img = self.preprocess_image(image=image, return_pad=False)
        img = img.to(self.device)
        roi = self.__get_roi_coordinates(lungs_mask=lungs_mask, preprocess_mask=True)
        width = roi[2] - roi[0]
        height = roi[3] - roi[1]
        patch_size = width // self.num_patches
        heatmap_height = int(np.ceil(height / patch_size))
        heatmap_width = self.num_patches

        xs = []
        samples = 0
        predictions = []
        for i in range(heatmap_height):
            for j in range(heatmap_width):
                # patch coordinates
                x1, x2 = np.minimum(roi[0] + np.array([j, j + 1]) * patch_size, roi[2])
                y1, y2 = np.minimum(roi[1] + np.array([i, i + 1]) * patch_size, roi[3])

                img_blacked = img.clone()
                img_blacked[:, :, y1:y2, x1:x2] = 0
                xs.append(img_blacked)

                samples += 1
                if samples == self.batch_size or (i == heatmap_height - 1 and j == heatmap_width - 1):
                    xs = torch.cat(xs, dim=0)
                    predictions.append(self.triage(xs).cpu().numpy())
                    samples = 0
                    xs = []

        predictions = np.concatenate(predictions, axis=0)
        return predictions, (heatmap_height, heatmap_width)

    def __get_roi_coordinates(self, lungs_mask: np.array, preprocess_mask: bool) -> Tuple[int, int, int, int]:
        """
        Method for getting region of interest coordinates.
        :param preprocess_mask: parameter indicating whether to preprocess mask or not.
        :param lungs_mask: lungs segmentation mask.
        :returns tuple containing region of interest coordinates.
        """
        if preprocess_mask:
            lungs_mask = self.mask_preprocess(mask=lungs_mask)

        lungs_contours, _ = cv.findContours(lungs_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(lungs_contours) > 2:
            lungs_contours = sorted(lungs_contours, key=lambda x: cv.contourArea(x), reverse=True)[:2]
        if len(lungs_contours) == 2:
            lungs_contour = np.array(lungs_contours[0].tolist() + lungs_contours[1].tolist())
        else:
            lungs_contour = lungs_contours[0]
        x_min = tuple(lungs_contour[lungs_contour[:, :, 0].argmin()][0])[0]
        x_max = tuple(lungs_contour[lungs_contour[:, :, 0].argmax()][0])[0]
        y_min = tuple(lungs_contour[lungs_contour[:, :, 1].argmin()][0])[1]
        y_max = tuple(lungs_contour[lungs_contour[:, :, 1].argmax()][0])[1]
        return x_min, y_min, x_max, y_max
