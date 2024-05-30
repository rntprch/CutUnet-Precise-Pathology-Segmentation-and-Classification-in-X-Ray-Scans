import numpy as np
from typing import Tuple
from typing import List
from collections import OrderedDict
import cv2 as cv
import torch

from Pipeline.results import LungsDXResult
from Pipeline.models.base import PreprocessingModel
from Pipeline.configs import *
import matplotlib.pyplot as plt


class LungsModel:
    def __init__(self, use_projection: List[str], device: torch.device) -> None:
        """
        Lungs segmentation nn model class.
        :parameter use_projection: list of parameters indicating which nn weights to load.
        :param device: the device on which the models working in the pipeline will be launched.
        """
        self.device = device
        if 'frontal' in use_projection:
            self.frontal_model = SEGMENTATION_NN(
                encoder_name=SEGMENTATION_ENCODER,
                encoder_weights=None,
                activation=SEGMENTATION_ACTIVATION,
                in_channels=SEGMENTER_TARGET_SIZE[0],
                classes=SEGMENTATION_NUM_CLASSES[0]
            )
            self.frontal_model.load_state_dict(self.load_pretrain(FRONTAL_SEGMENTATION_MODEL_PATH))
            self.frontal_model = self.frontal_model.to(self.device)
            self.frontal_model.eval()
        if 'lateral' in use_projection:
            self.lateral_model = SEGMENTATION_NN(
                encoder_name=SEGMENTATION_ENCODER,
                encoder_weights=None,
                activation=SEGMENTATION_ACTIVATION,
                in_channels=SEGMENTER_TARGET_SIZE[0],
                classes=SEGMENTATION_NUM_CLASSES[1]
            )
            self.lateral_model.load_state_dict(self.load_pretrain(LATERAL_SEGMENTATION_MODEL_PATH))
            self.lateral_model = self.frontal_model.to(self.device)
            self.lateral_model.eval()

    def forward(self, x: torch.Tensor, projection: str) -> np.array:
        """
        Method for making predictions for single image.
        :param x: tensor containing image under study.
        :param projection: parameter indicating which projection of the lungs on the image under study.
        :returns numpy array containing predicted mask.
        """
        if projection == 'frontal':
            x = self.frontal_model(x).cpu().detach().numpy()
        else:
            x = self.lateral_model(x).cpu().detach().numpy()
        return x

    def load_pretrain(self, path_to_model: str) -> OrderedDict:
        """
        Method for loading trained model.
        :param path_to_model: path to trained weights.
        :returns dict containing model weights.
        """
        weights = torch.load(path_to_model, map_location=self.device)['model_state_dict']
        new_weights = OrderedDict()
        for key, val in weights.items():
            if key.startswith("module.nn."):
                new_key = key[len("module.nn."):]
            elif key.startswith('nn.'):
                new_key = key[len('nn.'):]
            else:
                new_key = key
            new_weights[new_key] = val
        return new_weights


class LungsSegmentation(PreprocessingModel):
    def __init__(self, use_projection: List[str], device: torch.device) -> None:
        """
        Lungs segmentation class.
        :parameter use_projection: list of parameters indicating which nn weights to load.
        :param device: the device on which the models working in the pipeline will be launched.
        """
        super().__init__(
            SEGMENTER_TARGET_SIZE, smp.encoders.get_preprocessing_fn(SEGMENTATION_ENCODER, 'imagenet')
        )
        try:
            self.model = LungsModel(use_projection=use_projection, device=device)
        except FileNotFoundError:
            raise ValueError('There are no weights to load the trained model! Try to train the model first.')
        self.target_size = SEGMENTER_TARGET_SIZE
        self.device = device

    @torch.no_grad()
    def predictions(self, image_data: LungsDXResult) -> LungsDXResult:
        """
        Method for making predictions for image.
        :param image_data: LungsDXResult class object containing image data.
        :returns updated image_data object
        """
        image = self.preprocess_image(image=image_data.preprocess_image.copy())
        image = image.to(device=self.device)
        predict_mask = self.model.forward(image, projection=image_data.validation['stage_4'])

        lungs_mask = self.__mask_postprocessing(
            mask=predict_mask[0, 0, :, :],
            target_shape=(image_data.pathology_image.shape[1], image_data.pathology_image.shape[0])
        )

        image_data = self.__mask_filter(mask=lungs_mask, image_data=image_data, obj_type='lungs')

        if image_data.lungs_segmentation_mask.any():
            if REQUIRED_PATHOLOGIES['heart'] and image_data.validation['stage_4'] == 'frontal':
                heart_mask = self.__mask_postprocessing(
                    mask=predict_mask[0, 1, :, :],
                    target_shape=(image_data.pathology_image.shape[1], image_data.pathology_image.shape[0])
                )
                image_data = self.__mask_filter(mask=heart_mask, image_data=image_data, obj_type='heart')
            else:
                image_data.warnings.append(6)
        else:
            image_data.warnings.append(6)

        return image_data

    @staticmethod
    def __mask_filter(mask: np.array, image_data: LungsDXResult, obj_type: str) -> LungsDXResult:
        """
        Method for segmentation mask filtration.
        :param mask: numpy array containing mask.
        :param image_data: LungsDXResult class object containing image data.
        :param obj_type: type of segmentation mask.
        :return image_data object with filtered segmentation mask.
        """
        min_size = int(mask.shape[0] * mask.shape[1] * SEGMENTATION_SIZE_THRESHOLD)
        processed_mask = np.zeros_like(mask)

        contours, _ = cv.findContours(image=mask, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)
        approved_contours = [cnt for cnt in contours if cv.contourArea(cnt) >= min_size]
        approve_number = 2 if image_data.validation['stage_4'] == 'frontal' and obj_type == 'lungs' else 1

        if len(approved_contours) >= approve_number:
            cv.fillPoly(processed_mask, pts=approved_contours[:approve_number], color=1)
            if obj_type == 'lungs':
                image_data.lungs_segmentation_mask = processed_mask
            else:
                if MESH in REQUIRED_PATHOLOGIES['heart']:
                    image_data.mediastinal_shift.update(mask=processed_mask)
                if CTR in REQUIRED_PATHOLOGIES['heart']:
                    image_data.cardiomegaly.update(mask=processed_mask)
        else:
            image_data.warnings.append(6)
            if obj_type == 'lungs':
                image_data.warnings.append(7)
                if len(approved_contours) == 1:
                    cv.fillPoly(processed_mask, pts=approved_contours, color=1)
                    image_data.lungs_segmentation_mask = processed_mask

        return image_data

    @staticmethod
    def __mask_postprocessing(mask: np.array, target_shape: Tuple[int, int]) -> np.array:
        """
        Method for segmentation mask filtration and resize.
        :param mask: numpy array containing mask.
        :param target_shape: original image shape to which segmentation mask should be resized.
        :return filtered by threshold segmentation mask.
        """
        mask[mask < LUNGS_THRESHOLD * mask.max()] = 0
        mask[mask >= LUNGS_THRESHOLD * mask.max()] = 1
        mask = mask.astype('uint8')
        mask = cv.resize(mask, target_shape, interpolation=cv.INTER_AREA)

        return mask
