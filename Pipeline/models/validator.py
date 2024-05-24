import numpy as np
from typing import Tuple
import albumentations as aug
import torch
from albumentations.pytorch import ToTensorV2
import imutils
import timm

from Pipeline.models.base import PreprocessingModel
from Pipeline.results import LungsDXResult
from Pipeline.configs import *


class ValidatorModel:
    def __init__(self, device: torch.device) -> None:
        """
        Lungs validation nn model class.
        :param device: the device on which the models working in the pipeline will be launched.
        """
        self.nn_1 = timm.create_model(ENCODER_1, pretrained=False, num_classes=NUM_CLASSES_1)
        self.nn_2 = timm.create_model(ENCODER_2, pretrained=False, num_classes=NUM_CLASSES_2)
        self.nn_3 = timm.create_model(ENCODER_3, pretrained=False, num_classes=NUM_CLASSES_3)
        self.nn_4 = timm.create_model(ENCODER_4, pretrained=False, num_classes=NUM_CLASSES_4)
        self.nn_1.load_state_dict(torch.load(PATH_TO_MODEL_1, map_location=device)['model_state_dict'])
        self.nn_2.load_state_dict(torch.load(PATH_TO_MODEL_2, map_location=device)['model_state_dict'])
        self.nn_3.load_state_dict(torch.load(PATH_TO_MODEL_3, map_location=device)['model_state_dict'])
        self.nn_4.load_state_dict(torch.load(PATH_TO_MODEL_4, map_location=device)['model_state_dict'])
        self.nn_1 = self.nn_1.to(device)
        self.nn_2 = self.nn_2.to(device)
        self.nn_3 = self.nn_3.to(device)
        self.nn_4 = self.nn_4.to(device)
        self.nn_1.eval()
        self.nn_2.eval()
        self.nn_3.eval()
        self.nn_4.eval()

    def get_stage_1_predictions(self, image: torch.Tensor) -> np.array:
        """
        Method for getting predictions.
        :param image: torch tensor containing image.
        :return array containing predictions.
        """
        predictions = self.nn_1(image)
        inv_predictions = np.where(
            torch.sigmoid(predictions[:, 0]).cpu().detach().numpy().astype(float) >
            VALIDATOR_THRESHOLD['stage2'], 1., 0.
        )
        rot_predictions = np.where(
            torch.softmax(predictions[:, 1:], dim=1).cpu().detach().numpy().astype(float) >
            VALIDATOR_THRESHOLD['stage2'], 1., 0.
        )
        return inv_predictions, np.argmax(rot_predictions, axis=-1)

    def get_stage_2_predictions(self, image: torch.Tensor) -> np.array:
        """
        Method for getting predictions.
        :param image: torch tensor containing image.
        :return array containing predictions.
        """
        predictions = self.nn_2(image)
        return np.where(
            torch.sigmoid(predictions).cpu().detach().numpy().astype(float) > VALIDATOR_THRESHOLD['stage1'], 1, 0
        )

    def get_stage_3_predictions(self, image: torch.Tensor) -> np.array:
        """
        Method for getting predictions.
        :param image: torch tensor containing image.
        :return array containing predictions.
        """
        predictions = self.nn_3(image)
        predictions = np.where(
            torch.softmax(predictions, dim=1).cpu().detach().numpy().astype(float) > VALIDATOR_THRESHOLD['stage3'], 1, 0
        )
        return np.argmax(predictions, axis=-1)

    def get_stage_4_predictions(self, image: torch.Tensor) -> np.array:
        """
        Method for getting predictions.
        :param image: torch tensor containing image.
        :return array containing predictions.
        """
        predictions = self.nn_4(image)
        predictions = np.where(
            torch.softmax(predictions, dim=1).cpu().detach().numpy().astype(float) > VALIDATOR_THRESHOLD['stage4'], 1, 0
        )
        return np.argmax(predictions, axis=-1)


class LungsValidator(PreprocessingModel):

    def __init__(self, device: torch.device) -> None:
        """
        Image validator class.
        :param device: the device on which the models working in the pipeline will be launched.
        """
        super().__init__(
            target_size=VALIDATOR_TARGET_SIZE,
            preprocessing_fn=VALIDATOR_PREPROCESSING_FN,
        )
        self.device = device
        try:
            self.validator = ValidatorModel(device=self.device)
        except FileNotFoundError:
            raise ValueError('There are no weights to load the trained model! Try to train the model first.')

    @torch.no_grad()
    def predictions(self, image_data: LungsDXResult) -> LungsDXResult:
        """
        Method for making predictions for single image.
        :param image_data: LungsDXResult class object containing image data.
        :returns updated image_data object
        """
        orig_image = self.preprocess_image(image=image_data.preprocess_image.copy())
        image = orig_image.to(device=self.device)

        inv_predictions, rot_predictions = self.validator.get_stage_1_predictions(image)
        image, image_data = self.__update_results_1(
            inv_predictions=inv_predictions, rot_predictions=rot_predictions, image=image, image_data=image_data
        )
        if self.validator.get_stage_2_predictions(image).any():
            image_data.validation['stage_2'] = 'valid'
            crop_predictions = self.validator.get_stage_3_predictions(image)
            image_data = self.__update_results_2(image_data=image_data, predictions=crop_predictions)

            projection_predictions = self.validator.get_stage_4_predictions(image)
            image_data = self.__update_results_3(predictions=projection_predictions, image_data=image_data)
        else:
            image_data.validation['stage_2'] = 'non valid'
        return image_data

    def __update_results_1(
                self, inv_predictions: np.array, rot_predictions: np.array, image: torch.Tensor, image_data: LungsDXResult
        ) -> Tuple[torch.Tensor, LungsDXResult]:
        """
        Method using to update predictions.
        :param inv_predictions: array containing predictions of inversion validator (stage 2).
        :param rot_predictions: array containing predictions of rotation validator (stage 2).
        :param image: torch tensor containing image.
        :return processed image tensor
        """
        image = self.__tensor_to_image(image)

        if inv_predictions.any():
            results = 'inverted'
            image = image.max() - image
        else:
            results = 'non inverted'
        results = results + ', rotation degree: ' + str(90 * rot_predictions[0])
        if rot_predictions:
            image = imutils.rotate_bound(image, -90 * int(rot_predictions[0]))

        image_data.validation['stage_1'] = results
        return self.__image_to_tensor(image), image_data

    @staticmethod
    def __update_results_2(predictions: np.array, image_data: LungsDXResult) -> LungsDXResult:
        """
        Method using to update predictions.
        :param predictions: array containing predictions of view validator (stage 4).
        :return processed image tensor
        """
        class_names = ['bad crop', 'cropped', 'normal']
        image_data.validation['stage_3'] = class_names[int(predictions)]
        return image_data

    @staticmethod
    def __update_results_3(predictions: np.array, image_data: LungsDXResult) -> LungsDXResult:
        """
        Method using to update predictions.
        :param predictions: array containing predictions of view validator (stage 4).
        :return processed image tensor
        """
        class_names = ['frontal', 'lateral left', 'lateral']
        image_data.validation['stage_4'] = class_names[int(predictions)]
        return image_data

    @staticmethod
    def __tensor_to_image(image: torch.Tensor) -> np.array:
        """
        Method for converting image from tensor to array.
        :param image: torch tensor containing image.
        :return array containing image.
        """
        image = image[0, :, :, :].cpu().detach().numpy()
        image = np.moveaxis(image, 0, -1)
        image = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')
        return image

    def __image_to_tensor(self, image: np.array) -> torch.Tensor:
        """
        Method for converting image to tensor placing it on device.
        :param image: array containing image.
        :return torch tensor containing image.
        """
        transformer = aug.Compose([aug.Lambda(image=smp.encoders.get_preprocessing_fn('efficientnet-b0', 'imagenet')),
                                   ToTensorV2()])
        image = transformer(image=image)['image'].to(device=self.device)
        return torch.unsqueeze(image, 0).float()
