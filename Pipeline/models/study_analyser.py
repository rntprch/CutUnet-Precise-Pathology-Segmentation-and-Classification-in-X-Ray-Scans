import numpy as np
import torch
import timm

from Pipeline.models.base import PreprocessingModel
from Pipeline.results import LungsDXResult
from Pipeline.configs import *


class StudyTypeModel:
    def __init__(self, device: torch.device) -> None:
        """
        Lungs validation nn model class.
        :param device: the device on which the models working in the pipeline will be launched.
        """
        self.nn = timm.create_model(ENCODER_1, pretrained=False, num_classes=NUM_CLASSES_1)
        self.nn.load_state_dict(torch.load(PATH_TO_MODEL_1, map_location=device)['model_state_dict'])
        self.nn = self.nn.to(device)
        self.nn.eval()

    def forward(self, image: torch.Tensor) -> np.array:
        """
        Method for getting predictions.
        :param image: torch tensor containing image.
        :return array containing predictions.
        """
        predictions = self.nn(image)
        predictions = np.where(
            torch.softmax(predictions, dim=1).cpu().detach().numpy().astype(float) > THRESHOLD, 1., 0.
        )
        return np.argmax(predictions, axis=-1)

class StudyType(PreprocessingModel):

    def __init__(self, device: torch.device) -> None:
        """
        Image study type analyser class.
        :param device: the device on which the models working in the pipeline will be launched.
        """
        super().__init__(target_size=VALIDATOR_TARGET_SIZE, preprocessing_fn=VALIDATOR_PREPROCESSING_FN)
        self.device = device
        try:
            self.model = StudyTypeModel(device=self.device)
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
        predictions = self.model.forward(image)
        class_names = ['prl', 'flg']
        image_data.study_type = class_names[int(predictions)]
        return image_data
