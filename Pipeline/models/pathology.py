import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
import cv2 as cv
import torch

from Pipeline.results import LungsDXResult
from Pipeline.models.base import BaseModel
from Pipeline.calibration import ModelCalibrator
from Pipeline.configs import *


class PathologyModel(BaseModel):
    def __init__(
            self,
            num_classes: int,
            encoders: List[str],
            target_size: Tuple[int, int],
            model_paths: List[str],
            device: torch.device,
            study_type: str,
            model_architectures: List[str],
            use_projection: str,
            calibration_paths: List[str] = None,
            run_fp_16: bool = False,
            apply_calibration: bool = False,
    ) -> None:
        """
        Pathology segmentation model class.
        :param num_classes: number of model output classes.
        :param encoders: segmentation model encoder name.
        :param target_size: the size of the input images expected by the model.
        :param model_paths: path to pathology model.
        :param device: the device on which the models working in the pipeline will be launched.
        :param study_type: type of processed study (flg, prl).
        :param model_architectures: neural network model architecture.
        :param use_projection: list of parameters indicating which nn weights to load.
        :param calibration_paths: paths for models output calibration params.
        :param run_fp_16: param indicating whether to run models in fp-16 mode or not.
        :param apply_calibration: param indicating whether to apply calibration on model output or not.
        """
        super().__init__()
        self.target_size = target_size
        self.device = device
        self.side_size = target_size[0]
        self.pathologies_names = list()
        self.required_pathologies = list()
        self.study_type = study_type
        if calibration_paths:
            self.calibrators = list()
            for calibration_path in calibration_paths:
                self.calibrators.append(ModelCalibrator(calibration_path=calibration_path))
        else:
            self.calibrator_params = None
        self.nn_models = list()
        if 'frontal' == use_projection:
            for i, path in enumerate(model_paths):
                nn_model = MODELS[model_architectures[i]](
                    num_classes=num_classes, encoder=encoders[i], target_size=target_size
                )
                nn_model.load_weights(torch.load(path, map_location=device)['state_dict'])
                nn_model = nn_model.to(device)
                nn_model.eval()
                self.nn_models.append(nn_model)
        else:
            for i, path in enumerate(model_paths):
                nn_model = MODELS[model_architectures[i]](
                    num_classes=num_classes, encoder=encoders[i], target_size=target_size
                )
                nn_model.load_weights(torch.load(path, map_location=device)['state_dict'])
                nn_model = nn_model.to(device)
                nn_model.eval()
                self.nn_models.append(nn_model)
        self.run_fp_16 = run_fp_16
        self.apply_calibration = apply_calibration
        if run_fp_16:
             self.__fp16_convert()

    @torch.no_grad()
    def predictions(self, image_data: LungsDXResult) -> LungsDXResult:
        """
        Method for making predictions for image.
        :param image_data: LungsDXResult class object containing image data.
        :returns updated image_data object
        """
        h, w = image_data.pathology_image.shape[:-1]
        img, (h_pad_top, h_pad_bottom, w_pad_left, w_pad_right) = self.preprocess_image(
            image=image_data.pathology_image.copy(), return_pad=True
        )
        img = img.to(self.device)
        if self.run_fp_16:
            img = img.half()

        predicted_masks = list()
        for nn_model in self.nn_models:
            _, masks = nn_model(img)
            masks = masks.squeeze(dim=0).cpu().numpy()
            masks = masks[:, h_pad_top:self.side_size - h_pad_bottom, w_pad_left:self.side_size - w_pad_right]
            predicted_masks.append(masks)

        if self.calibrator_params:
            for masks in predicted_masks:
                for i, pathology in enumerate(self.pathologies_names):
                    masks[i, :, :] = self.calibrators[i].predict(y_prob=masks[i, :, :], pathology_name=pathology)

        predicted_masks = (sum(predicted_masks) / len(self.nn_models)).astype('float32')

        for pathology, mask in zip(self.pathologies_names, predicted_masks):
            if pathology in self.required_pathologies:
                mask = cv.resize(mask, (w, h), interpolation=cv.INTER_CUBIC)
                image_data = self.pathology_conclusion(image_data=image_data, pathology=pathology, mask=mask)
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
        image = self.longest_maxsize(image, max_size=self.target_size[0], interpolation=cv.INTER_AREA)
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

    def pathology_conclusion(
            self, image_data: LungsDXResult, pathology: str, mask: np.array
    ) -> LungsDXResult:
        """
        Method for making predictions for image.
        :param image_data: LungsDXResult class object containing image data.
        :param pathology: pathology name.
        :param mask: pathology segmentation mask.
        :returns updated image_data object
        """
        threshold = FLG_PATHOLOGY_THRESHOLD if self.study_type == 'flg' else RG_PATHOLOGY_THRESHOLD
        mask, prediction = self.__process_model_output(mask=mask, threshold=threshold[pathology], pathology=pathology)
        mask = cv.flip(mask, 1) if image_data.flip else mask
        if DIAPHM == pathology and 7 in image_data.warnings:
            prediction = None
        image_data.pathologies_result.update(
            pathology_name=pathology, confidence=prediction, mask=mask, threshold=threshold[pathology][1]
        )
        return image_data

    def __process_model_output(
            self, mask: np.ndarray, threshold: List[int], pathology: str
    ) -> Tuple[np.ndarray, np.array]:
        """
        Method for pathology model output calibration.
        :param mask: pathology mask array.
        :param threshold: list of thresholds value for current pathology.
        :param pathology: pathology name.
        :return processed mask and prediction output.
        """
        binary_mask = (mask > threshold[0])
        binary_mask = np.clip(binary_mask * 255, 0, 255).astype(np.uint8)
        polygons, _ = cv.findContours(binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        final_polygons = list()
        for polygon in polygons:
            object_mask = cv.fillPoly(np.zeros(mask.shape), [polygon], color=1)
            if (object_mask * mask).max() > threshold[1]:
                final_polygons.append(polygon)

        final_mask = cv.fillPoly(np.zeros(mask.shape), final_polygons, color=1)
        prediction = (final_mask * mask).max() if final_mask.any() else mask.max()
        if self.apply_calibration:
            prediction = self.__calibrate_prediction_by_threshold(prediction=prediction, threshold=threshold)
        return final_mask, prediction

    def __fp16_convert(self) -> None:
        """
        Method for converting models from float 32 to float 16.
        """
        fp16_models = list()
        for nn_model in self.nn_models:
            nn_model.half()
            for layer in nn_model.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.float()
            fp16_models.append(nn_model)
        self.nn_models = fp16_models

    @staticmethod
    def __calibrate_prediction_by_threshold(prediction: np.array, threshold: List[float]) -> np.array:
        """
        Method for model prediction output calibration.
        :param prediction: model confidence prediction.
        :param threshold: list of thresholds corresponding to the model.
        :return calibrated prediction.
        """
        if prediction > threshold[1]:
            prediction = 1 - (1 - prediction) / (1 - threshold[1]) * 0.5
        else:
            prediction = prediction / threshold[1] * 0.5
        return prediction


class Hemithorax(PathologyModel):

    def __init__(
            self, use_projection: str, device: torch.device, study_type: str, apply_calibration: bool, run_fp_16: bool
    ) -> None:
        """
        Hemithorax pathology class.
        :param use_projection: list of parameters indicating which nn weights to load.
        """
        super(Hemithorax, self).__init__(
            num_classes=HEMITHORAX_NUM_CLASSES, encoders=HEMITHORAX_ENCODER,
            target_size=HEMITHORAX_TARGET_SIZE, use_projection=use_projection,
            model_paths=PATH_TO_HEMITHORAX_MODEL, device=device, study_type=study_type,
            model_architectures=HEMITHORAX_NN,
            apply_calibration=apply_calibration, run_fp_16=run_fp_16
        )
        self.pathologies_names = PROJECTION_PATHOLOGIES[use_projection]['hemithorax']
        self.required_pathologies = REQUIRED_PROJECTION_PATHOLOGIES[use_projection]['hemithorax'].copy()


class ChestBones(PathologyModel):

    def __init__(
            self, use_projection: str, device: torch.device, study_type: str, apply_calibration: bool, run_fp_16: bool
    ) -> None:
        """
        Hemithorax pathology class.
        :param use_projection: list of parameters indicating which nn weights to load.
        """
        super(ChestBones, self).__init__(
            num_classes=BONES_NUM_CLASSES, encoders=BONES_ENCODER,
            target_size=BONES_TARGET_SIZE, use_projection=use_projection,
            model_paths=PATH_TO_BONES_MODEL, device=device, study_type=study_type,
            model_architectures=BONES_NN,
            apply_calibration=apply_calibration, run_fp_16=run_fp_16
        )
        self.pathologies_names = PROJECTION_PATHOLOGIES[use_projection]['bones']
        self.required_pathologies =  REQUIRED_PROJECTION_PATHOLOGIES[use_projection]['bones'].copy()


class Opacity(PathologyModel):
    def __init__(
            self, use_projection: str, device: torch.device, study_type: str, apply_calibration: bool, run_fp_16: bool
    ) -> None:
        """
        Opacity pathology class.
        :param use_projection: list of parameters indicating which nn weights to load.
        """
        super(Opacity, self).__init__(
            num_classes=OPACITY_NUM_CLASSES, encoders=OPACITY_ENCODER,
            target_size=OPACITY_TARGET_SIZE, use_projection=use_projection,
            model_paths=PATH_TO_OPACITY_MODEL, device=device, study_type=study_type,
            model_architectures=OPACITY_NN,
            apply_calibration=apply_calibration, run_fp_16=run_fp_16
        )
        self.pathologies_names = PROJECTION_PATHOLOGIES[use_projection]['opacity'].copy()
        if CAVITY in self.pathologies_names:
            self.pathologies_names.remove(CAVITY)
        if NDL in self.pathologies_names:
            self.pathologies_names.remove(NDL)
        self.required_pathologies =  REQUIRED_PROJECTION_PATHOLOGIES[use_projection]['opacity'].copy()
        if NDL in self.required_pathologies:
            self.required_pathologies.remove(NDL)
        if NDL in self.required_pathologies:
            self.required_pathologies.remove(NDL)


class Cavity(PathologyModel):
    def __init__(
            self, use_projection: str, device: torch.device, study_type: str, apply_calibration: bool, run_fp_16: bool
    ) -> None:
        """
        Opacity pathology class.
        :param use_projection: list of parameters indicating which nn weights to load.
        """
        super(Cavity, self).__init__(
            num_classes=CAVITY_NUM_CLASSES, encoders=CAVITY_ENCODER,
            target_size=CAVITY_TARGET_SIZE, use_projection=use_projection,
            model_paths=PATH_TO_CAVITY_MODEL, device=device,
            study_type=study_type, model_architectures=CAVITY_NN,
            apply_calibration=apply_calibration, run_fp_16=run_fp_16
        )
        self.pathologies_names = [CAVITY]
        self.required_pathologies = [CAVITY]


class Ndl(PathologyModel):
    def __init__(
            self, use_projection: str, device: torch.device, study_type: str, apply_calibration: bool, run_fp_16: bool
    ) -> None:
        """
        Opacity pathology class.
        :param use_projection: list of parameters indicating which nn weights to load.
        """
        super(Ndl, self).__init__(
            num_classes=NDL_NUM_CLASSES, encoders=NDL_ENCODER,
            target_size=NDL_TARGET_SIZE, use_projection=use_projection,
            model_paths=PATH_TO_NDL_MODEL, device=device,
            study_type=study_type, model_architectures=NDL_NN,
            apply_calibration=apply_calibration, run_fp_16=run_fp_16
        )
        self.pathologies_names = [NDL]
        self.required_pathologies = [NDL]

