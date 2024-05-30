from typing import List, Dict
import numpy as np
from pydicom import Dataset
import cv2 as cv
import os
import torch
import datetime as dt
import matplotlib.pyplot as plt

from Pipeline.models.triage import BinaryTriage
from Pipeline.results import LungsDXResult, StudyResult
from Pipeline.models.validator import LungsValidator
from Pipeline.models.segmenter import LungsSegmentation
from Pipeline.models.alignment import LungsAlignment
from Pipeline.models.study_analyser import StudyType
from Pipeline.models.pathology import Hemithorax, Opacity, Cavity, ChestBones, Ndl
from Pipeline.models.heart_pathology import HeartPathology
from Pipeline.dicom2image import ImageFromDicom
from Pipeline.filtration.LocalFiltration import MaskFiltration
from Pipeline.filtration.GlobalFiltration import PathologyFiltration
from Pipeline.localization import PathologyLocalization
from Pipeline.configs import *


class PipelineLungsDX:
    """
    Lungs DX pipline class.
    """
    def __init__(
            self,
            device_id: int,
            run_fp_16: bool = True,
            apply_calibration: bool = True,
            study_type: str = '',
            version: str = '1.11.0'
    ) -> None:
        """
        :param device_id:
        :param run_fp_16:
        :param apply_calibration:
        :param study_type:
        :param version:
        """
        self.image = None
        if study_type:
            assert study_type in ['prl', 'flg']
        self.study_type = study_type
        self.__set_device(device_id=device_id)
        self.images_data: List[LungsDXResult] = list()
        self.use_projection = USE_PROJECTION
        self.process_img_idx = list()
        self.study_result = StudyResult()
        self.version = version
        if device_id is None:
            self.run_fp_16 = False
        else:
            self.run_fp_16 = run_fp_16
        self.apply_calibration = apply_calibration
        self.__set_models()

    def run(self, images: List[Dataset], date_time: dt.datetime = None) -> None:
        """
        Method for making predictions for one study.
        :param images: list containing dicoms corresponding to study.
        :param date_time: data of study processing.
        """
        self._reset_data()
        self.__image_from_dicom(dicom_images=images)
        self._validator()
        self._process_validator_results()
        self._most_valid()

        if self.process_img_idx:
            for idx in self.process_img_idx:
                image_data = self.images_data[idx]
                image_data = self.segmenter.predictions(image_data=image_data)
                if REQUIRED_PREPROCESSING_MODELS['alignmenter']:
                    image_data = self.alignmenter.predictions(orig_image_data=image_data)
                if REQUIRED_PREPROCESSING_MODELS['study_type']:
                    image_data = self.study.predictions(image_data=image_data)
                for pathology_model in self.pathologies_models[image_data.validation['stage_4']].values():
                    image_data = pathology_model.predictions(image_data=image_data)
                image_data.processed = True
                image_data.update_conclusion()
                self.images_data[idx] = image_data

        self._study_conclusion(datetime=date_time)

    def _process_validator_results(self):
        """
        Method for updating warning flags due to validation results.
        """
        for i, image_data in enumerate(self.images_data):
            if image_data.validation['stage_2'] == 'non valid':
                image_data.warnings.append(2)
            else:
                if image_data.validation['stage_3'] == 'bad crop':
                    image_data.warnings.append(3)
                if image_data.validation['stage_4'] not in self.use_projection:
                    image_data.warnings.append(4)
                if image_data.validation['stage_3'] == 'cropped':
                    image_data.warnings.append(5)

    def _most_valid(self) -> None:
        """
        Method for selecting the most reliable image
        :return: None if there is no valid image in study, else index of the most valid image in study.
        """
        frontal_idx = [
            i for i, image_data in enumerate(self.images_data) if image_data.validation['stage_4'] == 'frontal'
        ]
        valid_frontal_images_idx = [
            i for i in frontal_idx if all(flag not in self.images_data[i].warnings for flag in [1, 2, 3, 4])
        ]
        if not valid_frontal_images_idx:
            return
        else:
            most_valid_frontal_idx = [i for i in valid_frontal_images_idx if 5 not in self.images_data[i].warnings]
            if most_valid_frontal_idx:
                self.process_img_idx.append(most_valid_frontal_idx[0])
            else:
                self.process_img_idx.append(valid_frontal_images_idx[0])
            if 'lateral' in USE_PROJECTION:
                lateral_idx = [
                    i for i, image_data in enumerate(self.images_data) if image_data.validation['stage_4'] == 'lateral'
                ]
                valid_lateral_images_idx = [
                    i for i in lateral_idx if all(flag not in self.images_data[i].warnings for flag in [1, 2, 4])
                ]
                if valid_lateral_images_idx:
                    most_valid_lateral_idx = [
                        i for i in valid_lateral_images_idx if 5 not in self.images_data[i].warnings
                    ]
                    if most_valid_lateral_idx:
                        self.process_img_idx.append(most_valid_lateral_idx[0])
                    else:
                        most_valid_lateral_idx = [
                            i for i in valid_lateral_images_idx if 3 not in self.images_data[i].warnings
                        ]
                        if most_valid_lateral_idx:
                            self.process_img_idx.append(most_valid_lateral_idx[0])
                        else:
                            self.process_img_idx.append(valid_lateral_images_idx[0])
                else:
                    valid_lateral_images_idx = list()
            else:
                valid_lateral_images_idx = list()

            for i in valid_frontal_images_idx + valid_lateral_images_idx:
                if i not in self.process_img_idx:
                    self.images_data[i].warnings.append(8)

    def _validator(self):
        """
        Method for images validation.
        """
        for i, image_data in enumerate(self.images_data):
            if 1 not in image_data.warnings:
                image_data = self.validator.predictions(image_data=image_data)
                if 'non inverted' not in image_data.validation['stage_1']:
                    image_data.triage_image = image_data.triage_image.max() - image_data.triage_image
                    image_data.preprocess_image = image_data.preprocess_image.max() - image_data.preprocess_image
                    image_data.pathology_image = image_data.pathology_image.max() - image_data.pathology_image
                if 'rotation degree: 90' in image_data.validation['stage_1']:
                    image_data.triage_image = cv.rotate(image_data.triage_image, cv.ROTATE_90_COUNTERCLOCKWISE)
                    image_data.preprocess_image = cv.rotate(
                        image_data.preprocess_image, cv.ROTATE_90_COUNTERCLOCKWISE
                    )
                    image_data.result_image = cv.rotate(
                        image_data.result_image, cv.ROTATE_90_COUNTERCLOCKWISE
                    )
                    image_data.pathology_image = cv.rotate(
                        image_data.pathology_image, cv.ROTATE_90_COUNTERCLOCKWISE
                    )
                if 'rotation degree: 180' in image_data.validation['stage_1']:
                    image_data.triage_image = cv.rotate(image_data.triage_image, cv.ROTATE_180)
                    image_data.preprocess_image = cv.rotate(image_data.preprocess_image, cv.ROTATE_180)
                    image_data.result_image = cv.rotate(image_data.result_image, cv.ROTATE_180)
                    image_data.pathology_image = cv.rotate(image_data.pathology_image, cv.ROTATE_180)
                if 'rotation degree: 270' in image_data.validation['stage_1']:
                    image_data.triage_image = cv.rotate(image_data.triage_image, cv.ROTATE_90_CLOCKWISE)
                    image_data.preprocess_image = cv.rotate(image_data.preprocess_image, cv.ROTATE_90_CLOCKWISE)
                    image_data.result_image = cv.rotate(image_data.result_image, cv.ROTATE_90_CLOCKWISE)
                    image_data.pathology_image = cv.rotate(image_data.pathology_image, cv.ROTATE_90_CLOCKWISE)
                if image_data.validation['stage_2'] == 'valid':
                    if 'lateral left' == image_data.validation['stage_4']:
                        image_data.triage_image = cv.flip(image_data.triage_image, 1)
                        image_data.pathology_image = cv.flip(image_data.preprocess_image, 1)
                        image_data.validation['stage_4'] = 'lateral'
                        image_data.flip = True

            self.images_data[i] = image_data

    def _reset_data(self) -> None:
        """
        Method for study data resetting.
        """
        self.images_data = list()
        self.study_result = StudyResult()
        self.process_img_idx = list()

    def __set_device(self, device_id: int):
        """
        Method for executable device setting.
        """
        if device_id is None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(f'cuda:{device_id}')

    def __set_models(self):
        """
        Method for loading required weights.
        """
        self.validator = LungsValidator(device=self.device)
        self.segmenter = LungsSegmentation(use_projection=self.use_projection, device=self.device)
        if REQUIRED_PREPROCESSING_MODELS['alignmenter']:
            self.alignmenter = LungsAlignment(use_projection=self.use_projection, device=self.device)
        if REQUIRED_PREPROCESSING_MODELS['study_type']:
            self.study = StudyType(device=self.device)

        self.pathologies_models = dict()
        for projection in USE_PROJECTION:
            self.pathologies_models[projection] = self.__set_pathologies_models_for_projection(
                required_pathologies=REQUIRED_PROJECTION_PATHOLOGIES[projection], use_projection=projection
            )

    def __set_pathologies_models_for_projection(self, required_pathologies: dict, use_projection: str) -> Dict:
        pathologies = dict()
        if required_pathologies['Other']:
            pathologies['Other'] = BinaryTriage(use_projection=use_projection, device=self.device)
        if required_pathologies['hemithorax']:
            pathologies['hemithorax'] = Hemithorax(
                use_projection=use_projection, device=self.device, study_type=self.study_type,
                apply_calibration=self.apply_calibration, run_fp_16=self.run_fp_16
            )
        if required_pathologies['bones']:
            pathologies['bones'] = ChestBones(
                use_projection=use_projection, device=self.device, study_type=self.study_type,
                apply_calibration=self.apply_calibration, run_fp_16=self.run_fp_16
            )
        if required_pathologies['opacity']:
            pathologies['opacity'] = Opacity(
                use_projection=use_projection, device=self.device, study_type=self.study_type,
                apply_calibration=self.apply_calibration, run_fp_16=self.run_fp_16
            )
        if CAVITY in required_pathologies['opacity']:
            pathologies[CAVITY] = Cavity(
                use_projection=use_projection, device=self.device, study_type=self.study_type,
                apply_calibration=self.apply_calibration, run_fp_16=self.run_fp_16
            )
        if NDL in required_pathologies['opacity']:
            pathologies[NDL] = Ndl(
                use_projection=use_projection, device=self.device, study_type=self.study_type,
                apply_calibration=self.apply_calibration, run_fp_16=self.run_fp_16
            )
        if required_pathologies['heart']:
            pathologies['heart'] = HeartPathology(use_projection=use_projection)
        return pathologies

    def _study_conclusion(self, datetime: dt.datetime) -> None:
        """
        Method for making study conclusion.
        :param datetime: data of study processing.
        """
        if self.process_img_idx:
            self.images_data = [
                self.__image_conclusion(image_data=image_data) if image_data.processed else image_data
                for image_data in self.images_data
            ]
            if len(USE_PROJECTION) > 1 and len(self.process_img_idx) > 1:
                self.__projections_data_sync()

            self.study_result.update_kafka_dictionary(
                dictionary=self.__fill_kafka_dictionary(
                    image_data=self.images_data[self.process_img_idx[0]], projection='frontal'
                )
            )
            if 'lateral' in USE_PROJECTION and len(self.process_img_idx) > 1:
                self.study_result.update_kafka_dictionary(
                    dictionary=self.__fill_kafka_dictionary(
                        image_data=self.images_data[self.process_img_idx[1]], projection='lateral'
                    )
                )

        for i, image_data in enumerate(self.images_data):
            self.study_result.dicoms[i] = image_data.dicom
            self.study_result.projections[i] = image_data.validation['stage_4']
            self.study_result.warnings[i] = image_data.warnings

    def __image_from_dicom(self, dicom_images: List[Dataset]) -> None:
        """
        Method for making predictions for one study.
        :param dicom_images: list containing dicoms corresponding to study.
        """
        for dicom_image in dicom_images:
            try:
                image = dicom_image.pixel_array
                if len(image.shape) == 3:
                    image = image[:, :, 0]
                image_data = LungsDXResult(
                    triage_image=image, dicom=dicom_image, apply_calibration=self.apply_calibration
                )
                image_data.result_image, image_data.pathology_image = ImageFromDicom(dicom=dicom_image).get_image()
                image_data.preprocess_image = image_data.result_image.copy()
            except:
                image_data = LungsDXResult(
                    triage_image=np.zeros(0), dicom=dicom_image, apply_calibration=self.apply_calibration
                )
                image_data.warnings.append(1)

            self.images_data.append(image_data)

    def __projections_data_sync(self):
        """
        Method for pathology data sync between frontal and lateral projections.
        """
        frontal_image_data = self.images_data[self.process_img_idx[0]]
        lateral_images_data = self.images_data[self.process_img_idx[1]]
        for pathology_name in [pathology for group in REQUIRED_FRONTAL_PATHOLOGIES.values() for pathology in group]:
            if pathology_name in [pathology for group in REQUIRED_LATERAL_PATHOLOGIES.values() for pathology in group]:
                if lateral_images_data.get_pathology(name=pathology_name).is_pathological:
                    if not frontal_image_data.get_pathology(name=pathology_name).is_pathological:
                        lateral_images_data.update_pathology_result(
                            name=pathology_name, confidence=0, mask=np.zeros_like(lateral_images_data.result_image)
                        )
        lateral_images_data.update_conclusion()
        self.images_data[self.process_img_idx[1]] = lateral_images_data


    @staticmethod
    def __fill_kafka_dictionary(image_data: LungsDXResult, projection: str) -> dict:
        """
        Method for processed image kafka dictionary filling.
        :param image_data: LungsDXResult class instance, containing information about image, after pipeline processing.
        :param projection: type of image projection: frontal, lateral.
        :return dictionary, containing all necessary information for kafka.
        """
        metadata = dict()
        metadata["pathology_flag"] = image_data.is_pathological
        metadata["confidence_level"] = image_data.confidence
        for _, group_value in REQUIRED_PROJECTION_PATHOLOGIES[projection].items():
            for pathology_name in group_value:
                pathology = image_data.get_pathology(name=pathology_name)
                if pathology.not_processed:
                    if pathology_name == CTR:
                        metadata[pathology_name] = {
                            "confidence_level": None, "pathology_flag": None, 'cti': None
                        }
                    elif pathology_name == MESH:
                        metadata[pathology_name] = {
                            "confidence_level": None, "pathology_flag": None, 'shift_param': None,
                            'left_shift_param': None, 'right_shift_param': None
                        }
                    else:
                        metadata[pathology_name] = {
                            "confidence_level": None, "pathology_flag": None, 'lateral_flag': None, 'localisation': None
                        }
                else:
                    if pathology_name == CTR:
                        metadata[pathology_name] = {
                            "confidence_level": pathology.confidence,
                            "pathology_flag": pathology.is_pathological,
                            'cti': pathology.cti
                        }
                    elif pathology_name == MESH:
                        metadata[pathology_name] = {
                            "confidence_level": pathology.confidence,
                            "pathology_flag":  pathology.is_pathological,
                            'shift_param': pathology.shift_param,
                            'left_shift_param': pathology.left_shift_param,
                            'right_shift_param': pathology.right_shift_param
                        }
                    else:
                        metadata[pathology_name] = {
                            "confidence_level": pathology.confidence,
                            "pathology_flag": pathology.is_pathological,
                            'lateral_flag': None,
                            'localisation': pathology.localization
                        }
        return metadata

    @staticmethod
    def __image_conclusion(image_data: LungsDXResult) -> LungsDXResult:
        """
        Method for making final study conclusion per image after pathologies mask filtration.
        :param image_data: LungsDXResult class object containing all the information about the image obtained when
         running through the pipeline.
        :return image_data object
        """
        required = REQUIRED_PROJECTION_PATHOLOGIES[image_data.validation['stage_4']]
        if image_data.is_pathological or (CTR in required['heart'] and not image_data.cardiomegaly.not_processed):
            if image_data.lungs_segmentation_mask.any():
                image_data = MaskFiltration(image_data=image_data).make_filtration()
                image_data.update_conclusion()
            if image_data.is_pathological:
                image_data = PathologyFiltration(image_data=image_data).make_filtration()
                image_data.update_conclusion()
                if not 7 in image_data.warnings and image_data.validation['stage_4'] == 'frontal':
                    image_data = PathologyLocalization(image_data=image_data).get_localisation()
        return image_data
