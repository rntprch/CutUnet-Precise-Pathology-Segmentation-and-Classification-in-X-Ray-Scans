from typing import Tuple, Any, List, Dict
import cv2 as cv
from pydicom import FileDataset, Dataset
import matplotlib.pyplot as plt
import numpy as np
from Pipeline.configs import *


class StudyResult:
    def __init__(self):
        """
        Class for combining and saving the results of study processing.
        """
        self.processed_images = dict()
        self.projections = dict()
        self.dicoms = dict()
        self.warnings = dict()
        self.kafka_dictionary: Dict[str: float] = None

    def get_all_warnings(self) -> set:
        """
        Method for grouping all study warnings.
        :return set of warnings corresponding to study.
        """
        study_warnings = list()
        for _, image_warnings in self.warnings.items():
            study_warnings += image_warnings
        return set(study_warnings)

    def get_processed_data(self) ->List[Tuple[FileDataset, np.array, str]] or None:
        """
        Method for getting SC image of processed data.
        :return processed data dicom and SC image.
        """
        if self.processed_images:
            best_data = list()
            for key, processed_image in self.processed_images.items():
                best_data.append((self.dicoms[key], processed_image, self.projections[key]))
            return best_data

    def update_kafka_dictionary(self, dictionary: dict) -> None:
        """
        Method for study result pathology conclusion updating.
        :param dictionary: kafka format dictionary.
        """
        if self.kafka_dictionary is not None:
            for pathology, result in dictionary.items():
                if pathology not in ['confidence_level', 'pathology_flag'] and pathology in self.kafka_dictionary.keys():
                    self.kafka_dictionary[pathology]['lateral_flag'] = result['pathology_flag']
        else:
            self.kafka_dictionary = dictionary


class PathologyResult:

    def __init__(self, pathology_name: str, pathology_confidence_threshold: Any, apply_calibration: bool) -> None:
        """
        Pathology result container class.
        :param pathology_name: pathology name
        """
        self.confidence = 0.
        self.is_pathological = False
        self.mask = np.zeros(0)
        self.blobs = np.zeros(0)
        self.name = pathology_name
        self.open_kernel, self.close_kernel = LOCAL_FILTRATION_PARAMS[pathology_name]
        if self.name not in [CTR, MESH]:
            self.pathology_confidence_threshold = 0.5 if apply_calibration else pathology_confidence_threshold
        else:
            self.pathology_confidence_threshold = pathology_confidence_threshold
        self.min_size = 0
        self.not_processed = True
        self.error = None
        self.localization = None

    def __lt__(self, other):
        self.mask[self.mask > 1] = 1
        other.mask[other.mask > 1] = 1
        param1 = np.sum(self.mask) / (self.mask.shape[0] * self.mask.shape[1])
        param2 = np.sum(other.mask) / (other.mask.shape[0] * other.mask.shape[1])
        return param1 > param2

    def update(self, confidence: float = None, mask: np.array = None, localization: str = None) -> None:
        """
        Method for updating result.
        :param confidence: pathology confidence.
        :param mask: segmentation pathology mask.
        :param localization: pathology localization tag.
        """
        if localization is not None:
            self.localization = localization
        if confidence is not None:
            self.confidence = confidence
            self.not_processed = False
        if mask is not None:
            self.mask = mask.astype('uint8')
            self.min_size = int(self.mask.shape[0] * self.mask.shape[1] * SEGMENTATION_SIZE_THRESHOLD)
        if self.confidence > self.pathology_confidence_threshold and self.mask.any():
            self.is_pathological = True
        else:
            self.confidence *= self.pathology_confidence_threshold
            self.is_pathological = False

    def lungs_filter(self, lungs_mask: np.array = None) -> None:
        """
        Method for pathology mask filtration.
        :param lungs_mask: lungs segmentation mask.
        """
        if lungs_mask is not None:
            self.mask *= lungs_mask.astype('uint8')
        self.__morphology_filter()
        self.__contour_smoothing()

    def create_blobs(self) -> None:
        """
        Method for pathology blobs separation.
        """
        blobs = list()
        contours, _ = cv.findContours(image=self.mask.astype('uint8'), mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            blobs.append(cv.fillPoly(np.zeros_like(self.mask), pts=[contour], color=1))
        self.blobs = np.stack(blobs, axis=-1)

    def __morphology_filter(self) -> None:
        """
        Method for pathology mask filtration.
        """
        side_size = max(self.mask.shape[:2])

        if self.open_kernel:
            open_kernel_size = int(self.open_kernel * side_size)
            open_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
            self.mask = cv.morphologyEx(self.mask, cv.MORPH_OPEN, open_kernel)

        if self.close_kernel:
            close_kernel_size = int(self.close_kernel * side_size)
            close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
            self.mask = cv.morphologyEx(self.mask, cv.MORPH_CLOSE, close_kernel)

    def __contour_smoothing(self):
        """
        Method for pathology blob smoothing.
        """
        if PATHOLOGY_SMOOTHING_PARAMS[self.name]:
            kern_size = int(max(self.mask.shape[:2]) * PATHOLOGY_SMOOTHING_PARAMS[self.name])
            kern_size = kern_size if kern_size % 2 == 1 else kern_size + 1
            self.mask = cv.GaussianBlur(
                src=self.mask, ksize=(kern_size, kern_size), sigmaX=kern_size, sigmaY=kern_size,
            )


class CardiomegalyResult(PathologyResult):

    def __init__(self, pathology_confidence_threshold: dict, apply_calibration: bool):
        """
        Cardiomegaly pathology result saving class.
        """
        super().__init__(
            pathology_name=CTR, pathology_confidence_threshold=pathology_confidence_threshold,
            apply_calibration=apply_calibration
        )
        self.line_points = list()
        self.cti = 0.
        self.visualize = False
        self.is_pathological = False

    def update(self, cti: float = None, mask: np.array = None) -> None:
        """
        Method for updating result.
        :param cti: pathology confidence.
        :param mask: segmentation pathology mask.
        """
        if cti is not None:
            self.cti = round(cti, 2)
            self.not_processed = False
        if mask is not None:
            self.mask = mask
            self.min_size = int(self.mask.shape[0] * self.mask.shape[1] * SEGMENTATION_SIZE_THRESHOLD)

        self.confidence = 0
        for threshold, score in self.pathology_confidence_threshold.items():
            if self.cti > threshold and self.confidence < score:
                self.confidence = score

        self.is_pathological = self.confidence > 0.5
        self.visualize = self.cti > 0.5


class MediastinalShiftResult(PathologyResult):

    def __init__(self, pathology_confidence_threshold: dict, apply_calibration: bool):
        """
        Mediastinal shift pathology result saving class.
        """
        super().__init__(
            pathology_name=MESH, pathology_confidence_threshold=pathology_confidence_threshold,
            apply_calibration=apply_calibration
        )
        self.line_points = list()
        self.shift_param = 0.
        self.left_shift_param = 0.
        self.right_shift_param = 0.
        self.confidence = 0
        self.threshold = 0.5
        self.is_pathological = False

    def update(self, shift_params: List[float] = None, mask: np.array = None) -> None:
        """
        Method for updating result.
        :param shift_params: values of heart shift in left and right directions.
        :param mask: segmentation pathology mask.
        """
        if shift_params is not None:
            self.left_shift_param = round(shift_params[0], 2)
            self.right_shift_param = round(shift_params[1], 2)
            self.shift_param = round(shift_params[2], 2)
            self.not_processed = False
        if mask is not None:
            self.mask = mask
            self.min_size = int(self.mask.shape[0] * self.mask.shape[1] * SEGMENTATION_SIZE_THRESHOLD)

        # if self.right_shift_param > MEDIASTINAL_LEFT_SHIFT_COEFFICIENT * self.left_shift_param:
        #     self.shift_param = self.right_shift_param - self.left_shift_param
        # else:
        #     self.shift_param = MEDIASTINAL_LEFT_SHIFT_COEFFICIENT * self.left_shift_param

        # self.shift_param = round(self.left_shift_param / self.right_shift_param, 2)

        for score, thresholds in self.pathology_confidence_threshold.items():
            if self.left_shift_param < thresholds[0] and self.confidence < score:
                self.confidence = score
            if self.right_shift_param < thresholds[1] and self.confidence < score:
                self.confidence = score

        self.is_pathological = self.confidence > self.threshold


class PathologiesResult:
    def __init__(self, apply_calibration: bool, pathologies_confidence_threshold: float = 0.5):
        """
        Pathologies results container class.
        :param pathologies_confidence_threshold: dict with confidence thresholds for current pathology group.
        """
        self.pathologies_confidence_threshold = pathologies_confidence_threshold
        self.pathologies: List[PathologyResult] = list()
        self.is_pathological = False
        self.confidence = 0.
        self.apply_calibration = apply_calibration
        self.localisation = None

    def update(
            self,
            pathology_name: str,
            confidence: float = None,
            mask: np.array = None,
            threshold: float = None,
            localization: str = None
    ) -> None:
        """
        Method for updating pathology result.
        :param pathology_name: pathology name.
        :param confidence: pathology confidence.
        :param mask: segmentation pathology mask.
        :param threshold: pathology threshold.
        :param localization: pathology localization tag.
        """
        if self.get_pathology(pathology_name=pathology_name):
            idx, pathology = self.get_pathology(pathology_name=pathology_name, drop_index=False)
            pathology.update(confidence=confidence, mask=mask, localization=localization)
            self.pathologies[idx] = pathology
        else:
            pathology = PathologyResult(
                pathology_name=pathology_name,
                pathology_confidence_threshold=threshold if threshold is not None else
                self.pathologies_confidence_threshold,
                apply_calibration=self.apply_calibration
            )
            pathology.update(confidence=confidence, mask=mask)
            self.pathologies.append(pathology)

        self.__is_pathological()
        self.__get_max_confidence()
        self.pathologies = sorted(self.pathologies)

    def filter_mask(self, pathology_name: str, lungs_mask: np.array) -> None:
        """
        Method for pathology mask filtration by lungs mask.
        :param pathology_name: name of pathology you want filter.
        :param lungs_mask: segmentation lungs mask.
        """
        idx, pathology = self.get_pathology(pathology_name=pathology_name, drop_index=False)
        pathology.lungs_filter(lungs_mask=lungs_mask)
        self.pathologies[idx] = pathology

    def get_result(self) -> List[PathologyResult]:
        """
        Method for getting all pathologies result.
        """
        return self.pathologies

    def get_pathology_item(self, pathology_name: str, item: str) -> Any:
        """
        Method for getting any pathology item.
        :parameter pathology_name: the name of the pathology for which you want to get some value of the pathology
        variable.
        :parameter item: the pathology variable whose value you want to retrieve.
        """
        return next(
            (getattr(pathology, item) for pathology in self.pathologies if pathology.name == pathology_name), None
        )

    def get_pathology(
            self, pathology_name: str, drop_index: bool = True
    ) -> PathologyResult or Tuple[int, PathologyResult]:
        """
        Method for getting any pathology.
        :parameter pathology_name: the name of the pathology for which you want to retrieve result.
        :parameter drop_index: parameter indicating whether to return list index of pathology result or not.
        """
        data = next(
            ((index, pathology) for (index, pathology)
             in enumerate(self.pathologies) if pathology.name == pathology_name),
            (None, None)
        )
        return data[1] if drop_index else data

    def __is_pathological(self):
        """
        Method for checking if any pathology from list of pathologies is pathological.
        """
        self.is_pathological = any(result.is_pathological for result in self.pathologies)

    def __get_max_confidence(self):
        """
        Method for getting maximum pathology confidence from list of pathologies.
        """
        self.confidence = max(
            pathology.confidence for pathology in self.pathologies if pathology.name not in [DIAPHM, CONSOLIDATED_FRACTURE]
        )


class LungsDXResult:
    def __init__(self, triage_image: np.array, dicom: Dataset, apply_calibration: bool) -> None:
        """
        Lungs chest X-ray diagnostic results container class.
        :param triage_image: numpy array containing original image.
        """
        self.dicom = dicom
        self.triage_image = triage_image
        self.pathology_image = np.zeros(0)
        self.preprocess_image = np.zeros(0)
        self.result_image = np.zeros(0)
        self.alignment_image = np.zeros(0)
        self.lungs_segmentation_mask = np.zeros(0)
        self.pleural_fluid_mask = np.zeros(0)
        self.roi_mask = np.zeros(0)
        self.validation = {'stage_1': '', 'stage_2': '', 'stage_3': '', 'stage_4': ''}
        self.study_type = ''
        self.cardiomegaly = CardiomegalyResult(
            pathology_confidence_threshold=CARDIOMEGALY_THRESHOLDS, apply_calibration=apply_calibration
        )
        self.mediastinal_shift = MediastinalShiftResult(
            pathology_confidence_threshold=MEDIASTINAL_THRESHOLD, apply_calibration=apply_calibration
        )
        self.triage_result = PathologyResult(
            pathology_name=TRIAGE, pathology_confidence_threshold=TRIAGE_THRESHOLD, apply_calibration=apply_calibration
        )
        self.pathologies_result = PathologiesResult(apply_calibration=apply_calibration)
        self.lungs_quality = 0.
        self.lungs_symmetry = 0.
        self.warnings: List[int] = []
        self.is_pathological = False
        self.confidence = 0
        self.pathology_blobs = list()
        self.processed: bool = False
        self.flip: bool = False

    def update_conclusion(self) -> None:
        """
        Method for global confidence calculation and pathologic detection
        """
        self.is_pathological = any([
            self.cardiomegaly.is_pathological,
            self.mediastinal_shift.is_pathological,
            self.triage_result.is_pathological,
            self.pathologies_result.is_pathological
        ])
        if self.is_pathological:
            self.confidence = round(max(
                self.triage_result.confidence,
                self.mediastinal_shift.confidence,
                self.cardiomegaly.confidence,
                self.pathologies_result.confidence
            ) * 100)
        else:
            self.confidence = round(self.triage_result.confidence * 100)

    def get_pathology(self, name: str):
        if name == CTR:
            return self.cardiomegaly
        elif name == MESH:
            return self.mediastinal_shift
        elif name == TRIAGE:
            return self.triage_result
        else:
            return self.pathologies_result.get_pathology(pathology_name=name)

    def update_pathology_result(
            self, name: str, confidence: float or List[float] = None, mask: np.array = None, threshold: float = None
    ) -> None:
        if name == CTR:
            self.cardiomegaly.update(cti=confidence, mask=mask)
        elif name == MESH:
            return self.mediastinal_shift.update(shift_params=confidence, mask=mask)
        elif name == TRIAGE:
            self.triage_result.update(confidence=confidence, mask=mask)
        else:
            self.pathologies_result.update(pathology_name=name, confidence=confidence, mask=mask, threshold=threshold)
