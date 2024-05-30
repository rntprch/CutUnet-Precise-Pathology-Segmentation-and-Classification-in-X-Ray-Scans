import numpy as np
from typing import List, Tuple
from more_itertools import locate
import cv2 as cv
import matplotlib.pyplot as plt
from Pipeline.results import LungsDXResult
from Pipeline.configs import *


class PathologyFiltration:
    def __init__(self, image_data: LungsDXResult) -> None:
        """
        Pathology blobs filtration class.
        :param image_data: LungsDXResult class object containing all the information about the image obtained when
         running through the pipeline.
        """
        self.image_data = image_data
        self.blobs = None
        self.pathologies_names = list()

    def make_filtration(self) -> LungsDXResult:
        """
        Method for pathologies filtration processing.
        """
        self.get_blobs()
        if self.blobs is not None:
            self.blobs[self.blobs > 1] = 1
            self.blobs = self.blobs.astype('float32')
            metrics = dict()
            metrics['iou'], metrics['ioa'], metrics['inv_ioa'] = self.get_metrics()
            for i, pathology_name in enumerate(self.pathologies_names):
                for j in range(self.blobs.shape[-1]):
                    key = f'{self.pathologies_names[i]}-{self.pathologies_names[j]}'
                    if key in PATHOLOGY_FILTRATION_PARAMS.keys():
                        if self.check_intersection(
                                filtration_params=PATHOLOGY_FILTRATION_PARAMS[key], metrics=metrics, i=i, j=j
                        ):
                            self.blobs[:, :, i] = np.zeros_like(self.blobs[:, :, i])
                            break
            self.__check_if_only_non_pathologic_pathologies()
            # self.__blob_border_smoothing()
            self.__update_image_data()
        return self.image_data

    def __update_image_data(self) -> None:
        """
        Method for image data_structure updating.
        """
        pathologies_indexes = {}
        for name in self.pathologies_names:
            if name not in pathologies_indexes.keys():
                pathologies_indexes[name] = list(locate(self.pathologies_names, lambda a: a == name))
        for name, indexes in pathologies_indexes.items():
            if name == CTR:
                pass
            if name == TRIAGE:
                self.image_data.triage_result.update(mask=self.__blobs_to_mask(indexes=indexes))
            else:
                self.image_data.pathologies_result.update(
                    pathology_name=name, mask=self.__blobs_to_mask(indexes=indexes)
                )

    def __check_if_only_non_pathologic_pathologies(self) -> None:
        """
        Method for diaphragm pathology filtration if nothing else was detected.
        """
        non_empty_pathologies_idx = np.nonzero(self.blobs.sum(axis=(0, 1)))
        if len(non_empty_pathologies_idx[0]) > 0:
            if all(p in NON_PATHOLOGICAL_PATHOLOGIES or p == CTR for p in self.pathologies_names):
                if not self.image_data.cardiomegaly.is_pathological:
                    self.blobs = np.zeros_like(self.blobs)

    def __blob_border_smoothing(self):
        """
        Method for pathology blob smoothing.
        """
        for i in range(self.blobs.shape[-1]):
            if self.pathologies_names[i] in PATHOLOGY_SMOOTHING_PARAMS:
                kern_size = int(max(self.blobs[:, :, i].shape) * PATHOLOGY_SMOOTHING_PARAMS[self.pathologies_names[i]])
                kern_size = kern_size if kern_size % 2 == 1 else kern_size + 1
                self.blobs[:, :, i] = cv.GaussianBlur(
                    src=self.blobs[:, :, i], ksize=(kern_size, kern_size), sigmaX=kern_size, sigmaY=kern_size,
                )

    def __blobs_to_mask(self, indexes: List[int]) -> np.array:
        """
        Method for pathologies blobs converting to mask.
        """
        mask = np.sum(np.take(self.blobs, np.array(indexes), axis=-1), axis=-1)
        return mask

    def get_blobs(self):
        """
        Method for pathologies blobs array creation.
        """
        if CTR in REQUIRED_PROJECTION_PATHOLOGIES[self.image_data.validation['stage_4']]['heart']:
            self.__get_cardiomegaly_blob()
        if MESH in REQUIRED_PROJECTION_PATHOLOGIES[self.image_data.validation['stage_4']]['heart']:
            self.__get_mediastinal_shift_blob()
        self.__get_triage_blobs()
        self.__get_pathologies_blobs()

    def __get_pathologies_blobs(self):
        """
        Method for pathologies blobs array creation.
        """
        for result in self.image_data.pathologies_result.get_result():
            if result.mask.any():
                result.create_blobs()
                self.pathologies_names += [result.name] * result.blobs.shape[-1]
                self.blobs = result.blobs if self.blobs is None else np.concatenate((self.blobs, result.blobs), axis=-1)

    def __get_cardiomegaly_blob(self):
        """
        Method for cardiomegaly blob array creation.
        """
        if self.image_data.cardiomegaly.is_pathological or self.image_data.cardiomegaly.visualize:
            self.image_data.cardiomegaly.create_blobs()
            self.pathologies_names += [self.image_data.cardiomegaly.name] * self.image_data.cardiomegaly.blobs.shape[-1]
            if self.blobs is not None:
                self.blobs = np.concatenate((self.blobs, self.image_data.cardiomegaly.blobs), axis=-1)
            else:
                self.blobs = self.image_data.cardiomegaly.blobs

    def __get_mediastinal_shift_blob(self):
        """
        Method for cardiomegaly blob array creation.
        """
        if self.image_data.mediastinal_shift.is_pathological:
            self.image_data.mediastinal_shift.create_blobs()
            self.pathologies_names += \
                [self.image_data.mediastinal_shift.name] * self.image_data.mediastinal_shift.blobs.shape[-1]
            if self.blobs is not None:
                self.blobs = np.concatenate((self.blobs, self.image_data.mediastinal_shift.blobs), axis=-1)
            else:
                self.blobs = self.image_data.mediastinal_shift.blobs

    def __get_triage_blobs(self):
        """
        Method for binary triage blobs array creation.
        """
        if self.image_data.triage_result.mask.any():
            self.image_data.triage_result.create_blobs()
            self.pathologies_names += \
                [self.image_data.triage_result.name] * self.image_data.triage_result.blobs.shape[-1]
            if self.blobs is not None:
                self.blobs = np.concatenate((self.blobs, self.image_data.triage_result.blobs), axis=-1)
            else:
                self.blobs = self.image_data.triage_result.blobs

    def get_metrics(self) -> Tuple[np.array, np.array, np.array]:
        """
        Method for IoU and IoA metrics calculation.
        :return numpy arrays containing calculated IoU and IoA metrics.
        """
        intersection = self.__intersection()
        area = self.__area()
        union = self.__union(area=area, intersection=intersection)

        iou = intersection / union
        ioa = intersection / area
        inv_ioa = intersection / area.T

        return iou, ioa, inv_ioa

    def __area(self) -> np.array:
        """
        Method for blobs area calculation.
        :return numpy array containing blobs area.
        """
        area = np.sum(self.blobs, axis=(0, 1))
        return np.tile(area, (len(area), 1)).T

    def __intersection(self) -> np.array:
        """
        Method for blobs intersection calculation.
        :return numpy array containing blobs intersection.
        """
        blobs = self.blobs.reshape((self.blobs.shape[0] * self.blobs.shape[1], self.blobs.shape[2]))
        intersection = np.matmul(np.transpose(blobs).astype(float), blobs.astype(float))
        return intersection

    @staticmethod
    def __union(area: np.array, intersection: np.array) -> np.array:
        """
        Method for blobs union calculation.
        :param area: numpy array containing blobs area.
        :param intersection: numpy array containing intersection.
        :return numpy array containing blobs union.
        """
        return area + area.T - intersection

    @staticmethod
    def check_intersection(filtration_params: dict, metrics: dict, i: int, j: int) -> bool:
        """
        Method for filtration threshold checking.
        :param filtration_params: filtration parameters of current pathologies pair.
        :param metrics: dictionary containing  intersection metrica for current pathology blobs pair.
        :param i: current filtered pathology position.
        :param j: current filtration pathology position.
        :return parameter indicating whether to filter pathology blob or not.
        """
        for metric, value in filtration_params.items():
            if metrics[metric][i, j] > value:
                return True
