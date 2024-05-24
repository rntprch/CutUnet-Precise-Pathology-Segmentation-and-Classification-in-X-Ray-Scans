from Pipeline.results import LungsDXResult
import numpy as np
import cv2 as cv
from typing import Tuple, List
import matplotlib.pyplot as plt

from Pipeline.configs.service import *


class PathologyLocalization:
    """
    Class for pathology localization.
    """
    def __init__(self, image_data: LungsDXResult) -> None:
        self.lungs_blobs = None
        self.lungs_lobe_blobs = None
        self.image_data = image_data
        self.blobs = None
        self.pathologies_names = list()

    def get_localisation(self) -> LungsDXResult:
        """
        Method for pathologies localization processing.
        :return image data with localization tags.
        """
        self.__get_pathologies_blobs()
        self.__get_lungs_blobs()

        if self.blobs is not None:
            self.blobs[self.blobs > 1] = 1
            self.blobs = self.blobs.astype('float32')
            intersections = dict()
            lungs_intersection, lungs_lobe_intersection = self.__get_metrics()
            for i, pathology_name in enumerate(self.pathologies_names):
                intersection = lungs_intersection[i, :] if lungs_intersection is not None else lungs_lobe_intersection[i, :]
                if pathology_name in intersections.keys():
                    intersections[pathology_name].append(self.__check_intersection(intersection=intersection))
                else:
                    intersections[pathology_name] = [self.__check_intersection(intersection=intersection)]
            self.__update_image_data(intersections=intersections)
        return self.image_data

    def __get_pathologies_blobs(self):
        """
        Method for pathologies blobs array creation.
        """
        for result in self.image_data.pathologies_result.get_result():
            if result.mask.any() and PATHOLOGY_LOCALIZATION[result.name]:
                result.create_blobs()
                self.pathologies_names += [result.name] * result.blobs.shape[-1]
                self.blobs = result.blobs if self.blobs is None else np.concatenate((self.blobs, result.blobs), axis=-1)

    def __get_lungs_blobs(self):
        """
        Method for lungs blobs creation.
        """
        if 'lungs' in PATHOLOGY_LOCALIZATION.values():
            lungs_contours, _ = cv.findContours(
                self.image_data.lungs_segmentation_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
            lung_index_l2r = np.argsort([cv.boundingRect(i)[0] for i in lungs_contours]).tolist()
            for i in lung_index_l2r:
                blob = np.zeros_like(self.image_data.lungs_segmentation_mask)
                cv.fillPoly(blob, pts=[lungs_contours[i]], color=1)
                self.lungs_blobs = blob if self.lungs_blobs is None else np.stack((self.lungs_blobs, blob), axis=-1)

        if 'lungs_lobe' in PATHOLOGY_LOCALIZATION.values():
            pass

    def __get_metrics(self) -> Tuple[np.array, np.array]:
        """
        Method for IoU and IoA metrics calculation.
        :return numpy arrays containing calculated IoU and IoA metrics.
        """
        area = self.__area()
        lungs_ioa, lungs_lobe_inv_ioa = None, None
        if 'lungs' in PATHOLOGY_LOCALIZATION.values():
            lungs_intersection = self._intersection(intersection_type='lungs')
            lungs_ioa = lungs_intersection / area
        if 'lungs_lobe' in PATHOLOGY_LOCALIZATION.values():
            lungs_lobe_intersection = self._intersection(intersection_type='lungs_lobe')
            lungs_lobe_inv_ioa = lungs_lobe_intersection / area.T

        return lungs_ioa, lungs_lobe_inv_ioa

    @staticmethod
    def __check_intersection(intersection: np.array) -> np.array:
        threshold = 0
        intersection[intersection > threshold] = 1
        intersection[intersection < threshold] = 0
        return np.argmax(intersection)

    def __update_image_data(self, intersections: dict) -> None:
        """
        Method for image data_structure updating.
        """
        for pathology_name, intersection in intersections.items():
            self.image_data.pathologies_result.update(
                pathology_name=pathology_name,
                localization=self.__get_localization_tag(
                    intersection_type=PATHOLOGY_LOCALIZATION[pathology_name], intersections=intersection
                )
                )

    def _intersection(self, intersection_type: str) -> np.array:
        """
        Method for blobs intersection calculation.
        :return numpy array containing blobs intersection.
        """
        if intersection_type == 'lungs':
            lungs_blobs = self.lungs_blobs.reshape(
                (self.lungs_blobs.shape[0] * self.lungs_blobs.shape[1], self.lungs_blobs.shape[2])
            )
        else:
            lungs_blobs = self.lungs_lobe_blobs.reshape(
                (self.lungs_lobe_blobs.shape[0] * self.lungs_lobe_blobs.shape[1], self.lungs_lobe_blobs.shape[2])
            )
        blobs = self.blobs.reshape((self.blobs.shape[0] * self.blobs.shape[1], self.blobs.shape[2]))
        intersection = np.matmul(np.transpose(blobs).astype(float), lungs_blobs.astype(float))
        return intersection

    def __area(self) -> np.array:
        """
        Method for blobs area calculation.
        :return numpy array containing blobs area.
        """
        area = np.sum(self.blobs, axis=(0, 1))
        return np.tile(area, (len(area), 1)).T

    @staticmethod
    def __get_localization_tag(intersection_type: str, intersections: List[np.array]) -> str:
        if intersection_type == 'lungs':
            if np.all(np.array(intersections) == 0):
                tag = 'left'
            elif np.all(np.array(intersections) == 1):
                tag = 'right'
            else:
                tag = 'both'
        else:
            tag=''
        return tag

