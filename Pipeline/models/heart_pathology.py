import numpy as np
from typing import Tuple, List
import cv2 as cv
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from Pipeline.results import LungsDXResult
from Pipeline.configs import *


class HeartPathology:
    def __init__(self, use_projection: str):
        """
        Mediastinal pathology detection class.
        """
        self.projection = use_projection
        self.required_pathologies = REQUIRED_PATHOLOGIES['heart']

    def predictions(self, image_data: LungsDXResult) -> LungsDXResult:
        """
        Method for heart pathologies detection.
        :param image_data: LungsDXResult class object containing image data.
        :returns updated image_data object.
        """
        if 7 not in image_data.warnings:
            image_data.lungs_quality = self.__lungs_quality(lungs_mask=image_data.lungs_segmentation_mask)
            image_data.lungs_symmetry = self.__lungs_symmetry(lungs_mask=image_data.lungs_segmentation_mask)
            if CTR in self.required_pathologies and image_data.lungs_quality > CARDIOMEGALY_LUNGS_SEGMENT_QUALITY:
                image_data.cardiomegaly.update(cti=self.__detect_cardiomegaly(
                    heart_mask=image_data.cardiomegaly.mask, lungs_mask=image_data.lungs_segmentation_mask
                ))
            elif image_data.lungs_quality < CARDIOMEGALY_LUNGS_SEGMENT_QUALITY:
                image_data.warnings.append(6)
            if MESH in self.required_pathologies:
                shift_params = MediastinalShift(
                    heart_mask=image_data.cardiomegaly.mask.copy(), lungs_mask=image_data.lungs_segmentation_mask.copy()
                ).calculate_shift()
                if shift_params:
                    image_data.mediastinal_shift.update(shift_params=shift_params, mask=image_data.cardiomegaly.mask)
        return image_data

    @staticmethod
    def __detect_cardiomegaly(heart_mask: np.array, lungs_mask: np.array) -> float:
        """
        Method cardiothoracic index value calculation.
        :param heart_mask: heart segmentation mask.
        :param lungs_mask: lungs segmentation mask.
        :returns cardiothoracic index
        """
        heart_contour, _ = cv.findContours(heart_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        heart_contour = heart_contour[0]
        heart_left = tuple(heart_contour[heart_contour[:, :, 0].argmin()][0])[0]
        heart_right = tuple(heart_contour[heart_contour[:, :, 0].argmax()][0])[0]
        heart_width = heart_right - heart_left

        lungs_contours, _ = cv.findContours(lungs_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        lungs_contour = np.array(lungs_contours[0].tolist() + lungs_contours[1].tolist())
        lungs_left_x = tuple(lungs_contour[lungs_contour[:, :, 0].argmin()][0])[0]
        lungs_right_x = tuple(lungs_contour[lungs_contour[:, :, 0].argmax()][0])[0]
        lungs_width = lungs_right_x - lungs_left_x
        return heart_width / lungs_width

    @staticmethod
    def __lungs_quality(lungs_mask: np.array) -> float:
        """
        Method for lungs quality param value calculation.
        :param lungs_mask: numpy array containing lungs segmentation mask.
        :return lungs quality param value.
        """
        lungs_contours, _ = cv.findContours(lungs_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        lungs_contour_1 = lungs_contours[0]
        lungs_contour_2 = lungs_contours[1]
        lung1_min_y = tuple(lungs_contour_1[lungs_contour_1[:, :, 1].argmin()][0])[1]
        lung1_max_y = tuple(lungs_contour_1[lungs_contour_1[:, :, 1].argmax()][0])[1]
        lung2_min_y = tuple(lungs_contour_2[lungs_contour_2[:, :, 1].argmin()][0])[1]
        lung2_max_y = tuple(lungs_contour_2[lungs_contour_2[:, :, 1].argmax()][0])[1]
        delta = abs(lung1_min_y - lung2_min_y) + abs(lung1_max_y - lung2_max_y)
        mean_height = max((lung1_max_y - lung1_min_y), (lung2_max_y - lung2_min_y))
        return 1 - delta / mean_height

    def __lungs_symmetry(self, lungs_mask: np.array) -> float:
        """
        Method for calculating the symmetry of lung fields.
        :param lungs_mask: numpy array containing lungs segmentation mask.
        :return lungs symmetry param value.
        """
        lungs_contours, _ = cv.findContours(lungs_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        lungs = np.array(lungs_contours[0].tolist() + lungs_contours[1].tolist())
        x_min = tuple(lungs[lungs[:, :, 0].argmin()][0])[0]
        x_max = tuple(lungs[lungs[:, :, 0].argmax()][0])[0]
        y_min = tuple(lungs[lungs[:, :, 1].argmin()][0])[1]
        y_max = tuple(lungs[lungs[:, :, 1].argmax()][0])[1]
        roi = lungs_mask[y_min:y_max, x_min:x_max]
        symmetry = self.__dice(mask1=roi, mask2=np.flip(roi, axis=1))
        return symmetry

    @staticmethod
    def __dice(mask1: np.array, mask2: np.array):
        dice = 2 * np.sum(mask1 * mask2) / np.sum(mask1 + mask2)
        return dice


class MediastinalShift:
    def __init__(self, heart_mask: np.array, lungs_mask: np.array) -> None:
        """
        Class for mediastinal shift pathology detection.
        :param heart_mask: heart segmentation mask.
        :param lungs_mask: lungs_segmentation mask.
        """
        self.heart = heart_mask
        self.lungs = lungs_mask

    def calculate_shift(self) -> Tuple[float, float, float]:
        """
        Method for heart shift calculation.
        :return: tuple of left and right shifts respectively.
        """
        heart_left_coordinates, heart_right_coordinates, heart_top_coordinates = self.__get_heart_coordinates()
        left_lung_contour, right_lung_contour = self.__get_contours(mask=self.lungs)
        left_shift = self.__shift(
            heart_extreme_point=heart_left_coordinates, lungs_contour=left_lung_contour, lung='left'
        )
        right_shift = self.__shift(
            heart_extreme_point=heart_right_coordinates, lungs_contour=right_lung_contour, lung='right'
        )
        shift = self.__shift(
            heart_extreme_point=heart_top_coordinates, lungs_contour=left_lung_contour, lung='left'
        )
        # left_lung, right_lung = np.zeros_like(self.lungs), np.zeros_like(self.lungs)
        # cv.fillPoly(left_lung, pts=[left_lung_contour], color=1)
        # cv.fillPoly(right_lung, pts=[right_lung_contour], color=1)
        # left_shift = (self.heart * left_lung).sum() / self.heart.sum()
        # right_shift = (self.heart * right_lung).sum() / self.heart.sum()

        return left_shift, right_shift, shift

    def __shift(self, heart_extreme_point: Tuple[int, int], lungs_contour: np.array, lung: str) -> float:
        """
        Method for shift calculation.
        :param heart_extreme_point: heart extreme point coordinate.
        :param lungs_contour: lungs contour coordinates.
        :param lung: lung position (left or right).
        :return: calculated shift value.
        """
        lung_left_x, lung_right_x = self.__get_lung_coordinates(
            heart_coordinate=heart_extreme_point, contour=lungs_contour
        )
        lung_width = lung_right_x - lung_left_x
        lung_coord = lung_left_x if lung == 'right' else lung_right_x
        heart_width = abs(heart_extreme_point[0] - lung_coord)

        # mask = cv.circle(img=self.lungs+self.heart, center=(lung_left_x, heart_extreme_point[1]), radius=10, thickness=-1, color=(10, 10, 10))
        # mask = cv.circle(img=mask, center=(lung_right_x, heart_extreme_point[1]), radius=10, thickness=-1, color=(10, 10, 10))
        # mask = cv.circle(img=mask, center=heart_extreme_point, radius=10, thickness=-1, color=(10, 10, 10))
        # plt.imshow(mask)
        # plt.show()
        return heart_width / lung_width

    def __get_heart_coordinates(self) -> List[Tuple[int, int]]:
        """
        Method for getting extreme heart coordinates.
        :return: list of tuples with left and right coordinates respectively.
        """
        heart_contour = self.__get_contours(mask=self.heart)
        top_y = tuple(heart_contour[heart_contour[:, :, 1].argmin()][0])[1]
        bottom_y = tuple(heart_contour[heart_contour[:, :, 1].argmax()][0])[1]
        center_y = int((top_y - bottom_y) * (1 - HEART_TOP_PARAM)) + bottom_y

        upper_heart_mask = self.heart.copy()
        upper_heart_mask[center_y:, :] = 0
        upper_heart_contour = self.__get_contours(mask=upper_heart_mask)

        # plt.imshow(upper_heart_mask)
        # plt.show()

        right_x, right_y = tuple(upper_heart_contour[upper_heart_contour[:, :, 0].argmax()][0])
        left_x, left_y = tuple(heart_contour[heart_contour[:, :, 0].argmin()][0])
        top_left_x, top_left_y = tuple(upper_heart_contour[upper_heart_contour[:, :, 0].argmin()][0])
        return [(left_x, left_y), (right_x, right_y), (top_left_x, top_left_y)]

    @staticmethod
    def __get_lung_contours(lung_contour: np.array) -> Tuple[int, int]:
        """
        Method for lung extreme coordinates extraction.
        :param lung_contour: lungs contour coordinates.
        :return: lung x-coordinates corresponding to heart coordinate.
        """
        lung_left_x, _ = tuple(lung_contour[lung_contour[:, :, 0].argmin()][0])
        lung_right_x, _ = tuple(lung_contour[lung_contour[:, :, 0].argmax()][0])
        return lung_right_x, lung_left_x

    def __get_lung_fwhm(self, lung_contour) -> int:
        """
        Method for lung width calculation.
        :param lung_contour: lungs contour coordinates.
        :return: calculated lungs width.
        """
        image = cv.fillPoly(img=np.zeros_like(self.lungs), pts=[lung_contour], color=1)
        moment = cv.moments(lung_contour)
        lungs_center_y = int(moment["m01"] / moment["m00"])

        width = cv.countNonZero(image[lungs_center_y, :])
        return width

    @staticmethod
    def __get_lung_coordinates(heart_coordinate: Tuple[int, int], contour: np.array) -> Tuple[int, int]:
        """
        method for lung x-coordinates corresponding to heart x extreme coordinate.
        :param heart_coordinate:
        :param contour:
        :return
        """
        distance = np.abs(contour.squeeze()[:, 1] - heart_coordinate[1])
        min_coord_idx = np.where(distance == distance.min())[0]
        if len(min_coord_idx) < 2:
            min_coord_idx = np.argpartition(distance, 2)
        min_x = contour.squeeze()[min_coord_idx][:, 0].min()
        max_x = contour.squeeze()[min_coord_idx][:, 0].max()
        return min_x, max_x

    @staticmethod
    def __get_contours(mask: np.array) -> np.array or List[np.array]:
        """
        Method for mask contour extraction.
        :param mask: numpy array containing mask.
        :return: numpy array or list of arrays containing contour coordinates sorted (left->right).
        """
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if len(contours) == 1:
            return contours[0]
        else:
            bounding_boxes = [cv.boundingRect(c) for c in contours]
            contours, _ = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0], reverse=False))
            return contours
