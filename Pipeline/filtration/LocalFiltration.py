import numpy as np
import cv2 as cv
from Pipeline.results import LungsDXResult
from Pipeline.configs import *
import matplotlib.pyplot as plt


class MaskFiltration:

    def __init__(self, image_data: LungsDXResult) -> None:
        """
        Lungs DX pathology mask filtration class.
        :param image_data: LungsDXResult class object containing all the information about the image obtained when
         running through the pipeline.
        """
        self.image_data = image_data
        self.__create_pleural_fluid_mask()
        self.__get_roi_mask()

    def make_filtration(self) -> LungsDXResult:
        """
        Method for making pathologies masks filtration.
        :return image_data object
        """
        if CTR in REQUIRED_PATHOLOGIES['heart'] and self.image_data.cardiomegaly.visualize:
            self.__make_cardiomegaly_line()

        if self.image_data.pathologies_result.is_pathological:
            self.__get_pathologies_masks()

        if REQUIRED_PATHOLOGIES['Other'] and self.image_data.triage_result.is_pathological:
            self.__get_triage_masks()

        return self.image_data

    def __get_pathologies_masks(self) -> None:
        """
        Method for hemithorax pathologies masks filtration.
        """
        for result in self.image_data.pathologies_result.get_result():
            if result.is_pathological:
                if result.name == PLEF:
                    mask = self.image_data.pleural_fluid_mask.copy()
                elif result.name in [CONSOLIDATED_FRACTURE, FRACTURE]:
                    mask = self.__create_fractures_masks()
                elif result.name == DIAPHM:
                    pathology_mask = self.__create_diaphragm_mask(diaphragm_mask=self.__float_to_uint(x=result.mask))
                    self.image_data.pathologies_result.update(pathology_name=result.name, mask=pathology_mask)
                    mask = self.image_data.pleural_fluid_mask.copy()
                else:
                    mask = self.image_data.lungs_segmentation_mask.copy()

                if FILTER_PATHOLOGY_MASK:
                    self.image_data.pathologies_result.filter_mask(pathology_name=result.name, lungs_mask=mask)
                    self.image_data.pathologies_result.update(pathology_name=result.name)

    def __get_triage_masks(self) -> None:
        """
        Method for binary triage pathologies mask filtration.
        """
        if FILTER_PATHOLOGY_MASK:
            self.image_data.triage_result.lungs_filter(lungs_mask=self.image_data.roi_mask)
            self.image_data.triage_result.update()

    def __create_fractures_masks(self) -> np.array:
        """
        Method for fractures pathology mask creation.
        :returns numpy array containing fractures mask
        """
        mask = self.image_data.lungs_segmentation_mask.copy()
        fr_mask = self.__build_up_mask(mask=mask, param=8)
        return fr_mask

    def __create_diaphragm_mask(self, diaphragm_mask: np.array) -> np.array:
        """
        Method for diaphragm pathology mask creation.
        :param diaphragm_mask: base diaphm pathologic mask.
        :returns numpy array containing diaphragm mask
        """
        lungs_contours, _ = cv.findContours(
            self.image_data.lungs_segmentation_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        diaphragm_contours, _ = cv.findContours(diaphragm_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        new_mask = np.zeros_like(self.image_data.lungs_segmentation_mask)
        for diaphragm_contour in diaphragm_contours:
            diaphragm_mask = np.zeros_like(self.image_data.lungs_segmentation_mask)
            for lung_contour in lungs_contours:
                lung_mask = np.zeros_like(self.image_data.lungs_segmentation_mask)
                if self.__ioa(cv.fillPoly(lung_mask, pts=[lung_contour], color=1),
                              cv.fillPoly(diaphragm_mask, pts=[diaphragm_contour], color=1)):
                    approx_points = cv.approxPolyDP(
                        lung_contour, CONTOUR_APPROX_PARAM * cv.arcLength(lung_contour, True), True
                    ).tolist()
                    bottom_lung_points = sorted(approx_points, key=lambda k: k[0][1])[-2:]
                    diaphragm_curve = self.__get_diaphragm_curve(
                        points=bottom_lung_points, contour=lung_contour.tolist()
                    )
                    new_mask = cv.polylines(
                        img=new_mask,
                        pts=[diaphragm_curve],
                        isClosed=False,
                        color=1,
                        thickness=int(max(diaphragm_mask.shape[:2]) * CONTOUR_LINE_WIDTH)
                    )
                    new_mask = self.__build_up_mask(mask=new_mask, param=DIAPHRAGM_MASK_THICKNESS)
                    break
        return self.__float_to_uint(x=new_mask)

    def __make_cardiomegaly_line(self) -> None:
        """
        Method for getting cardiomegaly line coordinates.
        """
        heart_contour, _ = cv.findContours(
            self.image_data.cardiomegaly.mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )
        heart_contour = heart_contour[0]
        heart_left_x, heart_left_y = tuple(heart_contour[heart_contour[:, :, 0].argmin()][0])
        heart_right_x, heart_right_y = tuple(heart_contour[heart_contour[:, :, 0].argmax()][0])

        lungs_contours, _ = cv.findContours(
            self.image_data.lungs_segmentation_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )
        lung_1 = lungs_contours[0]
        lung_2 = lungs_contours[1]
        lung_1_left_x = tuple(lung_1[lung_1[:, :, 0].argmin()][0])[0]
        lung_1_right_x = tuple(lung_1[lung_1[:, :, 0].argmax()][0])[0]
        lung_2_left_x = tuple(lung_2[lung_2[:, :, 0].argmin()][0])[0]
        lung_2_right_x = tuple(lung_2[lung_2[:, :, 0].argmax()][0])[0]

        left_lung_right_x = min(lung_1_right_x, lung_2_right_x)
        right_lung_left_x = max(lung_1_left_x, lung_2_left_x)

        center_x = (right_lung_left_x - left_lung_right_x) // 2 + left_lung_right_x

        self.image_data.cardiomegaly.line_points = [heart_left_x, heart_left_y, center_x, heart_right_x, heart_right_y]

    def __create_pleural_fluid_mask(self) -> None:
        """
        Method for pleural effusion pathology mask creation.
        """
        contours, _ = cv.findContours(self.image_data.lungs_segmentation_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        new_mask = np.zeros_like(self.image_data.lungs_segmentation_mask)
        if len(contours) == 2:
            for contour in contours:
                peri = cv.arcLength(contour, True)
                approx_points = cv.approxPolyDP(contour, CONTOUR_APPROX_PARAM * peri, True)
                approx_points = approx_points.tolist()
                bottom_points = sorted(approx_points, key=lambda x: x[0][1])[-2:]
                contour = contour.tolist()
                contour.append([[bottom_points[1][0][0], self.image_data.lungs_segmentation_mask.shape[0]]])
                contour.append([[bottom_points[0][0][0], self.image_data.lungs_segmentation_mask.shape[0]]])
                convex_points = cv.convexHull(np.array(contour))
                cv.fillPoly(new_mask, pts=[convex_points], color=1)
        else:
            lung_contour = contours[0]
            lung_left = tuple(lung_contour[lung_contour[:, :, 0].argmin()][0])[0]
            lung_right = tuple(lung_contour[lung_contour[:, :, 0].argmax()][0])[0]
            lung_contour = lung_contour.tolist()
            lung_contour.append([[lung_left, self.image_data.lungs_segmentation_mask.shape[0]]])
            lung_contour.append([[lung_right, self.image_data.lungs_segmentation_mask.shape[0]]])
            convex_points = cv.convexHull(np.array(lung_contour))
            cv.fillPoly(new_mask, pts=[convex_points], color=1)
        self.image_data.pleural_fluid_mask = new_mask

    def __get_roi_mask(self) -> None:
        """
        Method for region of interesting mask creation.
        """
        contours, _ = cv.findContours(self.image_data.lungs_segmentation_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        if len(contours) == 2:
            contour = np.array(contours[0].tolist() + contours[1].tolist())
            convex_points = cv.convexHull(np.array(contour))
            roi_mask = cv.fillPoly(np.zeros_like(self.image_data.lungs_segmentation_mask), pts=[convex_points], color=1)
            bottom_points = list()
            for lung_contour in contours:
                approx_points = cv.approxPolyDP(
                    lung_contour, CONTOUR_APPROX_PARAM * cv.arcLength(lung_contour, True), True
                ).tolist()
                bottom_lung_points = sorted(approx_points, key=lambda k: k[0][1])[-2:]
                diaphragm_curve = self.__get_diaphragm_curve(
                    points=bottom_lung_points, contour=lung_contour.tolist()
                )
                bottom_points.append(diaphragm_curve)
            bottom_points = bottom_points[0].tolist() + bottom_points[1].tolist()
            bottom_points.append([[0, self.image_data.lungs_segmentation_mask.shape[0]]])
            bottom_points.append(
                [[self.image_data.lungs_segmentation_mask.shape[1], self.image_data.lungs_segmentation_mask.shape[0]]]
            )

            convex_points = cv.convexHull(points=np.array(bottom_points))
            bottom_mask = cv.fillPoly(
                np.zeros_like(self.image_data.lungs_segmentation_mask), pts=[convex_points], color=1
            )
            bottom_mask = bottom_mask.max() - bottom_mask
            self.image_data.roi_mask = roi_mask * bottom_mask
        else:
            self.image_data.roi_mask = self.image_data.lungs_segmentation_mask

    @staticmethod
    def __get_diaphragm_curve(points: list, contour: list) -> np.array:
        """
        Method for getting lung diaphragm coordinates.
        :param points: coordinates of two lung mask bottom points.
        :param contour: list containing points of lung contour.
        :returns part of lung contour corresponding to its diaphragm.
        """
        points = sorted(points, key=lambda k: k[0][0])
        i = contour.index(points[0])
        j = contour.index(points[1])
        return np.array(contour[i:j])

    @staticmethod
    def __build_up_mask(mask: np.array, param: int) -> np.array:
        """
        Method for getting lung diaphragm coordinates.
        :param mask: numpy array containing mask.
        :param param: parameter that determines the strength of the mask increase.
        :returns enlarged mask.
        """
        for _ in range(param):
            mask = cv.dilate(
                mask, kernel=np.ones(
                    shape=(int(mask.shape[0] * 2 * CONTOUR_LINE_WIDTH), int(mask.shape[0] * 2 * CONTOUR_LINE_WIDTH)),
                    dtype=np.uint8
                ),
                iterations=1
            )
        return mask

    @staticmethod
    def __ioa(base_mask: np.array, mask: np.array) -> float:
        """
        Method for intersection over area parameter calculation.
        :param base_mask: area mask.
        :param mask: other mask.
        :returns intersection over area parameter value.
        """
        mask[mask > 1] = 1
        base_mask[base_mask > 1] = 1
        intersection = (mask * base_mask).sum()
        area = base_mask.sum()
        return intersection / area

    @staticmethod
    def __float_to_uint(x: np.array) -> np.array:
        """
        Method converting array from float to uint8 type.
        :param x: image or mask numpy array.
        :returns converted numpy array.
        """
        if x.any():
            return ((x - x.min()) * (1 / (x.max() - x.min())) * 255).astype('uint8')
        else:
            return np.zeros_like(x, dtype='int32')
