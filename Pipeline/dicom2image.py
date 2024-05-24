from pydicom import Dataset
import numpy as np
from typing import Tuple
from pydicom.multival import MultiValue


class ImageFromDicom:
    def __init__(self, dicom: Dataset) -> None:
        """
        Class for image array extraction from dicom.
        :param dicom: dicom file containing Chest X-ray image.
        """
        self.dicom = dicom

    def get_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method for getting chest x-ray images from dicom.
        :return tuple of arrays containing uint-8 and uint-16 images.
        """
        pixel_data = self.dicom.pixel_array.astype(np.float32)
        assert pixel_data.ndim == 2, (
            "Expected to work with 2D input: pipeline should check this "
            "constraint on extraction of pixel data from dataset"
        )
        ww, wc = self.__infer_render_window_parameters()
        low = wc - ww / 2
        pixel_data = (pixel_data - low) / ww
        image8 = self.__convert_from_float_image_to_uint8(pixel_data)
        image16 = self.__convert_from_float_image_to_uint16(pixel_data)
        image8, image16 = self.__check_if_automatic_invert(image8=image8, image16=image16)
        return self.__to_3channels_shape(image8), self.__to_3channels_shape(image16)

    def __infer_render_window_parameters(self) -> Tuple[float, float]:
        """
        Method for getting dicom image window params.
        :return tuple of window params.
        """
        # Try to infer WW/WC from DICOM tags
        try:
            ww, wc = self.dicom.WindowWidth, self.dicom.WindowCenter
        except AttributeError:
            ww, wc = None, None

        # If there is a list of WW/WC options, use the "NORMAL" option
        if isinstance(ww, MultiValue) and isinstance(wc, MultiValue):
            try:
                exp = self.dicom.WindowCenterWidthExplanation
                if isinstance(exp, MultiValue) and all(
                    isinstance(e, str) for e in exp
                ):
                    exp = [e.lower().strip() for e in exp]
                else:
                    exp = None
            except AttributeError:
                exp = None

            if exp is not None and "normal" in exp:
                normal_idx = exp.index("normal")
                try:
                    ww, wc = ww[normal_idx], wc[normal_idx]
                except IndexError:
                    ww, wc = None, None
            else:
                ww, wc = None, None

        # Try to coerce the WW/WC values to float (in case they are str or int)
        try:
            ww, wc = float(ww), float(wc)

            # Chest X-ray pixel_array data should have an unsigned type,
            # So ww < 1 and wc < 0 are most likely wrong/filler values
            if ww < 1 or wc < 0:
                ww, wc = None, None
        except (ValueError, TypeError):
            ww, wc = None, None
        region_data = self.dicom.pixel_array
        if wc is not None and wc > region_data.max():
            ww, wc = None, None
        if wc is not None and wc < region_data.min():
            ww, wc = ww, region_data.min() + wc

        # If inference from the DICOM tags failed, use roi-quantile approximation
        if ww is None or wc is None:
            lo = np.quantile(region_data, 0.05)
            hi = np.quantile(region_data, 0.95)
            ww = 1.75 * (hi - lo)
            wc = 0.5 * (hi + lo)
            ww = np.clip(ww, 1, None)

        return ww, wc

    def __check_if_automatic_invert(self, image8: np.array, image16: np.array) -> Tuple[np.array, np.array]:
        """
        Method for check if pydicom automatically invert image
        :param image8: numpy array containing 8-bit pixel data from current dicom.
        :param image16: numpy array containing 16-bit pixel data from current dicom.
        :return tuple of arrays corresponding to 8 and 16-bit images respectively.
        """
        if self.dicom.PhotometricInterpretation == 'MONOCHROME1':
            return image8.max() - image8, image16 - image16.max()
        else:
            return image8, image16

    @staticmethod
    def __convert_from_float_image_to_uint8(arr: np.ndarray) -> np.ndarray:
        """
        Method for converting float image into uint-8 format.
        :param arr: input image array.
        :return array containing image in uint-8.
        """
        return np.clip(arr * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def __convert_from_float_image_to_uint16(arr: np.ndarray) -> np.ndarray:
        """
        Method for converting float image into uint-16 format.
        :param arr: input image array.
        :return array containing image in uint-16.
        """
        return np.clip(arr * (2**16-1), 0, (2**16-1)).astype(np.uint16)

    @staticmethod
    def __to_3channels_shape(x: np.array) -> np.array:
        """
        Method for converting grayscale image into 3 channels image.
        :param x: input 2-d grayscale image.
        :return numpy array containing 3-channel image.
        """
        x = np.atleast_3d(np.squeeze(x))

        if x.shape[-1] > 3:
            x = x[:, :, x.shape[-1] // 2:x.shape[-1] // 2 + 1]

        if x.shape[-1] == 1:
            x = np.repeat(x, repeats=3, axis=-1)

        return x

