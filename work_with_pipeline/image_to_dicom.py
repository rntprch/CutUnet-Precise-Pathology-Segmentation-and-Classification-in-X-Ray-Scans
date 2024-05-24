from functools import partial
import imagecodecs
import os
import numpy as np
from pydicom import Dataset, Sequence
from pydicom._storage_sopclass_uids import SecondaryCaptureImageStorage
from pydicom.encaps import encapsulate
from pydicom.uid import (
    JPEG2000,
    PYDICOM_ROOT_UID,
    RE_VALID_UID_PREFIX,
    UID,
    ImplicitVRLittleEndian,
    JPEG2000Lossless,
    JPEGBaseline8Bit,
    JPEGLSLossless,
    RLELossless,
    generate_uid,
)
from typing import Any
import hashlib
import re
import uuid
import enum
from datetime import datetime, timezone
from typing import NamedTuple, Optional


class OperatingMode(enum.Enum):
    GENERIC = "GENERIC"
    MOSCOW_ERIS = "MOSCOW_ERIS"


class PipelineContext(NamedTuple):
    # input parameters
    operating_mode: OperatingMode
    tz: timezone
    full_software_version: str
    public_software_version: str
    output_ts: UID
    model_id: int
    processing_started_at: datetime
    processing_ended_at: Optional[datetime]

    # pipeline generated parameters
    analysis_dt: datetime
    src: Dataset
    metadata: Optional[dict] = None


class SCBuild:

    def __init__(
            self, context: PipelineContext, sc_image: np.array, number_of_images: int = 1, number_of_series: int = 2
    ) -> None:
        """
        Class for dicom creation.
        :param context:
        :param sc_image: numpy array containing image to be saved as dicom.
        :param number_of_images:
        :param number_of_series:
        """
        self.context = context
        self.img = sc_image
        self.img_num = number_of_images
        self.number_of_series = number_of_series
        self.result_sc = Dataset()

    def create(self) -> Dataset:
        """
        Method for dicom creation.
        :return dicom dataset
        """
        if self.context.output_ts == ImplicitVRLittleEndian:
            self.result_sc.is_implicit_VR = self.context.output_ts.is_implicit_VR
            self.result_sc.is_little_endian = self.context.output_ts.is_little_endian

        self.__fill_patient_module()
        self.__fill_general_study_module()
        self.__copy_tag(tag="Modality", tag_type=1)

        if self.context.operating_mode == OperatingMode.MOSCOW_ERIS:
            self.result_sc.SeriesInstanceUID = self.__generate_moscow_eris_series_iuid()
        else:
            self.result_sc.SeriesInstanceUID = self.__generate_series_iuid()

        self._fill_series_data()
        self._fill_related_series_data()

        # General Equipment Module
        self.result_sc.Manufacturer = "Third opinion"

        # SC Equipment Module
        self.result_sc.ConversionType = "SYN"

        # General Image Module
        self.result_sc.InstanceNumber = self.img_num
        self.result_sc.ImageType = ["DERIVED", "SECONDARY", "OTHER"]
        self.result_sc.AcquisitionDate = self.context.analysis_dt

        if self.context.operating_mode == OperatingMode.MOSCOW_ERIS:
            self.result_sc.AcquisitionTime = self.context.analysis_dt.replace(microsecond=0)
        else:
            self.result_sc.AcquisitionTime = self.context.analysis_dt

        self._fill_image_data()
        self._fill_sop_data()

        # ERIS
        self.__fill_other_tags()
        if self.context.operating_mode == OperatingMode.MOSCOW_ERIS:
            self.__fill_moscow_eris_sc_tags()

        self.result_sc.fix_meta_info()
        self.__add_version_private_tag()

        if self.context.output_ts == RLELossless:
            self.result_sc.compress(RLELossless)

        if self.context.output_ts in {JPEGBaseline8Bit, JPEGLSLossless, JPEG2000, JPEG2000Lossless}:
            self.compress_to_jpeg()

        return self.result_sc

    def _fill_series_data(self) -> None:
        """
        Method for dicom series data filling.
        """
        self.result_sc.SeriesNumber = None  # has type 2 (required, empty if unknown)
        self.result_sc.SeriesDate = self.context.analysis_dt
        self.result_sc.SeriesTime = self.context.analysis_dt
        self.result_sc.SeriesDescription = "Third Opinion authorized"

    def _fill_related_series_data(self) -> None:
        """
        Method for dicom related series data filling.
        """
        related_series = Dataset()
        related_series.StudyInstanceUID = self.__get_required_and_non_empty(tag="StudyInstanceUID")
        related_series.SeriesInstanceUID = self.__get_required_and_non_empty(tag="SeriesInstanceUID")
        related_series.PurposeOfReferenceCodeSequence = Sequence()
        self.result_sc.RelatedSeriesSequence = Sequence([related_series])

    def _fill_sop_data(self) -> None:
        """
        Method for dicom sop data filling.
        """
        self.result_sc.SOPClassUID = SecondaryCaptureImageStorage
        if self.context.operating_mode == OperatingMode.MOSCOW_ERIS:
            self.result_sc.SOPInstanceUID = generate_uid()
        else:
            self.result_sc.SOPInstanceUID = self.__generate_sop_iuid(
                self.__get_required_and_non_empty(tag="SOPInstanceUID")
            )

        self.result_sc.SpecificCharacterSet = "ISO_IR 192"  # it is UTF-8
        self.result_sc.TimezoneOffsetFromUTC = datetime.now(self.context.tz).strftime("%z")

    def _fill_image_data(self) -> None:
        """
        Method for dicom image data filling.
        """
        # General Reference Module
        ref_img_ds = Dataset()
        ref_img_ds.ReferencedSOPClassUID = self.__get_required_and_non_empty(tag="SOPClassUID")
        ref_img_ds.ReferencedSOPInstanceUID = self.__get_required_and_non_empty(tag="SOPInstanceUID")
        ref_img_ds.SpatialLocationsPreserved = "YES"
        self.result_sc.SourceImageSequence = Sequence([ref_img_ds])

        # Image Pixel Module
        assert self.img.ndim == 3, "Only 3 dimensional image arrays are supported"
        assert self.img.shape[-1] == 3, "Number of image channels should be equal to 3"
        self.result_sc.SamplesPerPixel = self.img.shape[2]
        self.result_sc.PhotometricInterpretation = {1: "MONOCHROME2", 3: "RGB"}[self.img.shape[2]]
        self.result_sc.Rows = self.img.shape[0]
        self.result_sc.Columns = self.img.shape[1]
        self.result_sc.BitsAllocated = 8 * self.img.dtype.itemsize
        self.result_sc.BitsStored = self.result_sc.BitsAllocated
        self.result_sc.HighBit = self.result_sc.BitsStored - 1
        self.result_sc.PixelRepresentation = {"u": 0, "i": 1}[np.iinfo(self.img.dtype).kind]
        # NOTE: PlanarConfiguration values mapping: 0 is "HWC", 1 is "CHW"
        self.result_sc.PlanarConfiguration = 0
        self.result_sc.PixelData = self.img.tobytes()

    def compress_to_jpeg(self, quality: int = 95) -> None:
        """
        Method for compressing dicom image into jpeg.
        :param quality: drop in quality compared to the original image.
        """
        encode_methods = {
            JPEGBaseline8Bit: partial(imagecodecs.jpeg8_encode, level=quality),
            JPEGLSLossless: imagecodecs.jpegls_encode,
            JPEG2000: partial(imagecodecs.jpeg2k_encode, level=quality),
            JPEG2000Lossless: imagecodecs.jpeg2k_encode,
        }

        jpeg_bytes = encode_methods[self.context.output_ts](self.result_sc.pixel_array)
        self.result_sc.PixelData = encapsulate([jpeg_bytes])
        self.result_sc.file_meta.TransferSyntaxUID = self.context.output_ts

        if not os.getenv("IS_FIX_FOR_COMPRESSION_DISABLED", False):
            self.result_sc = fix_compressed_ds(self.result_sc)

    def __fill_patient_module(self) -> None:
        """
        Method for patient information tags module filling.
        """
        for tag, tag_type in (("PatientName", 2), ("PatientID", 2), ("PatientBirthDate", 2), ("PatientSex", 2)):
            self.__copy_tag(tag=tag, tag_type=tag_type)

    def __fill_general_study_module(self) -> None:
        """
        Method for study information tags module filling.
        """
        for tag, tag_type in (
                ("StudyInstanceUID", 1),
                ("StudyDate", 2),
                ("StudyTime", 2),
                ("ReferringPhysicianName", 2),
                ("StudyID", 2),
                ("AccessionNumber", 2),
        ):
            self.__copy_tag(tag=tag, tag_type=tag_type)

    def __copy_tag(self, tag: str, tag_type: int) -> None:
        """
        Method for copying tags from original to new dicom.
        :param tag: tag name.
        :param tag_type: type of tag to be copied.
        """
        if tag_type == 1:
            setattr(self.result_sc, tag, self.__get_required_and_non_empty(tag=tag))
        elif tag_type == 2:
            setattr(self.result_sc, tag, self.context.src.get(tag))
        elif tag_type == 3:
            if tag in self.context.src:
                setattr(self.result_sc, tag, self.context.src.get(tag))
        else:
            raise ValueError(f"Got unsupported tag type {tag_type}")

    def __generate_series_iuid(self, prefix: Optional[str] = PYDICOM_ROOT_UID) -> UID:
        """Generates special DICOM UID with length up to 64 characters:
        SeriesInstanceUID: <prefix>.<hash_of_hash_iuid>.<series>

        For single image modalities (like CR, DX) SOPInstanceUID should be used as
        hash_uid, and for series of images modalities (like CT) SeriesInstanceUID
        should be used as hash_uid
        """
        hash_uid =  self.__get_required_and_non_empty(tag="SOPInstanceUID")
        suffix = f".{self.number_of_series}"
        return self.__generate_uid(hash_iuid=hash_uid, suffix=suffix, prefix=prefix)

    def __generate_sop_iuid(self, hash_iuid: UID, prefix: Optional[str] = PYDICOM_ROOT_UID) -> UID:
        """Generates special DICOM UID with length up to 64 characters:
        SOPInstanceUID: <prefix>.<hash_of_hash_iuid>.<series>.<image>

        For single image modalities (like CR, DX) SOPInstanceUID should be used as
        hash_uid, and for series of images modalities (like CT) SeriesInstanceUID
        should be used as hash_uid
        """
        suffix = f".{self.number_of_series}.{self.img_num}"
        return self.__generate_uid(hash_iuid=hash_iuid, suffix=suffix, prefix=prefix)

    def __fill_other_tags(self) -> None:
        """
        Method for filling dicom tags using tags from original dicom.
        """
        for tag, tag_type in (
                ("IssuerOfPatientID", 3),
                ("FillerOrderNumberImagingServiceRequest", 3),
                ("InstitutionName", 3),  # MEDSI
        ):
            self.__copy_tag(tag=tag, tag_type=tag_type)

    def __get_required_and_non_empty(self, tag: str) -> Any:
        """
        Method for checking if required tag is not empty in original dicom.
        :param tag: tag name to check.
        :return tag value.
        """
        value = self.context.src.get(tag)
        if value is None:
            raise Exception(f"{tag} required to be present and non-empty")
        return value

    def __add_version_private_tag(self) -> None:
        """
        Method for adding private service tags into result dicom.
        """
        private_group = 0x3DA1
        private_creator = "Third Opinion"
        fsv = self.context.full_software_version
        assert len(fsv) <= 64, (
            f"full software version {fsv!r} has length {len(fsv)} which doesn't "
            f"fit into Long String (LO) private tag with length limit of 64"
        )
        self.result_sc.private_block(private_group, private_creator, create=True).add_new(0x01, "LO", fsv)

    def __generate_moscow_eris_series_iuid(self) -> UID:
        """
        Method for custom series uid generation.
        """
        series_iuid =  self.__get_required_and_non_empty(tag="SeriesInstanceUID")[:56]
        if series_iuid[-1] == ".":
            series_iuid = series_iuid[:-1]
        series_iuid = f"{series_iuid}.{self.context.model_id}.{self.number_of_series}"
        if len(series_iuid) > 64:
            raise RuntimeError(
                f"SeriesInstanceUID for MOSCOW_ERIS operating mode has length "
                f"{len(series_iuid)} more than 64: maybe model_id '{self.context.model_id}' "
                f"has too many digits.",
            )
        return UID(series_iuid)

    def __fill_moscow_eris_sc_tags(self) -> None:
        """
        Method for filling institutional tags.
        """
        # WARN: usage of tags below violates their semantics, looks like a hack
        # to visualize information in viewers
        self.result_sc.InstitutionalDepartmentName = self.context.public_software_version
        self.result_sc.InstitutionName = self.result_sc.SeriesDescription

    @staticmethod
    def __generate_uid(hash_iuid: UID, suffix: str, prefix: Optional[str] = PYDICOM_ROOT_UID) -> UID:
        """
        Patched version of pydicom.uid.generate_uid function that allows to pass suffix.
        """
        if prefix is None:
            # UUID -> as 128-bit int -> max 39 characters long
            return UID("2.25.{}".format(uuid.uuid4().int))

        max_uid_len = 64
        if len(prefix) > max_uid_len - 1:
            raise ValueError("The prefix must be less than 63 chars")
        if not re.match(RE_VALID_UID_PREFIX, prefix):
            raise ValueError("The prefix is not in a valid format")

        # Expected max length of suffix like .<series_num>.<image_num> is
        # 1 (dot) + 1 (series_num) + 1 (dot) + 4 (image_num) = 7.
        # We expect series_num to be no more than 9, and image_num no more than
        # 9999.
        expected_max_suf_len = 7
        avail_digits = max_uid_len - len(prefix) - max(expected_max_suf_len, len(suffix))
        if avail_digits < 10:
            raise ValueError(f"Hash part of UID is shorter than 10 symbols: requested suffix {suffix} is too long")

        hash_val = hashlib.sha512(hash_iuid.encode("utf-8"))

        # Convert this to an int with the maximum available digits
        dicom_uid = prefix + str(int(hash_val.hexdigest(), 16))[:avail_digits] + suffix

        return UID(dicom_uid)


def fix_compressed_ds(dataset: Dataset, write_like_original=False) -> Dataset:
    """This code was copied from https://github.com/pydicom/pydicom/blob/2.2.X/pydicom/filewriter.py#L999-L1066"""

    # Ensure is_little_endian and is_implicit_VR are set
    if None in (dataset.is_little_endian, dataset.is_implicit_VR):
        has_tsyntax = False
        try:
            tsyntax = dataset.file_meta.TransferSyntaxUID
            if not tsyntax.is_private:
                dataset.is_little_endian = tsyntax.is_little_endian
                dataset.is_implicit_VR = tsyntax.is_implicit_VR
                has_tsyntax = True
        except AttributeError:
            pass

        if not has_tsyntax:
            name = dataset.__class__.__name__
            raise AttributeError(
                f"'{name}.is_little_endian' and '{name}.is_implicit_VR' must "
                f"be set appropriately before saving"
            )

    # Try and ensure that `is_undefined_length` is set correctly
    try:
        tsyntax = dataset.file_meta.TransferSyntaxUID
        if not tsyntax.is_private:
            dataset['PixelData'].is_undefined_length = tsyntax.is_compressed
    except (AttributeError, KeyError):
        pass

    # Check that dataset's group 0x0002 elements are only present in the
    #   `dataset.file_meta` Dataset - user may have added them to the wrong
    #   place
    if dataset.group_dataset(0x0002) != Dataset():
        raise ValueError(
            f"File Meta Information Group Elements (0002,eeee) should be in "
            f"their own Dataset object in the "
            f"'{dataset.__class__.__name__}.file_meta' attribute."
        )

    # A preamble is required under the DICOM standard, however if
    #   `write_like_original` is True we treat it as optional
    preamble = getattr(dataset, 'preamble', None)
    if preamble and len(preamble) != 128:
        raise ValueError(
            f"'{dataset.__class__.__name__}.preamble' must be 128-bytes long."
        )
    if not preamble and not write_like_original:
        # The default preamble is 128 0x00 bytes.
        preamble = b'\x00' * 128

    # File Meta Information is required under the DICOM standard, however if
    #   `write_like_original` is True we treat it as optional
    if not write_like_original:
        # the checks will be done in write_file_meta_info()
        dataset.fix_meta_info(enforce_standard=False)
    else:
        dataset.ensure_file_meta()

    # Check for decompression, give warnings if inconsistencies
    # If decompressed, then pixel_array is now used instead of PixelData
    if dataset.is_decompressed:
        if dataset.file_meta.TransferSyntaxUID.is_compressed:
            raise ValueError(
                f"The Transfer Syntax UID element in "
                f"'{dataset.__class__.__name__}.file_meta' is compressed "
                f"but the pixel data has been decompressed"
            )

        # Force PixelData to the decompressed version
        dataset.PixelData = dataset.pixel_array.tobytes()
    return dataset
