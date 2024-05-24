__all__ = ["Pipeline"]

from datetime import datetime, timezone
from typing import List, Optional, Union, cast

from loguru import logger
from Pipeline.configs.service import WARNING_FLAGS
from Pipeline.pipeline import PipelineLungsDX, StudyResult
from pydicom import FileDataset
from pydicom.uid import UID, ImplicitVRLittleEndian

from app.prl_pipeline import dicom_builder, errors
from app.prl_pipeline._filtration import fix_modality_tag, has_required_tags, is_modality_valid, is_sop_class_valid
from app.prl_pipeline.models import (
    DXParams,
    DXResult,
    ErrorReason,
    OperatingMode,
    PipelineContext,
    PipelineFail,
    PipelineSuccess,
)
from app.settings.config import Modality


class Pipeline:
    def __init__(self, gpu_id: Optional[int], study_type: Modality, service_version: str):
        gpu_id = cast(int, gpu_id)  # fix mypy, in fact gpu_id is Optional
        self.pipeline = PipelineLungsDX(version=service_version, study_type=study_type.value, device_id=gpu_id)

    def process(
        self,
        datasets: List[FileDataset],
        operating_mode: OperatingMode,
        tz: timezone,
        full_software_version: str,
        public_software_version: str,
        output_ts: UID = ImplicitVRLittleEndian,
        model_id: int = 1003,
    ) -> Union[PipelineSuccess, PipelineFail]:
        ctx = PipelineContext(
            operating_mode=operating_mode,
            tz=tz,
            full_software_version=full_software_version,
            public_software_version=public_software_version,
            output_ts=output_ts,
            model_id=model_id,
            analysis_dt=datetime.now(tz),
            src=datasets[0],
            processing_started_at=datetime.now(tz),
            processing_ended_at=None,
        )
        for ds in datasets:
            fix_modality_tag(ds)

        self.pipeline.run(images=datasets)
        if self.pipeline.study_result.processed_images:
            for src, image, projection in self.pipeline.study_result.get_processed_data():
                ctx = ctx._replace(
                    src=src,
                    analysis_dt=datetime.now(tz),
                    processing_ended_at=datetime.now(tz),
                    metadata=self.pipeline.study_result.kafka_dictionary,
                )

                logger.debug("Build SC")
                sc_ds = dicom_builder.build_secondary_capture(ctx=ctx, img=image)

            logger.debug("Build SR")
            sr_ds = dicom_builder.build_structured_report(ctx, sc_ds.SOPInstanceUID)

            dx_result = DXResult.parse_obj(self.pipeline.study_result.kafka_dictionary)

            return PipelineSuccess(
                ctx=ctx,
                structured_report=sr_ds,
                secondary_capture_series=[sc_ds],
                pathology_flag=dx_result.pathology_flag,
                confidence_level=dx_result.confidence_level,
                dx=DXParams(
                    dx_conf_level=dx_result.confidence_level,
                    dx_conf_level_pleural=dx_result.hemithorax.plef.confidence_level,
                    dx_conf_level_pneumothorax=dx_result.hemithorax.pntx.confidence_level,
                    dx_conf_level_blackout=dx_result.opacity.ndl.confidence_level,
                    dx_conf_level_infiltration=dx_result.opacity.sp.confidence_level,
                    dx_conf_level_lin=dx_result.opacity.lin.confidence_level,
                    dx_conf_level_diaphm=dx_result.hemithorax.diaphm.confidence_level,
                    dx_conf_level_cavity=dx_result.opacity.cavity.confidence_level,
                    dx_conf_level_cardiomegaly=dx_result.heart.ctr.confidence_level,
                ),
                dx_result=dx_result,
            )
        else:
            return self.handle_failure(datasets, study_result=self.pipeline.study_result)

    @staticmethod
    def handle_failure(datasets: List[FileDataset], study_result: StudyResult) -> PipelineFail:
        datasets = [ds for ds in datasets if has_required_tags(ds)]
        if not datasets:
            return errors.TAG_ERROR

        datasets = [ds for ds in datasets if is_sop_class_valid(ds)]
        if not datasets:
            return errors.SOP_CLASS_ERROR

        datasets = [ds for ds in datasets if is_modality_valid(ds)]
        if not datasets:
            return errors.MODALITY_ERROR

        warnings = study_result.get_all_warnings()

        map_pipeline_errors = {
            ErrorReason.IMAGES_ERROR: {1, 3, 7},
            ErrorReason.ANATOMIC_REGION_ERROR: {2},
            ErrorReason.PROJECTION_ERROR: {4},
        }
        for error, pipeline_errors in map_pipeline_errors.items():
            if pipeline_errors & warnings:
                descriptions = [WARNING_FLAGS[i].replace("\n", " ") for i in warnings]
                return PipelineFail(
                    error,
                    " ".join(descriptions),
                )

        return errors.OTHER_ERROR
