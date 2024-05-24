import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple, Dict
import json
import pydicom
from pydicom.uid import RLELossless
from pydicom.uid import ExplicitVRLittleEndian
from pydicom.dataset import Dataset
import cv2 as cv
import os

from Pipeline.pipeline import PipelineLungsDX
from Pipeline.configs import *
from test_config import *
from testing_modules import DataSelection, DataMetricsCalculation


class RunPipeline:

    def __init__(self) -> None:
        """
        Pipline test class.
        """
        self.path_to_images = PATH_TO_IMAGES
        self.results_path = RESULTS_PATH
        self.study_type = STUDY_TYPE
        self.lungs_dx = PipelineLungsDX(
            device_id=INFERENCE_DEVICE, study_type=STUDY_TYPE, apply_calibration=APPLY_CALIBRATION, run_fp_16=RUN_FP_16
        )
        self.test_mode = TEST_MODE
        self.save_dicom = SAVE_DICOM
        self.required_pathologies = list()
        if SAVE_PREDICTIONS:
            self.__create_pred_df()
        else:
            self.pred_confidence = None
            self.pred_labels = None
        self.pred_confidence_path = PRED_CONFIDENCE_PATH
        self.issues_images_data = 'wrong_images.json'
        self.wrong_images = dict()

    def run(self, read_series: bool = False) -> None:
        """
        Method for running the pipeline on the calibration dataset.
        :param read_series: parameter indicating whether to read a series of images or images one by one.
        """
        assert type(NUM_OF_SAMPLES) in [type(None), int]
        if any([FILTER_BY_COMMENT, GET_PATHOLOGICAL_STUDIES]):
            studies = DataSelection(study_type=self.study_type).select_data()
            images_paths = self.__read_series(filter_by_study=studies) if read_series else self.__read_dicoms(
                filter_by_study=studies
            )
        else:
            images_paths = self.__read_series() if read_series else self.__read_dicoms()

        timestamp = datetime.now().strftime('%Y.%m.%m %H:%M:%S')
        # Path(os.path.join(self.results_path, timestamp)).mkdir(parents=True, exist_ok=True)

        print_warning = False
        if NUM_OF_SAMPLES is not None:
            stream = tqdm(list(images_paths.keys())[:NUM_OF_SAMPLES])
        else:
            stream = tqdm(list(images_paths.keys()))

        for study in stream:
            dcm_images = list()
            for path in images_paths[study]:
                dcm_images.append(pydicom.dcmread(path))

            warning = self.__pipeline_process(
                dcm_images=dcm_images, paths=images_paths[study], timestamp=timestamp, study=study
            )
            if warning:
                print_warning = True

        if self.wrong_images:
            with open(self.issues_images_data, "w") as f:
                json.dump({'path': self.wrong_images}, f, indent=4)
        if self.pred_confidence is not None:
            Path(self.pred_confidence_path).mkdir(parents=True, exist_ok=True)
            self.pred_confidence.to_csv(os.path.join(self.pred_confidence_path, f'{STUDY_TYPE}_pred.csv'), index=False)
        if self.pred_labels is not None:
            self.pred_labels.to_csv(os.path.join(self.pred_confidence_path, f'{STUDY_TYPE}_pred_label.csv'), index=False)

        if print_warning:
            print('[INFO] Часть изображений не обработалась! Вызовите метод test_issues для отладки!')

    def run_on_image(self, image_name: str) -> None:
        """
        Method for single image pipline test.
        :param image_name: file name of dicom.
        :return: None.
        """
        timestamp = datetime.now().strftime('%Y.%m.%m %H:%M:%S')
        Path(os.path.join(self.results_path, timestamp)).mkdir(parents=True, exist_ok=True)
        image_path = Path(self.path_to_images).rglob(f'{image_name}')
        path = os.path.join(list(image_path)[0])
        dcm_image = pydicom.dcmread(path)
        self.lungs_dx.run(images=[dcm_image])
        for i, image in self.lungs_dx.study_result.processed_images.items():
            if 1 not in self.lungs_dx.study_result.get_all_warnings():

                self.__save_result(
                    image=image, timestamp=timestamp, name=os.path.basename(path), save_path=path,
                    study=self.lungs_dx.study_result.dicoms[0].StudyInstanceUID
                )

    def test_issues(self) -> None:
        """
        Method for running the pipeline on the calibration dataset for debugging.
        """
        with open(self.issues_images_data, "r") as file:
            data = json.load(file)
        timestamp = datetime.now().strftime('%Y.%m.%m %H:%M:%S')
        Path(os.path.join(self.results_path, timestamp)).mkdir(parents=True, exist_ok=True)

        for paths in tqdm(data['path'].values()):
            dcm_images = list()
            for path in paths:
                dcm_images.append(pydicom.dcmread(path))
            self.lungs_dx.run(images=dcm_images)
            for i, image in self.lungs_dx.study_result.processed_images.items():
                if 1 not in self.lungs_dx.study_result.get_all_warnings():
                    self.__save_result(
                        image=image, timestamp=timestamp, name=os.path.basename(paths[i]), save_path=paths[i],
                        study=self.lungs_dx.study_result.dicoms[0].StudyInstanceUID
                    )

    def __pipeline_process(
            self, dcm_images: List, timestamp: str, paths: List, study: str = ''
    ) -> Tuple[list, bool] or list:
        """
        Method for pipeline processing on the study.
        :param dcm_images: list of dicom images corresponding to the study.
        :param timestamp: date and time param.
        :param paths: paths to current study.
        :param study: study id.
        """
        try:
            self.lungs_dx.run(images=dcm_images)
        except Exception as err:
            print(err)
            self.wrong_images[len(self.wrong_images)] = paths
            return True

        projections = list()

        for idx, image in self.lungs_dx.study_result.processed_images.items():
            self.__save_result(
                image=image,
                timestamp=timestamp,
                name=os.path.basename(paths[idx]),
                save_path=paths[idx],
                study=study if READ_SERIES else self.lungs_dx.study_result.dicoms[idx].StudyInstanceUID,
                orig_dicom=self.lungs_dx.study_result.dicoms[idx]
            )
            projections.append(self.lungs_dx.study_result.projections[idx])

        if SAVE_PREDICTIONS and self.lungs_dx.study_result.processed_images.items():
            frontal_filename = os.path.basename(paths[projections.index('frontal')])
            try:
                self.__update_confidence_data(
                    study=study if READ_SERIES else self.lungs_dx.study_result.dicoms[0].StudyInstanceUID,
                    image_data=self.lungs_dx.study_result.kafka_dictionary,
                    filename=frontal_filename
                )
            except Exception as err:
                print('Error during confidence update: ', err)
                pass
        return False

    def __update_confidence_data(self, image_data: dict, study: str, filename: str) -> None:
        """
        Method for metrics updating.
        :param study: study id.
        :param image_data: LungsDXResult class object.
        """
        pred_labels, pred_confidence = self.__get_pred_labels(image_data=image_data)
        study.replace("study_", "")
        pred_labels.insert(0, study)
        pred_labels.append(filename)
        pred_confidence.insert(0, study)
        pred_confidence.append(filename)
        self.pred_confidence.loc[len(self.pred_confidence.index)] = pred_confidence
        self.pred_labels.loc[len(self.pred_labels.index)] = pred_labels

    def __create_pred_df(self) -> None:
        """
        Method for dataframe creation.
        """
        cols = ['study_id']
        self.__get_required_pathologies()
        for name in self.required_pathologies:
            if name in METRIC_NAME_TO_TAG.keys():
                cols.append(METRIC_NAME_TO_TAG[name])
        seen = set()
        seen_add = seen.add
        cols = [x for x in cols if not (x in seen or seen_add(x))]
        cols.append('filename')
        self.pred_confidence = pd.DataFrame(columns=cols)
        self.pred_labels = pd.DataFrame(columns=cols)

    def __get_custom_thresholds(self) -> List[float]:
        """
        Method for list with preset thresholds creation.
        """
        custom_thresholds = FLG_PATHOLOGY_THRESHOLD if self.study_type == 'flg' else RG_PATHOLOGY_THRESHOLD
        thresholds = list()
        for name in self.required_pathologies:
            if name in METRIC_NAME_TO_TAG.keys():
                thresholds.append(custom_thresholds[name][-1])
        return thresholds

    def __get_required_pathologies(self) -> None:
        """
        Method for listing pathologies in the correct order.
        """
        for group_name, group_value in REQUIRED_PATHOLOGIES.items():
            for pathology_name in group_value:
                self.required_pathologies.append(pathology_name)

    def __save_result(
            self,
            image: np.array,
            timestamp: str,
            name: str,
            study: str,
            orig_dicom: Dataset = None,
            save_path: str = '',
    ) -> None:
        """
        Method for saving result of pipeline processing.
        :param image: result image.
        :param timestamp: date and time param.
        :param name: name of image file corresponding to original dicom.
        :param save_path: path, where result file will be saved.
        """
        if self.save_dicom:
            if self.results_path == 'same':
                save_path = os.path.dirname(save_path)
                name = 'processed_' + name
            else:
                save_path = os.path.join(self.results_path, 'output', study)
                # save_path = os.path.join(self.results_path, timestamp, 'dicoms')
            Path(save_path).mkdir(parents=True, exist_ok=True)
            self.__save_dicom(save_path=save_path, name=name, image=image, orig_dicom=orig_dicom)
        else:
            if self.results_path == 'same':
                save_path = os.path.join(os.path.dirname(save_path), name.replace('dcm', 'png'))
            else:
                save_path = os.path.join(self.results_path, timestamp, study, name.replace('dcm', 'png'))
                # save_path = os.path.join(self.results_path, timestamp, name.replace('dcm', 'png'))
                Path(os.path.join(self.results_path, timestamp, study)).mkdir(parents=True, exist_ok=True)
                # Path(os.path.join(self.results_path, timestamp)).mkdir(parents=True, exist_ok=True)
            cv.imwrite(save_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))

    def __read_series(self, filter_by_study: List[str] = None) -> Dict:
        """
        Method for reading images from dicom series.
        """
        study_paths = dict()
        studies_ids = os.listdir(self.path_to_images)
        if filter_by_study:
            print('[WARNING] Study filtration works only with correct dataset storage format (SSS)!')
            studies_ids = [study_id for study_id in studies_ids if study_id.replace('study_', '') in filter_by_study]
        for study_id in studies_ids:
            study_dicom_paths = Path(os.path.join(self.path_to_images, study_id)).rglob('*.dcm')
            study_paths[study_id] = [str(path) for path in study_dicom_paths]
        # if filter_by_study:
        #     study_paths = {
        #         study_id: images_paths for study_id, images_paths in study_paths.copy().items()
        #         if study_id.replace('study_', '') in filter_by_study
        #     }
        return study_paths

    def __read_dicoms(self, filter_by_study: List[str] = None) -> Dict:
        """
        Method for finding all dicom files in all subdirectories of current directory.
        :param filter_by_study: list of studies to be processed.
        :return: list of paths to dicom files.
        """
        if len(USE_PROJECTION) == 2 and SAVE_PREDICTIONS:
            print('[WARNING] Predictions could be saved incorrectly! Use only frontal projection or read full studies!')
        images_paths = Path(self.path_to_images).rglob('*.dcm')
        images_paths = [[str(path)] for path in images_paths]
        if FILTER_BY_DIRECTORY:
            images_paths = [
                path for path in images_paths if any(directory in str(path) for directory in FILTER_BY_DIRECTORY)
            ]
        if filter_by_study:
            images_paths = [
                path for path in images_paths if any(directory in str(path) for directory in filter_by_study)
            ]
        images_paths = {idx: path for idx, path in enumerate(images_paths)}
        return images_paths

    def __get_pred_labels(self, image_data: dict) -> Tuple[list, list]:
        """
        Method for creation predicted labels vector.
        :param image_data: dictionary with pathologies predictions.
        :return: vector with service predictions.
        """
        pred_confidence = list()
        pred_labels = list()
        for pathology_name in self.required_pathologies:
            if pathology_name in METRIC_NAME_TO_TAG.keys():
                if image_data[pathology_name]['pathology_flag'] is not None:
                    pred_confidence.append(image_data[pathology_name]['confidence_level'])
                    pred_labels.append(int(image_data[pathology_name]['pathology_flag']))
                else:
                    pred_confidence.append(-1)
                    pred_labels.append(-1)
        return pred_labels, pred_confidence

    @staticmethod
    def __save_dicom(name: str, image: np.array, orig_dicom: Dataset, save_path: str):
        """
        Method for saving results in dicom files.
        """
        orig_dicom = orig_dicom

        orig_dicom.remove_private_tags()

        orig_dicom.Rows = image.shape[0]
        orig_dicom.Columns = image.shape[1]
        orig_dicom.PhotometricInterpretation = "RGB"
        orig_dicom.SamplesPerPixel = 3
        orig_dicom.BitsStored = 8
        orig_dicom.BitsAllocated = 8
        orig_dicom.HighBit = 7
        orig_dicom.PixelRepresentation = 0
        orig_dicom.PixelData = image.tobytes()
        orig_dicom.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'
        orig_dicom.Modality = 'SC'
        orig_dicom.SeriesDescription = "ThirdOpinion"
        orig_dicom.SeriesInstanceUID = orig_dicom.SeriesInstanceUID
        orig_dicom.ensure_file_meta()
        orig_dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        orig_dicom.fix_meta_info()
        orig_dicom.WindowWidth = 255
        orig_dicom.WindowCenter = 127
        orig_dicom.compress(RLELossless)
        orig_dicom.save_as(os.path.join(save_path, name), write_like_original=False)


def main():
    pipeline = RunPipeline()
    if RUN:
        pipeline.run(read_series=READ_SERIES)
    if RUN_ON_IMAGE:
        pipeline.run_on_image(image_name=IMAGE_NAME)
    if TEST_ISSUES:
        pipeline.test_issues()
    if CALCULATE_METRICS:
        pass


if __name__ == '__main__':
    main()
