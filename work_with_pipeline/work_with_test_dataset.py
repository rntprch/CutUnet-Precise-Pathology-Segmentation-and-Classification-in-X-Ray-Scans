import pydicom
import os
import glob
import shutil
from distutils.dir_util import copy_tree
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import collections
from metrics import CalculateMetrics
import matplotlib.pyplot as plt
import seaborn as sns
import random
from test_config import PATHOLOGIES, METRIC_NAME_TO_TAG, FRACTURE, CONSOLIDATED_FRACTURE


class CreateDataset:
    def __init__(self, save_path: str, path: str, flg_path: str = '', prl_path: str = '') -> None:
        """
        Class for test dataset update.
        :param save_path: path where refactored data would be saved.
        :param path: path to initial data.
        :param flg_path: path to flg annotations data.
        :param prl_path: path to prl annotations data.
        """
        self.output_dir = save_path
        self.path_to_images = path
        self.__get_study_id_from_annotations_data(flg_path=flg_path, prl_path=prl_path)

    def process_data(self, filter_by_directories: List[str] = None) -> None:
        """
        Method for running the pipeline on the calibration dataset.
        :param filter_by_directories: select specific names of directories from which you want to process dicoms.
        """
        images_paths = self.__read_dicoms(filter_by_directory=filter_by_directories)

        for paths in tqdm(images_paths, desc='Updating dataset'):
            for path in paths:
                dcm = pydicom.dcmread(path)
                self.__save_dataset(ds=dcm)

        self.__print_statistics()

    def __print_statistics(self) -> None:
        """
        Method for created dataset statistics printing.
        """
        if self.prl_studies_ids:
            self.__print_statistics_per_modality(modality='prl')

        if self.flg_studies_ids:
            self.__print_statistics_per_modality(modality='flg')

        if os.path.isdir(os.path.join(self.output_dir, 'error')):
            print(
                f"\n[WARNING] Были обнаружены не размеченные исследования: "
                f"{len(os.listdir(os.path.join(self.output_dir, 'error')))} штук."
            )

    def __print_statistics_per_modality(self, modality: str) -> None:
        """
        Method for printing dataset stats per modality.
        :param modality: dataset studies modality type.
        """
        saved_studies = [study.split('_')[-1] for study in os.listdir(os.path.join(self.output_dir, modality))]
        studies_ids = self.flg_studies_ids if modality == 'flg' else self.prl_studies_ids
        data = self.flg_data if modality == 'flg' else self.prl_data
        print(f'\n[INFO] Статистика по {modality}:')
        print(f'Количество размеченных исследований {modality}: {len(set(studies_ids))}')
        print(f"Количество исследований рентгенов в тестовом наборе: {len(saved_studies)}")

        if len(studies_ids) != len(set(studies_ids)):
            print('[WARNING] Обнаружены повторяющиеся исследования в разметке! Их study id:')
            data = data.drop_duplicates(subset=['Имя файла'], keep='first')
            for item, count in collections.Counter(studies_ids).items():
                if count > 1:
                    print(item)

        if abs(len(saved_studies) - len(set(studies_ids))):
            print(f'[WARNING] Отсутствует {abs(len(saved_studies) - len(set(studies_ids)))} размеченных '
                  f'исследований. Не были обнаружены следующие исследования:')
            extra_studies = list()
            for study_id in studies_ids:
                if study_id not in saved_studies:
                    extra_studies.append(study_id)
                    print(study_id)
            data = data[~data['Имя файла'].isin(extra_studies)]
        data.to_csv(os.path.join(self.output_dir, 'processed_df', f'{modality}_gt.csv'), index=False)

    def __get_study_id_from_annotations_data(self, flg_path: str, prl_path: str) -> None:
        """
        Method for studies id from annotations data extraction.
        :param flg_path: path to flg gt labels annotations.
        :param prl_path: path to prl gt labels annotations.
        """
        self.flg_studies_ids, self.flg_data = self.__get_studies_ids(path=flg_path) if flg_path else None
        self.prl_studies_ids, self.prl_data = self.__get_studies_ids(path=prl_path) if prl_path else None
        if not (self.flg_studies_ids and self.prl_studies_ids):
            raise ValueError('No annotations file! Could not get type of data!')

    @staticmethod
    def __get_studies_ids(path: str) -> Tuple[List[str], pd.DataFrame]:
        """
        Method for study id extraction.
        :param path: path to annotations file.
        """
        data = pd.read_csv(path)
        data = data[data['Имя файла'].notna()]
        study_ids = data['Имя файла'].tolist()
        return study_ids, data

    def __read_series(self):
        """
        Method for reading images from dicom series.
        """
        dicom_paths = list()
        directories = os.listdir(self.path_to_images)
        for directory in directories:
            for dir_path, _, files in os.walk(os.path.join(self.path_to_images, directory)):
                dicom_paths.append([os.path.join(dir_path, file) for file in files])
        return dicom_paths

    def __read_dicoms(self, filter_by_directory: List[str]):
        """
        Method for finding all dicom files in all subdirectories of current directory
        :param filter_by_directory:
        :return: list of paths to dicom files.
        """
        images_paths = Path(self.path_to_images).rglob('*.dcm')
        if filter_by_directory:
            images_paths = [
                [str(path)] for path in images_paths if not any(
                    directory in str(path) for directory in filter_by_directory
                )
            ]
        else:
            images_paths = [[str(path)] for path in images_paths]
        return images_paths

    def __save_dataset(self, ds: pydicom.Dataset) -> None:
        """
        Method for correct saving dicoms.
        :param ds: dicom file.
        """
        instance_number = f"_{ds.InstanceNumber}" if "InstanceNumber" in ds else ""
        if ds.StudyInstanceUID in self.prl_studies_ids:
            study_type = 'prl'
        elif ds.StudyInstanceUID in self.flg_studies_ids:
            study_type = 'flg'
        else:
            study_type = 'error'

        filename = os.path.join(
            self.output_dir,
            study_type,
            f"study_{ds.StudyInstanceUID}",
            f'series_{ds.get("Modality", "unknown")}_{ds.SeriesInstanceUID}',
            f"sop{instance_number}_{ds.SOPInstanceUID}.dcm",
        )
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        ds.save_as(filename, write_like_original=False)


class WorkWithData:
    def __init__(
            self,
            data_path: str,
            result_path: str,
            annotations_path: str,
            total_number_of_studies: int,
            pathology_test_weights: List[float],
            class_names: List[str],
            priority_class=None,
            pos_class_proportion: float = 0.5,
    ) -> None:
        """
        Class for processed data selection.
        :param data_path: path, where results of split will be saved.
        :param result_path: path, where results of selection will be saved.
        :param annotations_path: path, where results of split will be saved.
        :param total_number_of_studies: total number of studies in the test subset.
        :param class_names: pathology classes names.
        :param priority_class: main class of data distribution.
        :param pathology_test_weights: the ratio of the number of objects for each pathology class.
        :param pos_class_proportion: the ratio of the number of objects of positive and negative class.
        """
        self.markings = pd.read_csv(annotations_path)
        self.data_path = data_path
        self.save_path = result_path
        self.class_names = class_names
        self.pathology_weights = pathology_test_weights
        self.total_number_of_studies = total_number_of_studies
        self.pos_class_proportion = pos_class_proportion
        self.priority_class = priority_class

        # self.conf_df = pd.read_csv('/home/nikita27/LungsDX/Research/Data/pipline/prediction/flg_pred_conf.csv',
        #                            index_col=0)

    def select_studies(self) -> None:
        """
        Method for pathological studies selection.
        """
        data = self.__get_dist_data()
        self.__print_statistics(data=data)
        self.__copy_data(data=data)

    def __get_dist_data(self, group=None) -> pd.DataFrame:
        """
        :param group:
        return: dataset with test data.
        """
        cls_dist = self.__get_pathology_classes_distribution()
        samples = []
        pathology_data = self.markings.copy()
        while any(v > 0 for val in cls_dist.values() for v in val):
            maximum = 0
            target_cls = None
            target_binary = None
            random_cls = self.class_names.copy()
            random.shuffle(random_cls)
            for pathology_class in random_cls:
                binaries = [0, 1]
                random.shuffle(binaries)
                for i in binaries:
                    if cls_dist[pathology_class][i] > maximum:
                        maximum = cls_dist[pathology_class][i]
                        target_cls = pathology_class
                        target_binary = i

            df_cls = None
            if self.priority_class is not None:
                df_cls = pathology_data[
                    (pathology_data[target_cls] == target_binary) & (pathology_data[self.priority_class] == 1.0)
                ]

            if df_cls is None or not df_cls.shape[0]:
                df_cls = pathology_data[pathology_data[target_cls] == target_binary]
            assert df_cls.shape[0], f'Could not found enough images for {target_cls, target_binary}'
            sample = df_cls.sample(1)
            pathology_data = pathology_data.drop(sample.index)
            samples.append(sample)
            for pathology_class in cls_dist:
                cls_dist[pathology_class][int(sample[pathology_class].iloc[0])] -= 1

            if group is not None:
                group_samples = pathology_data[pathology_data[group] == sample[group].iloc[0]]
                pathology_data = pathology_data.drop(group_samples.index)
                for pathology_class in cls_dist:
                    for i in range(group_samples.shape[0]):
                        cls_dist[pathology_class][int(group_samples[pathology_class].iloc[i])] -= 1
                samples.append(group_samples)

        return pd.concat(samples).reset_index(drop=True)

    def __get_pathology_classes_distribution(self) -> dict:
        """
        Method for calculating number of positive and negative samples per class according to proportion to those given
        in the configuration set.
        :return  pathology class distribution dictionary.
        """
        pathologies_distribution = dict()
        for i, pathology in enumerate(self.class_names):
            positive_num = self.pathology_weights[i] * self.pos_class_proportion * self.total_number_of_studies
            negative_num = self.pathology_weights[i] * self.total_number_of_studies - positive_num
            if self.markings[pathology].sum() < positive_num:
                positive_num = self.markings[pathology].sum() * int(self.total_number_of_studies / len(self.markings))
                print(f'[WARNING] Not enough positive studies for pathology class {pathology}!\n'
                      f'Set {positive_num} studies for test set!')
            if len(self.markings[pathology]) - positive_num < negative_num:
                negative_num = (len(self.markings[pathology]) - positive_num) * \
                               int(self.total_number_of_studies / len(self.markings))
                print(f'[WARNING] Not enough negative studies for pathology class {pathology}!\n'
                      f'Set {negative_num} studies for test set!')
            pathologies_distribution[pathology] = [positive_num, negative_num]
        return pathologies_distribution

    def predict_data_stats(self):
        """
        Method for pathological studies stats visualisation.
        """
        labels_data = self.markings.copy().drop(['Имя файла', 'Комментарии'], axis=1, inplace=False)
        labels_data = labels_data.dropna(inplace=False)
        numer_of_studies = self.markings['Имя файла'].count()

        data = dict()
        data['pathologies'] = list()
        data['number of pathological'] = list()
        for pathology in labels_data:
            data['pathologies'].append(pathology)
            data['number of pathological'].append(int(labels_data[pathology].sum()))

        sns.set(rc={'figure.figsize': (len(data['pathologies']) * 2.75, 5)})
        sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
        hist = sns.barplot(
            data=data, x='pathologies', y='number of pathological', hue='number of pathological',
            saturation=0.57, dodge=False, palette=sns.color_palette("Spectral", n_colors=len(data['pathologies']))
        )
        hist.axes.set_title(
            f'Number of pathological images per pathology class (total number of studies: {numer_of_studies})',
            fontsize=16
        )
        hist.set_xlabel("Pathologies", fontsize=14)
        hist.set_ylabel("Number of pathological images", fontsize=14)

        fig = hist.get_figure()
        fig.savefig(os.path.join(self.save_path, 'prl_pathologies_dist.png'))
        plt.close()

        self.__pathological_studies()
        self.__pathological_per_group()

    def __pathological_studies(self) -> None:
        labels_data = self.markings.copy().drop(['Имя файла', 'Комментарии'], axis=1, inplace=False)
        labels_data = labels_data.dropna(inplace=False)
        numer_of_studies = self.markings['Имя файла'].count()

        labels_data['pathological studies'] = labels_data.iloc[:, :].sum(axis=1).astype(bool).astype(int)
        num_of_pathological = labels_data['pathological studies'].sum()

        print(f'pathological studies {num_of_pathological}, {num_of_pathological / numer_of_studies * 100} %')
        print(f'non pathological studies {numer_of_studies - num_of_pathological}, '
              f'{(numer_of_studies - num_of_pathological) / numer_of_studies * 100}%')

    def __pathological_per_group(self):
        labels_data = self.markings.copy().drop(['Имя файла', 'Комментарии'], axis=1, inplace=False)
        labels_data = labels_data.dropna(inplace=False)
        numer_of_studies = self.markings['Имя файла'].count()
        data = pd.DataFrame()

        for group_name, pathology_group in PATHOLOGIES.items():
            pathologies = [
                'Переломы' if pathology in [FRACTURE, CONSOLIDATED_FRACTURE]
                else METRIC_NAME_TO_TAG[pathology] for pathology in pathology_group
            ]
            data[group_name] = labels_data[pathologies].sum(axis=1).astype(bool).astype(int)

            num_of_pathological = data[group_name].sum()

            print(f'pathological studies for group {group_name}: {num_of_pathological}, {num_of_pathological / numer_of_studies * 100} %')
            print(f'non pathological studies for group {group_name}: {numer_of_studies - num_of_pathological}, '
                  f'{(numer_of_studies - num_of_pathological) / numer_of_studies * 100} %')



    def __copy_data(self, data: pd.DataFrame) -> None:
        for study in tqdm(data['Имя файла'].tolist(), desc='Data copying'):
            copy_tree(
                src=os.path.join(self.data_path, f'study_{study}'),
                dst=os.path.join(self.save_path)
            )

    def __print_statistics(self, data: pd.DataFrame) -> None:
        """
        Method for pathology classes distribution statistics print.
        :param data: pandas dataframe with pathology classification data.
        """
        for pathology_name in data.columns:
            if pathology_name in self.class_names:
                print(f'Collected studies for {pathology_name}:\n{data[pathology_name].value_counts()}')

    @staticmethod
    def __zip_data(result_path: str) -> None:
        """
        Method for data compression.
        :param result_path: path where compressed data will be stored.
        """
        shutil.make_archive(
            base_name=os.path.join(result_path, 'visualize_test_archive'),
            format='zip',
            root_dir=os.path.join(result_path, 'visualize_test')
        )


class WorkWithDataset:

    def __init__(self, path_to_data: str, study_type: str):
        assert study_type in ['flg', 'prl']
        self.gt_labels = pd.read_csv(os.path.join(path_to_data, f'{study_type}_gt.csv'))
        self.pred_labels = pd.read_csv(os.path.join(path_to_data, f'{study_type}_pred.csv'), index_col=0)
        self.gt_labels = self.gt_labels[self.gt_labels['Имя файла'].notna()]
        self.pred_labels = self.pred_labels[self.pred_labels['Имя файла'].notna()]
        self.path_to_data = path_to_data
        self.study_type = study_type

    def create_subset(self, required_pathologies: List[str], save_params: Dict) -> None:
        """
        Method for test dataset subset creation? based on saving parameters condition.
        :param required_pathologies: list of pathologies names.
        :param save_params: dictionary with saving configuration.
        """
        pathologies_names = list(self.gt_labels.columns.values)[1:]
        cols = ['Имя файла', 'Комментарии']
        for pathology_name in required_pathologies:
            if pathology_name in pathologies_names:
                cols.append(pathology_name)
            else:
                print(f'[WARNING] Could not find pathology f{pathology_name} annotations! Drop this pathology.')

        gt_labels = self.gt_labels[cols]
        pred_labels = self.pred_labels[cols]

        if save_params['pathologic_gt']:
            for _, gt in gt_labels.iterrows():
                study = gt.iloc[0]
                class_labels = gt.iloc[2:].to_numpy().astype(float)
                if class_labels.any():
                    copy_tree(
                        src=os.path.join(self.path_to_data, self.study_type, f'study_{study}'),
                        dst=os.path.join(self.path_to_data, f'pathologic_gt_{self.study_type}', )
                    )
            print(
                f"[INFO] Created subset with "
                f"{len(os.listdir(os.path.join(self.path_to_data, f'pathologic_gt_{self.study_type}')))} "
                f"pathological studies."
            )

        if save_params['non_pathologic_gt']:
            for _, gt in gt_labels.iterrows():
                study = gt.iloc[0]
                class_labels = gt.iloc[1:-1].to_numpy().astype(float)
                if not class_labels.any():
                    copy_tree(
                        src=os.path.join(self.path_to_data, self.study_type, f'study_{study}'),
                        dst=os.path.join(self.path_to_data, f'non_pathologic_gt_{self.study_type}')
                    )
        # if save_params['fp']:
        #     for _, gt in gt_labels.iterrows():
        #         study = gt.iloc[0]
        #         class_labels = gt.iloc[1:-1].to_numpy().astype(float)
        #         if class_labels.any():
        #             copy_tree(
        #                 src=os.path.join(self.path_to_data, self.study_type, f'study_{study}'),
        #                 dst=os.path.join(self.path_to_data, 'fp', f'study_{study}')
        #             )
        # if save_params['fn']:
        #     for _, gt in gt_labels.iterrows():
        #         study = gt.iloc[0]
        #         class_labels = gt.iloc[1:-1].to_numpy().astype(float)
        #         if class_labels.any():
        #             copy_tree(
        #                 src=os.path.join(self.path_to_data, self.study_type, f'study_{study}'),
        #                 dst=os.path.join(self.path_to_data, 'fn', f'study_{study}')
        #             )
        # if save_params['tp']:
        #     for _, gt in gt_labels.iterrows():
        #         study = gt.iloc[0]
        #         class_labels = gt.iloc[1:-1].to_numpy().astype(float)
        #         if class_labels.any():
        #             copy_tree(
        #                 src=os.path.join(self.path_to_data, self.study_type, f'study_{study}'),
        #                 dst=os.path.join(self.path_to_data, 'tp', f'study_{study}')
        #             )
        # if save_params['tn']:
        #     for _, gt in gt_labels.iterrows():
        #         study = gt.iloc[0]
        #         class_labels = gt.iloc[1:-1].to_numpy().astype(float)
        #         pred = pred_labels.loc[pred_labels['Имя файла'] == gt['Имя файла']].iloc[:, 1:-1]
        #         pred = pred.to_numpy().astype(float)
        #         if class_labels.any():
        #             copy_tree(
        #                 src=os.path.join(self.path_to_data, self.study_type, f'study_{study}'),
        #                 dst=os.path.join(self.path_to_data, 'tn', f'study_{study}')
        #             )

    def calculate_metrics(self, start_idx: int = None, end_idx: int = None, set_thresholds: bool = False) -> None:
        """
        Function for calculation metrics on test dataset.
        :param start_idx:
        :param end_idx:
        :param set_thresholds:
        """
        gt_labels = self.gt_labels.drop(['Комментарии'], axis=1, inplace=False)
        if 'Комментарии' in self.pred_labels.columns:
            pred_labels = self.pred_labels.drop(['Комментарии'], axis=1, inplace=False)
        else:
            pred_labels = self.pred_labels
        gt_labels = gt_labels.dropna(inplace=False)
        pred_labels = pred_labels.dropna(inplace=False)
        pred_labels = pred_labels.drop_duplicates(subset=['Имя файла'], inplace=False)
        gt_labels = gt_labels.drop_duplicates(subset=['Имя файла'], inplace=False)
        class_names = list(gt_labels.columns.values)[1:]
        num_classes = len(class_names)
        metrics = CalculateMetrics(num_classes=num_classes)

        if  start_idx is not None:
            gt_labels = gt_labels[start_idx:]
        if  end_idx is not None:
            gt_labels = gt_labels[:end_idx]

        print(gt_labels.astype(bool).sum(axis=0))

        for _, gt in gt_labels.iterrows():
            if gt['Имя файла'] in pred_labels['Имя файла'].tolist():
                pred = pred_labels.loc[pred_labels['Имя файла'] == gt['Имя файла']]
                pred = pred.iloc[:, 1:]
                pred = pred.to_numpy().astype(float)
                gt = np.reshape(gt.iloc[1:].to_numpy().astype(float), pred.shape)
                metrics.update(y_pred=pred, y_true=gt)

        metrics.compute(set_thresholds=[0.5] * num_classes) if set_thresholds else metrics.compute()

        metrics.plot_metrics(
            save_path=os.path.join(self.path_to_data, f'{self.study_type}_metrics'),
            per_class=True,
            class_names=class_names
        )


if __name__ == '__main__':
    # WorkWithData(
    #     data_path='/home/nikita27',
    #     annotations_path='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/processed_df/flg_gt.csv',
    #     result_path='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/distribution',
    #     class_names=['Снижение пневматизации', 'Линейные затемнения', 'Ограниченные затемнения'],
    #     total_number_of_studies = 200,
    #     pathology_test_weights=[1., 1. , 1.]
    # ).predict_data_stats()
    CreateDataset(
        path='/home/nikita27/LungsDX/Research/Data/pipline/input_КТ_Ndl ',
        save_path='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset',
        flg_path='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/processed_df/flg_gt.csv',
        prl_path='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/processed_df/prl_gt.csv'
    ).process_data()
    # WorkWithDataset(
    #     path_to_data='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/pred_conf', study_type='prl'
    # ).calculate_metrics()