from test_config import *
from Pipeline.configs import *
from typing import List
import pandas as pd
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import re
from tqdm import tqdm
import shutil

from metrics import CalculateMetrics


class DataSelection:

    def __init__(self, study_type: str):
        self.required_pathologies = list()
        self.filter_by_comment = FILTER_BY_COMMENT
        self.pred_confidence = None
        self.pred_labels = None
        self.get_pathological = GET_PATHOLOGICAL_STUDIES
        self.gt_labels = None
        self.study_type = study_type

    def select_data(self) -> List[str]:
        """
        Method for data selection.
        :return list of study ids selected by condition.
        """
        # studies = [str(filename) for filename in self.gt_labels['Имя файла'].tolist()]
        if GET_TP:
            return self._select_tp()
        if GET_TN:
            return self._select_tn()
        if GET_FP:
            return self._select_fp()
        if GET_FN:
            return self._select_fn()
        if GET_PATHOLOGICAL_STUDIES:
            return self._get_pathological()
        if FILTER_BY_COMMENT:
            return self._get_with_comment()

    def _select_tp(self) -> List[str]:
        pass

    def _select_tn(self) -> List[str]:
        pass

    def _select_fp(self) -> List[str]:
        pass

    def _select_fn(self) -> List[str]:
        pass

    def _get_pathological(self) -> List[str]:
        pass

    def _get_non_pathological(self) -> List[str]:
        pass

    def _get_with_comment(self) -> List[str]:
        pass

    def __get_gt_labels(self, path: str, filter_by_name: str = None) -> None:
        """
        Method for ground truth labels loading.
        :param path: path to directory with markings dataframe.
        """
        pattern = 'gt.csv' if filter_by_name is None else filter_by_name + '_gt.csv'
        self.gt_labels = None
        self.pred_confidence = None
        if path:
            files = os.listdir(path)
            self.__get_required_pathologies()
            for file in files:
                if (pattern in file) and (self.study_type in file):
                    cols = self.__get_gt_labels_from_csv(gt_label_path=os.path.join(path, file))
                    self.pred_confidence = pd.DataFrame(columns=cols)
                    self.pred_labels = pd.DataFrame(columns=cols)

    def __get_gt_labels_from_csv(self, gt_label_path: str) -> List[str]:
        """
        Method for ground truth labels loading.
        :param gt_label_path: path to markings dataframe.
        """
        gt_labels = pd.read_csv(gt_label_path)
        if 'Комментарии' in gt_labels.columns:
            gt_labels = gt_labels.drop(['Комментарии'], axis=1, inplace=False)
        if self.filter_by_comment:
            pred_label_path = gt_label_path.replace('gt', 'pred')
            labels = pd.read_csv(pred_label_path)
            labels = labels[labels['Комментарии'].str.contains(self.filter_by_comment) == True]
            gt_labels = gt_labels[gt_labels['Имя файла'].isin(labels['Имя файла'])]
        elif self.get_pathological:
            gt_labels = gt_labels[gt_labels[self.get_pathological] == 1]

        cols = ['Имя файла']
        for name in self.required_pathologies:
            if name in METRIC_NAME_TO_TAG.keys():
                cols.append(METRIC_NAME_TO_TAG[name])
        seen = set()
        seen_add = seen.add
        cols = [x for x in cols if not (x in seen or seen_add(x))]

        gt_cols = cols.copy()
        gt_cols.remove('Неконсолидированные переломы')
        gt_cols.remove('Консолидированные переломы')
        gt_cols.append('Переломы')

        if self.gt_labels is not None:
            self.gt_labels = pd.concat([self.gt_labels, gt_labels[gt_cols]], ignore_index=True, sort=False)
        else:
            self.gt_labels = gt_labels.reindex(columns=gt_cols)
        return cols

    def __get_required_pathologies(self) -> None:
        """
        Method for listing pathologies in the correct order.
        """
        self.required_pathologies = list()
        for group_name, group_value in REQUIRED_PATHOLOGIES.items():
            for pathology_name in group_value:
                self.required_pathologies.append(pathology_name)


class WorkWithPredictions:

    def __init__(self, predictions_path: str, study_path: str, save_path: str) -> None:
        """
        :param predictions_path:
        """
        self.__get_predictions(path=predictions_path)
        self.study_path = study_path
        self.save_path = save_path

    def copy_selected_studies(
            self, pathology_name: str, is_pathologic: bool, by_confidence: float = None, select_num: int = None
    ) -> None:
        """
        Method for studies selection.
        :param pathology_name:
        :param is_pathologic:
        :param by_confidence:
        :param select_num:
        """
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        selected_studies = self.select_studies(
            pathology_name=pathology_name, is_pathologic=is_pathologic, by_confidence=by_confidence
        )
        studies_paths = [str(path) for path in Path(self.study_path).rglob('study_*')]
        selected_studies = selected_studies[:select_num] if len(selected_studies) > select_num else selected_studies
        for study in tqdm(selected_studies, desc='Coping selected studies'):
            try:
                path = [x for x in studies_paths if re.search(study, x)][0]
                shutil.copytree(src=path, dst=os.path.join(self.save_path, study))
            except IndexError:
                pass

    def select_studies(self, pathology_name: str, is_pathologic: bool, by_confidence: float = None) -> List[str]:
        if is_pathologic:
            data = self.labels[self.labels[pathology_name] == 1]
        else:
            data = self.labels[self.labels[pathology_name] == 1]
        return [str(filename) for filename in data['study_id'].tolist()]


    def __get_predictions(self, path: str) -> None:
        """
        Method for pipeline predicted data loading.
        :param path:
        """
        self.labels = None
        self.pred_conf = None
        file_paths = [str(path) for path in Path(path).rglob('*.csv')]
        for file in tqdm(file_paths):
            if 'label' in file:
                labels = pd.read_csv(file)
                if self.labels is not None:
                    self.gt_labels = pd.concat([self.labels, labels], ignore_index=True, sort=False)
                else:
                    self.gt_labels = labels
            else:
                pred_conf = pd.read_csv(file)
                if self.pred_conf is not None:
                    self.pred_conf = pd.concat([self.pred_conf, pred_conf], ignore_index=True, sort=False)
                else:
                    self.pred_conf = pred_conf


class DataMetricsCalculation:

    def __init__(self, results_path: str, pred_confidence_path: str, study_type: str):
        self.results_path = results_path
        self.study_type = study_type
        self.pred_confidence = pd.read_csv(os.path.join(pred_confidence_path, 'prl_pred.csv'))
        self.__get_required_pathologies()

    def calculate(self, save_path: str = '') -> None:
        """
        Method for metrics calculation.
        :param save_path: path, where metrics plot will be saved.
        """
        if not save_path:
            save_path = os.path.join(self.results_path, datetime.now().strftime('%Y.%m.%m %H:%M:%S'))

        metrics = CalculateMetrics(num_classes=len(list(self.gt_labels.columns.values)[1:]))

        self.pred_confidence = self.pred_confidence.dropna(inplace=False)
        self.gt_labels = self.gt_labels.dropna(inplace=False)
        self.pred_confidence['Переломы'] = self.pred_confidence[
            ['Неконсолидированные переломы', 'Консолидированные переломы']
        ].max(axis=1)
        self.pred_confidence = self.pred_confidence.drop(['Неконсолидированные переломы'], axis=1, inplace=False)
        self.pred_confidence = self.pred_confidence.drop(['Консолидированные переломы'], axis=1, inplace=False)
        self.gt_labels = self.gt_labels.drop_duplicates(subset=['Имя файла'], inplace=False)
        #
        if len(self.pred_confidence.columns) > len(self.gt_labels.columns):
            self.pred_confidence = self.pred_confidence[self.gt_labels.columns].copy()
        else:
            self.gt_labels = self.gt_labels[self.pred_confidence.columns].copy()

        for _, pred in self.pred_confidence.iterrows():
            if pred['Имя файла'] in self.gt_labels['Имя файла'].tolist():
                gt = self.gt_labels.loc[self.gt_labels['Имя файла'] == pred['Имя файла']]
                gt = gt.iloc[:, 1:]
                gt = gt.to_numpy().astype(float)
                pred = np.reshape(pred.iloc[1:].to_numpy().astype(float), gt.shape)
                metrics.update(y_pred=pred, y_true=gt)

        metrics.compute(set_thresholds=self.__get_custom_thresholds()) if USE_CUSTOM_THRESHOLDS else metrics.compute()
        class_names = list(self.gt_labels.columns.values)[1:]
        metrics.plot_metrics(save_path=save_path, per_class=True, class_names=class_names)

        # for _, pred in self.pred_confidence.iterrows():
        #     if pred['Имя файла'] in self.gt_labels['Имя файла'].tolist():
        #         gt = self.gt_labels.loc[self.gt_labels['Имя файла'] == pred['Имя файла']]
        #         gt = gt.iloc[:, 1:]
        #         gt = gt.to_numpy().astype(float)
        #         gt = np.array([[float(gt.any())]])
        #         pred = np.array([[np.max(pred.iloc[1:])]])
        #         metrics.update(y_pred=pred, y_true=gt)
        #
        # metrics.compute(set_thresholds=self.__get_custom_thresholds()) if USE_CUSTOM_THRESHOLDS else metrics.compute()
        # class_names = list(self.gt_labels.columns.values)[1:]
        # metrics.plot_metrics(save_path=save_path, per_class=True, class_names=class_names)

        # thresholds = self.__get_custom_thresholds()
        # pathologies = list(self.pred_confidence.columns)
        # pathologies.remove('Имя файла')
        # for i, column_name in enumerate(pathologies):
        #     threshold = thresholds[i]
        #     if column_name != 'Имя файла':
        #         self.pred_confidence[column_name][self.pred_confidence[column_name] >= threshold] = 1
        #         self.pred_confidence[column_name][self.pred_confidence[column_name] < threshold] = 0
        #
        # self.pred_confidence.drop_duplicates('Имя файла', inplace=True)
        # self.gt_labels.drop_duplicates('Имя файла', inplace=True)
        # df = pd.merge(self.pred_confidence, self.gt_labels, how='outer', indicator=True, suffixes=('_pred', '_vitaliy'))
        # df.to_csv('pred_labels.csv')

    def __get_gt_labels(self, path: str, filter_by_name: str = None) -> None:
        """
        Method for ground truth labels loading.
        :param path: path to directory with markings dataframe.
        """
        pattern = 'gt.csv' if filter_by_name is None else filter_by_name + '_gt.csv'
        self.gt_labels = None
        self.pred_confidence = None
        if path:
            files = os.listdir(path)
            for file in files:
                if (pattern in file) and (self.study_type in file):
                    cols = self.__get_gt_labels_from_csv(gt_label_path=os.path.join(path, file))
                    self.pred_confidence = pd.DataFrame(columns=cols)
                    self.pred_labels = pd.DataFrame(columns=cols)

    def __get_gt_labels_from_csv(self, gt_label_path: str) -> List[str]:
        """
        Method for ground truth labels loading.
        :param gt_label_path: path to markings dataframe.
        """
        gt_labels = pd.read_csv(gt_label_path)
        if 'Комментарии' in gt_labels.columns:
            gt_labels = gt_labels.drop(['Комментарии'], axis=1, inplace=False)
        cols = ['Имя файла']
        for name in self.required_pathologies:
            if name in METRIC_NAME_TO_TAG.keys():
                cols.append(METRIC_NAME_TO_TAG[name])
        seen = set()
        seen_add = seen.add
        cols = [x for x in cols if not (x in seen or seen_add(x))]

        gt_cols = cols.copy()
        gt_cols.remove('Неконсолидированные переломы')
        gt_cols.remove('Консолидированные переломы')
        gt_cols.append('Переломы')

        if self.gt_labels is not None:
            self.gt_labels = pd.concat([self.gt_labels, gt_labels[gt_cols]], ignore_index=True, sort=False)
        else:
            self.gt_labels = gt_labels.reindex(columns=gt_cols)
        return cols

    def __get_required_pathologies(self) -> None:
        """
        Method for listing pathologies in the correct order.
        """
        self.required_pathologies = list()
        for group_name, group_value in REQUIRED_PATHOLOGIES.items():
            for pathology_name in group_value:
                self.required_pathologies.append(pathology_name)

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


if __name__ == '__main__':
    worker = WorkWithPredictions(predictions_path='', study_path='', save_path='')
    worker.copy_selected_studies(pathology_name='Ограниченные затемнения', is_pathologic=True, select_num=200)