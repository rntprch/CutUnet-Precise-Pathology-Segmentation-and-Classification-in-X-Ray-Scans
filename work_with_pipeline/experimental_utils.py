import pydicom
import os
import glob
import shutil
from distutils.dir_util import copy_tree
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import collections
from Pipeline.metrics import BalancedAccuracy, CalculateMetrics
from Pipeline.configs.test import *
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime





def series_to_studies(dataset_path: str) -> None:
    """
    Method for updating final dataset.
    :param: dataset path: path to final dataset.
    """
    all_studies = os.listdir(dataset_path)
    for study in all_studies:
        if 'series' in os.path.basename(study):
            new_name = 'study_' + os.path.basename(study).split('_')[-1]
            os.rename(src=os.path.join(dataset_path, study), dst=os.path.join(dataset_path, new_name))


def mediastinal_shift_stats(path: str):
    data = pd.read_csv(path).drop(['study', 'name'], axis=1, inplace=False)
    data = data.drop(data[data['right_shift'] == 0].index)
    right_threshold = 0.26
    left_threshold = 0.08
    threshold = 0.1
    new_threshold = 0.05
    # threshold = 0.03
    # coefficient1 = 1
    # coefficient1 = 1 / (data['left_shift'] + data['right_shift'])
    # coefficient2 = (1 / data['right_shift'])
    # coefficient1 = 1
    # coefficient2 = 1
    # coefficient = 1
    # data['right_shift'] *= coefficient1
    # data['left_shift'] *= coefficient1
    # data['shift'] = data['right_shift'] - data['left_shift']

    data['label'] = np.where(
        (data['left_shift'] <= left_threshold) | (threshold <= data['shift']) |
        (data['right_shift'] >= right_threshold) | (data['shift'] <= new_threshold), 1, 0
    )
    data['difference'] = np.abs(data['label'] - data['gt_label'])
    acc = BalancedAccuracy().compute_per_class(y=data['label'].to_numpy(), y_hat=data['gt_label'].to_numpy())[0]
    print(f'Balanced accuracy: {round(acc, 3)}')
    data = data.iloc[:, 1:]
    pd.set_option('display.max_columns', None)
    print(data.loc[(data['difference'] != 0) & (data['gt_label'] == 1)].head())
    print(data.loc[(data['difference'] != 0) & (data['gt_label'] == 0)].head())


def delete_sr(path: str):
    images_paths = glob.glob(path + '/*/*', recursive=True)
    for path in images_paths:
        if 'series_SR' in path:
            shutil.rmtree(path=path)


def cavity_data_create(path_to_images: str, annotations_path: str, save_path: str):
    gt_labels = pd.read_csv(annotations_path)
    for filename in tqdm(gt_labels['filename'].tolist()):
        gt = gt_labels.loc[gt_labels['filename'] == filename]['Воздушная полость'].to_numpy()
        if gt.any():
            filename = filename.split('.')[0]
            images_paths = Path(path_to_images).rglob(f'{filename}.dcm')
            images_paths = [[str(path)] for path in images_paths]
            if images_paths:
                image_path = images_paths[0][0]
                shutil.copyfile(src=image_path, dst=os.path.join(save_path, f'{filename}.dcm'))


def cavity_data_transform(path_to_images: str, annotations_path: str, save_path: str):
    images_paths = Path(path_to_images).rglob('*.dcm')
    images_paths = [str(path) for path in images_paths]
    new_data = {'study_id': list(), 'old_name': list()}
    for path in tqdm(images_paths):
        ds = pydicom.dcmread(path)
        instance_number = f"_{ds.InstanceNumber}" if "InstanceNumber" in ds else ""
        # if '.' not in ds.SeriesInstanceUID:
        #     study_id = ds.SeriesInstanceUID.copy()

        file_path = os.path.join(
            save_path,
            f"study_{ds.StudyInstanceUID}",
            f'series_{ds.get("Modality", "unknown")}_{ds.SeriesInstanceUID}',
            f"sop{instance_number}_{ds.SOPInstanceUID}.dcm",
        )
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        ds.save_as(file_path, write_like_original=False)
        new_data['study_id'].append(str(ds.StudyInstanceUID))
        new_data['old_name'].append(os.path.basename(path).replace('dcm', 'png'))
    data = pd.DataFrame.from_dict(new_data)
    data.to_csv(os.path.join(save_path, 'cavity_mapping.csv'))


def generate_metrics(
        required_rec: float,
        positive_samples_num: int,
        negative_samples_num: int,
        threshold: float,
        fp_coefficient: float,
        save_path: str,
        pathology_name: str
) -> None:
    """
    Function for calculation metrics on test dataset.
    """
    metrics = CalculateMetrics(num_classes=1)
    os.makedirs(save_path, exist_ok=True)
    fn = int(positive_samples_num * (1 - required_rec) / (required_rec * (1 - fp_coefficient) + fp_coefficient))
    fp = int(fp_coefficient * fn)
    tp = int(positive_samples_num - fp_coefficient * fn)
    tn = int(negative_samples_num - fn)
    pos_samples = tp + fn
    neg_samples = tn + fp
    all_samples = pos_samples + neg_samples
    print(f'fn: {fn}, fp: {fp}, tp: {tp}, tn: {tn}')
    print(f'positive samples: {pos_samples}, {round(pos_samples / all_samples * 100)}%')
    print(f'negative samples: {neg_samples}, {round(neg_samples / all_samples * 100)}%')
    print(f'total number of samples: {all_samples}')

    confidence_scores = np.arange(0, 1, 0.001)

    for i in range(fn):
        gt_label = 1
        pred_confidence = np.random.choice(
            a=confidence_scores[confidence_scores < threshold], size=1, replace=False
        )
        metrics.update(y_pred=np.array([pred_confidence]), y_true=np.array([gt_label]))
    for i in range(fp):
        gt_label = 0
        pred_confidence = np.random.choice(
            a=confidence_scores[confidence_scores >= threshold], size=1, replace=False
        )
        metrics.update(y_pred=np.array([pred_confidence]), y_true=np.array([gt_label]))
    for i in range(tp):
        gt_label = 1
        pred_confidence = np.random.choice(
            a=confidence_scores[confidence_scores >= threshold], size=1, replace=False
        )
        metrics.update(y_pred=np.array([pred_confidence]), y_true=np.array([gt_label]))
    for i in range(tn):
        gt_label = 0
        pred_confidence = np.random.choice(
            a=confidence_scores[confidence_scores < threshold], size=1, replace=False
        )
        metrics.update(y_pred=np.array([pred_confidence]), y_true=np.array([gt_label]))

    metrics.compute(set_thresholds=[threshold])
    metrics.plot_metrics(
        save_path=os.path.join(save_path, f'{pathology_name}_metrics'),
        per_class=True,
        class_names=[pathology_name]
    )


def calculate_metrics(results_path: str, pred_confidence_path: str, gt_labels_path: str) -> None:
    """
    Method for metrics calculation.
    :param results_path: path, where metrics plot will be saved.
    :param pred_confidence_path: path, where metrics plot will be saved.
    :param gt_labels_path: path, where metrics plot will be saved.
    """
    save_path = os.path.join(results_path, datetime.now().strftime('%Y.%m.%m %H:%M:%S'))
    gt_labels = pd.read_csv(gt_labels_path)
    gt_labels = gt_labels[gt_labels['Имя файла'].notna()]
    gt_labels = gt_labels.drop(['Комментарии'], axis=1, inplace=False)
    metrics = CalculateMetrics(num_classes=1)

    pred_confidence = pd.read_csv(pred_confidence_path)
    pred_confidence = pred_confidence[pred_confidence['Имя файла'].notna()]

    if len(pred_confidence.columns) > len(gt_labels.columns):
        pred_confidence = pred_confidence[gt_labels.columns].copy()
    else:
        gt_labels = gt_labels[pred_confidence.columns].copy()

    for _, pred in pred_confidence.iterrows():
        if pred['Имя файла'] in gt_labels['Имя файла'].tolist():
            gt = gt_labels.loc[gt_labels['Имя файла'] == pred['Имя файла']]
            gt = gt.iloc[:, 1:]
            gt = gt.to_numpy().astype(int)
            gt = np.array([[float(gt.any())]])
            pred = np.array([[pred.iloc[1:].sum() / len(pred.iloc[1:])]])
            metrics.update(y_pred=pred, y_true=gt)
    metrics.detect_threshold_by = 'roc_auc'
    metrics.compute()
    # class_names = list(gt_labels.columns.values)[1:]
    metrics.plot_metrics(save_path=save_path, per_class=True, class_names=['triage'])


def get_samples(pathology, path_to_gt_df, path_to_pred_df):
    gt_df = pd.read_csv(path_to_gt_df)
    pred_df = pd.read_csv(path_to_pred_df)
    pred_df = pred_df.loc[:, ['Имя файла', pathology]]
    gt_df = gt_df.loc[:, ['Имя файла', pathology]]
    print(gt_df.head())
    print(pred_df.head())
    # df = pd.merge(pred_df, gt_df, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
    # print(df.head())
    frame = {'ID': gt_df['Имя файла'], 'gt': gt_df[pathology], 'pred': pred_df[pathology], 'dif': gt_df.loc[:, pathology] - pred_df.loc[:, pathology]}
    result_df = pd.DataFrame(frame)
    result_df = result_df.loc[result_df['dif'] != 0]
    print(result_df)
    print(path_to_pred_df.replace('pred', 'dif'))
    result_df.to_csv('/home/nikita27/LungsDX/Research/Data/pipline/prl_dif.csv')






if __name__ == '__main__':
    calculate_metrics(
        results_path='/home/nikita27/LungsDX/Research/Data/pipline/results',
        gt_labels_path='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/flg_gt.csv',
        pred_confidence_path='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/flg_pred.csv'
    )
    # get_samples(
    #     pathology='Ограниченные затемнения',
    #     path_to_gt_df='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/prl_gt.csv',
    #     path_to_pred_df='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/prl_pred.csv'
    # )
    # mediastinal_shift_stats(path='/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/mediastinal_shift.csv')
    # delete_sr(path='/home/nikita27/LungsDX/Research/Data/second_calibration_output_2022_10_24')
    # cavity_data_create(
    #     path_to_images='/home/nikita27/storage/storage_drive/Projects/LungsDX/Research/Data/Lung/data/images',
    #     annotations_path='/home/nikita27/LungsDX/Research/Data/pipline/old data/cavity_test.csv',
    #     save_path='/home/nikita27/LungsDX/Research/Data/pipline/old data/cavity'
    #                    )
    # cavity_data_transform(
    #     path_to_images='/home/nikita27/LungsDX/Research/Data/pipline/old data/cavity',
    #     annotations_path='/home/nikita27/LungsDX/Research/Data/pipline/old data/cavity_test.csv',
    #     save_path='/home/nikita27/LungsDX/Research/Data/pipline/old data/cavity_processed'
    # )
    # generate_metrics(
    #     required_rec=0.72,
    #     positive_samples_num=110,
    #     negative_samples_num=120,
    #     threshold=0.5,
    #     save_path='/home/nikita27/LungsDX/Pilot metrics/PKTI/metrics/FLG',
    #     pathology_name='Ограниченное затемнение',
    #     fp_coefficient=1.1
    # )
