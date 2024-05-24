import numpy as np
from functools import wraps
import sklearn.metrics as sklm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from typing import List
from abc import abstractmethod, ABC
from pathlib import Path
import os
import warnings
from test_config import *


NUM_CLASSES = 2
CLASS_NAMES = ['no such pathology', 'pathology is present']
CONF_MATRIX_THRESHOLD = 0.5


def _ignore(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    indices = y_true > -1
    return y_true[indices], y_pred[indices]


def ignore(func):
    @wraps(func)
    def wrapped_function(y_true, y_pred, *args, **kwargs):
        y_true, y_pred = _ignore(y_true=y_true, y_pred=y_pred)
        return func(y_true=y_true, y_pred=y_pred, *args, **kwargs)

    return wrapped_function


def one_hot(labels, num_classes):
    num_samples = labels.shape[0]
    last_dim = labels.shape[-1]

    categorical_labels = np.zeros((num_samples, num_classes, labels.shape[-1]))

    dim1 = np.repeat(np.arange(num_samples), last_dim)
    dim3 = np.tile(np.arange(last_dim), num_samples)
    categorical_labels[dim1, labels.reshape(-1, order='C'), dim3] = 1.0
    return categorical_labels


@ignore
def confusion_matrix(y_true, y_pred, num_classes=2):
    y_true = y_true.reshape(-1).astype(int)
    y_pred = y_pred.reshape(-1).astype(int)
    # matrix = sklm.confusion_matrix(y_true=y_true, y_pred=y_pred)

    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            matrix[i, j] = ((y_true == i) & (y_pred == j)).sum()

    return matrix


class Metric(ABC):
    def __init__(self, progress_bar=False, display=True, mode="binary"):
        self.name = "Unknown"
        self.score = 0.0
        self.scores = None
        self.progress_bar = progress_bar
        self.print = display
        self.mode = mode
        assert self.mode in ["binary", "multi-binary", "multi"], f"Unknown mode {mode}"
        self.y_pred = None
        self.y_true = None

    @abstractmethod
    def compute(
            self,
            y_true: np.array = None,
            y_pred: np.array = None,
            reduction: str = "mean",
            thresholds: List[float] = None
    ) -> float or np.array:
        """ returns current value of metric"""
        return

    def update(self, y_pred: np.array, y_true: np.array):
        """ updates current metric"""
        if self.y_pred is not None and self.y_true is not None:
            self.y_pred = np.concatenate((self.y_pred, y_pred), axis=0)
            self.y_true = np.concatenate((self.y_true, y_true), axis=0)
        else:
            self.y_pred = y_pred
            self.y_true = y_true

    def reset(self):
        """ reset metric """
        self.score = 0.0
        return

    def compute_per_class(self, y_pred: np.array = None, y_true: np.array = None, thresholds: List[float] = None):
        return self.compute(y_pred=y_pred, y_true=y_true, reduction="none", thresholds=thresholds)


class BalancedAccuracy(Metric):
    def __init__(self, name="accuracy", mode="binary", *args, **kwargs):
        super(BalancedAccuracy, self).__init__(mode=mode, *args, **kwargs)
        self.name = name

    def compute(
            self,
            y_true: np.array = None,
            y_pred: np.array = None,
            reduction: str = "mean",
            thresholds: List[float] = None
    ) -> float or np.array:
        """
        Method for metric calculation.
        :param y_true:
        :param y_pred:
        :param reduction:
        :param thresholds: list of thresholds to convert confidence into class.
        :return computed metric value (or array of values in multi-binary case)
        """
        assert reduction in ["mean", "none"], f"Unknown reduction type: {reduction}"
        if y_true is not None and y_pred is not None:
            self.y_true, self.y_pred = y_true, y_pred
        num_samples = self.y_true.shape[0]

        if self.mode == "binary":
            num_classes = 1
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
        elif self.mode == "multi-binary":
            num_classes = self.y_true.shape[1]
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
        else:  # multi
            num_classes = self.y_pred.shape[-1]
            self.y_pred = self.y_pred.argmax(axis=1).reshape(num_samples, 1, -1)
            self.y_true = self.y_true.reshape(num_samples, 1, -1)

        if self.mode in ["binary", "multi-binary"]:
            self.score = []
            for i in range(num_classes):
                threshold = thresholds[i] if thresholds is not None else 0.5
                gt = self.y_true[:, i].copy().astype(int)
                pred = self.y_pred[:, i].copy()
                pred[pred >= threshold] = 1
                pred[pred < threshold] = 0
                self.score.append(self.__balanced_accuracy_score(y_true=gt, y_pred=pred, num_classes=2))

            if reduction == "mean":
                self.score = np.mean(self.score)

        else:  # multi
            gt = self.y_pred.copy().astype(int)
            pred = self.y_true.copy()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            self.score = self.__balanced_accuracy_score(y_true=gt, y_pred=pred, num_classes=num_classes)

        return self.score

    def compute_per_class(self, y: np.array = None, y_hat: np.array = None, thresholds: List[float] = None):
        return self.compute(y, y_hat, reduction="none", thresholds=thresholds)

    @staticmethod
    def __balanced_accuracy_score(y_true, y_pred, num_classes=2):
        num_classes = 2 if num_classes < 2 else num_classes
        conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred, num_classes=num_classes)
        eps = 1e-8
        indices = np.arange(num_classes)
        return 1.0 / num_classes * (conf_mat[indices, indices] / (conf_mat.sum(axis=-1) + eps)).sum()



class F1Score(Metric):
    def __init__(self, name="accuracy", mode="binary", *args, **kwargs):
        super(F1Score, self).__init__(mode=mode, *args, **kwargs)
        self.name = name

    def compute(
            self,
            y_pred: np.array = None,
            y_true: np.array = None,
            reduction: str = "mean",
            thresholds: List[float] = None
    ) -> float or np.array:
        """
        Method for metric calculation.
        :param y_true:
        :param y_pred:
        :param reduction:
        :param thresholds: list of thresholds to convert confidence into class.
        :return computed metric value (or array of values in multi-binary case)
        """
        assert reduction in ["mean", "none"], f"Unknown reduction type: {reduction}"
        if y_pred is not None and y_true is not None:
            self.y_pred, self.y_true = y_pred, y_true
        num_samples = self.y_pred.shape[0]

        if self.mode == "binary":
            num_classes = 1
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        elif self.mode == "multi-binary":
            num_classes = self.y_pred.shape[1]
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        else:  # multi
            num_classes = self.y_true.shape[-1]
            self.y_true = self.y_true.argmax(axis=1).reshape(num_samples, 1, -1)
            self.y_pred = self.y_pred.reshape(num_samples, 1, -1)

        if self.mode in ["binary", "multi-binary"]:
            self.score = []
            for i in range(num_classes):
                threshold = thresholds[i] if thresholds is not None else 0.5
                gt = self.y_true[:, i].copy().astype(int)
                pred = self.y_pred[:, i].copy()
                pred[pred >= threshold] = 1
                pred[pred < threshold] = 0
                self.score.append(self.__f1_score(y_true=gt, y_pred=pred))

            if reduction == "mean":
                self.score = np.mean(self.score)

        else:  # multi
            gt = self.y_true.copy().astype(int)
            pred = self.y_pred.copy()
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            self.score = self.__f1_score(y_true=gt, y_pred=pred)

        return self.score

    @staticmethod
    def __f1_score(y_true, y_pred, num_classes: int = 2) -> np.array:
        average = 'binary' if num_classes == 2 else 'weighted'
        return sklm.f1_score(y_true=y_true, y_pred=y_pred, average=average)


class Sensitivity(Metric):
    def __init__(self, name="accuracy", mode="binary", *args, **kwargs):
        super(Sensitivity, self).__init__(mode=mode, *args, **kwargs)
        self.name = name

    def compute(
            self,
            y_pred: np.array = None,
            y_true: np.array = None,
            reduction: str = "mean",
            thresholds: List[float] = None
    ) -> float or np.array:
        """
        Method for metric calculation.
        :param y_true:
        :param y_pred:
        :param reduction:
        :param thresholds: list of thresholds to convert confidence into class.
        :return computed metric value (or array of values in multi-binary case)
        """
        assert reduction in ["mean", "none"], f"Unknown reduction type: {reduction}"
        if y_pred is not None and y_true is not None:
            self.y_pred, self.y_true = y_pred, y_true
        num_samples = self.y_pred.shape[0]

        if self.mode == "binary":
            num_classes = 1
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        elif self.mode == "multi-binary":
            num_classes = self.y_pred.shape[1]
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        else:  # multi
            num_classes = self.y_true.shape[-1]
            self.y_true = self.y_true.argmax(axis=1).reshape(num_samples, 1, -1)
            self.y_pred = self.y_pred.reshape(num_samples, 1, -1)

        if self.mode in ["binary", "multi-binary"]:
            self.score = []
            for i in range(num_classes):
                threshold = thresholds[i] if thresholds is not None else 0.5
                gt = self.y_true[:, i].copy().astype(int)
                pred = self.y_pred[:, i].copy()
                pred[pred >= threshold] = 1
                pred[pred < threshold] = 0
                self.score.append(self.__sensitivity(y_true=gt, y_pred=pred))

            if reduction == "mean":
                self.score = np.mean(self.score)

        return self.score

    @staticmethod
    def __sensitivity(y_true, y_pred) -> np.array:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        return sensitivity


class Specificity(Metric):
    def __init__(self, name="accuracy", mode="binary", *args, **kwargs):
        super(Specificity, self).__init__(mode=mode, *args, **kwargs)
        self.name = name

    def compute(
            self,
            y_pred: np.array = None,
            y_true: np.array = None,
            reduction: str = "mean",
            thresholds: List[float] = None
    ) -> float or np.array:
        """
        Method for metric calculation.
        :param y_true:
        :param y_pred:
        :param reduction:
        :param thresholds: list of thresholds to convert confidence into class.
        :return computed metric value (or array of values in multi-binary case)
        """
        assert reduction in ["mean", "none"], f"Unknown reduction type: {reduction}"
        if y_pred is not None and y_true is not None:
            self.y_pred, self.y_true = y_pred, y_true
        num_samples = self.y_pred.shape[0]

        if self.mode == "binary":
            num_classes = 1
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        elif self.mode == "multi-binary":
            num_classes = self.y_pred.shape[1]
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        else:  # multi
            num_classes = self.y_true.shape[-1]
            self.y_true = self.y_true.argmax(axis=1).reshape(num_samples, 1, -1)
            self.y_pred = self.y_pred.reshape(num_samples, 1, -1)

        if self.mode in ["binary", "multi-binary"]:
            self.score = []
            for i in range(num_classes):
                threshold = thresholds[i] if thresholds is not None else 0.5
                gt = self.y_true[:, i].copy().astype(int)
                pred = self.y_pred[:, i].copy()
                pred[pred >= threshold] = 1
                pred[pred < threshold] = 0
                self.score.append(self.__specificity(y_true=gt, y_pred=pred))

            if reduction == "mean":
                self.score = np.mean(self.score)

        return self.score

    @staticmethod
    def __specificity(y_true, y_pred) -> np.array:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        return specificity


class FDR(Metric):
    def __init__(self, name="accuracy", mode="binary", *args, **kwargs):
        super(FDR, self).__init__(mode=mode, *args, **kwargs)
        self.name = name

    def compute(
            self,
            y_pred: np.array = None,
            y_true: np.array = None,
            reduction: str = "mean",
            thresholds: List[float] = None
    ) -> float or np.array:
        """
        Method for metric calculation.
        :param y_true:
        :param y_pred:
        :param reduction:
        :param thresholds: list of thresholds to convert confidence into class.
        :return computed metric value (or array of values in multi-binary case)
        """
        assert reduction in ["mean", "none"], f"Unknown reduction type: {reduction}"
        if y_pred is not None and y_true is not None:
            self.y_pred, self.y_true = y_pred, y_true
        num_samples = self.y_pred.shape[0]

        if self.mode == "binary":
            num_classes = 1
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        elif self.mode == "multi-binary":
            num_classes = self.y_pred.shape[1]
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        else:  # multi
            num_classes = self.y_true.shape[-1]
            self.y_true = self.y_true.argmax(axis=1).reshape(num_samples, 1, -1)
            self.y_pred = self.y_pred.reshape(num_samples, 1, -1)

        if self.mode in ["binary", "multi-binary"]:
            self.score = []
            for i in range(num_classes):
                threshold = thresholds[i] if thresholds is not None else 0.5
                gt = self.y_true[:, i].copy().astype(int)
                pred = self.y_pred[:, i].copy()
                pred[pred >= threshold] = 1
                pred[pred < threshold] = 0
                self.score.append(self.__fdr(y_true=gt, y_pred=pred))

            if reduction == "mean":
                self.score = np.mean(self.score)

        return self.score

    @staticmethod
    def __fdr(y_true, y_pred) -> np.array:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
        fdr = fp / (fp + tp) if (tp + fp) != 0 else 0
        return fdr


class FOR(Metric):
    def __init__(self, name="accuracy", mode="binary", *args, **kwargs):
        super(FOR, self).__init__(mode=mode, *args, **kwargs)
        self.name = name

    def compute(
            self,
            y_pred: np.array = None,
            y_true: np.array = None,
            reduction: str = "mean",
            thresholds: List[float] = None
    ) -> float or np.array:
        """
        Method for metric calculation.
        :param y_true:
        :param y_pred:
        :param reduction:
        :param thresholds: list of thresholds to convert confidence into class.
        :return computed metric value (or array of values in multi-binary case)
        """
        assert reduction in ["mean", "none"], f"Unknown reduction type: {reduction}"
        if y_pred is not None and y_true is not None:
            self.y_pred, self.y_true = y_pred, y_true
        num_samples = self.y_pred.shape[0]

        if self.mode == "binary":
            num_classes = 1
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        elif self.mode == "multi-binary":
            num_classes = self.y_pred.shape[1]
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        else:  # multi
            num_classes = self.y_true.shape[-1]
            self.y_true = self.y_true.argmax(axis=1).reshape(num_samples, 1, -1)
            self.y_pred = self.y_pred.reshape(num_samples, 1, -1)

        if self.mode in ["binary", "multi-binary"]:
            self.score = []
            for i in range(num_classes):
                threshold = thresholds[i] if thresholds is not None else 0.5
                gt = self.y_true[:, i].copy().astype(int)
                pred = self.y_pred[:, i].copy()
                pred[pred >= threshold] = 1
                pred[pred < threshold] = 0
                self.score.append(self.__for(y_true=gt, y_pred=pred))

            if reduction == "mean":
                self.score = np.mean(self.score)

        return self.score

    @staticmethod
    def __for(y_true, y_pred) -> np.array:
        tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
        for_val = fn / (fn + tn) if (tn + fn) != 0 else 0
        return for_val


class RocAUC(Metric):
    def __init__(self, name='roc_auc', mode='binary', *args, **kwargs):
        super(RocAUC, self).__init__(mode=mode, *args, **kwargs)
        self.name = name

    def compute(
            self, y_pred: np.array = None, y_true: np.array = None, reduction="mean", thresholds: List[float] = None
    ):
        assert reduction in ["mean", "none"], f"Unknown reduction type: {reduction}"
        if y_pred is not None and y_true is not None:
            self.y_pred, self.y_true = y_pred, y_true
        num_samples = self.y_pred.shape[0]

        if self.mode == "binary":
            num_classes = 1
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        else:  # multi-binary
            num_classes = self.y_pred.shape[1]
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)

        self.score = []
        for i in range(num_classes):
            gt = self.y_true[:, i].copy().astype(int)
            pred = self.y_pred[:, i].copy()
            self.score.append(self.roc_auc_score(y_true=gt, y_pred=pred))

        if reduction == "mean":
            self.score = np.mean(self.score)

        return self.score

    @staticmethod
    @ignore
    def roc_auc_score(y_true, y_pred):
        if (y_true == 0).all() or (y_true == 1).all():
            return 0.0
        try:
            return sklm.roc_auc_score(y_true=y_true, y_score=y_pred)
        except:
            return 0.0


class AP(Metric):
    def __init__(self, name='ap_score', mode="binary", *args, **kwargs):
        super(AP, self).__init__(mode=mode, *args, **kwargs)
        self.name = name

    def compute(
            self, y_pred: np.array = None, y_true: np.array = None, reduction="mean", thresholds: List[float] = None
    ) -> float or np.array:
        assert reduction in ["mean", "none"], f"Unknown reduction type: {reduction}"
        num_samples = self.y_pred.shape[0]

        if self.mode == "binary":
            num_classes = 1
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)
        else:  # multi-binary
            num_classes = self.y_pred.shape[1]
            self.y_pred = self.y_pred.reshape(num_samples, num_classes, -1)
            self.y_true = self.y_true.reshape(num_samples, num_classes, -1)

        self.score = []
        for i in range(num_classes):
            gt = self.y_true[:, i].copy().astype(int)
            pred = self.y_pred[:, i].copy()
            self.score.append(self.average_precision_score(y_true=gt, y_pred=pred))

        if reduction == "mean":
            self.score = np.mean(self.score)
        return self.score

    @staticmethod
    @ignore
    def average_precision_score(y_true: np.array, y_pred: np.array) -> np.array:
        if (y_true == 0).all() or (y_true == 1).all():
            return 0.0

        precision, recall, thresholds = sklm.precision_recall_curve(y_true=y_true.astype(int), probas_pred=y_pred)
        for j in range(len(precision)):
            precision[j] = np.max(precision[:j + 1])
        average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])
        return average_precision


class CalculateMetrics:
    def __init__(self, num_classes: int = 2) -> None:
        """
        Creating results annotations class.
        :param num_classes: number of classes.
        """
        self.num_classes = num_classes
        self.mode = 'binary' if self.num_classes == 1 else "multi-binary"
        self.accuracy = BalancedAccuracy(mode=self.mode)
        self.ap = AP(mode=self.mode)
        self.roc_auc = RocAUC(mode=self.mode)
        self.f1 = F1Score(mode=self.mode)
        self.sensitivity = Sensitivity(mode=self.mode)
        self.specificity = Specificity(mode=self.mode)
        self.fdr = FDR(mode=self.mode)
        self.FOR = FOR(mode=self.mode)
        self.accuracy_score = 0
        self.ap_score = 0
        self.roc_auc_score = 0
        self.f1_score = 0
        self.for_score = 0
        self.fdr_score = 0
        self.sensitivity_score = 0
        self.specificity_score = 0
        self.conf_matrix = None
        self.opt_thresholds = list()
        self.threshold_range = THRESHOLD_RANGE
        self.threshold_balance_coefficient = THRESHOLD_BALANCE_COEFFICIENT
        self.detect_threshold_by = DETECT_THRESHOLD_BY

    def update(self, y_pred: np.array, y_true: np.array) -> None:
        self.accuracy.update(y_pred=y_pred, y_true=y_true)
        self.roc_auc.update(y_pred=y_pred, y_true=y_true)
        self.ap.update(y_pred=y_pred, y_true=y_true)
        self.f1.update(y_pred=y_pred, y_true=y_true)
        self.sensitivity.update(y_pred=y_pred, y_true=y_true)
        self.specificity.update(y_pred=y_pred, y_true=y_true)
        self.fdr.update(y_pred=y_pred, y_true=y_true)
        self.FOR.update(y_pred=y_pred, y_true=y_true)

    def compute(self, set_thresholds: List[float] = None) -> None:
        if not set_thresholds:
            self.__get_optimum_thresholds()
        else:
            self.opt_thresholds = set_thresholds
        self.accuracy_score = self.accuracy.compute(thresholds=self.opt_thresholds)
        self.ap_score = self.ap.compute(thresholds=self.opt_thresholds)
        self.roc_auc_score = self.roc_auc.compute(thresholds=self.opt_thresholds)
        self.f1_score = self.f1.compute(thresholds=self.opt_thresholds)
        self.sensitivity_score = self.sensitivity.compute(thresholds=self.opt_thresholds)
        self.specificity_score = self.specificity.compute(thresholds=self.opt_thresholds)
        self.for_score = self.FOR.compute(thresholds=self.opt_thresholds)
        self.fdr_score = self.fdr.compute(thresholds=self.opt_thresholds)
        print(
            'Усреднённые метрики по всем патологиям:\n'
            f'accuracy: {self.accuracy_score * 100:.2f}%,\naverage precision: {self.ap_score.item():.3f},\n'
            f'roc-auc score: {self.roc_auc_score.item():.3f},\nf1: {self.f1_score.item() * 100:.2f}%,\n'
            f'sensitivity: {self.sensitivity_score * 100:.2f}%\nspecificity: {self.specificity_score * 100:.2f}%\n'
            f'FDR: {self.fdr_score * 100:.2f}%\nFOR: {self.for_score * 100:.2f}%'
        )

    def plot_metrics(self, save_path: str, per_class: bool = False, class_names: List[str] = None) -> None:
        """
        Method for confusion matrix plotting.
        :param save_path: path to save results.
        :param class_names: path to save results.
        :param per_class: path to save results.
        """
        path = save_path
        # path = os.path.join(save_path, 'metrics')
        Path(path).mkdir(parents=True, exist_ok=True)
        if per_class and self.mode in ['multi-binary', 'binary']:
            accuracy_score = self.accuracy.compute_per_class(thresholds=self.opt_thresholds)
            f1_score = self.f1.compute_per_class(thresholds=self.opt_thresholds)
            sensitivity_score = self.sensitivity.compute_per_class(thresholds=self.opt_thresholds)
            roc_auc_score = self.roc_auc.compute_per_class(thresholds=self.opt_thresholds)
            for i in range(self.num_classes):
                conf_matrix = self.__confusion_matrix(
                    y_true=self.accuracy.y_true[:, i, :].copy(),
                    y_pred=self.accuracy.y_pred[:, i, :].copy(),
                    num_classes=2,
                    threshold=self.opt_thresholds[i]
                )
                self.__plot_confusion_matrix(
                    save_path=path,
                    class_names=[class_names[i]],
                    acc_score=accuracy_score[i],
                    conf_matrix=conf_matrix,
                    per_class=per_class,
                    threshold=self.opt_thresholds[i],
                    f1_score=f1_score[i],
                    sensitivity_score=sensitivity_score[i]
                )
                if (self.accuracy.y_pred[:, i, :] == 0).all() or (self.accuracy.y_pred[:, i, :] == 1).all():
                    print(
                        f'Невозможно построить ROC-AUC & Precision-Recall для патологии {class_names[0]}'
                        f' - все объекты одного класса!'
                    )
                else:
                    print(f'Оптимальный порог для патологии {class_names[i]}: {self.opt_thresholds[i]:.3f}')
                    self.__plot_roc_auc(
                        save_path=path,
                        class_names=[class_names[i]],
                        score=roc_auc_score[i],
                        i=i
                    )
                    self.__plot_precision_recall(
                        save_path=path,
                        class_names=[class_names[i]],
                        y_pred=self.accuracy.y_pred[:, i, :].copy(),
                        y_true=self.accuracy.y_true[:, i, :].copy()
                    )
        else:
            conf_matrix = self.__confusion_matrix(
                y_true=self.accuracy.y_true, y_pred=self.accuracy.y_pred, num_classes=self.num_classes, threshold=0.5
            )
            self.__plot_confusion_matrix(
                save_path=path,
                class_names=class_names,
                acc_score=self.accuracy_score,
                f1_score=self.f1_score,
                sensitivity_score=self.sensitivity_score,
                conf_matrix=conf_matrix,
                per_class=per_class
            )

    def __get_optimum_thresholds(self):
        """
        Method for optimum threshold calculation.
        """
        num_samples = self.accuracy.y_pred.shape[0]
        if self.mode == "binary":
            num_classes = 1
            y_pred = self.accuracy.y_pred.reshape(num_samples, num_classes, -1)
            y_true = self.accuracy.y_true.reshape(num_samples, num_classes, -1)
        else:  # multi-binary
            num_classes = self.accuracy.y_pred.shape[1]
            y_pred = self.accuracy.y_pred.reshape(num_samples, num_classes, -1)
            y_true = self.accuracy.y_true.reshape(num_samples, num_classes, -1)

        if self.detect_threshold_by in ['roc_auc', 'pre_rec']:
            for i in range(self.num_classes):
                if self.detect_threshold_by == 'roc_auc':
                    fpr, tpr, thresholds = sklm.roc_curve(y_score=y_pred[:, i, :], y_true=y_true[:, i, :])
                    metric = (tpr - self.threshold_balance_coefficient * fpr)[:len(thresholds)]
                else:
                    precision, recall, thresholds = sklm.precision_recall_curve(
                        probas_pred=y_pred[:, i, :], y_true=y_true[:, i, :]
                    )
                    metric = (self.threshold_balance_coefficient * precision + recall)[:len(thresholds)]

                metric = metric[(self.threshold_range[0] <= thresholds) & (thresholds <= self.threshold_range[1])]
                thresholds = thresholds[
                    (self.threshold_range[0] <= thresholds) & (thresholds <= self.threshold_range[1])
                ]
                if thresholds.any():
                    max_idx = np.array(np.where(metric == metric.max())).max()
                    self.opt_thresholds.append(thresholds[max_idx])
                else:
                    self.opt_thresholds.append(0.5)
                    warnings.warn('Could not find optimum threshold in the given range! Set it to 0.5.')

        elif self.detect_threshold_by in ['f1', 'accuracy', 'sensitivity']:
            thresholds = list(np.arange(self.threshold_range[0], self.threshold_range[1], 0.001))
            metric = None
            for confidence_lvl in thresholds:
                threshold = list(np.repeat(confidence_lvl, self.num_classes))
                if self.detect_threshold_by == 'f1':
                    score = self.f1.compute_per_class(thresholds=threshold)
                elif self.detect_threshold_by == 'sensitivity':
                    score = self.sensitivity.compute_per_class(thresholds=threshold)
                else:
                    score = self.accuracy.compute_per_class(thresholds=threshold)
                metric = np.vstack((metric, score)) if metric is not None else score

            # metric = np.flip(metric, axis=0)
            # opt_indexes = len(thresholds) - np.argmax(metric, axis=0) - 1
            opt_indexes = np.argmax(metric, axis=0)
            self.opt_thresholds = np.array(thresholds)[opt_indexes].tolist()

    @staticmethod
    def __plot_confusion_matrix(
            save_path: str,
            class_names: List[str],
            per_class: bool,
            conf_matrix: np.array,
            acc_score: float,
            f1_score: float,
            sensitivity_score: float,
            threshold: float = 0.5
    ) -> None:
        """
        Method for platting confusion matrix.
        :param save_path:
        :param class_names:
        :param per_class:
        """
        tn, fp, fn, tp = conf_matrix.ravel()
        positive_labels_num = tp + fn
        negative_labels_num = tn + fp

        sn.set(font_scale=2)
        plt.figure(figsize=(20, 15), dpi=101)
        ax = plt.subplot()
        cfm = pd.DataFrame(conf_matrix)
        cfm.columns.name = 'PREDICTION'
        cfm.index.name = 'GROUND TRUTH'
        if per_class:
            text = f"{class_names[0]} balanced accuracy = {acc_score * 100:.2f}%, F1 = {f1_score * 100:.2f}%,\n" \
                   f"Sensitivity = {sensitivity_score * 100:.2f}% for threshold = {threshold:.2f}\n" \
                   f"Label distribution in dataset: positive: {positive_labels_num}, negative: {negative_labels_num}\n"
            labels = CLASS_NAMES
            name = f"{class_names[0]}_confusion_matrix_{threshold:.2f}.png"
        else:
            text = f"Balanced accuracy = {acc_score * 100:.2f}%, F1 = {f1_score * 100:.2f}%,\n" \
                   f"Sensitivity = {sensitivity_score * 100:.2f}% for threshold = {threshold:.2f}\n" \
                   f"Label distribution in dataset: positive: {positive_labels_num}, negative: {positive_labels_num}\n"
            labels = class_names
            name = f"confusion_matrix_{threshold:.2f}.png"

        sn.heatmap(
            cfm, ax=ax, annot=True, fmt='.0f', annot_kws={"size": 50},
            cmap="YlGnBu", xticklabels=labels, yticklabels=labels
        )
        ax.set_title(text, fontsize=24)
        plt.savefig(os.path.join(save_path, name))
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)

    def __plot_roc_auc(self, save_path: str, class_names: List[str], score: float, i: int) -> None:
        """
        Method for roc auc curve plotting.
        """
        fpr, tpr, _ = sklm.roc_curve(y_true=self.accuracy.y_true[:, i, :], y_score=self.accuracy.y_pred[:, i, :])

        plt.figure(figsize=(20, 20), dpi=101)
        ax = plt.subplot()

        ax.plot(100. * fpr, 100. * tpr,
                label='ROC curve (area = {0:0.4f})'.format(score), marker='.', lw=5,
                aa=True, alpha=0.9, )

        ax.plot([0, 100], [0, 100], 'k--', lw=2)
        ax.set_xlim([-1, 101])
        ax.set_ylim([-1, 101])
        ax.set_xlabel('False Positive Rate, %', fontsize=24)
        ax.set_ylabel('True Positive Rate, %', fontsize=24)
        ax.set_aspect('equal')
        ax.set_xticks(range(0, 101, 5), minor=False)
        ax.set_xticks(range(0, 101, 1), minor=True)
        ax.set_yticks(range(0, 101, 5), minor=False)
        ax.set_yticks(range(0, 101, 1), minor=True)
        ax.tick_params(axis='both', which='major', labelsize=24)
        ax.tick_params(axis='both', which='minor', labelsize=24)
        ax.set_title('ROC', fontsize=24)
        ax.legend(loc="lower right", fontsize=24)
        ax.grid(which='major', alpha=1.0, linewidth=2, linestyle='--')
        ax.grid(which='minor', alpha=0.5, linestyle='--')

        name = f"{class_names[0]}_roc_curve.png"
        plt.savefig(os.path.join(save_path, name))
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)

    @staticmethod
    def __plot_precision_recall(save_path: str, class_names: List[str], y_pred: np.array, y_true: np.array) -> None:
        """
        Method precision recall curve plotting.
        """
        y_true = y_true.astype(int)
        precision, recall, thresholds = sklm.precision_recall_curve(probas_pred=y_pred, y_true=y_true)
        for j in range(len(precision)):
            precision[j] = np.max(precision[:j + 1])
        average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])

        color = 'slateblue'
        y_values = [100. * x for x in precision]
        x_values = [100. * x for x in recall]

        plt.figure(figsize=(20, 20), dpi=101)
        axes = plt.subplot()
        axes.set_xlim([-1, 101])
        axes.set_ylim([-1, 101])
        axes.set_xlabel('Recall, %', fontsize=24)
        axes.set_ylabel('Precision, %', fontsize=24)
        axes.set_aspect('equal')
        axes.set_xticks(range(0, 101, 5), minor=False)
        axes.set_xticks(range(0, 101, 1), minor=True)
        axes.set_yticks(range(0, 101, 5), minor=False)
        axes.set_yticks(range(0, 101, 1), minor=True)
        axes.tick_params(axis='both', which='major', labelsize=24)
        axes.tick_params(axis='both', which='minor', labelsize=24)
        axes.grid(which='major', alpha=1.0, linewidth=2, linestyle='--')
        axes.grid(which='minor', alpha=0.5, linestyle='--')
        axes.set_title('Precision-Recall for {}'.format(class_names[0]), fontsize=24)

        plt.plot(x_values, y_values, marker='.', color=color, aa=True, alpha=0.9, linewidth=5,
                 label=f'AP: {average_precision:.04f}')
        plt.legend(loc='lower center', fontsize=24)

        name = f"{class_names[0]}_pre_rec_curve.png"
        plt.savefig(os.path.join(save_path, name))
        plt.close()
        plt.rcParams.update(plt.rcParamsDefault)

    @staticmethod
    def __confusion_matrix(y_pred, y_true, num_classes: int, threshold: float):
        y_pred = y_pred.reshape(-1)
        y_pred[y_pred >= threshold] = 1
        y_pred[y_pred < threshold] = 0
        y_true = y_true.reshape(-1).astype(int)
        # matrix = sklm.confusion_matrix(y_true=y_true, y_pred=y_pred)

        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for i in range(num_classes):
            for j in range(num_classes):
                matrix[i, j] = ((y_true == i) & (y_pred == j)).sum()
        return matrix
