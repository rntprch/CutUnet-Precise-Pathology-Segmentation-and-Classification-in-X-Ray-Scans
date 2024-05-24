import numpy as np
import json

class ModelCalibrator:

    def __init__(self, calibration_path: str) -> None:
        """
        Method for calibration model inference.
        :param calibration_path: path to model output calibration params.
        """
        self.__load_params(calibration_path=calibration_path)

    def predict(self, y_prob: np.ndarray, pathology_name: str) -> np.ndarray:
        """
        Method for data calibration.
        :param y_prob: predicted probabilities.
        :param pathology_name: predicted probabilities.
        """
        y_prob = self._convert_to_log_odds(y_prob)
        output = self._transform(y_prob, pathology_name=pathology_name)
        return output

    def _transform(self, y_prob: np.ndarray, pathology_name: str) -> np.ndarray:
        """
        Method for data calibration.
        :param y_prob: predicted probabilities.
        """
        coef = self.calibrator_params[pathology_name]["coef"]
        intercept = self.calibrator_params[pathology_name]["intercept"]
        output = y_prob * coef + intercept
        output = 1 / (1 + np.exp(-output))
        return output

    @staticmethod
    def _convert_to_log_odds(y_prob: np.ndarray) -> np.ndarray:
        """
        Method for data calibration.
        :param y_prob: predicted probabilities.
        """
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1 - eps)
        y_prob = np.log(y_prob / (1 - y_prob))
        return y_prob

    def __load_params(self, calibration_path: str) -> None:
        """
        Method for calibration params loading.
        """
        with open(calibration_path, "r") as inp:
            self.calibrator_params = json.load(inp)
