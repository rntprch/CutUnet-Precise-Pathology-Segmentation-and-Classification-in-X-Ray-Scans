import torch
from typing import List, Tuple, Any

from Pipeline.results import LungsDXResult
from Pipeline.models.base import PreprocessingModel
from Pipeline.configs import *


class LungsAlignment(PreprocessingModel):

    def __init__(self, use_projection: List[str], device: torch.device) -> None:
        super().__init__(target_size=ALIGNMENT_TARGET_SIZE, preprocessing_fn=ALIGNMENT_PREPROCESSING_FN)
        self.use_projection = use_projection
        self.device = device

    def predictions(self, orig_image_data: LungsDXResult) -> LungsDXResult:
        return orig_image_data
