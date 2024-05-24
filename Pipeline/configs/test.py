from Pipeline.configs.service import TRIAGE, NDL, PNTX, PLEF, CTR, MESH, SP, LIN, CAVITY, CONSOLIDATED_FRACTURE, \
    FRACTURE


# Path to initial data, save directory and markings dataframe
PATH_TO_IMAGES = '/workspace/storage/Projects/LungsDX/Research/Data/OPACITY_FP_REDUCTION/data/images'
RESULTS_PATH = '/workspace/storage/Projects/LungsDX/Research/Data/OPACITY_FP_REDUCTION/data/images/studies'
GT_DATA_PATH = ''
# PRED_CONFIDENCE_PATH = 'pred_confidence2.csv'
PRED_CONFIDENCE_PATH = '/workspace/storage/Projects/LungsDX/Research/Data/OPACITY_FP_REDUCTION/data/images/prediction'
# actions to do
RUN = True
CALCULATE_METRICS = False
RUN_ON_IMAGE = False
TEST_ISSUES = False
SAVE_PREDICTIONS = True
# pipeline params
STUDY_TYPE = 'flg'
READ_SERIES = True
APPLY_CALIBRATION = True
TEST_MODE = False
SAVE_DICOM = False
INFERENCE_DEVICE = 5
FILTER_BY_NAME = ''
FILTER_BY_COMMENT = ''
FILTER_BY_DIRECTORY = []
GET_TP = False
GET_FP = False
GET_TN = False
GET_FN = False
NUM_OF_SAMPLES = 10
GET_PATHOLOGICAL_STUDIES = ''
IMAGE_NAME = 'sop_0_3095911919.56123.17720.166014160120205230030219.0.dcm'
# Metrics calculation params
# The range in which the search for the optimal threshold will be carried out
THRESHOLD_RANGE = (0.3, 1.)
DETECT_THRESHOLD_BY = 'roc_auc'
THRESHOLD_BALANCE_COEFFICIENT = 1.
USE_CUSTOM_THRESHOLDS = True
METRIC_NAME_TO_TAG = {
    TRIAGE: 'Иное',
    CTR: 'Кардиомегалия',
    MESH: 'Смещение средостения',
    PLEF: 'Гидроторакс',
    PNTX: 'Пневмоторакс',
    SP: 'Снижение пневматизации',
    LIN: "Линейные затемнения",
    NDL: 'Ограниченные затемнения',
    CAVITY: 'Воздушные полости',
    CONSOLIDATED_FRACTURE: 'Консолидированные переломы',
    FRACTURE: 'Переломы'
}

SAVE_PARAMS = {
    'pathologic_gt': True,
    'non_pathologic_gt': False,
    'fp': False,
    'fn': False,
    'tp': False,
    'tn': False,
    'comment': ''
}
