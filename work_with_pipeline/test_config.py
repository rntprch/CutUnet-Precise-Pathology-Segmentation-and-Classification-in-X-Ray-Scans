from Pipeline.configs.service import TRIAGE, NDL, PNTX, PLEF, CTR, MESH, SP, LIN, CAVITY, CONSOLIDATED_FRACTURE, \
    FRACTURE


# Path to initial data, save directory and markings dataframe
# PATH_TO_IMAGES = '/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/prl'
PATH_TO_IMAGES = '/home/nikita27/LungsDX/Research/Data/pipline/IndexError'
RESULTS_PATH = '/home/nikita27/LungsDX/Research/Data/pipline/results'
# RESULTS_PATH = '//home/nikita27/LungsDX/Research/Data/pipline/flg_opacity/'
GT_DATA_PATH = '/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/pred_conf'
# PRED_CONFIDENCE_PATH = 'pred_confidence2.csv'
PRED_CONFIDENCE_PATH = '/home/nikita27/LungsDX/Research/Data/pipline/test_dataset/pred_conf'
# actions to do
RUN = True
CALCULATE_METRICS = False
RUN_ON_IMAGE = False
TEST_ISSUES = False
SAVE_PREDICTIONS = False
# pipeline params
STUDY_TYPE = 'prl'
READ_SERIES = True
APPLY_CALIBRATION = True
RUN_FP_16 = True
TEST_MODE = False
SAVE_DICOM = False
INFERENCE_DEVICE = 0
FILTER_BY_NAME = ''
FILTER_BY_COMMENT = ''
FILTER_BY_DIRECTORY = []
GET_TP = False
GET_FP = False
GET_TN = False
GET_FN = False
NUM_OF_SAMPLES = 10
GET_PATHOLOGICAL_STUDIES = ''
IMAGE_NAME = 'sop_3c58145fc5651bd028aaee3d5f3d6c40.dcm'
# Metrics calculation params
# The range in which the search for the optimal threshold will be carried out
THRESHOLD_RANGE = (0.0, 1.)
DETECT_THRESHOLD_BY = 'roc_auc'
THRESHOLD_BALANCE_COEFFICIENT = 1.
USE_CUSTOM_THRESHOLDS = False
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
    FRACTURE: 'Неконсолидированные переломы'
}
PATHOLOGIES = {
    'opacity': [SP, LIN, NDL, CAVITY],
    'hemithorax': [PLEF, PNTX],
    'bones': [FRACTURE, CONSOLIDATED_FRACTURE],
    'heart': [CTR, MESH],
    'Other': [TRIAGE]
}

# SAVE_PARAMS = {
#     'pathologic_gt': True,
#     'non_pathologic_gt': False,
#     'fp': False,
#     'fn': False,
#     'tp': False,
#     'tn': False,
#     'comment': ''
# }
