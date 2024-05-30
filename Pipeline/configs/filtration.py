from Pipeline.configs.service import *


# Parameter indicating whether it is necessary to filter pathology masks by lungs' segmentation mask.
FILTER_PATHOLOGY_MASK = True
# Threshold below which the CTI calculation will not be performed.
CARDIOMEGALY_LUNGS_SEGMENT_QUALITY = 0.85
# Thresholds used in image validator
VALIDATOR_THRESHOLD = {'stage1': 0.5, 'stage2': 0.5, 'stage3': 0.5, 'stage4': 0.5}
# Thresholds used for lungs and heart segmentation masks
LUNGS_THRESHOLD = 0.5
# Cardiomegaly CTI threshold
CARDIOMEGALY_THRESHOLDS = {
    0.51: 0.6,
    0.54: 0.75,
    0.55: 0.95,
    0.56: 1.0
}

MEDIASTINAL_THRESHOLD = {
    0.6: [0.07, 0.2],
    1.0: [0.03, 0.1]
}
HEART_TOP_PARAM = 0.25

# Binary triage threshold
TRIAGE_THRESHOLD = 0.5
# Areas whose dimensions are less than this threshold will be discarded when filtering the segmental mask of the lungs
# and heart.
SEGMENTATION_SIZE_THRESHOLD = 0.005
# Threshold by the area of intersection for filtering the mask of binary triage by pathologies of hemithorax and
# cardiomegaly.
TRIAGE_INTERSECTION_THRESHOLD = 0.75
# pathologies thresholds.
RG_PATHOLOGY_THRESHOLD = {
    PLEF: [0.4, 0.508],
    DIAPHM: [0.55, 0.55],
    PNTX: [0.35, 0.486],
    SP: [0.3, 0.487],
    LIN: [0.2, 0.450],
    NDL: [0.35, 0.453],
    CAVITY: [0.35, 0.5],
    CONSOLIDATED_FRACTURE: [0.5, 0.7],
    FRACTURE: [0.5, 0.7],
    TRIAGE: [0.5],
    CTR: [0.6],
    MESH: [0.6]
}

FLG_PATHOLOGY_THRESHOLD = {
    PLEF: [0.4, 0.45],
    DIAPHM: [0.65, 0.65],
    PNTX: [0.25, 0.45],
    SP: [0.4, 0.5],
    LIN: [0.25, 0.4],
    NDL: [0.35, 0.45],
    CONSOLIDATED_FRACTURE: [0.83, 0.963],
    FRACTURE: [0.755, 0.873],
    TRIAGE: [0.5],
    CTR: [0.6],
    CAVITY: [0.35, 0.5],
    MESH: [0.6]
}
# RG_PATHOLOGY_THRESHOLD = FLG_PATHOLOGY_THRESHOLD
# Parameter that sets number of points in approximated contour. Used for diaphragm, roi and pleural fluid masks
# creation.
CONTOUR_APPROX_PARAM = 0.08

LOCAL_FILTRATION_PARAMS = {
    PLEF: [0.05, 0.02],
    PNTX: [0.03, 0.02],
    DIAPHM: [0.02, 0.02],
    SP: [0.05, 0.02],
    LIN: [0.03, 0.02],
    NDL: [0.02, 0.02],
    CONSOLIDATED_FRACTURE: [0.02, 0.02],
    FRACTURE: [0.02, 0.02],
    CAVITY: [0.02, 0.02],
    CTR: [0, 0],
    MESH: [0, 0],
    TRIAGE: [0.05, 0.02]
}

PATHOLOGY_FILTRATION_PARAMS = {
    f'{TRIAGE}-{CTR}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{PLEF}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{DIAPHM}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{PNTX}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{SP}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{LIN}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{NDL}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{CAVITY}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{FRACTURE}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{TRIAGE}-{CONSOLIDATED_FRACTURE}': {'ioa': 0.3, 'inv_ioa': 0.3},
    f'{DIAPHM}-{TRIAGE}': {'ioa': 0.01, 'inv_ioa': 0.01},
    f'{DIAPHM}-{PLEF}': {'ioa': 0.01, 'inv_ioa': 0.01},
    f'{DIAPHM}-{PNTX}': {'ioa': 0.01, 'inv_ioa': 0.01},
    f'{DIAPHM}-{SP}': {'ioa': 0.01, 'inv_ioa': 0.01},
    f'{DIAPHM}-{LIN}': {'ioa': 0.01, 'inv_ioa': 0.01},
    f'{DIAPHM}-{NDL}': {'ioa': 0.01, 'inv_ioa': 0.01}
}
# PATHOLOGY_FILTRATION_PARAMS = {}
