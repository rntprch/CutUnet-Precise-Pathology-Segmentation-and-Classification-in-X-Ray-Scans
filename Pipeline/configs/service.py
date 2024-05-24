# Pathologies tag setting
SP = 'SP'
LIN = 'LIN'
NDL = 'NDL'
PLEF = 'PL_EF'
DIAPHM = 'DIAPHM'
PNTX = 'PNTX'
CTR = 'CTR'
MESH = 'MED_SH'
TRIAGE = 'OTHER'
CAVITY = 'CAVITY'
FRACTURE = 'FR'
CONSOLIDATED_FRACTURE = 'CFR'

PATHOLOGIES_IMPORTANCE = {
    PNTX: 10,
    NDL: 9,
    CAVITY: 8,
    SP: 7,
    LIN: 6,
    FRACTURE: 5,
    CONSOLIDATED_FRACTURE: 4,
    CTR: 4,
    TRIAGE: 3,
    PLEF: 2,
    MESH: 1,
    DIAPHM: 1,
}

# Here the preprocessing and pathologies models that will be used in service are set.
REQUIRED_PREPROCESSING_MODELS = {
    'alignmenter': False,
    'study_type': False
}
# Projection should be one of: frontal, lateral.
USE_PROJECTION = ['frontal', 'lateral']
AVALIABLE_PROJECTIONS = {
    'frontal': [TRIAGE, CTR, MESH, PLEF, DIAPHM, PNTX, SP, LIN, NDL, CAVITY, FRACTURE, CONSOLIDATED_FRACTURE],
    'lateral': [PLEF]
}
#  All pathologies
PATHOLOGIES = {
    'opacity': [SP, LIN, NDL, CAVITY],
    'hemithorax': [PLEF, DIAPHM, PNTX],
    'bones': [FRACTURE, CONSOLIDATED_FRACTURE],
    'heart': [CTR, MESH],
    'Other': [TRIAGE]
}
FRONTAL_PATHOLOGIES = {
    group: [
        pathology for pathology in pathologies if pathology in AVALIABLE_PROJECTIONS['frontal']
    ] for group, pathologies in PATHOLOGIES.items()
}
LATERAL_PATHOLOGIES = {
    group: [
        pathology for pathology in pathologies if pathology in AVALIABLE_PROJECTIONS['lateral']
    ] for group, pathologies in PATHOLOGIES.items()
}
PROJECTION_PATHOLOGIES = {'frontal': FRONTAL_PATHOLOGIES, 'lateral': LATERAL_PATHOLOGIES}
#  Required pathologies to be processed
REQUIRED_PATHOLOGIES = {
    'opacity': [SP, LIN, NDL, CAVITY],
    'hemithorax': [PLEF, DIAPHM, PNTX],
    'bones': [FRACTURE, CONSOLIDATED_FRACTURE],
    'heart': [CTR],
    'Other': [TRIAGE]
}

REQUIRED_FRONTAL_PATHOLOGIES = {
    group: [
        pathology for pathology in pathologies if pathology in AVALIABLE_PROJECTIONS['frontal']
    ] for group, pathologies in REQUIRED_PATHOLOGIES.items()
}
REQUIRED_LATERAL_PATHOLOGIES = {
    group: [
        pathology for pathology in pathologies if pathology in AVALIABLE_PROJECTIONS['lateral']
    ] for group, pathologies in REQUIRED_PATHOLOGIES.items()
}
REQUIRED_PROJECTION_PATHOLOGIES = {'frontal': REQUIRED_FRONTAL_PATHOLOGIES, 'lateral': REQUIRED_LATERAL_PATHOLOGIES}

NON_PATHOLOGICAL_PATHOLOGIES = [DIAPHM, CONSOLIDATED_FRACTURE]

PATHOLOGY_LOCALIZATION = {
    PNTX: '',
    NDL: '',
    CAVITY: '',
    SP: '',
    LIN: '',
    FRACTURE: '',
    CONSOLIDATED_FRACTURE: '',
    CTR: '',
    TRIAGE: '',
    PLEF: 'lungs',
    MESH: '',
    DIAPHM: '',
}

PATHOLOGY_TAG_NAME = {
    TRIAGE: f"Недифференцируемые\nсервисом признаки ({TRIAGE})",
    CTR: f'Кардиомегалия ({CTR})',
    MESH: f'Смещение средостения ({MESH})',
    PLEF: f'Плевральный выпот ({PLEF})',
    DIAPHM: f'Изменения диафрагмы ({DIAPHM})',
    PNTX: f'Пневмоторакс ({PNTX})',
    SP: f"Снижение пневматизации ({SP})",
    LIN: f"Линейное затемнение ({LIN})",
    NDL: f'Очаговое затемнение ({NDL})',
    CAVITY: f'Воздушная полость ({CAVITY})',
    FRACTURE: f'Перелом ({FRACTURE})',
    CONSOLIDATED_FRACTURE: f'Консолидированный перелом ({CONSOLIDATED_FRACTURE})',
}

GROUP_TAG_NAME = {
    "opacity": "Изменения в лёгких:",
    "hemithorax": "Патологии плевры и диафрагмы:",
    'bones': 'Патологии костей:',
    "heart": "Патологии сердца:",
    "Other": "Другие изменения:"
}

# Mapping of warning flags
WARNING_FLAGS = {
    1: 'Невозможно считать изображение!', # Images error
    2: 'Входное изображение не является\nрентгеном грудной клетки!', # Body part error
    3: 'Рентген грудной клетки сильно обрезан\n'
       'или имеет иные серьёзные дефекты\nне позволяющие произвести диагностику!', # Images error
    4: 'Данная проекция лёгких на рентгене\nне обрабатывается сервисом!', # Images error
    5: 'Рентген грудной клетки может быть\nнезначительно обрезан или иметь другие '
       '\nдефекты, которые затрудняют диагностику!', # ---
    6: 'Невозможно определить CTR\nдля данного изображения!', # ---
    7: 'Сегментация легких работает не корректно!\n',#---
    8: 'Выбран более валидный снимок!', # Other
    9: 'На снимке есть патологии,\nно они не были обнаружены!', # only for testing research pipeline!
    10: 'На снимке нет патологий,\nложноположительное срабатывание!', # only for testing research pipeline!
    11: 'Симметрия лёгочных полей:' # only for testing research pipeline.
}

WARNING_FLAGS_ENGLISH = {
    1: "Can't read the image",
    2: "Not the X-ray of lungs",
    3: "Heavily cropped image",
    4: "Wrong projection",
    5: "Slightly cropped image",
    6: "Can't count CTR",
    7: "Lungs segmentation didn't work",
    8: 'Found more valid X-ray',
    9: 'False negative!', # only for testing research pipeline!
    10: 'False positive!' # only for testing research pipeline!
}
