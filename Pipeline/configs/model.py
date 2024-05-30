import segmentation_models_pytorch as smp
from pkg_resources import resource_stream
from Pipeline.models.pathology_nn_models import CutUnet, DeeplabV3Plus, UnetPlusPlus, Unet, MANet, DeeplabV3, PSPNet, \
    PAN, LinkNet, FPN

PROJECT_NAME = 'Pipeline'
# Set device
# DEVICE = torch.device('cuda:0') if torch.cuda.is_available else torch.device('cpu')

# Projection should be one of: frontal, lateral.
USE_PROJECTION = ['frontal']

# Available models
MODELS = {
    'cut_unet': CutUnet,
    'deeplab': DeeplabV3Plus,
    'unet++': UnetPlusPlus,
    'unet': Unet,
    'deeplabv3': DeeplabV3,
    'manet': MANet,
    'linknet': LinkNet,
    'pspnet': PSPNet,
    'pan': PAN,
    'fpn': FPN
}

# Validator config
# Paths to validator weights
PATH_TO_MODEL_1 = resource_stream(PROJECT_NAME, 'weights/validator_stage1.pt').name
PATH_TO_MODEL_2 = resource_stream(PROJECT_NAME, 'weights/validator_stage2.pt').name
PATH_TO_MODEL_3 = resource_stream(PROJECT_NAME, 'weights/validator_stage3.pt').name
PATH_TO_MODEL_4 = resource_stream(PROJECT_NAME, 'weights/validator_stage4.pt').name
# Validator nn model names
ENCODER_1 = 'efficientnet_b0'
ENCODER_2 = 'efficientnet_b0'
ENCODER_3 = 'efficientnet_b3'
ENCODER_4 = 'efficientnet_b0'
# Number of classes for each validator model output
NUM_CLASSES_1 = 5
NUM_CLASSES_2 = 1
NUM_CLASSES_3 = 3
NUM_CLASSES_4 = 3
# Required image size by the model input
VALIDATOR_TARGET_SIZE = (3, 224, 224)
# Required by pretrain weights image preprocess function
VALIDATOR_PREPROCESSING_FN = smp.encoders.get_preprocessing_fn('efficientnet-b0', 'imagenet')

# Study type model config
# Paths to validator weights
PATH_TO_MODEL = resource_stream(PROJECT_NAME, 'weights/prl_flg.pt').name
# Neural network model name
ENCODER = 'efficientnet_b0'
# Number of classes for each validator model output
NUM_CLASSES = 1
# Required image size by the model input
TARGET_SIZE = (3, 224, 224)
# Required by pretrain weights image preprocess function
PREPROCESSING_FN = smp.encoders.get_preprocessing_fn('efficientnet-b0', 'imagenet')
THRESHOLD = 0.5

# Segmentation config
# Segmentation nn model encoder name
SEGMENTATION_ENCODER = 'timm-efficientnet-b5'
# Segmentation nn model name
SEGMENTATION_NN = smp.DeepLabV3Plus
# Required image size by the model input
SEGMENTER_TARGET_SIZE = (3, 640, 640)
# Activation function applied to the model output
SEGMENTATION_ACTIVATION = 'sigmoid'
# Number of classes for model output. It depends on which chest X-ray projection we are using the model for.
SEGMENTATION_NUM_CLASSES = [2, 1]
# Paths to segmentation model weights
FRONTAL_SEGMENTATION_MODEL_PATH = resource_stream(PROJECT_NAME, 'weights/frontal_lungs_segmentation.pt').name
LATERAL_SEGMENTATION_MODEL_PATH = resource_stream(PROJECT_NAME, 'weights/lateral_lungs_segmentation.pt').name

# alignment config
# Required image size by the model input
ALIGNMENT_TARGET_SIZE = (3, 256, 256)
# Required by pretrain weights image preprocess function
ALIGNMENT_PREPROCESSING_FN = 1 / 255

# Binary triage config
# Required image size by the model input
TRIAGE_TARGET_SIZE = (640, 640)
# Paths to binary triage model weights
PATH_TO_TRIAGE_MODEL = resource_stream(PROJECT_NAME, 'weights/bt.pth').name

# Hemithorax config
# Required image size by the model input
HEMITHORAX_TARGET_SIZE = (640, 640)
# Number of classes for model output
HEMITHORAX_NUM_CLASSES = 3
# Hemithorax nn model encoder name
HEMITHORAX_ENCODER = ['timm-efficientnet-b5']
# Paths to hemithorax model weights
PATH_TO_HEMITHORAX_MODEL = [
    resource_stream(PROJECT_NAME, 'weights/hemithorax_1.ckpt').name,
]
# pathology model architecture
HEMITHORAX_NN = ['cut_unet']

# Opacity config
# Required image size by the model input
OPACITY_TARGET_SIZE = (640, 640)
# Number of classes for model output
OPACITY_NUM_CLASSES = 2
# Hemithorax nn model encoder name
OPACITY_ENCODER = ['timm-efficientnet-b5', 'timm-efficientnet-b5']
# Paths to hemithorax model weights
PATH_TO_OPACITY_MODEL = [
    resource_stream(PROJECT_NAME, 'weights/model_1.ckpt').name,
    resource_stream(PROJECT_NAME, 'weights/model_2.ckpt').name
]
# pathology model architecture
OPACITY_NN = ['deeplab', 'unet++']
# calibration params path
OPACITY_CALIBRATION_PATH = [
    resource_stream(PROJECT_NAME, 'calibration_params/splin_1.json').name,
    resource_stream(PROJECT_NAME, 'calibration_params/splin_2.json').name
]

# Chest bones config
# Required image size by the model input
BONES_TARGET_SIZE = (640, 640)
# Number of classes for model output
BONES_NUM_CLASSES = 2
# Hemithorax nn model encoder name
BONES_ENCODER = ['timm-efficientnet-b5', 'timm-efficientnet-b5']
# Paths to hemithorax model weights
PATH_TO_BONES_MODEL = [
    resource_stream(PROJECT_NAME, 'weights/fracture_1.ckpt').name,
    resource_stream(PROJECT_NAME, 'weights/fracture_2.ckpt').name
]
# pathology model architecture
BONES_NN = ['deeplab', 'unet++']
# calibration params path
BONES_CALIBRATION_PATH = [
    resource_stream(PROJECT_NAME, 'calibration_params/fracture_1.json').name,
    resource_stream(PROJECT_NAME, 'calibration_params/fracture_2.json').name
]

# Cavity config
# Required image size by the model input
CAVITY_TARGET_SIZE = (640, 640)
# Number of classes for model output
CAVITY_NUM_CLASSES = 1
# Hemithorax nn model encoder name
CAVITY_ENCODER = ['timm-efficientnet-b5']
# Paths to  model weights
PATH_TO_CAVITY_MODEL = [resource_stream(PROJECT_NAME, 'weights/cavity.ckpt').name]
# pathology model architecture
CAVITY_NN = ['cut_unet']


# Ndl config
# Required image size by the model input
NDL_TARGET_SIZE = (640, 640)
# Number of classes for model output
NDL_NUM_CLASSES = 1
# Hemithorax nn model encoder name
NDL_ENCODER = ['timm-efficientnet-b5', 'timm-efficientnet-b5']
# Paths to  model weights
PATH_TO_NDL_MODEL = [
    resource_stream(PROJECT_NAME, 'weights/ndl_1.ckpt').name,
    resource_stream(PROJECT_NAME, 'weights/ndl_2.ckpt').name
]
# pathology model architecture
NDL_NN = ['deeplab', 'unet++']
# calibration params path
NDL_CALIBRATION_PATH = [
    resource_stream(PROJECT_NAME, 'calibration_params/ndl_1.json').name,
    resource_stream(PROJECT_NAME, 'calibration_params/ndl_2.json').name
]