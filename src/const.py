from features import Features
from .model import AllModelsEnum


SILH_MODEL_CHOICES = [
    Features.POINTREND,
    Features.MASK_RCNN
]

FEATURE_TYPE_CHOICES = [
    'density', 
    'slices', 
    'fragments'
]

SEG_POSITION_CHOICES = [
    'both', 
    'front', 
    'side'
]

OUTPUT_SET_CHOICES = [
    'all', 
    'volume'
    ] + \
    [chr(x) for x in range(ord('A'), ord('P') + 1)]

MODEL_CHOICES = AllModelsEnum.get_all_model_strings()
