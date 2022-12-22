from enum import Enum
import sys
sys.path.append('/media/kristijan/kristijan-hdd-ex/ShapeFromImages')

from src.model import AllModelsEnum


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
