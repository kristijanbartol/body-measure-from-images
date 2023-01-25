from enum import Enum
from typing import List
import sys
import os
sys.path.append('/media/kristijan/kristijan-hdd-ex/ShapeFromImages')

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso
)
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, 
    AdaBoostRegressor,
    RandomForestRegressor
)
from xgboost import Booster as XGBoostRegressor


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

    
class AllModelsEnum():
    
    # Linear regression models
    linear_regressor = LinearRegression
    logistic_regressor = LogisticRegression
    ridge_regressor = Ridge
    lasso_regressor = Lasso
    
    # Neural network models
    # TODO: Add transformers and LSTMs.
    mlp_regressor = MLPRegressor
    
    # Ensemble models
    gradient_boosting_regressor = GradientBoostingRegressor
    ada_boost_regressor = AdaBoostRegressor
    random_forest_regressor = RandomForestRegressor
    xgboost_regressor = XGBoostRegressor
    
    @staticmethod
    def get_all_model_strings() -> List[str]:
        return [x for x in dir(AllModelsEnum) if 'regressor' in x]


MODEL_CHOICES = AllModelsEnum.get_all_model_strings()

BACKBONE_MODELS_CHOICES = [
    'gt',
    'deeplabv3',
    'maskrcnn',
    'pointrend'
]

CONFIGS_DIR = './configs/'
PRETRAINED_MODELS_DIR = './pretrained/'

POINTREND_CFG_PATH = os.path.join(CONFIGS_DIR, 'pointrend_rcnn_R_50_FPN_3x_coco.yaml')
MASKRCNN_CFG_PATH = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

MASKRCNN_MODEL_PATH = os.path.join(PRETRAINED_MODELS_DIR, 'mask_rcnn_balloon.h5')
POINTREND_MODEL_PATH = os.path.join(PRETRAINED_MODELS_DIR, 'model_final_edd263.pkl')
