from re import sub
import numpy as np
from typing import List, Union, Optional
from sklearn.base import BaseEstimator
from sklearn.linear_model import (
    LinearRegression,
)
import xgboost as xgb
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.projects import point_rend
from detectron2.engine import DefaultPredictor

from .const import (
    MASKRCNN_CFG_PATH,
    POINTREND_CFG_PATH,
    POINTREND_MODEL_PATH
)


class Model(BaseEstimator):
    
    def __init__(
        self, 
        model: Optional[Union[BaseEstimator, xgb.Booster]] = LinearRegression
    ) -> None:
        self.model = model()
        self.fitted_model = None

    def fit(self, X, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.fitted_model = self.model.fit(X, y)
    
    def predict(self, X):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        return self.fitted_model.predict(X)
    
    def log(self):
        num_coefs = self.model.coef_.shape[0]
        for i in range(num_coefs):
            _coef = self.model.coef_[i][0]
            _intercept = self.model.intercept_[i]
            print(f'y{i} = {_coef:.2e}x + {_intercept:.2e}')


class BackboneModel():
    
    PERSON_CLASSES = {
        'deeplabv3': 15,
        'maskrcnn': 0,
        'pointrend': 0
    }
    
    def __init__(self, model_name: str = 'deeplabv3'):
        self.model_name = model_name

        self._model = None
        if model_name == 'deeplabv3':
            self._model = torch.hub.load(
                'pytorch/vision:v0.7.0', 
                'deeplabv3_resnet50', 
                pretrained=True
            )
            self._model.eval()
        elif model_name == 'maskrcnn':
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(MASKRCNN_CFG_PATH))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MASKRCNN_CFG_PATH)
            self._model = DefaultPredictor(cfg)
        elif model_name == 'pointrend':
            print('WARNING: PointRend not implemented.')
            self.model_name = 'gt'
            '''
            pointrend_config_file = "PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
            pointrend_cfg = get_cfg()
            add_pointrend_config(pointrend_cfg)
            pointrend_cfg.merge_from_file(pointrend_config_file)
            pointrend_cfg.MODEL.WEIGHTS = "checkpoints/pointrend_rcnn_R_50_fpn.pkl"
            pointrend_cfg.freeze()
            silhouette_predictor = DefaultPredictor(pointrend_cfg)
            '''
        else:
            print('NOTE: None of the silhouette models specified. Using GT silhouettes.')
            self.model_name = 'gt'
            
    def __call__(self, input_img):
        if self.model_name == 'deeplabv3':
            with torch.no_grad():
                silh_array = self._model(input_img)['out'][0].cpu().detach().numpy()
            argmax_array = np.argmax(silh_array, axis=0)
            return argmax_array == self.PERSON_CLASSES[self.model_name]
        else:
            silh_array = self._model(input_img)['instances'].pred_masks.cpu().detach().numpy()
            if silh_array.shape[0] > 0:
                return silh_array[self.PERSON_CLASSES[self.model_name]]
            else:
                return np.zeros((256, 256), dtype=bool)
