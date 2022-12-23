import os
import torch
from typing import Dict, List, Tuple
from numbers import Number
import numpy as np
from torchvision.models.segmentation.deeplabv3 import DeepLabV3

if __name__ == '__main__':
    import sys
    sys.path.append('/media/kristijan/kristijan-hdd-ex/ShapeFromImages')

from src.features import extract_features, get_features_array
from src.measures import extract_measures, get_measurements_array


DATA_ROOT = './dataset/generated/'
BETAS_PATH = os.path.join(DATA_ROOT, 'betas.npy')
GENDERS_PATH = os.path.join(DATA_ROOT, 'genders.npy')
    
    
class Dataset():
    
    def __init__(
            self, 
            silh_model: DeepLabV3 = None,
            img_size: int = None
        ) -> None:
        self.img_size = img_size
        self.data_root = './dataset/generated/'
        self.training_data = self._extract_data(
            silh_model)
    
    def _extract_data(
            self,
            silh_model: DeepLabV3 = None
        ) -> Tuple[List[Dict[str, Dict[str, Number]]], 
                   List[Dict[str, Dict[str, Number]]]]:
        all_betas = np.load(BETAS_PATH)
        all_genders = np.load(GENDERS_PATH)
        
        features_list = []
        measures_list = []

        for sample_idx in range(all_betas.shape[0]):
            features_list.append(
                extract_features(
                    self.data_root, 
                    sample_idx, 
                    self.img_size, 
                    silh_model
                )
            )
            measures_list.append(
                extract_measures(
                    gender=all_genders[sample_idx],
                    betas=all_betas[sample_idx]
                )
            )

        return {
            'features': features_list,
            'measures': measures_list
        }
        
    def get_data(
            self, 
            feature_type: str = 'density',
            seg_position: str = 'both',
            output_set: str = 'all'
        ) -> Tuple[np.ndarray, np.ndarray]:
        features_array = get_features_array(
            self.training_data['features'], feature_type, seg_position)      
        measures_array = get_measurements_array(
            self.training_data['measures'], output_set)
        
        return features_array, measures_array


if __name__ == '__main__':
    dataset = Dataset()
