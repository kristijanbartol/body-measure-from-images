import os
import torch
from typing import Tuple
import numpy as np

if __name__ == '__main__':
    import sys
    sys.path.append('/media/kristijan/kristijan-hdd-ex/ShapeFromImages')

from src.features import Features, FeaturesCollection
from src.measures import MeasurementsCollection, MeshMeasurements


DATA_ROOT = './dataset/generated/'
BETAS_PATH = os.path.join(DATA_ROOT, 'betas.npy')
GENDERS_PATH = os.path.join(DATA_ROOT, 'genders.npy')
    

class DatasetSpecs():
    
    def __init__(
            self,
            gt_features: bool = True,
            feature_type: str = 'density',
            seg_position: str = 'both',
            output_set: str = 'all'     # all_measures
        ) -> None:
        self.gt_features = gt_features      # bool
        self.feature_type = feature_type    # density, slices, or fragments
        self.seg_position = seg_position    # front, side, or both
        self.output_set = output_set        # all, volume, or (A, B, ...)
    
    
class Dataset():
    
    def __init__(
            self, 
            are_gt_features: bool = True,
            feature_type: str = 'density'
        ) -> None:
        self.training_data = self._extract_data(
            are_gt_features)
    
    @staticmethod
    def _extract_data(
            are_gt: bool
        ) -> Tuple[FeaturesCollection, MeasurementsCollection]:
        all_betas = np.load(BETAS_PATH)
        all_genders = np.load(GENDERS_PATH)
        
        features_array = []
        measures_array = []

        for sample_idx in range(all_betas.shape[0]):
            features_array.append(
                Features(
                    sample_idx=sample_idx,
                    are_gt=are_gt)
                )
            measures_array.append(
                MeshMeasurements(
                    gender=MeshMeasurements.INT_TO_GENDER[all_genders[sample_idx]], 
                    shape=torch.tensor(all_betas[sample_idx], 
                                    dtype=torch.float32).unsqueeze(0))
                )

        return (
            FeaturesCollection(features_array), 
            MeasurementsCollection(measures_array)
        )
        
    def get_data(
            self, 
            feature_type: str = 'density',
            seg_position: str = 'both',
            output_set: str = 'all'
        ) -> Tuple[np.ndarray, np.ndarray]:
        features = self.training_data[0].seg_density
        if seg_position == 'front':
            features = features[:, Features.POSITION_INDICES['front']]
        elif seg_position == 'side':
            features = features[:, Features.POSITION_INDICES['side']]
        else:
            if not seg_position == 'both':
                print('WARNING: Wrong seg_position provided, returning all.')
        
        measures = self.training_data[1]
        if output_set == 'volume':
            measures = measures.volume
        elif len(output_set) == 1:
            measures = getattr(measures, output_set)
        elif output_set == 'all':
            measures = measures.all
        else:
            print('WARNING: Wrong output set label, returning all.')
            measures = measures.all
        
        return features, measures


if __name__ == '__main__':
    dataset = Dataset()
