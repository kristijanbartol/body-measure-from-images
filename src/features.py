from functools import cached_property
import os
from typing import List, Tuple
import cv2
import numpy as np
import torch

if __name__ == '__main__':
    import sys
    sys.path.append('/media/kristijan/kristijan-hdd-ex/ShapeFromImages')

from image_utils import pad_to_square


IMG_DIR = './demo'
RESULT_DIR = os.path.join(IMG_DIR, 'result/')
INPUT_WH = 512

DATA_ROOT = './data/generated/'


class Features():
    
    _INT_TO_GENDER = {
        1: 'male',
        2: 'female'
    }
    
    POSITION_INDICES = {
        'front': 0,
        'side': 1
    }
    
    def __init__(
            self, 
            sample_idx: int = 0, 
            are_gt: bool = True, 
            silh_model = '',
            resize: int = None, 
            data_root: str = DATA_ROOT,
            auto_flush: bool = True
        ) -> None:
        '''Constructor for the single-sample features.
        
        Args:
            sample_idx: int
                The index of a sample used as an ID.
            are_gt: bool (default=True)
                Whether to use GT features for the sample.
            resize: int (default=None)
                The size to which to resize the square image.
            data_root: str (default=`DATA_ROOT`)
                The root data directory.
        '''
        self.sample_idx = sample_idx
        self.are_gt = are_gt
        self.resize = resize
        self.silh_model = silh_model
        if self.are_gt is False and silh_model == '':
            print(f'Setting default silhouette detector: {Features.POINTREND}')
            self.silh_model = Features.POINTREND
        if self.are_gt is True and silh_model != '':
            print('WARNING: GT features selected, but silhouette detector is not None!')
            print(f'NOTE: Setting GT features to False and using detector: {self.silh_model}')
            self.are_gt = False
            assert(silh_model in [Features.POINTREND, Features.MASK_RCNN])
        self.data_root = data_root
        self.rgb_dir = os.path.join(data_root, 'rgb/')
        
        if not self.are_gt:
            #self.joints_predictor, self.silhouette_predictor = \
            #    self.setup_detectron2_predictors(self.silh_model)
            self.silhouette_predictor = torch.hub.load(
                'pytorch/vision:v0.7.0', 'deeplabv3_resnet50', pretrained=True)
            self.seg_dir = os.path.join(self.data_root, f'seg_{self.silh_model}/')
        else:
            self.seg_dir = os.path.join(self.data_root, 'seg_gt/')
            
        self.cached_attributes = {}
        self.auto_flush = auto_flush
        
    def flush(self):
        self.cached_attributes = {}
    
    @cached_property
    def silhouettes(self):
        cached_attr_name = f'silhouettes_{self.silh_model}'
        if cached_attr_name not in self.cached_attributes:    
            fname_front = f'{self.sample_idx:04d}_front.png'
            fname_side = f'{self.sample_idx:04d}_side.png'
            silh_name_front = f'{self.sample_idx:04d}{self.silh_model}_front.png'
            silh_name_side = f'{self.sample_idx:04d}{self.silh_model}_side.png'
            seg_path_front = os.path.join(self.seg_dir, silh_name_front)
            seg_path_side = os.path.join(self.seg_dir, silh_name_side)
            
            if not self.are_gt:
                if not os.path.exists(seg_path_front) \
                        or not os.path.exists(seg_path_side):
                    rgb_img_front = cv2.imread(
                        os.path.join(self.rgb_dir, fname_front))
                    rgb_img_side = cv2.imread(
                        os.path.join(self.rgb_dir, fname_side))
                
                    # Preprocess for 2D detectors.
                    rgb_img_front = pad_to_square(rgb_img_front)
                    rgb_img_side = pad_to_square(rgb_img_side)
                    
                    if self.resize is not None:
                        rgb_img_front = cv2.resize(
                            rgb_img_front, 
                            (self.resize, self.resize),
                            interpolation=cv2.INTER_LINEAR)
                        rgb_img_side = cv2.resize(
                            rgb_img_side, 
                            (self.resize, self.resize),
                            interpolation=cv2.INTER_LINEAR)
                    seg_front, _ = self.predict_silhouette(
                        rgb_img_front, 
                        self.silhouette_predictor)
                    seg_side, _ = self.predict_silhouette(
                        rgb_img_side, 
                        self.silhouette_predictor)
                else:
                    print(f'Loading pre-extracted segmentations: {seg_path_front}')
                    seg_front = cv2.imread(seg_path_front)
                    seg_side = cv2.imread(seg_path_side)
            else:
                print(f'Loading pre-calculated GT segmentations: {seg_path_front}')
                seg_front = cv2.imread(seg_path_front)
                seg_side = cv2.imread(seg_path_side)
                
            self.cached_attributes[cached_attr_name] = [
                seg_front,
                seg_side
            ]
        
        return self.cached_attributes[cached_attr_name]
    
    @cached_property
    def human_pixels_count(self) -> Tuple[int, int]:
        return [np.count_nonzero(
            np.count_nonzero(
                x, 
                axis=2)
            ) for x in self.silhouettes]
        
    @cached_property
    def human_pixels_per_row(self) -> Tuple[np.ndarray, np.ndarray]:
        return [np.count_nonzero(
            np.count_nonzero(
                x, 
                axis=2), 
            axis=1) for x in self.silhouettes]
    
    @cached_property
    def human_height_in_image(self) -> Tuple[int, int]:
        return [np.count_nonzero(
            np.count_nonzero(
                np.count_nonzero(
                    x, 
                    axis=2), 
                axis=1)
            ) for x in self.silhouettes]
        
    @cached_property
    def seg_density(self) -> Tuple[float, float]:
        density = [float(x) / float(y) for x, y in zip(
            self.human_pixels_count, self.human_height_in_image)]
        if self.auto_flush:
            self.flush()
        return density
    
    @cached_property
    def seg_fragments(self) -> Tuple[np.ndarray, np.ndarray]:
        fragments = [x / y for x, y in zip(
            self.human_pixels_per_row, self.human_height_in_image
        )]
        if self.auto_flush:
            self.flush()
        return fragments
    
    
class FeaturesCollection():
    
    def __init__(self, features_objects: List[Features]):
        self._objects = features_objects
        
    @cached_property
    def seg_density(self):
        return np.array([x.seg_density for x in self._objects])
    
    @cached_property
    def seg_fragments(self):
        return np.array([x.seg_fragments for x in self._objects])


if __name__ == '__main__':
    silh_model = 'pointrend'
    if silh_model == 'pointrend':
        # .venv pip environment
        features = Features(0, are_gt=False)
        print(features.seg_density)
        
        #features_array = [Features(x, auto_flush=False) for x in range(10)]
        #collection = FeaturesCollection(features_array)
        #print(collection.seg_density)
        #print(collection.seg_density.shape)
        
        #print(collection.seg_fragments.shape)
        
        #Features(are_gt=False).silhouettes
    else:
        # shape-from-images conda environment
        mask_rcnn_features = Features(0, silh_model='mask_rcnn')
        print(mask_rcnn_features.seg_density)
