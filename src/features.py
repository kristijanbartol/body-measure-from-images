import os
from typing import Dict, List
import cv2
import numpy as np
import torch

if __name__ == '__main__':
    import sys
    sys.path.append('/media/kristijan/kristijan-hdd-ex/ShapeFromImages')

from src.image_utils import prepare_img
from src.utils import prepare_paths


IMG_DIR = './demo'
RESULT_DIR = os.path.join(IMG_DIR, 'result/')
INPUT_WH = 512

DATA_ROOT = './dataset/generated/'

POSITION_INDICES = {
    'front': 0,
    'side': 1
}


def get_human_pixels_count(silh: np.ndarray) -> int:
    return np.count_nonzero(np.count_nonzero(silh, axis=2))
    

def get_human_pixels_per_row(silh: np.ndarray) -> np.ndarray:
    return np.count_nonzero(np.count_nonzero(silh, axis=2), axis=1)


def get_human_height_in_image(silh: np.ndarray) -> int:
    return np.count_nonzero(np.count_nonzero(
        np.count_nonzero(silh, axis=2), 
            axis=1))
    

def get_seg_density(
        pixels_count: int, 
        height_in_img: int
    ) -> float:
    return float(pixels_count) / float(height_in_img)


def get_seg_fragments(
        pixels_per_row: np.ndarray, 
        height_in_img: int
    ) -> np.ndarray:
    return pixels_per_row.astype(np.float32) / float(height_in_img)


def collect_features(silh):
    pixels_count = get_human_pixels_count(silh)
    pixels_per_row = get_human_pixels_per_row(silh)
    height_in_image = get_human_height_in_image(silh)
    seg_density = get_seg_density(pixels_count, height_in_image)
    seg_fragments = get_seg_fragments(pixels_per_row, height_in_image)
    return {
        'human_pixels_count': pixels_count,
        'human_pixels_per_row': pixels_per_row,
        'human_height_in_image': height_in_image,
        'seg_density': seg_density,
        'seg_fragments': seg_fragments
    }


def extract_silhouette(silh_model, rgb_path, resize):
    input_batch = prepare_img(rgb_path, resize)
    with torch.no_grad():
        silh = silh_model(input_batch)['out'][0]
    return silh


def extract_features(data_root, sample_idx, resize, silh_model=None):
    paths = prepare_paths(data_root, sample_idx, silh_model is None)
    
    if not os.path.exists(paths['seg_path_front']) \
            or not os.path.exists(paths['seg_path_side']):
        print(f'Extracting silhouette #{sample_idx} using DeepLabv3...')
        front_silh = extract_silhouette(
            silh_model, paths['rgb_path_front'], resize)
        side_silh = extract_silhouette(
            silh_model, paths['rgb_path_side'], resize)
    else:
        print(f'Loading pre-extracted silhouettes...')
        front_silh = cv2.imread(paths['seg_path_front'])
        side_silh = cv2.imread(paths['seg_path_side'])
        
    front_feats = collect_features(front_silh)
    side_feats = collect_features(side_silh)
    
    return {
        'front': front_feats,
        'side': side_feats
    }
    

def get_features_array(
        features_list: List[Dict], 
        feature_type: str, 
        seg_position: str
    ) -> np.ndarray:
    if seg_position == 'both':
        seg_positions = ['front', 'side']
    else:
        seg_positions = [seg_position]
    features_array = np.empty(
        (len(seg_positions), len(features_list)), dtype=np.float32
    )
    for pos_idx, pos in enumerate(seg_positions):
        for sub_idx in len(features_list):
            features_array[pos_idx][sub_idx] = features_list[sub_idx][pos][feature_type]
    return features_array
