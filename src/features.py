import os
from typing import Dict, List
import cv2
import numpy as np
import torch

from src.image_utils import (
    prepare_img, 
    bool_to_img,
    img_to_bool
)
from src.utils import prepare_paths


IMG_DIR = './demo'
RESULT_DIR = os.path.join(IMG_DIR, 'result/')
INPUT_WH = 512
PERSON_CLASS = 15

DATA_ROOT = './dataset/generated/'

POSITION_INDICES = {
    'front': 0,
    'side': 1
}


def get_human_pixels_count(silh: np.ndarray) -> int:
    return np.sum(silh)
    

def get_human_pixels_per_row(silh: np.ndarray) -> np.ndarray:
    return np.count_nonzero(silh, axis=1)


def get_pixel_height(silh: np.ndarray) -> int:
    return np.count_nonzero(np.count_nonzero(silh, axis=1), axis=0)
    

def get_seg_density(
        pixels_count: int, 
        height_in_img: int
    ) -> float:
    if pixels_count > 0:
        return float(pixels_count) / float(height_in_img)
    else:
        return 0.


def get_seg_fragments(
        pixels_per_row: np.ndarray, 
        height_in_img: int
    ) -> np.ndarray:
    if height_in_img > 0:
        return pixels_per_row.astype(np.float32) / float(height_in_img)
    else:
        return 0


def collect_features(silh):
    pixels_count = get_human_pixels_count(silh)
    pixels_per_row = get_human_pixels_per_row(silh)
    pixel_height = get_pixel_height(silh)
    seg_density = get_seg_density(pixels_count, pixel_height)
    seg_fragments = get_seg_fragments(pixels_per_row, pixel_height)
    return {
        'pixels_count': pixels_count,
        'pixels_per_row': pixels_per_row,
        'pixel_height': pixel_height,
        'density': seg_density,
        'fragments': seg_fragments
    }


def extract_silhouette(silh_model, rgb_path, resize):
    input_batch = prepare_img(rgb_path, resize)
    with torch.no_grad():
        silh_array = silh_model(input_batch)['out'][0].cpu().detach().numpy()
    argmax_array = np.argmax(silh_array, axis=0)
    return argmax_array == PERSON_CLASS


def extract_features(data_root, sample_idx, resize, silh_model=None):
    paths = prepare_paths(data_root, sample_idx, silh_model is None)
    
    if not os.path.exists(paths['seg_path_front']) \
            or not os.path.exists(paths['seg_path_side']):
        print(f'Extracting silhouette #{sample_idx} using DeepLabv3...')
        front_silh = extract_silhouette(
            silh_model, paths['rgb_path_front'], resize)
        side_silh = extract_silhouette(
            silh_model, paths['rgb_path_side'], resize)
        cv2.imwrite(paths['seg_path_front'], bool_to_img(front_silh))
        cv2.imwrite(paths['seg_path_side'], bool_to_img(side_silh))
    else:
        print(f'Loading pre-extracted silhouette #{sample_idx}...')
        front_silh = img_to_bool(cv2.imread(paths['seg_path_front']))
        side_silh = img_to_bool(cv2.imread(paths['seg_path_side']))
        
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
        (len(features_list), len(seg_positions)), dtype=np.float32
    )
    for sub_idx in range(len(features_list)):
        for pos_idx, pos in enumerate(seg_positions):
            features_array[sub_idx][pos_idx] = features_list[sub_idx][pos][feature_type]
    return features_array
