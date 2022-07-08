from dataclasses import dataclass, fields
import os
import torch
from typing import Union
import cv2
import numpy as np

from src.features import count_human_pixels, get_height_in_image
from src.measures import MeasurementsCollection, MeshMeasurements


DATA_ROOT = './data/generated/'
BETAS_PATH = os.path.join(DATA_ROOT, 'betas.npy')
GENDERS_PATH = os.path.join(DATA_ROOT, 'genders.npy')
RGB_DIR = os.path.join(DATA_ROOT, 'rgb/')
SEG_DIR = os.path.join(DATA_ROOT, 'seg/')

INT_TO_GENDER = {
    1: 'male',
    2: 'female'
}


@dataclass
class TrainingData:
    front_densities: np.array
    side_densities: np.array
    measures: MeshMeasurements

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)


def extract_input(sample_idx: int) -> Union[float, float]:
    fname_front = f'{sample_idx:04d}_front.png'
    fname_side = f'{sample_idx:04d}_side.png'
    seg_front = cv2.imread(os.path.join(SEG_DIR, fname_front))
    seg_side = cv2.imread(os.path.join(SEG_DIR, fname_side))

    num_human_pixels_front = count_human_pixels(seg_front)
    num_human_pixels_side = count_human_pixels(seg_side)
    height_in_front_image = get_height_in_image(seg_front)
    height_in_side_image = get_height_in_image(seg_side)
    
    seg_density_front = float(num_human_pixels_front) / float(height_in_front_image)
    seg_density_side = float(num_human_pixels_side) / float(height_in_side_image)
    
    return seg_density_front, seg_density_side


def extract_measures(
        gender: str, 
        betas: np.ndarray
    ) -> MeshMeasurements:
    return MeshMeasurements(gender=gender, shape=betas)


def extract_training_data(data_type: str ='gt') -> TrainingData:
    all_betas = np.load(BETAS_PATH)
    all_genders = np.load(GENDERS_PATH)
    
    front_densities = []
    side_densities = []
    measurements_collection = []

    for sample_idx in range(all_betas.shape[0]):
        betas = torch.tensor(all_betas[sample_idx], dtype=torch.float32).unsqueeze(0)
        gender = INT_TO_GENDER[all_genders[sample_idx]]
        
        density_front, density_side = extract_input(sample_idx)
        
        front_densities.append(density_front)
        side_densities.append(density_side)
        
        measurements_collection.append(extract_measures(gender, betas))
        
    front_densities = np.array(front_densities)
    side_densities = np.array(side_densities)

    return TrainingData(
        front_densities=front_densities,
        side_densities=side_densities,
        measures=MeasurementsCollection(measurements_collection)
    )
    

if __name__ == '__main__':
    gt_data = extract_training_data()
