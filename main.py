import argparse
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from src.model import Model, BackboneModel
from src.evaluate import evaluate
from src.data import Dataset
from src.utils import dotdict
from src.const import (
    MODEL_CHOICES,
    OUTPUT_SET_CHOICES,
    FEATURE_TYPE_CHOICES,
    SEG_POSITION_CHOICES,
    BACKBONE_MODELS_CHOICES,
    AllModelsEnum
)


def run(args):
    silh_model = BackboneModel(args.backbone_model)
    X, y = Dataset(silh_model).get_data(
        feature_type=args.feature_type,
        seg_position=args.seg_position,
        output_set=args.output_set
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    model = Model(getattr(AllModelsEnum, args.model))
    model.fit(X_train, y_train)
    model.log()
    
    y_pred = model.predict(X_test)
    evaluate(y_pred, y_test, args.output_set)
    
    
def run_all():
    for are_gt_features in [True, False]:
        for feature_type in FEATURE_TYPE_CHOICES:
            for seg_position in SEG_POSITION_CHOICES:
                for output_set in OUTPUT_SET_CHOICES:
                    for model_str in MODEL_CHOICES:
                        args_dict = {
                            'gt_features': are_gt_features,
                            'feature_type': feature_type,
                            'seg_position': seg_position,
                            'output_set': output_set,
                            'model': model_str
                        }
                        args_dotdict = dotdict(args_dict)
                        run(args_dotdict)


if __name__ == '__main__':
    np.random.seed(2022)
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_model', type=str, default='gt',
                        choices=BACKBONE_MODELS_CHOICES,
                        help='which silhouette extraction model to use. In case of GT, ground-truth is used')
    parser.add_argument('--feature_type', type=str, default='density', 
                        choices=FEATURE_TYPE_CHOICES,
                        help='density is a single number per silhouette, slices use kpts, '
                             'fragments take each row of the silhouettes')
    parser.add_argument('--seg_position', type=str, default='both', 
                        choices=SEG_POSITION_CHOICES,
                        help='camera viewpoint, either front or side, or both')
    parser.add_argument('--output_set', type=str, default='all',
                        choices=OUTPUT_SET_CHOICES,
                        help='output data, either all measures, volume-only, or particular measure (A-P)')
    parser.add_argument('--model', type=str, default='linear_regressor',
                        choices=MODEL_CHOICES,
                        help='which regression model to fit')
    parser.add_argument('--run_all', dest='run_all', action='store_true',
                        help='whether to run all possible setup configurations')
    args = parser.parse_args()
    
    if args.run_all:
        run_all()
    else:
        run(args)
