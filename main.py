import os
import cv2
import numpy as np
import torch
from smplx.lbs import batch_rodrigues

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from PointRend.point_rend import add_pointrend_config

from utils.image_utils import pad_to_square


IMG_DIR = './demo'
RESULT_DIR = os.path.join(IMG_DIR, 'result/')
INPUT_WH = 512


def setup_detectron2_predictors():
    # Keypoint-RCNN
    kprcnn_config_file = "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
    kprcnn_cfg = get_cfg()
    kprcnn_cfg.merge_from_file(model_zoo.get_config_file(kprcnn_config_file))
    kprcnn_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
    kprcnn_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(kprcnn_config_file)
    kprcnn_cfg.freeze()
    joints2D_predictor = DefaultPredictor(kprcnn_cfg)

    # PointRend-RCNN-R50-FPN
    pointrend_config_file = "PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
    pointrend_cfg = get_cfg()
    add_pointrend_config(pointrend_cfg)
    pointrend_cfg.merge_from_file(pointrend_config_file)
    pointrend_cfg.MODEL.WEIGHTS = "checkpoints/pointrend_rcnn_R_50_fpn.pkl"
    pointrend_cfg.freeze()
    silhouette_predictor = DefaultPredictor(pointrend_cfg)

    return joints2D_predictor, silhouette_predictor


def get_largest_centred_bounding_box(bboxes, orig_w, orig_h):
    """
    Given an array of bounding boxes, return the index of the largest + roughly-centred
    bounding box.
    :param bboxes: (N, 4) array of [x1 y1 x2 y2] bounding boxes
    :param orig_w: original image width
    :param orig_h: original image height
    """
    bboxes_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_bbox_indices = np.argsort(bboxes_area)[::-1]  # Indices of bboxes sorted by area.
    bbox_found = False
    i = 0
    while not bbox_found and i < sorted_bbox_indices.shape[0]:
        bbox_index = sorted_bbox_indices[i]
        bbox = bboxes[bbox_index]
        bbox_centre = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)  # Centre (width, height)
        if abs(bbox_centre[0] - orig_w / 2.0) < orig_w/6.0 and abs(bbox_centre[1] - orig_h / 2.0) < orig_w/6.0:
            largest_centred_bbox_index = bbox_index
            bbox_found = True
        i += 1

    # If can't find bbox sufficiently close to centre, just use biggest bbox as prediction
    if not bbox_found:
        largest_centred_bbox_index = sorted_bbox_indices[0]

    return largest_centred_bbox_index


def predict_joints2D(input_image, predictor):
    """
    Predicts 2D joints (17 2D joints in COCO convention along with prediction confidence)
    given a cropped and centred input image.
    :param input_images: (wh, wh)
    :param predictor: instance of detectron2 DefaultPredictor class, created with the
    appropriate config file.
    """
    image = np.copy(input_image)
    orig_h, orig_w = image.shape[:2]
    outputs = predictor(image)  # Multiple bboxes + keypoints predictions if there are multiple people in the image
    bboxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    if bboxes.shape[0] == 0:  # Can't find any people in image
        keypoints = np.zeros((17, 3))
    else:
        largest_centred_bbox_index = get_largest_centred_bounding_box(bboxes, orig_w, orig_h)  # Picks out centred person that is largest in the image.
        keypoints = outputs['instances'].pred_keypoints.cpu().numpy()
        keypoints = keypoints[largest_centred_bbox_index]
        
        print(keypoints.dtype)

        for j in range(keypoints.shape[0]):
            cv2.circle(image, (int(keypoints[j, 0]), int(keypoints[j, 1])), 5, (0, 255, 0), -1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            fontColor = (0, 0, 255)
            cv2.putText(image, str(j), (int(keypoints[j, 0]), int(keypoints[j, 1])),
                                     font, fontScale, fontColor, lineType=2)

    return keypoints, image


def get_largest_centred_mask(human_masks, orig_w, orig_h):
    """
    Given an array of human segmentation masks, return the index of the largest +
    roughly-centred mask.
    :param human_masks: (N, img_wh, img_wh) human segmentation masks.
    :param orig_w: original image width
    :param orig_h: original image height
    """
    mask_areas = np.sum(human_masks, axis=(1, 2))
    sorted_mask_indices = np.argsort(mask_areas)[::-1]  # Indices of masks sorted by area.
    mask_found = False
    i = 0
    while not mask_found and i < sorted_mask_indices.shape[0]:
        mask_index = sorted_mask_indices[i]
        mask = human_masks[mask_index, :, :]
        mask_pixels = np.argwhere(mask != 0)
        bbox_corners = np.amin(mask_pixels, axis=0), np.amax(mask_pixels, axis=0)  # (row_min, col_min), (row_max, col_max)
        bbox_centre = ((bbox_corners[0][0] + bbox_corners[1][0]) / 2.0,
                       (bbox_corners[0][1] + bbox_corners[1][1]) / 2.0)  # Centre in rows, columns (i.e. height, width)

        if abs(bbox_centre[0] - orig_h / 2.0) < orig_w/4.0 and abs(bbox_centre[1] - orig_w / 2.0) < orig_w/6.0:
            largest_centred_mask_index = mask_index
            mask_found = True
        i += 1

    # If can't find mask sufficiently close to centre, just use biggest mask as prediction
    if not mask_found:
        largest_centred_mask_index = sorted_mask_indices[0]

    return largest_centred_mask_index


def predict_silhouette_pointrend(input_image, predictor):
    """
    Predicts human silhouette (binary segmetnation) given a cropped and centred input image.
    :param input_images: (wh, wh)
    :param predictor: instance of detectron2 DefaultPredictor class, created with the
    appropriate config file.
    """
    orig_h, orig_w = input_image.shape[:2]
    outputs = predictor(input_image)['instances']  # Multiple silhouette predictions if there are multiple people in the image
    classes = outputs.pred_classes
    masks = outputs.pred_masks
    human_masks = masks[classes == 0]
    human_masks = human_masks.cpu().detach().numpy()
    largest_centred_mask_index = get_largest_centred_mask(human_masks, orig_w, orig_h)  # Picks out centred person that is largest in the image.
    human_mask = human_masks[largest_centred_mask_index, :, :].astype(np.uint8)
    overlay_vis = cv2.addWeighted(input_image, 1.0,
                              255 * np.tile(human_mask[:, :, None], [1, 1, 3]),
                              0.5, gamma=0)

    return human_mask, overlay_vis


if __name__ == '__main__':
    image_fnames = [f for f in sorted(os.listdir(IMG_DIR)) if f.endswith('.png') or
                        f.endswith('.jpg')]
    
    # Set-up proxy representation predictors.
    joints2D_predictor, silhouette_predictor = setup_detectron2_predictors()
    
    for fname in image_fnames:
        print("Predicting on:", fname)
        image = cv2.imread(os.path.join(IMG_DIR, fname))
        
        # Preprocess for 2D detectors.
        image = pad_to_square(image)
        image = cv2.resize(image, (INPUT_WH, INPUT_WH),
                        interpolation=cv2.INTER_LINEAR)
        
        print(type(image))
        
        # Predict joints.
        joints2D, joints2D_vis = predict_joints2D(image, joints2D_predictor)

        # Predict silhouette.
        silhouette, silhouette_vis = predict_silhouette_pointrend(image,
                                                                silhouette_predictor)
        
        cv2.imwrite(os.path.join(RESULT_DIR, f'{fname}_joints.png'), joints2D_vis)
        cv2.imwrite(os.path.join(RESULT_DIR, f'{fname}_silh.png'), silhouette_vis)
