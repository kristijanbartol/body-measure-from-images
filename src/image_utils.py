import numpy as np
import cv2

from torchvision import transforms


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def pad_to_square(image):
    """
    Pad image to square shape.
    """
    height, width = image.shape[:2]

    if width < height:
        border_width = (height - width) // 2
        image = cv2.copyMakeBorder(image, 0, 0, border_width, border_width,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        border_width = (width - height) // 2
        image = cv2.copyMakeBorder(image, border_width, border_width, 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)

    return image


def prepare_img(rgb_path, resize, to_tensor):
    rgb_img = cv2.imread(rgb_path)
    rgb_img = pad_to_square(rgb_img)
    if resize is not None:
        rgb_img_front = cv2.resize(
            rgb_img_front, 
            (resize, resize),
            interpolation=cv2.INTER_LINEAR)
    if to_tensor:
        rgb_img = preprocess(rgb_img).unsqueeze(0)
    return rgb_img


def bool_to_img(silh: np.array) -> np.array:
    return np.expand_dims(silh.astype(np.int8) * 255, axis=-1)


def img_to_bool(img: np.array) -> np.array:
    return img[:, :, 0].astype(bool)
