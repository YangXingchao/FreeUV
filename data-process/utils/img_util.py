# -*- coding: utf-8 -*-

import cv2
import numpy as np


def tensor_to_img(img_tensor):
    img_numpy = img_tensor.clone().detach().cpu().numpy()
    img_numpy = (255.0 * img_numpy).clip(0, 255).astype(np.uint8)
    return img_numpy

def save_tensor(img_tensor, save_path):
    save_img = tensor_to_img(img_tensor)
    cv2.imwrite(save_path, save_img)
