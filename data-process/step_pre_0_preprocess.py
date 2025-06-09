import argparse
import os

import cv2
import numpy as np

from libs.face_parsing.segment import segment
from libs.face_landmark import landmark
import config


def mirror_and_blur_border(img, top, bottom, left, right, blur_size=(11, 11)):
    # Step 1: Perform mirror padding using BORDER_REFLECT
    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    
    # Step 2: Apply Gaussian blur to the border regions
    if top > 0:
        padded_img[:top, :] = cv2.GaussianBlur(padded_img[:top, :], blur_size, 0)
    if bottom > 0:
        padded_img[-bottom:, :] = cv2.GaussianBlur(padded_img[-bottom:, :], blur_size, 0)
    if left > 0:
        padded_img[:, :left] = cv2.GaussianBlur(padded_img[:, :left], blur_size, 0)
    if right > 0:
        padded_img[:, -right:] = cv2.GaussianBlur(padded_img[:, -right:], blur_size, 0)

    return padded_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess image')
    parser.add_argument('--img_path', '-i', default='./sample/sample_img.jpg', help='Input image')
    parser.add_argument('--out_dir', '-o', default='./results', help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # crop
    img = cv2.imread(args.img_path)
    img_size = img.shape[0]
    kpt_size = config.KPT_SIZE
    resized_img = cv2.resize(img, (kpt_size, kpt_size))

    predicts = landmark.detect_landmark(resized_img)
    kpt = predicts * img_size / kpt_size
    kpt_width = abs(np.max(kpt[:, 0]) - np.min(kpt[:, 0]))
    kpt_hight = abs(np.max(kpt[:, 1]) - np.min(kpt[:, 1]))
    kpt_range = max(kpt_width, kpt_hight)

    crop_size = (kpt_range * config.CROP_RATIO).astype(np.int32)
    kpt_center = kpt[30].astype(np.int32)

    up = kpt_center[1] - crop_size // 2
    down = kpt_center[1] + crop_size // 2
    left = kpt_center[0] - crop_size // 2
    right = kpt_center[0] + crop_size // 2

    pad_u = abs(up - 0) if up - 0 < 0 else 0
    pad_d = abs(img.shape[1] - down) if img.shape[1] - down < 0 else 0
    pad_l = abs(left - 0) if left - 0 < 0 else 0
    pad_r = abs(img.shape[0] - right) if img.shape[0] - right < 0 else 0

    kpt_pad_center = kpt_center.copy()
    kpt_pad_center[1] = kpt_center[1] + pad_u
    kpt_pad_center[0] = kpt_center[0] + pad_l

    pad_img = mirror_and_blur_border(img, pad_u, pad_d, pad_l, pad_r)

    crop_img = pad_img[kpt_pad_center[1] - crop_size // 2:kpt_pad_center[1] + crop_size // 2,
               kpt_pad_center[0] - crop_size // 2:kpt_pad_center[0] + crop_size // 2, :]

    crop_size_h = crop_img.shape[0]
    crop_size_w = crop_img.shape[1]
    half_min_crop_size = min(crop_size_h, crop_size_w) // 2
    half_max_crop_size = max(crop_size_h, crop_size_w) // 2
    margin = half_max_crop_size - half_min_crop_size
    if crop_size_w > crop_size_h:
        aligned_crop_img = crop_img[:, margin:crop_size_w - margin]
    else:
        aligned_crop_img = crop_img[margin:crop_size_h - margin, :]
    out_aligned_img_path = f'{out_dir}/aligned_img.jpg'
    cv2.imwrite(out_aligned_img_path, aligned_crop_img)

    # segment
    seg_size = config.SEG_SIZE
    seg_img, mask_img = segment(out_aligned_img_path, seg_size)
    cv2.imwrite(f'{out_dir}/segmented_img.jpg', seg_img)
    cv2.imwrite(f'{out_dir}/mask_img.jpg', mask_img)