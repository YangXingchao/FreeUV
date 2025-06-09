# -*- coding: utf-8 -*-

import cv2
import numpy as np
from python_color_transfer.masked_color_transfer import ColorTransfer


def color_transfer():
    
    img_path = './results/complete_uv.jpg'
    ref_path = './results/segmented_img.jpg'
    mask_ref_path = './results/mask_img.jpg'
    out_path = './results/color_transfer_uv.jpg'
    
    mask_in_path = './resources/croped_mask.jpg'
    mask_in = cv2.imread(mask_in_path, 0) 

    # cls init
    PT = ColorTransfer()
    try:
        # read input img
        img_arr_in = cv2.imread(img_path)
        [h, w, c] = img_arr_in.shape
        print(f"{img_path}: {h}x{w}x{c}")
        # read reference img
        img_arr_ref = cv2.imread(ref_path)
        [h, w, c] = img_arr_ref.shape
        print(f"{ref_path}: {h}x{w}x{c}")
        # read masks
        mask_ref = cv2.imread(mask_ref_path, 0)
        mask_ref = cv2.resize(mask_ref, (h, w))
        print(f"Mask_in non-zero count: {np.count_nonzero(mask_in)}")
        print(f"Mask_ref non-zero count: {np.count_nonzero(mask_ref)}")

        img_arr_lt = PT.lab_transfer(img_arr_in=img_arr_in,
                                    img_arr_ref=img_arr_ref,
                                    mask_in=mask_in,
                                    mask_ref=mask_ref)
        cv2.imwrite(out_path, img_arr_lt)
    except:
        print(f"Error processing {img_path} or {ref_path}")
        return
        

if __name__ == "__main__":
    color_transfer()