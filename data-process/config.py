# -*- coding: utf-8 -*-

import os

# 3DMM definition
MM_PATH = './resources/generic_model.pkl'
LMK_PATH = './resources/landmark_embedding.npy'
MM_SPECULAR_PATH = './resources/albedoModel2020_FLAME_albedoPart.npz'
MM_DIFFUSE_PATH = './resources/FLAME_texture.npz'
MM_MASK_PATH = './resources/FLAME_masks.pkl'
SKIN_MASK_PATH = './resources/mask.jpg'
CROPED_MASKED_UV_PATH = '/resources/croped_masked_uv.jpg'

n_shape = 200
n_tex = 100
n_exp = 100
n_make = 100
MM_SCALE = 1600

UV_CROP_W = 320
UV_CROP_H = 320
UV_CROP_POS = [96, 96]

CROP_RATIO = 1.6
KPT_SIZE = 256
FIT_SIZE = 256
UV_SIZE = 256
SEG_SIZE = 512
RENDER_SIZE = 512

RENDER_UV_SIZE = 1024
RENDER_UV_CROP_POS = [192, 192]
RENDER_UV_CROP_SIZE = 640