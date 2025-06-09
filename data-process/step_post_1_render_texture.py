import argparse

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

import config
from utils.img_util import save_tensor, tensor_to_img
from utils.mm_layer import get_mm
from utils.mm_util import compute_mm
from utils.render_util import compute_img_render, compute_uv_render


def get_uncrop_tex(tex_path, uncrop_mask, tex_w, tex_h, uv_size, uv_crop_pos):
    tex_raw = cv2.imread(tex_path)
    tex_raw = cv2.resize(tex_raw, (tex_w, tex_h))
    tex_raw = tex_raw * uncrop_mask
    tex = np.zeros([uv_size, uv_size, 3])
    tex[uv_crop_pos[1]:uv_crop_pos[1] + tex_h, uv_crop_pos[0]:uv_crop_pos[0] + tex_w, :] = tex_raw
    tex = torch.Tensor(tex.astype(np.float32) / 255.0)
    return tex


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='render texture')
    parser.add_argument('--mm_param_path', default='./results/mm_param.npz',
                        help='Input fitted 3DMM parameter')
    parser.add_argument('--uv_texture_path', default='./results/color_transfer_uv.jpg',
                        help='Complete color transfered UV texture path')
    parser.add_argument('--input_aligned_image', default='./results/aligned_img.jpg',
                        help='Input aligned image')
    parser.add_argument('--combined_mask_path', default='./results/combined_mask.jpg',
                        help='Input combined mask')
    args = parser.parse_args()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    fit_size = config.FIT_SIZE
    render_size = config.RENDER_SIZE
    render_uv_size = config.RENDER_UV_SIZE
    render_uv_crop_size = config.RENDER_UV_CROP_SIZE
    render_uv_crop_pos = config.RENDER_UV_CROP_POS

    # get mm components
    mm = get_mm()
    shape_layer = mm['shape_layer'].to(device)
    tex_layer = mm['tex_layer'].to(device)
    spec_layer = mm['spec_layer'].to(device)
    tri = mm['tri'].to(device)
    uvs = mm['uvs'].to(device)
    uv_coords = mm['uv_coords'].to(device)
    uv_ids = mm['uv_ids'].to(device)

    glctx = dr.RasterizeGLContext()
    rast_uv, _ = dr.rasterize(glctx, uv_coords.contiguous(), uv_ids.contiguous(), resolution=[render_uv_size, render_uv_size])

    fit_uv_mask = cv2.imread(config.SKIN_MASK_PATH)
    fit_uv_mask = cv2.resize(fit_uv_mask, (render_uv_size, render_uv_size))
    fit_uv_mask_render = torch.Tensor(fit_uv_mask.astype(np.float32) / 255.0).to(device)
    fit_uv_mask = torch.Tensor(fit_uv_mask[..., :2].astype(np.float32) / 255.0)[None].to(device)
    
    tex = cv2.imread(args.uv_texture_path)
    tex = cv2.resize(tex, (render_uv_crop_size, render_uv_crop_size))
    tex_uv_img_crop = torch.Tensor(tex.astype(np.float32)).to(device)
    
    # mm param
    fit_param = np.load(args.mm_param_path)
    id = torch.from_numpy(fit_param['id']).float().to(device)
    ex = torch.from_numpy(fit_param['ex']).float().to(device)
    tx = torch.from_numpy(fit_param['tx']).float().to(device)
    sp = torch.from_numpy(fit_param['sp']).float().to(device)
    r = torch.from_numpy(fit_param['r']).float().to(device)
    tr = torch.from_numpy(fit_param['tr']).float().to(device)
    s = torch.from_numpy(fit_param['s']).float().to(device)
    sh = torch.from_numpy(fit_param['sh']).float().to(device)
    p = torch.from_numpy(fit_param['p']).float().to(device)
    ln = torch.from_numpy(fit_param['ln']).float().to(device)
    gain = torch.from_numpy(fit_param['gain']).float().to(device)
    bias = torch.from_numpy(fit_param['bias']).float().to(device)

    # compute mm
    mm_ret = compute_mm(shape_layer, tex_layer, spec_layer, tri, id, ex, tx, sp, r, tr, s, sh, p, ln, gain, bias)

    v_cam = mm_ret['v_cam']
    v_cam = v_cam / fit_size
    v_cam = torch.cat([v_cam, torch.ones([v_cam.shape[0], v_cam.shape[1], 1]).cuda()], axis=2)

    # # uv 
    tex_uv_img = np.zeros((render_uv_size, render_uv_size, 3), dtype=np.uint8)
    tex_uv_img = torch.Tensor(tex_uv_img.astype(np.float32)).to(device)
    tex_uv_img[render_uv_crop_pos[1]:render_uv_crop_pos[1] + render_uv_crop_size,
                render_uv_crop_pos[0]:render_uv_crop_pos[0] + render_uv_crop_size] = tex_uv_img_crop
    
    tex_uv_img = tex_uv_img * fit_uv_mask_render

    rast_out, _ = dr.rasterize(glctx, v_cam, tri, resolution=[render_uv_size, render_uv_size])
    texc, _ = dr.interpolate(uvs.contiguous(), rast_out, uv_ids.contiguous())
    masked_face_img = dr.texture(tex_uv_img[None], texc, filter_mode='linear')
    masked_face_img = masked_face_img * torch.clamp(rast_out[..., -1:], 0, 1) / 255.
    save_tensor(masked_face_img[0], './results/img_compose.jpg')
    
    # blend rendered image and background
    img_aligned = cv2.imread(args.input_aligned_image)
    img_aligned = cv2.resize(img_aligned, (render_size, render_size))
    img_aligned_mask = cv2.imread(args.combined_mask_path) 
    img_aligned_mask = cv2.resize(img_aligned_mask, (render_size, render_size)) / 255.
    img_aligned_mask = cv2.erode(img_aligned_mask, np.ones((3, 3), np.uint8), iterations=1)
    img_aligned_mask = cv2.GaussianBlur(img_aligned_mask, (3, 3), 1)
    img_aligned_compose = tensor_to_img(masked_face_img[0])
    
    img_aligned_compose = cv2.resize(img_aligned_compose, (render_size, render_size))
    img_with_bg = img_aligned_compose * img_aligned_mask + img_aligned * (1 - img_aligned_mask)
    cv2.imwrite('./results/img_compose_with_bg.jpg', img_with_bg)