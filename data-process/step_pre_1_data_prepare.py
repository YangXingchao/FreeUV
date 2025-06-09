import argparse
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch

import config
from networks import CoarseReconsNet
from utils.img_util import save_tensor
from utils.mm_layer import get_mm
from utils.mm_util import compute_mm
from utils.render_util import compute_uv_render, compute_img_render


def save_cropped_sampling_material(uv_tensor, uv_crop_pos, uv_crop_h, uv_crop_w, save_path):
    uv_img = uv_tensor.clone().detach().cpu().numpy()
    uv_img = (255.0 * uv_img).clip(0, 255).astype(np.uint8)
    uv_img = uv_img[uv_crop_pos[1]:uv_crop_pos[1] + uv_crop_h,
             uv_crop_pos[0]:uv_crop_pos[0] + uv_crop_w,
             :]
    cv2.imwrite(save_path, uv_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='data prepare for inference')
    parser.add_argument('--aligned_img_path', default='./results/aligned_img.jpg', help='Input aligned image')
    parser.add_argument('--mask_img_path', default='./results/mask_img.jpg',
                        help='Input mask image')
    parser.add_argument('--segmented_img_path', default='./results/segmented_img.jpg',
                        help='Input segmented image')
    parser.add_argument('--out_dir', '-o', default='./results', help='Output directory')
    args = parser.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    fit_size = config.FIT_SIZE
    uv_size = config.UV_SIZE
    render_size = config.RENDER_SIZE
    uv_crop_pos = config.UV_CROP_POS
    uv_crop_w = config.UV_CROP_W
    uv_crop_h = config.UV_CROP_H

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
    rast_uv, _ = dr.rasterize(glctx, uv_coords.contiguous(), uv_ids.contiguous(), resolution=[render_size, render_size])

    fit_uv_mask = cv2.imread(config.SKIN_MASK_PATH)
    resized_fit_uv_mask = cv2.resize(fit_uv_mask, (render_size, render_size))
    fit_uv_mask = torch.tensor(resized_fit_uv_mask[..., :2].astype(np.float32) / 255.0)[None].to(device)
    rend_uv_mask = torch.tensor(resized_fit_uv_mask.astype(np.float32) / 255.0)[None].to(device)

    net = CoarseReconsNet(config.n_shape, config.n_exp, config.n_tex, config.n_tex).to(device)
    net.load_state_dict(torch.load('checkpoints/coarse_reconstruction.pkl'))
    net.eval()

    img = cv2.imread(args.aligned_img_path)
    img = cv2.resize(img, (fit_size, fit_size))
    img = torch.tensor(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)

    seg_img = cv2.imread(args.segmented_img_path)
    seg_img = cv2.resize(seg_img, (render_size, render_size))
    seg_img = torch.tensor(seg_img.astype(np.float32)).to(device)
    
    res = net(img)
    mm_ret = compute_mm(shape_layer, tex_layer, spec_layer, tri, res['id'], res['ex'], res['tx'], res['sp'], res['r'],
                        res['tr'], res['s'], res['sh'], res['p'], res['ln'], res['gain'], res['bias'])
    uv_ret = compute_uv_render(mm_ret, rast_uv, tri, rend_uv_mask)
    rend_ret = compute_img_render(glctx, mm_ret, uv_ret, tri, uvs, uv_ids, fit_size)

    v_cam = mm_ret['v_cam']
    v_cam = v_cam / fit_size
    v_cam = torch.cat([v_cam, torch.ones([v_cam.shape[0], v_cam.shape[1], 1]).cuda()], axis=2)
    rast, _ = dr.rasterize(glctx, v_cam, tri, resolution=[render_size, render_size])

    # sampling uv tex
    v_cam_uv = v_cam[..., :2].detach().contiguous()
    v_cam_uv = (v_cam_uv + 1) * 0.5
    
    texc, _ = dr.interpolate(uvs.contiguous(), rast, uv_ids.contiguous())
    mm_uv_mask = dr.texture(fit_uv_mask.contiguous(), texc, filter_mode='auto')

    texc_inv, _ = dr.interpolate(v_cam_uv, rast_uv, tri)
    tex_uv = dr.texture(seg_img[None] * mm_uv_mask[..., 0, None], texc_inv, filter_mode='linear')
    
    tex_uv = tex_uv * rend_uv_mask 
    tex_uv_img = tex_uv[0].detach().cpu().numpy().astype(np.uint8)

    # crop uv tex
    tex_uv_crop = tex_uv_img[uv_crop_pos[1]:uv_crop_pos[1] + uv_crop_h,
                  uv_crop_pos[0]:uv_crop_pos[0] + uv_crop_w,
                  :]
    cv2.imwrite(f'{out_dir}/flaw_uv.jpg', tex_uv_crop)
    
    # save mask
    mask_img = cv2.imread(args.mask_img_path)
    mask_img = cv2.resize(mask_img, (render_size, render_size)) / 255.0
    mm_mask = mm_uv_mask[..., 0, None][0].detach().cpu().numpy().astype(np.uint8)
    combined_mask = mm_mask * mask_img * 255.
    cv2.imwrite(f'{out_dir}/combined_mask.jpg', combined_mask)

    # save mm params
    id_param = res['id'].detach().cpu().tolist()
    tx_param = res['tx'].detach().cpu().tolist()
    sp_param = res['sp'].detach().cpu().tolist()
    ex_param = res['ex'].detach().cpu().tolist()
    r_param = res['r'].detach().cpu().tolist()
    tr_param = res['tr'].detach().cpu().tolist()
    s_param = res['s'].detach().cpu().tolist()
    sh_param = res['sh'].detach().cpu().tolist()
    p_param = res['p'].detach().cpu().tolist()
    ln_param = res['ln'].detach().cpu().tolist()
    gain_param = res['gain'].detach().cpu().tolist()
    bias_param = res['bias'].detach().cpu().tolist()
    np.savez(f'{out_dir}/mm_param.npz', id=id_param, tx=tx_param, ex=ex_param, sp=sp_param, r=r_param,
             tr=tr_param,
             s=s_param, sh=sh_param, p=p_param, ln=ln_param, gain=gain_param, bias=bias_param)