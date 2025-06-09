import os
import torch
import random
from PIL import Image
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, ControlNetModel
from diffusers.utils import load_image
from detail_encoder.encoder_freeuv import detail_encoder

model_id = "your_sdv1-5_path" # your sdv1-5 path
detial_extractor_path = "./checkpoints/flaw_tolerant_facial_detail_extractor.bin"
uv_aligner_path = "./checkpoints/uv_structure_aligner.bin"

Unet = OriginalUNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to("cuda")

uv_aligner = ControlNetModel.from_unet(Unet)
detial_extractor = detail_encoder(Unet, "./models/image_encoder_l/", "cuda", dtype=torch.float32)
detial_extractor_state_dict = torch.load(detial_extractor_path)
uv_aligner_state_dict = torch.load(uv_aligner_path)
uv_aligner.load_state_dict(uv_aligner_state_dict, strict=False)
detial_extractor.load_state_dict(detial_extractor_state_dict, strict=False)
uv_aligner.to("cuda")
detial_extractor.to("cuda")

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id,
    safety_checker=None,
    unet=Unet,
    controlnet=uv_aligner,
    torch_dtype=torch.float32).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

def infer():
    uv_img_path = "./data-process/resources/uv.jpg"
    flaw_uv_image_path = "./data-process/results/flaw_uv.jpg"
    out_image_path = "./data-process/results/complete_uv.jpg"
    
    uv_img = load_image(uv_img_path).resize((512, 512))
    flaw_uv_image = load_image(flaw_uv_image_path).resize((512, 512))
    result_img = detial_extractor.generate(uv_structure_image=uv_img, flaw_uv_image=flaw_uv_image,
                                                    pipe=pipe, guidance_scale=1.4)
    result_img.save(out_image_path)
        
                
if __name__ == '__main__':
    infer()
