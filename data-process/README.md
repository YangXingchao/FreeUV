The data process is based on and extends the following projects and uses some of the code and models from them:
> [**Makeup Extraction of 3D Representation via Illumination-Aware Image Decomposition**](https://yangxingchao.github.io/makeup-extract-page),  
> Xingchao Yang, Takafumi Taketomi, Yoshihiro Kanamori,   
> *Computer Graphics Forum (Proc. of Eurographics 2023)*

> [**Color transfer between images**](https://ieeexplore.ieee.org/document/946629),  
> E. Reinhard, M. Adhikhmin, B. Gooch, P. Shirley   
> *IEEE Computer Graphics and Applications (2001)*


### Prerequisites
1. Python3
2. PyTorch with CUDA
4. Nvdiffrast

### Installation
Run the following commands for installing other packages:
```
pip install -r requirements.txt
```

### Prepare 3DMM models
Download 3DMM model from [FLAME](https://flame.is.tue.mpg.de/) and put them into ```resources``` folder

We need the following models for our project:
```
albedoModel2020_FLAME_albedoPart.npz
FLAME_masks.pkl
FLAME_texture.npz
generic_model.pkl (from FLAME2020)
```

### Pretrained 3DMM fitting models
Put the [trained models](https://drive.google.com/drive/folders/1lwkR9JcrbZ7fNylTSJQQEiGnt3s2LQYq?usp=sharing) to ```checkpoints/```.

### Demo  
Perform a sequence of processes on ```sample_img.jpg``` in the ```sample``` folder

#### Pre-process
1. Detect the landmark, and crop the image so that it aligns with the face.
Then obtain an image of the skin area:
```
python step_pre_0_preprocess.py
```

2. 3DMM fitting and obtain incomplete uv texture (```flaw_uv.jpg```).
```
python step_pre_1_data_prepare.py
```

#### Post-process
1. After obtain complete uv texture (```complete_uv.jpg```), perform color adjustment.
```
python step_post_0_color_transfer.py
```

2. Render the UV texture to image space.
```
python step_post_1_render_texture.py
```

### Acknowledgements
Here are some of the resources we benefit from:

* [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
* [pytorch_face_landmark](https://github.com/cunjian/pytorch_face_landmark)
* [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
* [DECA](https://github.com/yfeng95/DECA)
* [python-color-transfer](https://github.com/pengbo-learn/python-color-transfer)

