<h1 align="center">🤘Temporal diffuser: Timing scale-aware modulation for sign language production</h1>

<p align="center"><a href="https://pdf.sciencedirectassets.com/271095/1-s2.0-S0952197625X0035X/1-s2.0-S0952197625027708/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEGgaCXVzLWVhc3QtMSJHMEUCIQDsqavUVnXNjb%2F3GWE29bVQDxTEdRXezzQd5bPrXJ%2FDAQIgdXLTQ6%2FntAZSSJuXGJSt%2BdDLeGqfsrdYw8u%2FHzc16jIqsgUIMBAFGgwwNTkwMDM1NDY4NjUiDPe2Ys2O%2Fv9%2FPF3dniqPBZ0Ej0arRDVVk%2B4syORhOLVupX9RVGGC3v9GSgp3Em15LPAe5aE%2BMt118aTrQ5y%2FORSr%2FAD1p3cyRwET50VYbl%2BjioEunzn9DvQW1S1%2FKKtqG3HUla3haij0h%2BnDLIpWrZB9UCbYdSj%2Bs%2F6LvzKHjpSM2xOZVxc%2F47wxapHM1w%2FrDfLuI%2FYHGxD8KTXGr3OMAtUiI%2FDb%2FwdpjOHeqNiiJ3x6a4plFoqFZIChPzvWfeBReinJojYqBD2B1Zq3rt%2FlZNTyeQ0Qc3Vob0rs9chCnHMlhWwIwDSLmktt3qo%2FMyck9i%2BSJuavaGKlOd2lfsdqRk40y8sKb7CfWJPobw5E53HE248m67WV%2BYgTm%2FpMjFJkOLrGBYAN%2FNemyEZoluNfV104WIJcojNs7IEfYCipL7ge%2Fd2JSCsuS2DpVi2L%2BDwf7VvvrpPDFnyBW2VxJQmiO0SqOQuOSkJ%2BFAgTcbQNA0Xi2VHPwGwHFlBYkmbZjEvaSMu%2BVvUfL%2BFQC2HQ%2BoztWKh5CtGxcX%2FuHqea88hC%2F5JizVmjGoBXpqn%2F%2Fsl3swk2OqUt1j7eDTwF5haecLTFs%2FBrdmc8U5yXygyOMKNRp3xUWRHjypNi%2BzXjclmRBNPlfvrFYQgl2wk5mHDsFb3WGgYNXZI3WV8SczFR%2BJVcvCKhnWHWDN2QjCf7M8ZSZ130TVJFBNPAStkSOmb0J9RXMFfSdKmK4xQlxOiS2xWy%2B1FYLSfULylbCdDZbVXJ2sW2%2F3CGz0t93HIvEGwOalgzGRZSFC5NCq%2BcDrpJGJo3nvtYIjdFmBjOixFPlWlPoehb%2Bgj08S1Q2hXS7O0gnPUt5MKUzeEmpm53uxy3TCC9MxT9Qb8vD72a2%2Ft3E6UknvYwuMP5yQY6sQEvM6Sc2OT7aRM4d0a6rG6B%2BunR3%2Fg89ccpnAnJBBRHh07S7ikqxtK1QtVS27PlWwA7j8ld%2BM95X2v00lYs7q3WzFjawdKZWeFfpSg1JscGQP0NhJTtYmjzhpubJ4qF5O8%2FT8ytCz80EfLO5PFggIRONi29kNf1LBnJh%2FSrHLv5H2RwCWUn3FYEtTh9k0uttNQm5pIGBD%2F4%2FFBtVJmnGrlpAFkbtDACl9Tu5FoTJ3nmrZo%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20251214T080453Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYQ65GFTX5%2F20251214%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=e4e5a3861ea0476364e0d34877e71a0a828e7947633e13661c650ed1210f000c&hash=a42d79e4c6262155bdfb7556a49de3952f9f9c9f11bde67841e7413532d3bd6f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0952197625027708&tid=spdf-8de509d5-bc2b-4db4-a2b5-0e15a3e3f877&sid=4e3ddde575731643182a5cb95d010aeec1d0gxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=180e560600035c57&rr=9adc31699ec6fe23&cc=vn"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://kha-kim-thuy.github.io/SLP-Demo/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>

<p align="center"><img src="assets/exp.png" width="100%"></p>

Sign Language Production with Scale-Aware Modulation (SignSAM), 
a novel single-stage, gloss-free SLP framework that directly synthesizes motion in continuous space, preserving 
fine temporal details. At its core is a Space–Time U-Net that learns compact temporal features by jointly 
downscaling the frame and sign feature dimensions, thereby reducing computational cost compared to a no
pyramid UNet or a pyramid UNet without consistency between dimensions. To further enhance temporal 
precision, we propose a Timing Scale-Aware Modulation module that fuses multiscale temporal resolutions 
for better motion coherence

<!-- Announcement -->
## 📢 Announcement
- **`09 Oct, 2025`:** SignSAM is accepted to EAAI 2026!
- **`14 Dec, 2025`:** Release code

<!-- Installation -->
## 📦 Installation

This code was tested on `Ubuntu 18.04.5 LTS` and requires:

* Python 3.7
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

Clone this repo:
```shell
git clone https://github.com/kha-kim-thuy/Sign-Diffusion-Model.git SignSAM
cd SignSAM
```

Setup conda env:
```shell
conda env create -f environment.yml
conda activate signsam
```

Download dependencies:

```shell
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
pip install chumpy
pip install blobfile
pip install ffmpeg
pip install numpy==1.23.5
pip install sentence_transformers
pip install accelerator
```

<!-- Dataset -->
## 📚 Dataset

### PHOENIX14T
Download PHOENIX14T
```bash
bash prepare/download_phoenix_dataset.sh
```

### YouTube Sign Language (MediaPipe 77 joints)
Prepare your own YouTube Sign Language dataset using MediaPipe Holistic to extract keypoints.
Each `.npy` file should have shape `(T, 231)` where `T` is number of frames and `231 = 77 joints × 3 coords`.

The 77 joints layout:
- **Joints 0–32**: MediaPipe Pose (33 landmarks)
- **Joints 33–53**: Left Hand (21 landmarks)
- **Joints 54–74**: Right Hand (21 landmarks)
- **Joints 75–76**: Extra face reference (2 landmarks)

Each `.txt` caption file must follow the format:
```
caption text here#token1 token2 token3#0.0#0.0
```

### Data structure
```shell
|- dataset
    |- PHOENIX           # Phoenix dataset (50 joints × 3 = 150 dim)
    |   |- new_joints/
    |   |   |- sample_001.npy
    |   |   |- ...
    |   |- texts/
    |   |   |- sample_001.txt
    |   |   |- ...
    |   |- Mean.npy      # (optional, auto-computed if missing)
    |   |- Std.npy       # (optional, auto-computed if missing)
    |   |- train.txt
    |   |- val.txt
    |   |- test.txt
    |
    |- YOUTUBE_SIGN      # YouTube Sign dataset (77 joints × 3 = 231 dim)
        |- new_joints/
        |   |- video_001.npy   # shape: (T, 231)
        |   |- ...
        |- texts/
        |   |- video_001.txt   # format: caption#tokens#0.0#0.0
        |   |- ...
        |- Mean.npy      # (optional, auto-computed if missing)
        |- Std.npy       # (optional, auto-computed if missing)
        |- train.txt     # list of sample names (without .npy)
        |- val.txt
        |- test.txt
```

> **Note:** `Mean.npy` and `Std.npy` are optional. If not provided, they will be automatically computed from the training split on first run.
<!-- Training -->
## 🏋️‍♂️ Training

### Train on Phoenix
```shell
python -m train.train_mdm \
--arch sam_stunet \
--lr 1e-4 \
--overwrite \
--save_interval 1000 \
--num_steps 400000 \
--dataset phoenix \
--save_dir ./save/phoenix_OURs \
--batch_size 64 \
--diffusion_steps 1000
```

### Train on YouTube Sign
```shell
python -m train.train_mdm \
--arch sam_stunet \
--lr 1e-4 \
--overwrite \
--save_interval 1000 \
--num_steps 400000 \
--dataset youtube_sign \
--save_dir ./save/youtube_sign_OURs \
--batch_size 64 \
--diffusion_steps 1000
```

To resume training from a checkpoint:
```shell
python -m train.train_mdm \
--arch sam_stunet \
--resume_checkpoint ./save/youtube_sign_OURs/model000050000.pt \
--lr 1e-4 \
--overwrite \
--save_interval 1000 \
--num_steps 400000 \
--dataset youtube_sign \
--save_dir ./save/youtube_sign_OURs \
--batch_size 64 \
--diffusion_steps 1000
```

## 🔍 Evaluate

Evaluate model with Ground Truth comparison (DTW metric):

### Phoenix
```shell
python -m sample.phoenix_generate \
--model_path ./save/phoenix_OURs/<your-model>.pt \
--num_repetitions 1 \
--guidance_param 5.5 \
--num_samples 10
```

### YouTube Sign
```shell
python -m sample.yt_generate \
--model_path ./save/youtube_sign_OURs/<your-model>.pt \
--num_repetitions 1 \
--guidance_param 5.5 \
--num_samples 10
```

## 🙆‍♂️ Inference

Generate sign language motion from text input.

### Phoenix
```shell
python -m sample.phoenix_generate \
--model_path ./save/phoenix_OURs/<your-model>.pt \
--num_repetitions 1 \
--guidance_param 5.5 \
--motion_length 5 \
--text_prompt "a person signing hello"
```

### YouTube Sign
```shell
python -m sample.yt_generate \
--model_path ./save/youtube_sign_OURs/<your-model>.pt \
--num_repetitions 1 \
--guidance_param 5.5 \
--motion_length 5 \
--text_prompt "a person signing hello"
```

> **⚠️ Important:** Do **NOT** use `sample.generate` for YouTube Sign data — it hardcodes 50-joint reshape and will crash.
> Use `sample.yt_generate` instead, which supports 77-joint MediaPipe skeleton visualization.

<!-- Acknowledgments -->
## 🙇‍♂️ Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[motion-diffusion-model](https://github.com/GuyTevet/motion-diffusion-model), [single-motion-diffusion-model](https://github.com/SinMDM/SinMDM.git), [guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi), [SMT](https://github.com/AFeng-x/SMT.git).

<!-- License -->
## ⚖️ License

This code is distributed under an [MIT License](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.

<!-- Citation -->
## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{kha2026temporal,
  title={Temporal diffuser: Timing scale-aware modulation for sign language production},
  author={Kha, Kim-Thuy and Vo, Anh H and Le, Van-Vang and Song, Oh-Young and Kim, Yong-Guk},
  journal={Engineering Applications of Artificial Intelligence},
  volume={163},
  pages={112739},
  year={2026},
  publisher={Elsevier}
}
```