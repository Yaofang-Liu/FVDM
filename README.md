# 🎬 FVDM

Official Code for Paper **_Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach_** 


> **Authors: [Yaofang Liu](https://scholar.google.com/citations?user=WWb7Y7AAAAAJ&hl=zh-CN), Yumeng REN, [Xiaodong Cun](https://github.com/vinthony), Aitor Artola, Yang Liu, Tieyong Zeng, Raymond H. Chan, [Jean-michel Morel](https://scholar.google.fr/citations?user=BlEbdeEAAAAJ&hl=en)**

[![arXiv](https://img.shields.io/badge/arXiv-2409.06666-b31b1b.svg?logo=arXiv)]([https://arxiv.org/abs/2409.06666](https://arxiv.org/abs/2410.03160))
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/Yaofang-Liu/FVDM)

FVDM (Frame-aware Video Diffusion Model) introduces a novel vectorized timestep variable (VTV) to revolutionize video generation, addressing limitations in current video diffusion models (VDMs).  Unlike previous VDMs, our approach allows each frame to follow an independent noise schedule, enhancing the model's capacity to capture fine-grained temporal dependencies. FVDM's flexibility is demonstrated across multiple tasks, including standard video generation, image-to-video generation, video interpolation, and long video synthesis. Through a diverse set of VTV configurations, we achieve superior quality in generated videos, overcoming challenges such as catastrophic forgetting during fine-tuning and limited generalizability in zero-shot methods.

<div align="center"><img src="https://github.com/Yaofang-Liu/FVDM/blob/7053489819c7dae13f4a3def6e97f5a0c65b5e03/Teaser.png" width="75%"/></div>

## 💡 Highlights
- 🎞️ **Vectorized Timestep Variable (VTV) for fine-grained temporal modeling**
- 🔄 **Great flexibility across a wide range of video generation tasks (in a zero-shot way)**
- 🚀 **Superior quality in generated videos**

## 🎥 Demos
With different VTV configurations, FVDM can be extended to numerous tasks (in a zero-shot way). 
<div align="center"><img src="https://github.com/Yaofang-Liu/FVDM/blob/6eca425bf0bbef8f2ae6e42310105ec98c115fdf/Pipeline.png" width="75%"/></div>

Below are FVDM generated videos w.r.t. datasets FaceForensics, SkyTimelapse, Taichi-HD, and UCF101. Note that the models/checkpoints are the same across different tasks (reflects strong zero-shot capabilities), and currently they are only trained with 2*A6000 GPUs.
https://private-user-images.githubusercontent.com/45255738/374870253-3bba0983-0453-4684-9ad2-e8325135a678.mp4
- **Standard Video Generation** (From noise)
  ![VidGen](https://github.com/Yaofang-Liu/FVDM/blob/26205b2a3cbda1bdac632b40f6aee8b690412169/output_video_compressed.gif) 
- **Video Interpolation** (First frame and last frame are given)
  ![Interpolation](https://github.com/Yaofang-Liu/FVDM/blob/26205b2a3cbda1bdac632b40f6aee8b690412169/output_video_interpolation_compressed.gif)
- **Image-to-Video Generation** (First frame is given)
  ![Image-to-video](https://github.com/Yaofang-Liu/FVDM/blob/26205b2a3cbda1bdac632b40f6aee8b690412169/output_video_i2v_compressed.gif)
- **Long Video Generation** (Take 128 frames as an example, model trained on 16 frames)
  ![Long](https://github.com/Yaofang-Liu/FVDM/blob/26205b2a3cbda1bdac632b40f6aee8b690412169/output_video_long_compressed.gif)
  
## 🚀 Quick Start (Coming Soon)
```bash
git clone https://github.com/Yaofang-Liu/FVDM.git
cd FVDM
```


## 📜 Citation
If you find our work useful, please consider citing:
```bibtex
@misc{liu2024redefiningtemporalmodelingvideo,
      title={Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach}, 
      author={Yaofang Liu and Yumeng Ren and Xiaodong Cun and Aitor Artola and Yang Liu and Tieyong Zeng and Raymond H. Chan and Jean-michel Morel},
      year={2024},
      eprint={2410.03160},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.03160}, 
}
```

## 📞 Contact
For any questions or feedback, please contact yaofanliu2-c@my.cityu.edu.hk.
