# 🎬 FVDM

Official Code for Paper **_Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach_** 


> **Authors: [Yaofang Liu](https://scholar.google.com/citations?user=WWb7Y7AAAAAJ&hl=zh-CN), [Yumeng REN](https://scholars.cityu.edu.hk/en/persons/yumeng-ren(88862739-c2ce-47ea-b202-c147e7c07bd6).html), [Xiaodong Cun](https://github.com/vinthony), [Aitor Artola](https://scholar.google.com/citations?user=yDqVoN0AAAAJ&hl=en), [Yang Liu](https://scholar.google.com/citations?user=z2eEUuwAAAAJ), [Tieyong Zeng](https://scholar.google.com/citations?user=2yyTgRwAAAAJ&hl=fr), [Raymond H. Chan](https://scholar.google.com/citations?user=ICiiEOAAAAAJ&hl=en), [Jean-michel Morel](https://scholar.google.fr/citations?user=BlEbdeEAAAAJ&hl=en)**

[![arXiv](https://img.shields.io/badge/arXiv-2409.06666-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.03160)
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/Yaofang-Liu/FVDM)

FVDM (Frame-aware Video Diffusion Model) introduces a novel vectorized timestep variable (VTV) to revolutionize video generation, addressing limitations in current video diffusion models (VDMs).  Unlike previous VDMs, our approach allows each frame to follow an independent noise schedule, enhancing the model's capacity to capture fine-grained temporal dependencies. FVDM's flexibility is demonstrated across multiple tasks, including standard video generation, image-to-video generation, video interpolation, and long video synthesis. Through a diverse set of VTV configurations, we achieve superior quality in generated videos, overcoming challenges such as catastrophic forgetting during fine-tuning and limited generalizability in zero-shot methods.

<div align="center"><img src="https://github.com/Yaofang-Liu/FVDM/blob/7053489819c7dae13f4a3def6e97f5a0c65b5e03/Teaser.png" width="75%"/></div>

## 💡 Highlights
- 🎞️ **Vectorized Timestep Variable (VTV) for fine-grained temporal modeling**
- 🔄 **Great flexibility across a wide range of video generation tasks (in a zero-shot way)**
- 🚀 **Superior quality in generated videos**
- 🙌 **No additional computation cost during training and inference**


## 🎥 Demos
With different VTV configurations, FVDM can be extended to numerous tasks (in a zero-shot way). 
<div align="center"><img src="https://github.com/Yaofang-Liu/FVDM/blob/6eca425bf0bbef8f2ae6e42310105ec98c115fdf/Pipeline.png" width="75%"/></div>

Below are FVDM generated videos w.r.t. datasets FaceForensics, SkyTimelapse, Taichi-HD, and UCF101. Note that the models/checkpoints are the same across different tasks (reflects strong zero-shot capabilities), and currently they are only trained with 2*A6000 GPUs.

https://github.com/user-attachments/assets/1a2c988b-d231-4e7b-9a2d-be1f96e98502

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
