# 🎬 FVDM

Official Code for Paper **_Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach_** 


> **Authors: [Yaofang Liu](https://github.com/Yaofang-Liu), [Yumeng REN](https://github.com/YumengREN), [Xiaodong Cun](https://github.com/vinthony), Aitor Artola, Yang Liu, Tieyong Zeng, Raymond H. Chan, Jean-michel Morel**

[![arXiv](https://img.shields.io/badge/arXiv-2409.06666-b31b1b.svg?logo=arXiv)]([https://arxiv.org/abs/2409.06666](https://arxiv.org/abs/2410.03160))
[![code](https://img.shields.io/badge/Github-Code-keygen.svg?logo=github)](https://github.com/Yaofang-Liu/FVDM)

FVDM (Frame-aware Video Diffusion Model) introduces a novel vectorized timestep variable (VTV) to revolutionize video generation, addressing limitations in current video diffusion models (VDMs).  Unlike previous VDMs, our approach allows each frame to follow an independent noise schedule, enhancing the model's capacity to capture fine-grained temporal dependencies. FVDM's flexibility is demonstrated across multiple tasks, including standard video generation, image-to-video generation, video interpolation, and long video synthesis. Through a diverse set of VTV configurations, we achieve superior quality in generated videos, overcoming challenges such as catastrophic forgetting during fine-tuning and limited generalizability in zero-shot methods.

<div align="center"><img src="https://github.com/Yaofang-Liu/FVDM/blob/7053489819c7dae13f4a3def6e97f5a0c65b5e03/Teaser.png" width="75%"/></div>

## 💡 Highlights
- 🎞️ **Vectorized Timestep Variable (VTV) for fine-grained temporal modeling**
- 🔄 **Great flexibity across a wide range of video generation tasks (in a zero-shot way)**
- 🚀 **Superior quality in generated videos**

## 🎥 Demos
With different VTV configurations, FVDM can be extended to numerous tasks (in a zeros-hot way). Below are some examples.
<div align="center"><img src="https://github.com/Yaofang-Liu/FVDM/blob/6eca425bf0bbef8f2ae6e42310105ec98c115fdf/Pipeline.png" width="75%"/></div>

- **Video Generation**
  ![VidGen](https://github.com/Yaofang-Liu/FVDM/blob/7f7bee998a15e7f9fec8d567769af7b187646fc1/output_video.gif)
- **Video Interpolation**
  ![Interpolation](https://github.com/Yaofang-Liu/FVDM/blob/40706ee56bf51542f2de2478444e9af8a0dd7f46/output_video_interpolation.gif)
- **Image-to-Video Generation**
  ![Image-to-video](https://github.com/Yaofang-Liu/FVDM/blob/d64fbb7f71a947c33030c776185cd30d8e2359ef/output_video_i2v.gif)
- **Long Video Generation**
  ![Long](https://github.com/Yaofang-Liu/FVDM/blob/ebc10418bbbc8a8fb2928f757545e21062f3ed97/output_video_long.gif)
  
## 🚀 Quick Start (Work in Progress)
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

## 🙏 Acknowledgements
We thank [relevant organizations/people] for their support and contributions.

## 📞 Contact
For any questions or feedback, please contact yaofanliu2-c@my.cityu.edu.hk.
