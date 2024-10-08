# FVDM
Official Code for Paper **_Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach_** 
[arXiv](https://arxiv.org/abs/2410.03160)

## Zero-Shot Applications
- **Video interpolation**
  ![Interpolation](https://github.com/Yaofang-Liu/FVDM/blob/40706ee56bf51542f2de2478444e9af8a0dd7f46/output_video_interpolation.gif)
- **Image-to-video generation**
  ![Image-to-video](https://github.com/Yaofang-Liu/FVDM/blob/d64fbb7f71a947c33030c776185cd30d8e2359ef/output_video_i2v.gif)
- **Long video generation**
  ![Long](https://github.com/Yaofang-Liu/FVDM/blob/ebc10418bbbc8a8fb2928f757545e21062f3ed97/output_video_long.gif)

Certainly! I'll update the README with the corresponding GIF links for the demos. Here's the updated version:

# 🎬 FVDM: Frame-aware Video Diffusion Model

> **Authors: [Yaofang Liu](https://github.com/Yaofang-Liu), [Yumeng REN](https://github.com/YumengREN), [Xiaodong Cun](https://github.com/vinthony), Aitor Artola, Yang Liu, Tieyong Zeng, Raymond H. Chan, Jean-michel Morel**

[![arXiv][]](https://arxiv.org/abs/2410.03160)[![GitHub][]](https://github.com/Yaofang-Liu/FVDM)

FVDM (Frame-aware Video Diffusion Model) introduces a novel vectorized timestep variable (VTV) to revolutionize video generation, addressing limitations in current video diffusion models.

<div align="center"><img src="path_to_your_model_architecture_image.png" width="75%"/></div>

## 💡 Highlights

- 🎞️ **Vectorized Timestep Variable (VTV) for fine-grained temporal modeling**
- 🔄 **Flexible across multiple video generation tasks**
- 🚀 **Superior quality in generated videos**
- 🧠 **Overcomes catastrophic forgetting during fine-tuning**
- 🌐 **Enhanced generalizability in zero-shot methods**

## 🎥 Demos

### Video Generation![VidGen][]

### Video Interpolation![Interpolation][]

### Image-to-Video Generation![Image-to-video][]

### Long Video Generation![Long][]

## 🚀 Quick Start```bash
git clone https://github.com/Yaofang-Liu/FVDM.git
cd FVDM
pip install -r requirements.txt```

## 📊 Results

FVDM outperforms state-of-the-art methods in:
- Video generation quality
- Extended tasks performance
- Temporal dependency modeling

## 📜 Citation

If you find our work useful, please consider citing:```bibtex
@article{liu2024fvdm,
  title={FVDM: Frame-aware Video Diffusion Model},
  author={Liu, Yaofang and Ren, Yumeng and Cun, Xiaodong and Artola, Aitor and Liu, Yang and Zeng, Tieyong and Chan, Raymond H. and Morel, Jean-michel},
  journal={arXiv preprint arXiv:2410.03160},  year={2024}
}
```

## 🙏 Acknowledgements

We thank [relevant organizations/people] for their support and contributions.

## 📄 License

This project is licensed under the [Your License] - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any questions or feedback, please contact [your contact information].
