# CS523 Final Project

This is the Final project for CS 523 at Boston University. We re-implemented the [Real-ESRGAN](https://arxiv.org/abs/2107.10833). We note that the official implementation is found [here](https://github.com/xinntao/Real-ESRGAN)

## Acknowledgements:

### In the degradation code: we refer to

1)Xintao Wang, Ke Yu, Kelvin C.K. Chan, Chao Dong and Chen Change Loy. BasicSR: Open Source Image and Video Restoration Toolbox, 2018. [Official BasicSR Github](https://github.com/xinntao/BasicSR)


2)[Filters tutorial](https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter)
for the sinc filter.

### For the Generator code we refer to:
 1) Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, Chen Change Loy. Enhanced Super-Resolution Generative Adversarial Networks. ECCV2018 PIRM Workshop, 2018. [Official ESRGAN Github](https://github.com/xinntao/ESRGAN)
2) Wang, Xintao, et al. "Real-esrgan: Training real-world blind super-resolution with pure synthetic data." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. [Official Real-ESRGAN Implementation](https://github.com/xinntao/Real-ESRGAN)

### For the Discriminator code we refer to:
 1) Xintao Wang, Ke Yu, Shixiang Wu, Jinjin Gu, Yihao Liu, Chao Dong, Yu Qiao, Chen Change Loy. Enhanced Super-Resolution Generative Adversarial Networks. ECCV2018 PIRM Workshop, 2018. [Official ESRGAN Github](https://github.com/xinntao/ESRGAN)
2) Wang, Xintao, et al. "Real-esrgan: Training real-world blind super-resolution with pure synthetic data." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. [Official Real-ESRGAN Implementation](https://github.com/xinntao/Real-ESRGAN)
3) Schonfeld, Edgar, Bernt Schiele, and Anna Khoreva. "A u-net based discriminator for generative adversarial networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
[Official U-Net Discriminator Implementation](https://github.com/boschresearch/unetgan)

### The FID code is influenced by:
2) [Official Pytorch Implementation of FID](https://github.com/mseitzer/pytorch-fid). **We emphasize that this implementation does not have the same Inception v3 weights as the official implementation. As such, it is not suitable to use our implementation to compare to other benchmarks**
