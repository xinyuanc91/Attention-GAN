# Attention-GAN
This repository provides the PyTorch code for our paper “Attention-GAN for object transfiguration in wild images”([ECCV2018](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Xinyuan_Chen_Attention-GAN_for_Object_ECCV_2018_paper.pdf)). This code is based on the PyTorch (0.4.1) implementation of [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). You may need to train several times as the quality of the results are sensitive to the initialization.
### Data Prepare
    bash datasets/download_cyclegan_dataset.sh horse2zebra
## Train
    bash scripts/train_attngan.sh
