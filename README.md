# Image Upscaling CGAN
A Deep Convolution Generative Adversarial Network for image upscaling and denoising complex biological organisms such as leaf in particular.

## Problem Statement / Usage Scenario
The main goal or diriving prurpose of this model is to upscale & denoise *low-res images* obtained from a remote location with contstrained network enviroment where sending *high-res images is not possible.  
In this particular project the model has to been trained to upscale & denoise leaf images obtained from these location for disease inspection.

## Model Architecture
### Generator Model
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Generator                                [1, 3, 128, 128]          --
├─Sequential: 1-1                        [1, 64, 32, 32]           --
│    └─Conv2d: 2-1                       [1, 64, 32, 32]           9,472
│    └─ReLU: 2-2                         [1, 64, 32, 32]           --
├─Sequential: 1-2                        [1, 64, 32, 32]           --
│    └─ResidualBlock: 2-3                [1, 64, 32, 32]           --
│    │    └─Sequential: 3-1              [1, 64, 32, 32]           74,112
│    └─ResidualBlock: 2-4                [1, 64, 32, 32]           --
│    │    └─Sequential: 3-2              [1, 64, 32, 32]           74,112
│    └─ResidualBlock: 2-5                [1, 64, 32, 32]           --
│    │    └─Sequential: 3-3              [1, 64, 32, 32]           74,112
│    └─ResidualBlock: 2-6                [1, 64, 32, 32]           --
│    │    └─Sequential: 3-4              [1, 64, 32, 32]           74,112
│    └─ResidualBlock: 2-7                [1, 64, 32, 32]           --
│    │    └─Sequential: 3-5              [1, 64, 32, 32]           74,112
│    └─ResidualBlock: 2-8                [1, 64, 32, 32]           --
│    │    └─Sequential: 3-6              [1, 64, 32, 32]           74,112
│    └─ResidualBlock: 2-9                [1, 64, 32, 32]           --
│    │    └─Sequential: 3-7              [1, 64, 32, 32]           74,112
│    └─ResidualBlock: 2-10               [1, 64, 32, 32]           --
│    │    └─Sequential: 3-8              [1, 64, 32, 32]           74,112
├─Sequential: 1-3                        [1, 64, 64, 64]           --
│    └─Upsample: 2-11                    [1, 64, 64, 64]           --
│    └─Conv2d: 2-12                      [1, 64, 64, 64]           36,928
│    └─ReLU: 2-13                        [1, 64, 64, 64]           --
├─Sequential: 1-4                        [1, 32, 128, 128]         --
│    └─Upsample: 2-14                    [1, 64, 128, 128]         --
│    └─Conv2d: 2-15                      [1, 32, 128, 128]         18,464
│    └─ReLU: 2-16                        [1, 32, 128, 128]         --
├─Sequential: 1-5                        [1, 3, 128, 128]          --
│    └─Conv2d: 2-17                      [1, 32, 128, 128]         9,248
│    └─ReLU: 2-18                        [1, 32, 128, 128]         --
│    └─Conv2d: 2-19                      [1, 3, 128, 128]          867
│    └─Sigmoid: 2-20                     [1, 3, 128, 128]          --
==========================================================================================
Total params: 667,875
Trainable params: 667,875
Non-trainable params: 0
Total mult-adds (Units.GIGABYTES): 1.23
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 28.18
Params size (MB): 2.67
Estimated Total Size (MB): 30.86
==========================================================================================
```
### Discriminator Model
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Discriminator                            [1, 1, 10, 10]            --
├─Sequential: 1-1                        [1, 1, 10, 10]            --
│    └─Conv2d: 2-1                       [1, 64, 63, 63]           6,208
│    └─LeakyReLU: 2-2                    [1, 64, 63, 63]           --
│    └─Conv2d: 2-3                       [1, 128, 30, 30]          131,200
│    └─BatchNorm2d: 2-4                  [1, 128, 30, 30]          256
│    └─LeakyReLU: 2-5                    [1, 128, 30, 30]          --
│    └─Conv2d: 2-6                       [1, 256, 14, 14]          524,544
│    └─BatchNorm2d: 2-7                  [1, 256, 14, 14]          512
│    └─LeakyReLU: 2-8                    [1, 256, 14, 14]          --
│    └─Conv2d: 2-9                       [1, 512, 11, 11]          2,097,664
│    └─BatchNorm2d: 2-10                 [1, 512, 11, 11]          1,024
│    └─LeakyReLU: 2-11                   [1, 512, 11, 11]          --
│    └─Conv2d: 2-12                      [1, 1, 10, 10]            8,193
==========================================================================================
Total params: 2,769,601
Trainable params: 2,769,601
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 500.17
==========================================================================================
Input size (MB): 0.39
Forward/backward pass size (MB): 5.67
Params size (MB): 11.08
Estimated Total Size (MB): 17.14
==========================================================================================
```

## Training Info
- This models is trained on a dataset of **1650 images** and tested on **500 images**.
- This models utilized **VGGPreceptualLoss** loss function.
- The **Mean Absolute Error (MAE)** achieved is **`0.0712`** in **100 EPOCHS**.

Further info on implementation, optimizer, dataloaders, etc can be found in notebook.
