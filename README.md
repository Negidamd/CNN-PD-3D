# CNN-PD-3D

[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b%2B-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-3D%20CNN-green.svg)]()

**Convolutional Neural Network for Automated Parkinson's Disease Detection from Structural 3D MRI**

A MATLAB-based deep learning toolbox that uses 3D ResNet-18 transfer learning for binary classification of Parkinson's disease (PD) vs. healthy controls from structural MRI scans.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [GUI Application](#gui-application)
  - [Training Pipeline](#training-pipeline)
  - [Inference](#inference)
- [Data Format](#data-format)
- [Model Architecture](#model-architecture)
- [File Descriptions](#file-descriptions)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

CNN-PD-3D is a deep learning framework designed for automated detection of Parkinson's disease from structural 3D MRI brain scans. The tool leverages transfer learning from a pre-trained 3D ResNet-18 architecture, adapted for volumetric neuroimaging data.

### Key Contributions
- **3D Transfer Learning**: Adapts 2D ResNet-18 weights for 3D volumetric MRI analysis
- **Data Augmentation**: Implements 3D spatial transformations (rotation, scaling, translation) for robust training
- **User-Friendly GUI**: Provides an intuitive graphical interface for clinical use
- **Two-Stage Training**: Supports iterative training to improve model accuracy

---

## ✨ Features

- 🧠 **3D Convolutional Neural Network** based on ResNet-18 architecture
- 📊 **Transfer Learning** from pre-trained ImageNet weights adapted for 3D
- 🔄 **Data Augmentation Pipeline** with randomized 3D affine transformations
- 🖥️ **MATLAB GUI** for easy clinical deployment
- 📈 **GPU Acceleration** for faster training and inference
- 📁 **NIfTI Support** for standard neuroimaging file formats

---

## 💻 Requirements

### Software
- **MATLAB** R2020b or later
- **Deep Learning Toolbox**
- **Image Processing Toolbox**
- **Computer Vision Toolbox** (optional, for some augmentations)

### Hardware
- **RAM**: Minimum 16 GB (32 GB recommended)
- **GPU**: NVIDIA GPU with CUDA support (recommended for training)
- **Storage**: ~5 GB for model weights and sample data

### MATLAB Toolboxes
```matlab
% Check required toolboxes
ver('nnet')           % Deep Learning Toolbox
ver('images')         % Image Processing Toolbox
```

---

## 📥 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/CNN-PD-3D.git
   cd CNN-PD-3D
   ```

2. **Download pre-trained weights**
   - Ensure `params.mat` (pre-trained 3D ResNet-18 weights) is in the root directory
   - The trained PD model `3DPretrainedModel.mat` will be generated after training

3. **Add to MATLAB path**
   ```matlab
   addpath(genpath('CNN-PD-3D'));
   ```

---

## 🚀 Usage

### GUI Application

Launch the graphical user interface for single-subject inference:

```matlab
CNN_3D_PD
```

1. Click the **Load Image** button
2. Select a NIfTI (.nii) file containing the 3D MRI scan
3. View the probability of Parkinson's disease

### Training Pipeline

#### First-Time Training

For initial model training with your dataset:

```matlab
% Run the first training pipeline
run('Pipeline3D_for1stTimeONLY_Training.m')
```

This script will:
- Load MRI data from `Dataset_Updated/Processed images - 1st session/`
- Initialize 3D ResNet-18 with transfer learning
- Train with data augmentation
- Save the model as `3DPretrainedModel.mat`

#### Iterative Training (Fine-tuning)

To improve accuracy with additional data sessions:

```matlab
% Run 2-3 times for improved accuracy
run('Pipeline3D_forREST_Training.m')
```

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initLearnRate` | 0.001 | Initial learning rate |
| `maxEpochs` | 15 | Maximum training epochs |
| `miniBatchSize` | Auto | Calculated based on dataset size |
| `valFrequency` | 4 | Validation frequency per epoch |

### Inference

For programmatic inference on new data:

```matlab
% Load trained model
load('3DPretrainedModel.mat');

% Set up datastore for test image
inputSize = mriNet.Layers(1).InputSize;
imds = imageDatastore('path/to/image.nii', ...
    'FileExtensions', '.nii', ...
    'ReadFcn', @niftiread, ...
    'ReadSize', 10);

% Preprocess and classify
augimdsTest = transform(imds, @(data)classification3DAugmentationPipeline1(data, inputSize, 'test'));
[label, probs] = classify(mriNet, augimdsTest);

% Display result
fprintf('PD Probability: %.2f%%\n', 100*probs(2));
```

---

## 📁 Data Format

### Expected Directory Structure

```
Dataset_Updated/
├── Processed images - 1st session/
│   ├── Subject001_Healthy.nii
│   ├── Subject002_Healthy.nii
│   ├── ...
│   ├── Subject026_Parkinson.nii
│   └── Subject027_Parkinson.nii
└── Processed images - 2nd session/
    └── [Same structure]
```

### Input Specifications

| Specification | Value |
|--------------|-------|
| File Format | NIfTI (.nii) |
| Input Size | 224 × 224 × 224 × 1 |
| Preprocessing | Z-score normalization |
| Cropping Region | [7, 6, 6] to [94, 78, 78] |

---

## 🏗️ Model Architecture

### 3D ResNet-18 Structure

```
Input Layer (224×224×224×1)
    │
    ▼
Conv3D (7×7×7, 64) → BN → ReLU → MaxPool3D
    │
    ▼
[ResBlock × 2] (64 filters)
    │
    ▼
[ResBlock × 2] (128 filters)
    │
    ▼
[ResBlock × 2] (256 filters)
    │
    ▼
[ResBlock × 2] (512 filters)
    │
    ▼
Global Average Pooling 3D
    │
    ▼
Fully Connected (2 classes)
    │
    ▼
Softmax → Classification Output
```

### Data Augmentation (Training)

- **Rotation**: ±15°
- **Scaling**: 0.85× to 1.15×
- **Translation**: ±15 voxels (X, Y, Z)

---

## 📄 File Descriptions

| File | Description |
|------|-------------|
| `CNN_3D_PD.m` | Main GUI application for PD classification |
| `CNN_3D_PD.fig` | MATLAB GUI layout file |
| `Pipeline3D_for1stTimeONLY_Training.m` | Initial training pipeline script |
| `Pipeline3D_forREST_Training.m` | Iterative fine-tuning pipeline |
| `classification3DAugmentationPipeline.m` | Data augmentation for train/val/test |
| `classification3DAugmentationPipeline1.m` | Simplified preprocessing for inference |
| `resnet18TL3Dfunction.m` | 3D ResNet-18 architecture definition |
| `params.mat` | Pre-trained 3D ResNet-18 weights |
| `3DPretrainedModel.mat` | Trained PD classification model |

---

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{cnn_pd_3d,
  author = {Negida, Ahmed},
  title = {CNN-PD-3D: Convolutional Neural Network for Automated Parkinson's Disease Detection from Structural 3D MRI},
  year = {2026},
  url = {https://github.com/yourusername/CNN-PD-3D}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Ahmed Negida, MD, MS**  

- 🌐 Website: [negida.net](https://negida.net)
- 🔬 ORCID: [0000-0001-5363-6369](https://orcid.org/0000-0001-5363-6369)

---

## 🙏 Acknowledgments

- MATLAB Deep Learning Toolbox team for the transfer learning framework
- Contributors to the original ResNet architecture

---

<p align="center">
  <i>Advancing precision medicine in Parkinson's disease through computational neuroscience</i>
</p>
