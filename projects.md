---
layout: page
title: Projects
---
<head>
  <link href="../images/fontawesome/css/fontawesome.css" rel="stylesheet">
  <link href="../images/fontawesome/css/brands.css" rel="stylesheet">
  <link href="../images/fontawesome/css/solid.css" rel="stylesheet">
</head>
<style type="text/css">
  .fab {
    color: black;
  }
  .fas {
    color: black;
  }
</style>

### **Group-DRO for Automatic Speech Recognition** <a href="https://github.com/erictang000/groupdro-asr"><i class="fab fa-github"></i></a> <a href="../images/CS_224S_Final_Project.pdf"><i class="fas fa-file"></i></a>
Improved the robustness of automatic speech recognition models on the ML-SUPERB benchmark by applying Group-DRO. Via experiments on the Bantu family of languages, and on English audio across various ML-SUPERB datasets, we showed that Group-DRO can help improve worst case performance across challenging group shifts while maintaining high average performance across the test set. 

### **Do Objectives Matter OOD? Understanding the Impact of Self-Supervised Objectives on Robustness of Vision Transformers** <a href="https://github.com/erictang000/wilds"><i class="fab fa-github"></i></a> <a href="../images/CS_329D_Project.pdf"><i class="fas fa-file"></i></a>
Investigated the effect of self supervised pretraining objectives (Contrastive Learning vs Masked Image Modelling) on the robustness of Vision Transformer representations to distribution shift. Found that linear probing over CL pretrained ViTs show the strongest OOD robustness for image classification due to the stronger out of the box classification performance and lack of overfitting to in-distribution training data that can be found during fine tuning.

### **Instance Specific Data Augmentation for Meta Learning** <a href="https://github.com/erictang000/instance_aug_meta_learning"><i class="fab fa-github"></i></a> <a href="../images/330_final_proj.pdf"><i class="fas fa-file"></i></a>
Explored applying Instance Specific data augmentations to the meta-learning setting. Found that although instance specific augmentations are still able to beat a random baseline, the augmentations tend to overfit to the training task distribution, and do not perform as well as augmentations with stronger regularizing effects such as CutMix. Final project for CS 330 - Deep Multi-Task and Meta Learning.

### [**CS 194-26 Project Pages**](/projects/194)
Projects for CS 194-26 - Computational Photography and Computer Vision. Explored image alignment, blending, filters, morphing, warping, and applications of deep learning in computer vision.

### **GPU-BSW Work Stealing** <a href="https://github.com/erictang000/GPU-BSW-Work-Stealing"><i class="fab fa-github"></i></a> <a href="../images/gpu_bsw_report.pdf"><i class="fas fa-file"></i></a>
Implemented work stealing for a batched GPU implementation of the Smith-Waterman DNA sequence alignment Algorithm using **CUDA** and **OpenMP** in **C++** for heterogenous computing environments. Final project for CS 267 - Parallel Computing.


### **Tiny ImageNet Classifier** <a href="https://github.com/erictang000/182cvproj"><i class="fab fa-github"></i></a> <a href="../images/182_report.pdf"><i class="fas fa-file"></i></a>
Built custom architecture for classification on Tiny ImageNet based on Vision Transformers using PyTorch. Achieved 85% top 1 accuracy, and explored use of different attention mechanisms, data augmentation methods, and pretrained base models for improving out of distribution robustness. Computer Vision final project for CS 182 - Deep Learning.