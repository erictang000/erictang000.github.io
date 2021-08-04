---
layout: page
title: Projects
---
## GPU-BSW Work Stealing
![image](../images/work_stealing.png)

In this project, we aimed to implement CPU work stealing from a shared work queue for the Smith-Waterman algorithm. Research has showed that the most optimal CPU implementation of Smith-Waterman is competitive with GPU implementations for an instruction set (SSE2) an implementation from 2013 - thus it is important to both be able to take advantage of the parallel computation ability of GPUs, and the lack of the need for transfer latency on CPU, in order to optimize the Smith-Waterman algorithm on high performance compute clusters, where the compute environment is heterogeneous.

For the project, we worked off of an implementation of GPU-BSW, which is a batched version of the Smith-Waterman algorithm, for running parallel alignment computations on GPU owning threads. For non GPU owning threads, we used a SIMD version of Smith-Waterman. We integrated the two by updating the kernel calls of the GPU-BSW library to pop work off a shared atomic work queue used by all of the currently running threads. We showed steady performance improvements with this split CPU/GPU approach to tackling Smith-Waterman, and showed how these two implementations can be used in conjunction with one another rather than independently.

In addition to this basic integration of CPU and GPU, we attempted additional optimizations, including work stealing on GPU owning threads, and tuning batch sizes for CPU and GPUs to take or block work off of the shared queue.

## Tiny ImageNet Classifier
![image](../images/imagenet_architecture.png)
The emergence of Vision Transformers late last year has brought on a variety of new architectures into the space of image classification - however, work on robustness of these models as compared to more traditional convolution based approaches is still emerging.

Motivated by trying to gain understanding of the robustness of ViT based models, in this project, we attempt a number of different techniques for building a robust classifier for the Tiny ImageNet challenge. Vision Transformers have been shown to potentially help improve robustness against out of distribution examples from both white box and black box attackers. We report a top 1 validation accuracy of 81% on our architecture from fine tuning on Tiny ImageNet, using vision transformer blocks that were pretrained with ImageNet 1k, and using standard data augmentations along with AugMix. Our architecture, loosely based off of CrossViT from Chen et al., shows performance improvements over a standard ViT model via parallel vision transformers attending to different image patch sizes combined with cross attention and an MLP head. We also observe faster training and higher clean accuracy compared with deeper stacked ViT architectures with similar numbers of parameters. We benchmark robustness and accuracy of our model against a variety of ViT and ResNet based models on Tiny Imagenet-C and with adversarial attacks from Foolbox, and evaluate the addition of cross attention and varying patch sizes, as well as the use of sparse attention, to classifying out of distribution images.