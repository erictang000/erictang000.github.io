---
layout: post
title: GPU-BSW
---
## Work Stealing for GPU Batched Smith-Waterman Algorithm on HPC
In this project, I (along with a partner) aimed to implement CPU work stealing from a shared work queue for the Smith-Waterman algorithm. Research showed that the most optimal CPU implementation of Smith-Waterman was competitive with GPU implementations for an instruction set (SSE2) an implementation from 2013 - thus it was important to both be able to take advantage of the parallel computation ability of GPUs, and the lack of the need for transfer latency on CPU, in order to optimize the Smith-Waterman algorithm on high performance compute clusters, where the compute environment was heterogeneous. 

For the project, we worked off of an implementation of GPU-BSW, which is a batched version of the Smith-Waterman algorithm, for running parallel alignment computations on GPU owning threads. For non GPU owning threads, we used a SIMD version of Smith-Waterman. We integrated the two by updating the kernel calls of the GPU-BSW library to pop work off a shared atomic work queue used by all of the currently running threads. We showed steady performance improvements with this split CPU/GPU approach to tackling Smith-Waterman, and showed how these two implementations can be used in conjunction with one another rather than independently.

In addition to this basic integration of CPU and GPU, we attempted additional optimizations, including work stealing on GPU owning threads, and tuning batch sizes for CPU and GPUs to take or block work off of the shared queue.

The results of our work can be found at https://github.com/erictang000/GPU-BSW-Work-Stealing