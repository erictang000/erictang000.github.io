---
layout: post
title: Dataset Distillation and Learning to Augment
---
## Dataset Distillation
### [Dataset Distillation (Wang et al. 2017)](https://arxiv.org/abs/1811.10959)
* Main idea is to distill an image dataset to some much smaller set of synthetic images
    * One motivation is to understand how much data is encoded in a given training set, and to see how compressible that data is
    * Another motivation is to be able to load up a given network much more efficiently (pretraining on effectively larger dataset much more quickly)
* Main algorithm: gradient descent on a randomly sampled subset of the data, and on a learning rate for single gradient stepping over that subset of the data
    * We first pick some subset of the data $$\tilde{x}$$, and some initial single gradient step learning rate $$\tilde{\mu}$$
    * Then for each step in our algorithm:
        * We fix a minibatch of the data $$x$$ sampled at random from our dataset.
        * We then sample a batch of models parameterized by $$\theta_0$$ from a fixed distribution $$p(\theta)$$. 
        * For each model in the batch of models, we calculate $$\theta_1$$ by doing a single gradient step over our current $$\tilde{x}$$ and $$\tilde{\eta}$$.
            * We then evaluate a model parameterized by $$\theta_1$$ on the minibatch $$x$$ from the original training data
            * We aggregate the resulting loss over the batch of models
        * We then calculate the gradients with respect to our synthetic data $$\tilde{x}$$, and $$\tilde{\eta}$$, and update them with fixed learning rate $$\alpha$$
    * We end by outputting our final distilled dataset $$\tilde{x}$$ and learning rate $$\tilde{\eta}$$
* Main algorithm is optimized by doing multiple gradient steps on different batches of the synthetic dataset $$\tilde{x}$$, and also doing multiple epochs of training on the synthetic data within each step of the algorithm
* Works best with a fixed network initialization.

### [Dataset Condensation via Efficient Synthetic-Data Parameterization](https://arxiv.org/abs/2205.14959)
* Dataset distillation also has interesting applications in terms of continual learning and accelerating neural architecture search (finding smaller dataset representation helps these go faster).







Dataset distillation
https://arxiv.org/abs/1811.10959
https://arxiv.org/abs/2205.14959
https://arxiv.org/abs/2208.10494
Others in https://github.com/Guang000/Awesome-Dataset-Distillation

Learning augmentations
https://arxiv.org/pdf/2010.11882.pdf
https://arxiv.org/pdf/2206.00051.pdf
