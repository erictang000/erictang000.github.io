---
layout: post
title: Generative Modeling
---
Generative modeling (as opposed to discriminative modeling) is an area of machine learning which deals with modeling distributions over an input. For the common example of distinguishing fake images from real images, we can imagine a generative model would learn a distribution over input images, while a discriminitive model would learn a decision boundary instead. But how do we also synthesize images given an existing distribution of images? Or in other words, how do we train and sample from a generative model $$p(x)$$ based on that image distribution $$X$$?

More generally, in generative modeling, there are three typical evaluation tasks:
1. **Density Estimation** Given a datapoint $$x$$, what is the probability of the data point assigned by the model, $$p_{\theta}(x)$$?
2. **Sampling** How can we generate a new data point from the model distribution, $$x_{new}$$ ~ $$p_{\theta}(x)$$?
3. **Unsupervised Representation Learning** How can we learn meaningful feature representations given a datapoint x (without labels)?

### Variational Autoencoders
We are given a latent variable model, with x and z as the observed and latent variables respectively, defined as $$p_{\theta}(x, z) = p(x\mid z)p(x)$$. Consider the set of joint probability distributions over x and z given by 

$$P_{x,z} = \{p(x,z) \: \mid \: p(z) \in P_z, p(x\mid z) \in P_{x\mid z}\}$$ 

Given a dataset $$X = \{x^{(1)}, ..., x^{(n)}\}$$, our objective is to select the $$p \in P_{x,z}$$ that best fits D. In addition, given a sample $$x$$ and a model $$p \in P_{x,z}$$, we want to find the posterior distribution over the latent variables, $$p(z \mid x)$$.

One way to fit $$p(x,z)$$ to the data $$X$$ is minimizing the KL divergence between the two distributions.

$$\min_{p \in P_{x,z}} D_{KL}(p_{data}(x)\|p(x))$$

This is equivalent to maximizing the marginal log likelihood $$\log{p(x)}$$. More concretely, it is equivalent to

$$\max_{p \in P_{x,z}} \sum_{x\in X} \log{p(x)} = \max_{p \in P_{x,z}} log \int_z p(x\mid z)dz$$

However, solving for the integral in this equation is computationally infeasible. This also makes it difficult to compute the posterior distribution we earlier stated we were interested in - $$p(z\mid x)$$, since $$p(z\mid x) = \frac{p(z,x)}{p(x)}$$, and we just showed $$p(x)$$ to be difficult to solve for. An alternative is attempting to solve for a $$q \in Q$$ such that $$q$$ most closely approximates $$p(z\mid x)$$. We can do this by once again trying to minimize the KL divergence. 


$$\min_{q\in Q} D_{KL}(q(z)\|p(z\mid x))$$

<!-- $$D_{KL} -->


#### Autoencoders
Autoencoders are a type of deep neural network consisting of an encoder and a decoder, with an intermediate lower dimensional representation between the two pieces often referred to as a latent representation. Autoencoders are generally trained with a simple MSE loss, for example $$L(x, x') = \|x - x'\|^2$$. However, the latent representation of an autoencoder trained in this naive way suffers from a lack of continuity (points in the latent space nearby to each other should have similar properties) and completeness (points in the latent space should generate meaningful content). Variational autoencoders regularize the learned latent space to generate a more continuous and complete representation from which it's possible to generate content.

<!-- ### Generative Adversarial Networks

### Diffusion Models -->


### Sources
* [Carl Doersch Notes](https://arxiv.org/pdf/1606.05908.pdf)
* [Stanford CS 236 Notes](https://deepgenerativemodels.github.io/notes/index.html)
* [Joseph Rocca Medium Post](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
* [Gregory Gundersen Blog](https://gregorygundersen.com/blog/2021/04/16/variational-inference/)