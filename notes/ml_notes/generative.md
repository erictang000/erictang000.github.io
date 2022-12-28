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
#### Variational Inference
We are given a latent variable model, with x and z as the observed and latent variables respectively, defined as $$p_{\theta}(x, z) = p(x\mid z)p(x)$$. Consider the set of joint probability distributions over x and z given by 

$$P_{x,z} = \{p(x,z) \: \mid \: p(z) \in P_z, p(x\mid z) \in P_{x\mid z}\}$$ 

Given a dataset $$X = \{x^{(1)}, ..., x^{(n)}\}$$, our objective is to select the $$p \in P_{x,z}$$ that best fits D. In addition, given a sample $$x$$ and a model $$p \in P_{x,z}$$, we want to find the posterior distribution over the latent variables, $$p(z \mid x)$$.

One way to fit $$p(x,z)$$ to the data $$X$$ is minimizing the KL divergence between the two distributions.

$$\min_{p \in P_{x,z}} D_{KL}(p_{data}(x)\|p(x))$$

This is equivalent to maximizing the marginal log likelihood $$\log{p(x)}$$. More concretely, it is equivalent to

$$\max_{p \in P_{x,z}} \sum_{x\in X} \log{p(x)} = \max_{p \in P_{x,z}} log \int_z p(x\mid z)dz$$

However, solving for the integral in this equation is computationally infeasible. This also makes it difficult to compute the posterior distribution we earlier stated we were interested in - $$p(z\mid x)$$, since $$p(z\mid x) = \frac{p(z,x)}{p(x)}$$, and we just showed $$p(x)$$ to be difficult to solve for. An alternative is attempting to solve for a $$q \in Q$$ such that $$q$$ most closely approximates $$p(z\mid x)$$. We can do this by once again trying to minimize the KL divergence. 

$$\min_{q\in Q} D_{KL}(q(z)\|p(z\mid x))$$

Expanding based on the definition of KL divergence, we get the following

$$
\begin{align*}
D_{KL}(q(z)\|p(z\mid x)) &= \mathbb{E}[-\log(p(z\mid x)] - \mathbb{E}[-\log(q(z)] \\
                    &= \mathbb{E}[\log{\frac{q(z)}{p(z\mid x)}}] \\
                    &=\int_q q(z)\log{\frac{q(z)}{p(z\mid x)}} \\
                    &= \mathbb{E}_{q(z)}[\log{\frac{q(z)}{p(z\mid x)}}] \\
                    &= \mathbb{E}[\log{q(z)}] - \mathbb{E}[\log{p(z\mid x)}] \\
                    &= \mathbb{E}[\log{q(z)}] - (\mathbb{E}[\log{p(z, x)p(x)}]) \\ 
                    &= \mathbb{E}[\log{q(z)}] - \mathbb{E}[\log{p(z, x)}] - \mathbb{E}[\log{p(x)}]
\end{align*}
$$

The Evidence Lower Bound (ELBO) is optimized for instead of the KL Divergence, since we cannot compute the value of $$p(x)$$ in the above expression. The Evidence Lower Bound is defined as:

$$
\begin{align*}
ELBO(q) := \mathbb{E}[\log{p(z,x)}] - \mathbb{E}[\log{q(z)}]
\end{align*}
$$

We can then write

$$
\begin{align*}
\log(p(x)) = ELBO(q) + D_{KL}[q(z) \| p(z\mid x)]
\end{align*}
$$

Since the KL divergence is non-negative, we know that $$\log{p(x)} \geq ELBO(q)$$, which is why we call it the Evidence lower bound. Thus, if we maximize the ELBO, we minimize the KL divergence, which is the core of variational inference.

#### Autoencoders
In general, Autoencoding is a classical method for learning representations, with an encoder mapping an input to a latent representation, and a decoder that then reconstructs that input from the latent representation. PCA and k-means are examples of autoencoders.

In the context of deep learning specifically, autoencoder architectures consist of an encoder and decoder that are both deep neural networks. The intermediate lower dimensional representation (a vector that is the output of the encoder network) between the two pieces is the latent representation. Autoencoders are generally trained with a simple MSE loss, for example $$L(x, x') = \|x - x'\|^2$$. However, the latent representation of an autoencoder trained in this naive way suffers from a lack of continuity (points in the latent space nearby to each other should have similar properties) and completeness (points in the latent space should generate meaningful content). Variational autoencoders regularize the learned latent space to generate a more continuous and complete representation from which it's possible to generate content.

#### Variational Autoencoders

<!-- ### Generative Adversarial Networks

### Diffusion Models -->


### Sources
* [Carl Doersch Notes](https://arxiv.org/pdf/1606.05908.pdf)
* [Stanford CS 236 Notes](https://deepgenerativemodels.github.io/notes/index.html)
* [Joseph Rocca Medium Post](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
* [Gregory Gundersen Blog](https://gregorygundersen.com/blog/2021/04/16/variational-inference/)