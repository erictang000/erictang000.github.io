---
layout: post
title: Optimizers
---
For the following, we use a learning rate of $$\alpha$$, a momentum term of $$\mu$$, use $$k$$ to denote timestep, and use $$L$$ to denote the loss function we define over the parameters $$\theta$$ we are optimizing.
### Momentum
Update rule consisting of not only the gradients of the loss function, but also the previous direction of gradient descent. The idea is to make a correction to the loss at the current step by summing with the direction taken in the previous gradient step. The momentum term $$\mu$$ is usually set to around $$0.9$$.

$$ g_{k+1} = \nabla_ \theta L(\theta_k) + \mu g_k$$ 

$$ \theta_{k + 1} = \theta_k -\alpha g_{k+1}$$

### Nesterov Momentum
Modification of momentum that holds stronger theoretical convergence guarantees, and works better in practice as well. The intuition is that since we are going to make a large step in our parameters based on our previous momentum, $$\mu g_k$$, it makes more sense to calculate our gradients after this initial momentum based step.

$$ g_{k+1} = \nabla_ \theta L(\theta_k - \mu g_k) + \mu g_k$$

$$ \theta_{k + 1} = \theta_k -\alpha g_{k+1}$$

## Adaptive Learning Rate Optimizers
Vanilla SGD and Momentum based SGD use the same learning rate for each parameter - we want to be able to adjust the learning rate for each parameter to be higher or lower based on their importance (elements that receive higher gradients should have their learning rates lowered, and elements that receive lower gradients should have theirs increased)
### AdaGrad
AdaGrad uses the square root of the cumulative sum of squared gradients to normalize gradient updates. It performs best for convex problems, since it decreases the learning rate over time (since the cumulative sum of square gradients continually increases). It also is particularly suited for sparse data, since it helps to ensure that parameters associated with infrequent features are able to converge to optimal values by increasing their assigned learning rate.

For a set of parameters $$\theta$$ of dimension $$d$$, we maintain a diagonal matrix $$S \in \mathbb{R}^{d\times d}$$ where $$S_{k,ii}$$ contains the cumulative sum of squared gradients with respect to $$\theta_{k,i}$$ (This is given as an alternative to $$S_{k}$$ containing the full sum of all outer products of gradients, since taking the square root of a full non diagonal $$d\times d$$ matrix is computationally infeasible). At each gradient step we then compute the following.

$$S_{k + 1,ii} = S_{k,ii} + (\nabla_\theta L(\theta_{k,i}))^2$$

$$\theta_{k+1, i} = \theta_{k, i} - \frac{\alpha}{\sqrt{S_{k,ii}+ \epsilon}} \nabla_\theta L(\theta_{k,i})$$

Vectorizing the above yields the following:

$$\theta_{k+1} = \theta_{k} - \frac{\alpha}{\sqrt{S_{k}+ \epsilon}} \nabla_\theta L(\theta_{k})$$

### RMSProp
RMSProp uses a running "average" of the squared gradients for each parameter in order to avoid decaying learning rate, as in AdaGrad. As such, it works better than AdaGrad for non-convex optimization problems - namely for deep neural networks.

We maintain a vector $$s_k$$ that we update at each gradient step as below (like the running average term for momentum, $$0.9$$ is a common choice for $$\beta$$)

$$s_{k+1} = \beta s_k + (1 - \beta) (\nabla_\theta L(\theta_{k}))^2$$

We then update the gradients similarly to in AdaGrad:

$$\theta_{k+1} = \theta_{k} - \frac{\alpha}{\sqrt{s_{k}+ \epsilon}} \nabla_\theta L(\theta_{k})$$

### Adam
Adam is a combination of momentum and RMSProp. We keep running "average" terms for both the previous gradients, and the previous gradients squared. Common default values for $$\beta_1$$ and $$\beta_2$$ are $$0.9$$ and $$0.999$$, with $$\epsilon=10^{-8}$$

$$m_{k+1} = (1 - \beta_1)\nabla_\theta L(\theta_{k}) + \beta_1 m_k $$

$$v_{k+1} = (1 - \beta_2)(\nabla_\theta L(\theta_{k}))^2 + \beta_2 v_k $$

To correct for initial behavior when $$m_0$$ and $$v_0$$ are both zero, we compute bias corrected estimates:

$$\hat{m}_{k+1}= \frac{m_{k+1}}{1 - \beta_1^k}$$

$$\hat{v}_{k+1}= \frac{v_{k+1}}{1 - \beta_2^k}$$

We then compute the gradient update 

$$\theta_{k+1} = \theta_{k} - \frac{\alpha}{\sqrt{\hat{v}_k} + \epsilon}\hat{m}_k$$


## Sources
* Sebastian Ruder's Blog - [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html)
* CS 231N - [Neural Networks Part 3](https://cs231n.github.io/neural-networks-3/)
* CS 182 - [Lecture 4 - Optimization](https://cs182sp21.github.io/static/slides/lec-4.pdf)
