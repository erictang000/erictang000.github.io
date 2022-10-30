---
layout: post
title: ML Fundamental Interview Review
---
Prepping for ML interviews - brushing up on fundamental stuff that may or may not come up in interviews.

## CS 189 Notes
Link: [Sahai Notes](https://www.eecs189.org/notes/) 
$$\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}$$


### Linear Regression
* We are trying to use a parametric model, where we have some $$w_d \in \mathbb{R}^d$$ that controls the behavior of the function. 
* We pick out some cost function that measures the quality of the hypothesis function's predictions against the true output, and then solve by looking for $$w^* = \underset{w}{\argmax} L(w)$$
    * The most basic way to solve is using OLS (Ordinary Least Squares), which assumes that our hypothesis function takes the form $$h_w(x) = x^Tw$$, and we can solve by minimizing the mean squared error ($$L(w) = \sum\limits_{i=1}^n (x_i^Tw-y_i)^2 = \underset{w}{\min} \|Xw-y\|_2^2$$
        * We can arrive at a closed form solution (one way is taking the gradient of the loss equal to 0 and solving for $$w^*$$) that tells us that $$w^*_{OLS} = (X^TX)^{-1}X^Ty$$

### Ridge Regression
* OLS has numerical instability and generalization issues
    * Numerical instability can arise when the features are close to collinear, which causes the input matrix X to lose its rank, or have close to 0 singular values. This causes the $$(X^TX)^{-1}$$ term to have very large singular values, which can lead to very high values for $$w$$, which can prevent generalization.
    * The solution here is to penalize $$w$$ from being too big, by adding a term to the loss to penalize the norm of $$w$$. Thus the ridge regression loss is $$L(w) = \underset{w}{\min} \|Xw-y\|_2^2 + \lambda\|w\|_2^2$$. The ridge regression solution (from taking the gradient of L to be 0), is then $$w^*_{RIDGE} = (X^TX + \lambda I)^{-1}X^Ty$$.
        * In addition, the hessian of the loss function is positive definite, so the loss function is strongly convex, and the ridge regression solution is unique

### Feature Engineering
* If we want to model non linear functions, we can still use least squares - we just have to first construct a feature map that maps each raw data point $$x \in \mathbb{R}^l$$ into a vector of features $$\phi(x)$$. Note that $$\phi$$ does not have to be a linear function. We can then solve for our hypothesis function $$h_w(x) = w^T\Phi(x)$$, with least squares solution $$\underset{w}{\min}\|\Phi w - y\|_2^2$$ (where $$\Phi$$ is $$\phi$$ applied to each data point in $$X$$).
    * For example, if we want to solve for the equation of an ellipse, we can construct an equation of the form $$w_1 x_1^2 + w_2 x^2 + w_3 x_1 x_2 + w_4x_1 + w_5x_2 = 1$$, and then formulate the problem with least squares, where the feature map is given by $$\phi(x) = (x_1^2, x_2^2, x_1x_2, x_1, x_2)$$
* In general, polynomials are universal approximators, but the problem with large polynomials is that the number of terms increases exponentially with the degree of the polynomial.
    * The kernel trick can help get around this cost in certain cases
* In general hyperparameters can fall into two catagories - model hyperparameters that determine the structure of a model, and optimization hyperparameters that determine the optimization procedure

### The Kernel Trick
* When we model some polynomial, we 


### Previous Questions
* Given a machine with infinite GPU memory would we be able to train a deep learning model faster?
    * Maybe not - if we want to do stochastic gradient descent, we need some fixed smaller batch size, and we can't do it in parallel since we need to update the weights based on the gradients from each batch. 
* What is the difference between LayerNorm and BatchNorm, why is normalization helpful, and when do we use one vs another and why?
    * In batchnorm we normalize over the layers 
    * The reason why normalization is useful in the middle of a deep learning model has a similar intuition to why it's useful at all in the beginning - it keeps some numerical stability, and keeps the magnitude of the gradients similar throughout the model of the layers
    * BatchNorm is generally used in CNNs and other image models. LayerNorm is more common in language and sequence modelling, since BatchNorm is tricky for RNNs, and doesn't work well with small batch sizes.
* Explain Adam
    * RMSProp + Momentum. RMSProp keeps a running average over the squared gradients, and normalizes the gradient with the square root of the running average. Momentum keeps a running average over the gradients, and uses it to inform the direction of the gradient.
* How to combat overfitting/What is regularization
    * Reduce model size
    * Regularization (L1, L2, Dropout, etc)
    * Ensembling models
* Explain the problem of vanishing and exploding gradients
    * Gradients can either drastically increase or tend to zero as backprop happens - neither is desirable
    * Vanishing gradients can be identified if the weights of early layers do not change very much relative to later layers
    * Exploding gradients can be identified by NaN model weights or outputs due to overflow
    * Solutions can include normalization (Batch/LayerNorm), reducing the number of layers, gradient clipping, and using better weight initialization
        * ResNets also were motivated by solving this problem - the residual connections help gradient flow

### Generic ML Questions
* Explain the Bias Variance Tradeoff
    * Bias quantifies how close our model fits to the given training set. Low bias indicates that our model fits well to the given training set, while high bias indicates that we haven't fit well to the training set.
    * Variance quantifies how well our model generalizes to an unseen test set. Low variance indicates that our model performs similarly on other datasets as it does on the training set. High variance indicates that the model performs worse on unseen data compared to its performance on the training set.
    * In general, using regularization helps decrease variance while increasing bias. Increasing model complexity usually reduces bias while increasing variance.
    * The bias variance tradeoff in classical ML setting suggests that lower bias will lead to higher variance, and vice versa. The double descent phenomenon in deep learning shows that the bias variance tradeoff may not be as applicable for non-classical ML settings (i.e. making models bigger and bigger can be good)
* Explain the difference between precision and recall
    * Given a binary classifier:
        * Precision is defined as the number of correct positive predictions divided by the number of positive predictions
        * Recall is the number of correct positive predictions divided by the number of actual positives