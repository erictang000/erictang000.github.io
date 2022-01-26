---
layout: post
title: data2vec - A General Framework for Self-supervised Learning in Speech, Vision and Language
---

[**Paper**](https://scontent-sjc3-1.xx.fbcdn.net/v/t39.8562-6/271974914_483120576492438_4239522333319653600_n.pdf?_nc_cat=107&ccb=1-5&_nc_sid=ae5e01&_nc_ohc=4-cMR5tUq4QAX8J1QtU&_nc_ht=scontent-sjc3-1.xx&oh=00_AT-iZqzaxCXfwCEY2Cd5XaeC_H2da0xAtWLH3bN9hbTL5g&oe=61F3F7D1)

## Notes while Reading
* Main concept is self supervised learning framework that uses the same architecture and learning objective across speech, language, and vision, with the main difference between models being the embedding layers and the output layer for domain specific tasks.
* Combination of masked prediction and learning of latent target representations.
* **Method + Architecture Overview**
    * The main idea is to use a base transformer as the backbone of the network, with domain specific embedding layers. There are 2 versions of the network - one in "student" mode and one in "teacher mode". The teacher mode network uses a moving average of the student model's weights as its weights, with the teacher being updated more frequently at the beginning of training, and slowing down updates after the teacher has already learned better parameters.
    * The training target in the case of all domains is the same - the student, given a masked input, tries to predict the latent representation of the full input that is passed through the teacher model. This latent representation is normalized over the output of the last K layers of the teacher network. A smooth L1 loss is used to regress to these targets.

    $$L(y_t, f_t(x)) = \begin{cases}\frac{1}{2}(y_t - f_t(x))^2/\beta & \lvert y_t - f_t(x)\rvert \leq \beta \\ (\lvert y_t - f_t(x)\rvert - \frac{1}{2}\beta) & \text{otherwise}\end{cases}$$

    * Masking is done using block-wise masking for vision, span-masking for speech, and BERT style masking for language.
    * The main advantage of having the target be predicting latent representations is that the target includes context - when a model predicts a single patch of an image, or the content of a single word, that output does not contain contextual information. By predicting a latent representation, the model is forced to also predict a contextualized representation due to the way self attention works in Transformer models.
    * **Question** - Do we perform inference with the teacher or student model? Teacher could make sense since it has been operating on unmasked content, while student makes more sense from the side that it's weights are directly being updated via gradient descent.

## Summary
The authors present a general framework for self supervised learning across modalities using a Transformer backbone with a student teacher model that trains the student to predict contexualized latent representations given masked inputs. This method outperforms the SOTA in specific benchmarks across all three.