---
layout: post
title: Masked Autoencoders are Scalable Vision Learners
---

[**Paper**](https://arxiv.org/pdf/2111.06377.pdf?ref=https://githubhelp.com)

## Notes while Reading
* Masking data for autoencoding to learn representations appeared in vision prior to success in NLP (i.e. [BERT](https://arxiv.org/pdf/1810.04805.pdf)). However due to CNNs being the dominant architecture in vision, data masking was difficult to apply.
    *  The introduction of Vision Transformers and their success on traditional vision tasks allow for the easier application of masked autoencoding of image data.
* Language is very dense and every word is on average fairly imporant for the meaning of a sentence. This is why BERT only masks ~10% of the words while pretraining to obtain good text representations.
    * Vision is much less dense - spatial inference is much easier. In this paper, the authors mask ~75% of patches at random.
* Authors use asymmetric encoder-decoder architecture, since the reconstruction of pixels is a task that is at a lower semantic level than reconstructing words in text.
    * The encoder is a [ViT](https://arxiv.org/pdf/2010.11929.pdf) that only operates on the unmasked patches - unlike BERT, no mask tokens are used. This reduces the amount of compute necessary.
    * The decoder is another set of transfomer blocks that takes in a combination of the encoded visible patches, and mask tokens that encode the positional embeddings of each masked token from the original image.
    * MSE Loss over the predicted pixels from the decoder for masked patches is used to train the model.
* Architecture used is ViT-Large (ViT-L/16) with self supervised pretraining on Imagenet-1K's training set.
* The lack of a mask token during the encoder step is important - accuracy drops sharply if it is included in this step and not only in the decoder step. This is because when operating on real unmasked images, there are no mask tokens. Constraining the encoder to operate only on real patches helps fix this.
* The target for reconstruction that works the best is pixel wise with normalization. The authors also try token wise prediction (e.g. picking the right token index from the set of masked tokens, as in [BEiT](https://arxiv.org/pdf/2106.08254.pdf)) - this performs similarly, but decreases linear probing accuracy, and tokenizing requires extra compute and data for learning quality representations.
* MAE performs decently well w/o data aug - flipping and cropping help, but not color jitter. This makes sense, since the role data augmentation usually plays is satisfied by random masking.
    * Unlike contrastive Learning which relies heavily on data augmentation
* Training runs for 800 epochs, but steadily increases to 1600 epochs, unlike contrastive learning methods which saturate more quickly.
    * Training for 1600 epochs took 31 hours on 128 TPU-v3 cores (lol).
    
## Summary
This paper introduces a method for self supervised pretraining on images using an autoencoder architecture operating on heavily masked images, with the encoder operating on visible patches, and a lightweight decoder operating on masked tokens and visible patches, trained with a pixelwise MSE loss. It outperforms the previous SOTA of models pretrained on Imagenet-1K, with significantly less compute required.