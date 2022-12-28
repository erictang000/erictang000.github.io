---
layout: post
title: Decision Trees Notes
---
* Function that maps a vector to a decision
* In order to learn a decision tree from examples, we can generally take a greedy strategy, where at each step in our decision tree, we split our dataset based on the attribute that gives us the most information (i.e. the one that splits yes/no the most)
    * We can quantify the amount of information an attribute gives us by calculating its entropy $$H = -\sum p_ilog(p_i)$$. The information gain is then the expected reduction in entropy.
        * Let's calculate this for a given set of example vectors, where we have boolean yes/no labels for each vector
        * We can define the entropy of a bernoulli RV that has probability q as 
            *  $$B(q)=-(qlog_2(q) + (1-q)log_2(1-q))$$
        * Then the entropy of a group of vectors with p positive examples, and n negative examples is $$B(\frac{p}{p + n})$$
        * Let's say we split our vector on some attribute A, that splits our set of vectors $$E$$ into subsets $$E_1, E_2, ..., E_k$$, then we can quantify the amount of entropy left as the amount of entropy in each of the k sets, multiplied by the probability of each of the k sets, giving us $$Remainder(A) = \sum_{k=1}^d \frac{p_k + n_k}{p + n} * B(\frac{p_k}{n_k + p_k})$$