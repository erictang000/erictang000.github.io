---
layout: post
title: ML Interview Answers
---
Answers to questions from Chip Huyen's [book](https://huyenchip.com/ml-interviews-book/contents/part-ii.-questions.html).
## Section I - Math
1. Dot product
    1. [E] What’s the geometric interpretation of the dot product of two vectors?

        The dot product of two vectors a and b, expressed as ab, represents the projection of a onto b, scaled by the magnitude of b.
    2. [E] Given a vector u, find vector v of unit length such that the dot product of u and v is maximum.

        The dot product between u and v is defined as the product of the magnitudes of the vectors and the consine of the angle between them. Cosine is maximized when the angle is zero, and so to maximize the dot product, we would pick v to have the same direction as the vector u. Thus we would pick v = u / \|\|u\|\|.
2. Outer product
    1. [E] Given two vectors  a=[3,2,1]  and  b=[−1,0,1] . Calculate the outer product aTb?
    2. [M] Give an example of how the outer product can be useful in ML.
3. [E] What does it mean for two vectors to be linearly independent?
4. [M] Given two sets of vectors  A=a1,a2,a3,...,an  and  B=b1,b2,b3,...,bm . How do you check that they share the same basis?
5. [M] Given  n  vectors, each of  d  dimensions. What is the dimension of their span?
6. Norms and metrics
    1. [E] What's a norm? What is  L0,L1,L2,Lnorm ?
    2. [M] How do norm and metric differ? Given a norm, make a metric. Given a metric, can we make a norm?