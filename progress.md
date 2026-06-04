# Document purpose and overivew

This document purpose is to document the core ideas, recent progress, and obstacles. 

Other files will be used for a more comprehensive overview of all aspects of the project. This document is the key knowledge, ideas and direction we currently working in. Its the base for orienting towards new ideas we may want to experiment with, new tests to perform, and new research directiosn to explore.

The document assumes alot of knowledge that can bew found in the other markdown files and is not supposed to be read by someone who is not familiar with the project.

## Current state of TRM + G3M

Self-conditioning is implemented, working and improves performance quite significantly, especially as to do with producing stable molecules.

latent recursion (TRM) is implemented and working, but does not improve performance. It uses more compute but without improving performance on a per epoch basis, so on a per compute basis it hurts it.

### TRM  / latent recursion

We have solved the numerical instability issues that we had with the original implementation. The z representation is now reasonable stable.

Z shares information with "s"
The latent recursion layer interracts with s for each message passing round. 

##### The Z representation 
Z = (M x z_dim)
Each token (M) does NOT corrospond to a specific atom, but rather an arbitrary latent representation of the molecule.

Currently: Trained models have the properties that
1 Z remains quite stable over timesteps during denoising (not perfectly, but representations are highly stable)
2 Each token is different from the others (vector points in different directions)
3 Each token is HIGHLY correlated with the learned nn.embeddings, ie. the embedding is 95%+ correlated with Z representation even at later timesteps. 



#### experiments

Models for all experiments hae not been trained fully to convergence, however, we still get close enough that we can draw some conclusions.

We have tested:
Higher M, Lower z dim and vice versa, no change in performance

n=2, n=1 and n=0 have all been tested. Similar performance on a per epoch basis, but n=2 is slower to train. 

self-conditioning helps across the board. Latent recursion does not help either with or without self-conditioning, atleast not to a degree we cant say its just noise.



## What to do

We want to improve performance of latent recursion.

Possibly, this has to do with the model not learning to use Z during the trajectories. 
However, increasing n does not seem to increase performance which is what we backpropagate through.



# ideas:

1 Enforce that Z learns orthogonal representations
Penalize representations of Z that are too similar to s representations. 
fx cosine similarity between s and each token in Z. 
Would require all vs all comparison, which is expensive. 

2 Prime Z with timestep t
Z = z_t + mlp(t)
where mlp(t) is a learnable function of t. 

s takes t directly so maybe not that important?

3 Adding direct prediction of position via Z representation. 

4 Positional embeddings for M: 
- currently cross attention is invariant to permutation of M tokens. 

5 Matching M to the number of atoms / making each token corrospond to a specific atom

6 Masking read operation from s padded tokens when updating z

7 Varying the number of "warmup" iterations (k) where Z or Z,X are not backpropagated through. -> greater stability?



# experiments
1 Baseline: 
    n=1
    no warmup (k=1)
    no self-conditioning, no warmup
    
2 Test 

    