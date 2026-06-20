### Ordered Linear Shape & Scale Conditioning

**1. Feature Extraction & Sorting**
Given mean-centered atom coordinates $X_c \in \mathbb{R}^{N \times 3}$, compute the $3 \times 3$ scatter matrix and extract the raw eigenvalues $\boldsymbol{\lambda} \in \mathbb{R}^3$. Sort them in descending order to guarantee the network always processes the primary, secondary, and tertiary axes consistently:


$$\boldsymbol{\lambda}_{raw} = \text{eig}(X_c^T X_c)$$

$$\boldsymbol{\lambda} = \text{sort}(\boldsymbol{\lambda}_{raw}, \text{descending})$$


*(Ensuring $\lambda_1 \geq \lambda_2 \geq \lambda_3$)*

**2. Linearization & Scale Decoupling**
Convert the spatial variance (quadratic) to physical extents (linear), and extract the global scale scalar $S \in \mathbb{R}$:


$$E_i = \sqrt{\lambda_i}$$

$$S = \sum_{i=1}^3 E_i$$

**3. Fractional Shape Normalization**
Compute the size-invariant geometric proportions $\hat{\mathbf{s}} \in \mathbb{R}^3$:


$$\hat{s}_i = \frac{E_i}{S}$$


**4. With hydrogens**

Eigenvalues are computed using all atoms including hydrogens.