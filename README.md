Toy implementations of some Gaussian discriminant analysis algorithms.  Allows for estimation via maximum likelihood, posterior mode or posterior mean, or a full bayes discriminant analysis.  It is common to place restrictions on covariance matrices (e.g. tie them accross classes, or restrict to diagonal), but this is not implemented.

In low dimensions and with enough data, there is little difference between the full Bayesian estimate and the ML estimate:

![alt tag](https://github.com/RJTK/gaussian_discriminant_analysis/blob/master/figures/ML_boundaries.png)

![alt tag](https://github.com/RJTK/gaussian_discriminant_analysis/blob/master/figures/FULL_boundaries.png)

In higher dimensions, ML is unstable / overfit for small samples:

![alt tag](https://github.com/RJTK/gaussian_discriminant_analysis/blob/master/figures/learning_curves002.png)

Here I have applied the DA algorithm on the classic handwritten digits data, after projecting into 18 dimensions (18 being chosen ad-hoc) with PCA.

![alt tag](https://github.com/RJTK/gaussian_discriminant_analysis/blob/master/figures/learning_curves_digits.png)