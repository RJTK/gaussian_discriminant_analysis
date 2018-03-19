Toy implementations of some Gaussian discriminant analysis algorithms.  Allows for estimation via maximum likelihood, posterior mode or posterior mean, or a full bayes discriminant analysis.  It is common to place restrictions on covariance matrices (e.g. tie them accross classes, or restrict to diagonal), but this is not implemented.

In low dimensions and with enough data, there is little difference between the full Bayesian estimate and the ML estimate.

![alt tag](https://github.com/RJTK/gaussian_discriminant_analysis/blob/master/figures/ML_boundaries.png)

![alt tag](https://github.com/RJTK/gaussian_discriminant_analysis/blob/master/figures/FULL_boundaries.png)

In higher dimensions, ML is unstable / overfit for small samples

![alt tag](https://github.com/RJTK/gaussian_discriminant_analysis/blob/master/figures/learning_curves002.png)

PCA is not a good linear projection for QDA, p=25 is not optimized, and tied covariances might do better here.  This is also another demonstration of ML's failure with limited data and higher dimensions.  I have also done a hard prediction of class labels here using log probabilities since floating point does not have enough dynamic range to directly work with the actual probabilities in a straightforward way.

![alt tag](https://github.com/RJTK/gaussian_discriminant_analysis/blob/master/figures/learning_curves_digits.png)