Toy implementations of some Gaussian discriminant analysis algorithms.

D = {y_i, x_i} for points x_i \in \R^p and labels y_i \in {1, 2, ..., |C|}

p(y=c | x) = p(x | y=c) p(y=c) / p(x)  # Posterior label probability

Gaussian Discriminant analysis:
x|y=c ~ N(mu_c, Sigma_c)
y ~ Cat(pi)

This yields the well known classifiers:

  -"Quadratic Discriminant Analysis" fit the full covariance for each class
  -"Linear Discriminant Analysis" tie all the covariances together

This is easy to fit via Maximum likelihood.

A fully Bayesian model is also tractable with the conjugate priors

pi ~ Dir
(mu_c, Sigma_c) ~ NIW

Under the full posterior p(x | y=c) has a T-distribution.
The MAP or Posterior mean in this model can also be used in replace of ML.