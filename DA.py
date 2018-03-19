import numpy as np

import scipy as sp
from scipy import stats
from scipy.special import gammaln
from sklearn.base import BaseEstimator, ClassifierMixin
import logging

# Available fitting methods
MAXIMUM_LIKELIHOOD = "ML"
BAYES_MAP = "MAP"
BAYES_MEAN = "Mean"
BAYES_FULL = "Bayes"


def mvar_t_pdf(X, mu, Sigma, nu):
    """
    Evaluate the density function of a multivariate student t
    distribution at the points X

    params:
      X (np.array n x p): The points at which to evaluate the density
      mu (np.array p): The mean parameter
      Sigma (np.array p x p): The scale matrix
      nu (int): degrees of freedom parameter
    """
    p = X.shape[1]
    log_ret = 0
    # log Z (normalizing constant)
    logZ = ((p / 2) * np.log(np.pi) + gammaln(nu / 2) -
            gammaln((nu / 2) + (p / 2)))
    log_ret -= logZ

    V = nu * Sigma
    L = np.linalg.cholesky(V)  # V = L @ L.T
    half_logdetV = np.sum(np.log(np.diag(L)))  # log det V
    log_ret -= half_logdetV

    # z = L_inv @ (x - mu) Remember: (LL.T)^-1 = L.T^-1 L^-1
    # L.T @ z = x - mu
    # => L.T @ z = U.T

    U = X - mu[None, :]
    z = sp.linalg.solve_triangular(L, U.T, lower=True)
    log_quad = np.log1p(np.linalg.norm(z, ord=2, axis=0)**2)
    log_ret -= 0.5 * (nu + p) * log_quad
    return np.exp(log_ret)


class DiscriminantAnalysis(BaseEstimator, ClassifierMixin):
    def __init__(self, do_logging=False, fit_method=MAXIMUM_LIKELIHOOD):
        self._fitted = False
        self.do_logging = do_logging
        self.fit_method = fit_method
        if do_logging:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)
            fh = logging.FileHandler('discriminant_analysis.log')
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        return

    def fit(self, X, y, method=None,
            alpha=1.0, nu=2, k=1./100):
        """
        params:
          X (np.array N x p): The data matrix
          y (np.array N): Class labels for X, need not be numeric
          method: The method to fit the model with.  One of
            -"ML": Fit via maximum likelihood
            -"MAP": The Bayesian MAP estimate with conjugate priors
            -"Mean": Use Bayesian posterior mean point estimates
            -"Bayes": Fit a fully Bayesian model with conjugate priors
          alpha (float): Prior value for Dir(alpha), ignored for ML
          nu (float): Covariance pseudo will be p + nu, ignored for ML
          k (float): Mean pseudo data, ignored for ML.

        Increasing alpha, nu, or k will increase the amount of regularization,
        or in other words, add more weight to the prior belief.  We use the
        data dependent priors mean(X) and diag(cov(X)) for each prior.
        """
        N = X.shape[0]  # Number of samples
        p = X.shape[1]  # Data dimension
        self.N, self.p = N, p
        if len(y.shape) != 1:
            if self.do_logging:
                self.logger.error("Attempted to call fit() with a "
                                  "multidimensional y vector")
            raise ValueError("y must have a single dimension!")
        if len(X.shape) != 2:
            if self.do_logging:
                self.logger.error("Attempted to call fit() with "
                                  "dim(X) > 2")
            raise ValueError("X must have a single dimension!")
        if len(y) != N:
            if self.do_logging:
                self.logger.error("Attempted to call fit() with "
                                  "len(y) != X.shape[0]")
            raise ValueError("X, y length mismatch %d != %d"
                             % (X.shape[0], len(y)))

        C, N_c = np.unique(y, return_counts=True)  # Class labels
        N_c = {c: N_c[i] for i, c in enumerate(C)}  # Label : count table
        self.C, self.N_c = C, N_c
        self.num_catagories = len(C)  # Number of unique labels

        # Label to class data look up table
        X = {c: np.vstack([X[i, :] for i in range(N) if y[i] == c])
             for c in C}

        if method is None:
            method = self.fit_method

        if method == MAXIMUM_LIKELIHOOD:
            self._fit_ML(X, y)
        else:  # Bayesian method
            # Specify priors
            self.m = {c: np.mean(
                np.vstack([np.mean(X[c], axis=0) for c in self.C]), axis=0)
                      for c in self.C}
            self.S = {c: np.diag(
                np.mean(np.vstack([np.std(X[c], axis=0)]), axis=0))
                      for c in self.C}

            # TODO: I should check that these are numeric values?
            self.alpha = alpha  # Dir(alpha) prior on pi
            self.nu = p + nu  # Covariance pseudo-data
            self.k = k  # Mean pseudo-data

            if method == BAYES_MAP:
                self._fit_MAP(X, y)
            elif method == BAYES_MEAN:
                self._fit_mean(X, y)
            elif method == BAYES_FULL:
                self._fit_full(X, y)
            else:
                if self.do_logging:
                    self.logger.error("Attempted to fit model via "
                                      "the full bayesian procedure.  "
                                      "This is not yet available.")
                raise NotImplementedError

        if self.do_logging:
            self.logger.debug("Successfully fit DA model via %s" % method)
            self.logger.debug("pi = %s" % self.pi)
            self.logger.debug("mu = %s" % self.mu)
            self.logger.debug("Sigma = %s" % self.Sigma)
        return self

    def predict_proba(self, X, ret_array=True):
        """
        Compute the predicted probability of the data in X for each class.

        params:
          X (np.array n x p): Points at which to evaluate the density
          ret_array (bool): Whether to return a {cls: prob} dictionary (false)
            or an n x num_catagories array.  In the latter case, the
            ordering will be given by the order of self.C
        """
        if not self._fitted:
            raise ValueError("Must fit the model first!")
        if self._fitted != BAYES_FULL:  # Gaussian posterior
            def prob(c):
                try:
                    return self.pi[c] *\
                        stats.multivariate_normal.pdf(X, mean=self.mu[c],
                                                      cov=self.Sigma[c])
                except np.linalg.LinAlgError as e:  # Singular matrix
                    if self.do_logging:
                        self.loger.error("LinAlgError on normal.pdf "
                                         "mean = %s, cov = %s"
                                         % (self.mu[c], self.Sigma[c]))
                    raise e

        else:  # T-distributed posterior
            def prob(c):
                try:
                    return self.pi[c] *\
                        mvar_t_pdf(X, self.mu[c], self.Sigma[c],
                                   self.nu_post[c])
                except np.linalg.LinAlgError as e:  # Singular matrix
                    if self.do_logging:
                        self.loger.error("LinAlgError on normal.pdf "
                                         "mean = %s, cov = %s"
                                         % (self.mu[c], self.Sigma[c]))
                    raise e

        density = {}
        normalizer = np.zeros(X.shape[0])
        for c in self.C:
            try:
                P = prob(c)
            except np.linalg.LinAlgError:  # Singular matrix
                P = np.nan * np.empty(X.shape[0])
                if self.do_logging:
                    self.logger.error("Caught LinAlgError when evaluating "
                                      "probability, attempting to continue")
                density[c] = np.zeros_like(P)
                continue  # Skip updating the normalizer

            density[c] = P
            normalizer += P

            if self.do_logging:
                self.logger.debug("Unnormalized density for class %s: %s" %
                                  (c, P))

        for c in self.C:
            # Could raise a zerodivision runtime warning
            density[c] = density[c] / normalizer

        if self.do_logging:
            self.logger.debug("density normalizer: %s" % normalizer)
        if ret_array:
            density = np.vstack(density[c] for c in density.keys()).T
        return density

    # Augment this to consider a loss function?
    def predict(self, X):
        """
        Predicts the class label of each point in X by simply picking the
        most probable label.

        params:
          X (np.array N x p): The data to label
        """
        D = self.predict_proba(X)
        predictions = self.C[np.argmax([D[c] for c in self.C], axis=0)]
        return predictions

    def _fit_ML(self, X, y=None):
        # Fits via ML.  X should be a {label: Array} look up table.
        self.pi = {c: self.N_c[c] / self.N for c in self.C}  # Class probs
        self.mu = {c: np.mean(X[c], axis=0) for c in self.C}  # Class means
        self.Sigma = {c: np.cov(X[c], bias=True, rowvar=False) for c in self.C}
        self._fitted = MAXIMUM_LIKELIHOOD
        return

    def _bayes_update(self, X, y=None):
        C = self.C
        x_bar = {c: np.mean(X[c], axis=0) for c in C}

        alpha_hat = {c: self.alpha + self.N_c[c] for c in C}
        k_hat = {c: self.k + self.N_c[c] for c in C}
        m_hat = {c: (self.k * self.m[c] + self.N_c[c] * x_bar[c])
                 / k_hat[c] for c in C}
        # Note: S is just a scatter matrix, not a normalized covariance
        S_hat = {c: self.S[c] + self.N_c[c] * np.cov(X[c],
                                                     bias=True, rowvar=False) +
                 (self.k * self.N_c[c] / (self.k + self.N_c[c])) *
                 np.outer(x_bar[c] - self.m[c], x_bar[c] - self.m[c])
                 for c in C}
        nu_hat = {c: self.nu + self.N_c[c] for c in C}
        return alpha_hat, k_hat, m_hat, nu_hat, S_hat

    def _fit_MAP(self, X, y=None):
        # Fits via Bayesian MAP.  X should be a {label: Array} look up table.

        p, C = self.p, self.C
        alpha_hat, k_hat, m_hat, nu_hat, S_hat = self._bayes_update(X, y)
        alpha0_hat = sum(alpha_hat[c] for c in C)
        pi_norm = alpha0_hat - self.num_catagories
        self.pi = {c: (alpha_hat[c] - 1) / pi_norm for c in C}
        self.Sigma = {c: S_hat[c] / (nu_hat[c] + p + 1) for c in C}
        self.mu = m_hat
        self._fitted = BAYES_MAP
        return

    def _fit_mean(self, X, y=None):
        # Fits via Bayesian posterior mean.  X is {label: Array} look up table.

        p, C = self.p, self.C
        alpha_hat, k_hat, m_hat, nu_hat, S_hat = self._bayes_update(X, y)
        alpha0_hat = sum(alpha_hat[c] for c in C)
        self.pi = {c: alpha_hat[c] / alpha0_hat for c in C}
        self.Sigma = {c: S_hat[c] / (nu_hat[c] - p - 1) for c in C}
        self.mu = m_hat
        self._fitted = BAYES_MEAN
        return

    def _fit_full(self, X, y=None):
        p, C = self.p, self.C
        alpha_hat, k_hat, m_hat, nu_hat, S_hat = self._bayes_update(X, y)
        alpha0_hat = sum(alpha_hat[c] for c in C)
        self.pi = {c: alpha_hat[c] / alpha0_hat for c in C}
        self.mu = m_hat
        self.Sigma = {c: ((k_hat[c] + 1) / (k_hat[c] * (nu_hat[c] - p + 1))) *
                      S_hat[c] for c in C}
        self.nu_post = {c: nu_hat[c] - p + 1 for c in C}
        self._fitted = BAYES_FULL
        return
