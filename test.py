import unittest
import numpy as np

from sklearn import datasets
from matplotlib import pyplot as plt

from DA import DiscriminantAnalysis, MAXIMUM_LIKELIHOOD, BAYES_MAP,\
    BAYES_MEAN, BAYES_FULL

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


class testDA_fits(unittest.TestCase):
    """Basic tests to check if methods work at all"""
    def test000_ML(self):
        N = 100
        p = 3
        C = ["class1", "class2"]
        X = np.random.normal(size=(N, p))
        y = np.random.choice(C, size=N)

        DA = DiscriminantAnalysis(do_logging=True)
        DA.fit(X, y, method=MAXIMUM_LIKELIHOOD)
        P = DA.density(X)
        assert np.allclose(np.sum(P[c] for c in C), 1.0)
        return

    def test001_MAP(self):
        N = 100
        p = 3
        C = ["class1", "class2"]
        X = np.random.normal(size=(N, p))
        y = np.random.choice(C, size=N)

        DA = DiscriminantAnalysis(do_logging=True)
        DA.fit(X, y, method=BAYES_MAP)
        P = DA.density(X)
        assert np.allclose(np.sum(P[c] for c in C), 1.0)
        return

    def test002_mean(self):
        N = 100
        p = 3
        C = ["class1", "class2"]
        X = np.random.normal(size=(N, p))
        y = np.random.choice(C, size=N)

        DA = DiscriminantAnalysis(do_logging=True)
        DA.fit(X, y, method=BAYES_MEAN, do_logging=True)
        P = DA.density(X)
        assert np.allclose(np.sum(P[c] for c in C), 1.0)
        return

    def test003_full(self):
        N = 100
        p = 3
        C = ["class1", "class2"]
        X = np.random.normal(size=(N, p))
        y = np.random.choice(C, size=N)

        DA = DiscriminantAnalysis()
        DA.fit(X, y, method=BAYES_FULL, do_logging=True)
        # DA.density(X)
        return


class testDA_visualize(unittest.TestCase):
    """Some 2D classification visualizations"""
    def test001(self):
        np.random.seed(1)
        N = 150
        p = 2
        n_classes = 3
        X, y_cls = datasets.make_classification(n_samples=N, n_features=p,
                                                n_informative=p, n_redundant=0,
                                                n_clusters_per_class=1,
                                                n_classes=n_classes)
        DA = DiscriminantAnalysis()
        DA.fit(X, y_cls, method=BAYES_MEAN)

        x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
        y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
        x = np.linspace(1.1 * x_min, 1.1 * x_max, 500)
        y = np.linspace(1.1 * y_min, 1.1 * y_max, 500)
        xx, yy = np.meshgrid(x, y)
        xxyy = np.array([xx.ravel(), yy.ravel()]).T
        P = DA.density(xxyy)
        P = {c: P[c].reshape(xx.shape) for c in P.keys()}
        Z = np.stack((P[c] for c in P.keys()), -1)

        plt.scatter(X[:, 0], X[:, 1], c=np.array(['r', 'g', 'b'])[y_cls],
                    edgecolors=(0, 0, 0))
        plt.imshow(Z, origin='lower', alpha=0.5,
                   extent=(x_min, x_max, y_min, y_max))
        plt.xlabel(r"$x$", fontsize=14)
        plt.ylabel(r"$y$", fontsize=14)
        plt.title("ML Decision Surface", fontsize=14)
        plt.show()
        return
