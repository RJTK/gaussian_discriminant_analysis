import unittest
import numpy as np

from sklearn import datasets
from sklearn.metrics import log_loss
from sklearn.model_selection import learning_curve
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
        P = DA.predict_proba(X)
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
        P = DA.predict_proba(X)
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
        P = DA.predict_proba(X)
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
        # DA.predict_proba(X)
        return


def visualize_3class(X, y_cls, DA, title=None, save_file=None):
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
    x = np.linspace(1.1 * x_min, 1.1 * x_max, 500)
    y = np.linspace(1.1 * y_min, 1.1 * y_max, 500)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.array([xx.ravel(), yy.ravel()]).T
    P = DA.predict_proba(xxyy)
    P = {c: P[c].reshape(xx.shape) for c in P.keys()}
    Z = np.stack((P[c] for c in P.keys()), -1)

    plt.scatter(X[:, 0], X[:, 1], c=np.array(['r', 'g', 'b'])[y_cls],
                edgecolors=(0, 0, 0))
    plt.imshow(Z, origin='lower', alpha=0.5,
               extent=(x_min, x_max, y_min, y_max))
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$y$", fontsize=14)
    if title is not None:
        plt.title(title, fontsize=14)

    if save_file is not None:
        plt.savefig(save_file)

    plt.show()
    return


class testDA_visualize(unittest.TestCase):
    """Some 2D classification visualizations"""
    @classmethod
    def setUpClass(cls):
        np.random.seed(0)
        N = 150
        p = 2
        n_classes = 3
        X, y_cls = datasets.make_classification(n_samples=N, n_features=p,
                                                n_informative=p, n_redundant=0,
                                                n_clusters_per_class=1,
                                                n_classes=n_classes,
                                                weights=[0.3, 0.5])
        cls.N, cls.p, cls.n_classes = N, p, n_classes
        cls.X, cls.y_cls = X, y_cls
        return

    def test000_ML(self):
        X, y_cls = self.X, self.y_cls
        DA = DiscriminantAnalysis()
        DA.fit(X, y_cls, method=MAXIMUM_LIKELIHOOD)

        visualize_3class(X, y_cls, DA, title="ML Decision Boundaries",
                         save_file="./figures/ML_boundaries.png")
        return

    def test001_MAP(self):
        X, y_cls = self.X, self.y_cls

        DA = DiscriminantAnalysis()
        DA.fit(X, y_cls, method=BAYES_MAP)

        visualize_3class(X, y_cls, DA, title="MAP Decision Boundaries",
                         save_file="./figures/MAP_boundaries.png")
        return

    def test002_MEAN(self):
        X, y_cls = self.X, self.y_cls

        DA = DiscriminantAnalysis()
        DA.fit(X, y_cls, method=BAYES_MEAN)

        visualize_3class(X, y_cls, DA,
                         title="Posterior Mean Decision Boundaries",
                         save_file="./figures/MEAN_boundaries.png")
        return


class testDA_ll(unittest.TestCase):
    def plot_learning_curve(self, estimator, X, y, ax, train_sizes,
                            color="b"):
        """
        Generate a simple plot of the test and training learning curve.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=10, n_jobs=4, train_sizes=train_sizes,
            scoring="neg_log_loss")
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color=color)
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color=color)
        ax.plot(train_sizes, train_scores_mean, 'o--', color=color)
        ax.plot(train_sizes, test_scores_mean, 'o-', color=color)
        return ax

    def learning_curve_plots(self, X, y, save_file=None):
        fig, ax = plt.subplots(1, 1)
        DA_ML = DiscriminantAnalysis(fit_method=MAXIMUM_LIKELIHOOD)
        DA_MAP = DiscriminantAnalysis(fit_method=BAYES_MAP)
        DA_MEAN = DiscriminantAnalysis(fit_method=BAYES_MEAN)

        ax = self.plot_learning_curve(DA_ML, X, y, ax,
                                      train_sizes=np.linspace(.1, 1.0, 10),
                                      color="b")
        ax = self.plot_learning_curve(DA_MAP, X, y, ax,
                                      train_sizes=np.linspace(.1, 1.0, 10),
                                      color="r")
        ax = self.plot_learning_curve(DA_MEAN, X, y, ax,
                                      train_sizes=np.linspace(.1, 1.0, 10),
                                      color="g")
        ax.plot([], label="Training", linestyle="--", color="k")
        ax.plot([], label="Testing", linestyle="-", color="k")
        ax.plot([], label="ML", color="b")
        ax.plot([], label="MAP", color="r")
        ax.plot([], label="MEAN", color="g")

        plt.legend()
        plt.grid()
        plt.xlabel("num samples", fontsize=14)
        plt.ylabel("Log Loss", fontsize=14)
        plt.title("Learning Curves", fontsize=14)
        if save_file is not None:
            plt.savefig(save_file)
        plt.show()
        return

    def test000(self):
        np.random.seed(0)
        N = 2000
        p = 2
        n_classes = 3
        X, y = datasets.make_classification(n_samples=N, n_features=p,
                                            n_informative=p, n_redundant=0,
                                            n_clusters_per_class=1,
                                            n_classes=n_classes)
        self.learning_curve_plots(X, y, save_file="./figures/"
                                  "learning_curves001.png")
        return

    def test001(self):
        np.random.seed(0)
        N = 2000
        p = 10
        n_classes = 3
        X, y = datasets.make_classification(n_samples=N, n_features=p,
                                            n_informative=p,
                                            n_redundant=0,
                                            n_clusters_per_class=2,
                                            n_classes=n_classes)
        self.learning_curve_plots(X, y, save_file="./figures/"
                                  "learning_curves002.png")
        return
