import unittest
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
        P = DA.predict_proba(X, ret_array=False)
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
        P = DA.predict_proba(X, ret_array=False)
        assert np.allclose(np.sum(P[c] for c in C), 1.0)
        return

    def test002_mean(self):
        N = 100
        p = 3
        C = ["class1", "class2"]
        X = np.random.normal(size=(N, p))
        y = np.random.choice(C, size=N)

        DA = DiscriminantAnalysis(do_logging=True)
        DA.fit(X, y, method=BAYES_MEAN)
        P = DA.predict_proba(X, ret_array=False)
        assert np.allclose(np.sum(P[c] for c in C), 1.0)
        return

    def test003_full(self):
        N = 100
        p = 3
        C = ["class1", "class2"]
        X = np.random.normal(size=(N, p))
        y = np.random.choice(C, size=N)

        DA = DiscriminantAnalysis(do_logging=True)
        DA.fit(X, y, method=BAYES_FULL)
        P = DA.predict_proba(X, ret_array=False)
        assert np.allclose(np.sum(P[c] for c in C), 1.0)
        return


def visualize_3class(X, y_cls, DA, title=None, save_file=None):
    x_min, x_max = np.min(X[:, 0]), np.max(X[:, 0])
    y_min, y_max = np.min(X[:, 1]), np.max(X[:, 1])
    x = np.linspace(1.1 * x_min, 1.1 * x_max, 500)
    y = np.linspace(1.1 * y_min, 1.1 * y_max, 500)
    xx, yy = np.meshgrid(x, y)
    xxyy = np.array([xx.ravel(), yy.ravel()]).T
    P = DA.predict_proba(xxyy, ret_array=False)
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

    def test003_FULL(self):
        X, y_cls = self.X, self.y_cls

        DA = DiscriminantAnalysis()
        DA.fit(X, y_cls, method=BAYES_FULL)

        visualize_3class(X, y_cls, DA,
                         title="Full Bayes Decision Boundaries",
                         save_file="./figures/FULL_boundaries.png")
        return


class testDA_learning_curves(unittest.TestCase):
    def plot_learning_curve(self, estimator, X, y, ax, train_sizes,
                            color="b", scoring="neg_log_loss",
                            ylabel=None):
        """
        Generate a simple plot of the test and training learning curve.
        """
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=4, train_sizes=train_sizes,
            scoring=scoring)
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

    def learning_curve_plots(self, X, y, title=None, save_file=None,
                             scoring="neg_log_loss", ylabel=None):
        fig, ax = plt.subplots(1, 1)
        DA_ML = DiscriminantAnalysis(fit_method=MAXIMUM_LIKELIHOOD)
        DA_MAP = DiscriminantAnalysis(fit_method=BAYES_MAP)
        DA_MEAN = DiscriminantAnalysis(fit_method=BAYES_MEAN)
        DA_FULL = DiscriminantAnalysis(fit_method=BAYES_FULL)

        train_sizes = np.linspace(0.1, 1.0, 15)
        ax = self.plot_learning_curve(DA_ML, X, y, ax, scoring=scoring,
                                      train_sizes=train_sizes,
                                      color="b")
        ax = self.plot_learning_curve(DA_MAP, X, y, ax, scoring=scoring,
                                      train_sizes=train_sizes,
                                      color="r")
        ax = self.plot_learning_curve(DA_MEAN, X, y, ax, scoring=scoring,
                                      train_sizes=train_sizes,
                                      color="g")
        ax = self.plot_learning_curve(DA_FULL, X, y, ax, scoring=scoring,
                                      train_sizes=train_sizes,
                                      color="m")

        ax.plot([], label="Training", linestyle="--", color="k")
        ax.plot([], label="Testing", linestyle="-", color="k")
        ax.plot([], label="ML", color="b")
        ax.plot([], label="MAP", color="r")
        ax.plot([], label="MEAN", color="g")
        ax.plot([], label="FULL", color="m")

        plt.legend()
        plt.grid()
        plt.xlabel("num samples", fontsize=14)
        if ylabel is None:
            plt.ylabel("loss", fontsize=14)
        else:
            plt.ylabel(ylabel, fontsize=14)

        if title is None:
            plt.title("Learning Curves", fontsize=14)
        else:
            plt.title(title, fontsize=14)

        if save_file is not None:
            plt.savefig(save_file)
        plt.show()
        return

    def test001(self):
        np.random.seed(0)
        N = 500
        p = 2
        n_classes = 3
        X, y = datasets.make_classification(n_samples=N, n_features=p,
                                            n_informative=p, n_redundant=0,
                                            n_clusters_per_class=1,
                                            n_classes=n_classes)
        self.learning_curve_plots(X, y, title=r"Learning Curves, $p = %d$" % p,
                                  save_file="./figures/learning_curves001.png",
                                  ylabel="Log Loss")
        return

    def test002(self):
        np.random.seed(0)
        N = 500
        p = 10
        n_classes = 3
        X, y = datasets.make_classification(n_samples=N, n_features=p,
                                            n_informative=p,
                                            n_redundant=0,
                                            n_clusters_per_class=2,
                                            n_classes=n_classes)
        self.learning_curve_plots(X, y, title=r"Learning Curves, $p = %d$" % p,
                                  save_file="./figures/learning_curves002.png",
                                  ylabel="Log Loss")
        return

    def test003(self):
        p = 25

        # Load, then shuffle the dataset
        X, y = datasets.load_digits(10, True)
        Xy = np.random.permutation(np.hstack((X, y[:, None])))
        X, y = Xy[:, :-1], Xy[:, -1]

        ss = StandardScaler()
        X = ss.fit_transform(X)
        # PCA is not the best projection for this algorithm
        pca = PCA(n_components=p)
        X = pca.fit_transform(X)
        DA = DiscriminantAnalysis()
        DA.fit(X, y)
        self.learning_curve_plots(X, y, title=r"Learning Curves "
                                  "(digits data with PCA: $p = %d$)" % p,
                                  save_file="./figures/"
                                  "learning_curves_digits.png",
                                  scoring="accuracy",
                                  ylabel="Accuracy")
        return
