from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    magnitude_factor = 1
    for n in ["linearly_inseparable.npy", "linearly_separable.npy"]:
        miss_classed = []
        X , y = load_dataset(f"../datasets/{n}")
        my_per = Perceptron(callback = lambda model , _, __: miss_classed.append(model.loss(X,y)*100))
        # when fitting calls callback so error percentage is added to miss classed array
        my_per.fit(X,y)
        # sets data name as a string to be added to the graph title
        fig = go.Figure(data=go.Scatter(x=list(range(len(miss_classed))), y=miss_classed, mode="lines",
                                        line = dict(width = 0.3*magnitude_factor, color= "black")),
                        layout=go.Layout(
                            title={"x": 0.5, "text": r"$\text{Perceptron model error percentage over - %s dataset}$"
                                                     % n.split(".")[0].replace("_", " ")},
                            xaxis_title=r"$\text{Number of allowed fixing iterations}$",
                            yaxis_title=r"$\text{percentage of misclassified samples (%)}$"))
        fig.write_image(f"C:/Users/itani/BioInformatics/2-B/IML/IML_EX3/perceptron_fit_{n}.png")
        magnitude_factor += 1



def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    # for f in ["gaussian1.npy", "gaussian2.npy"]:
    #     # Load dataset
    #     samples, true_y = load_dataset(f"../datasets/{f}")
    #     lda =  LDA().fit(samples, true_y)
    #     bayes =  GaussianNaiveBayes().fit(samples, true_y)
    #
    #     lda_res = lda.predict(samples)
    #     bayes_res = bayes.predict(samples)
    #
    #
    #     # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
    #     # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
    #     # Create subplots
    #     from IMLearn.metrics import accuracy
    #     fig = make_subplots(rows=1, cols=2,
    #                         subplot_titles=(
    #                             rf"$\text{{LDA (accuracy={round(accuracy(true_y, lda_res), )}%)}}$",
    #                         rf"$\text{{Gaussian Naive Bayes (accuracy={round(accuracy(true_y, bayes_res), 2)}%)}}$"))
    #     fig.write_image(f"C:/Users/itani/BioInformatics/2-B/IML/IML_EX3/LDA_vs_naive_bayes_{f}.png")
    #
    #     fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
    #                                marker=dict(color=bayes_res, symbol=class_symbols[true_y], colorscale=class_colors(3))),
    #                     go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
    #                                marker=dict(color=lda_res, symbol=class_symbols[true_y], colorscale=class_colors(3)))],
    #                    rows=[1, 1], cols=[1, 2])
    # from plotly.validators.scatter.marker import SymbolValidator
    # raw_symbols = SymbolValidator().values
    # symbols = []
    # for i in range(0, len(raw_symbols), 3):
    #     symbols.append(raw_symbols[i])

    symbols = ['circle', 'x', 'diamond']

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        naive, lda = GaussianNaiveBayes().fit(X, y), LDA().fit(X, y)
        naive_preds, lda_preds = naive.predict(X), lda.predict(X)

        # Create a mapping of class labels to symbols
        class_to_symbol = {class_label: symbol for class_label, symbol in zip(set(y), symbols)}

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=(
                                rf"$\text{{Gaussian Naive Bayes (accuracy={round(accuracy(y, naive_preds), 2)}%)}}$",
                                rf"$\text{{LDA (accuracy={round(accuracy(y, lda_preds), )}%)}}$"))
        f.split(".")[0]
        fig.update_layout(title=f'Naive Bayse vs. LDA estimators comparison over {f.split(".")[0]}')

        # Add traces for data-points setting symbols and colors
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=naive_preds, symbol=[class_to_symbol[label] for label in y],
                                               colorscale="Pinkyl", )),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=lda_preds, symbol=[class_to_symbol[label] for label in y],
                                               colorscale="Pinkyl"))],
                       rows=[1, 1], cols=[1, 2])
        fig.add_traces([go.Scatter(x=naive.mu_[:, 0], y=naive.mu_[:, 1], mode="markers",
                                   marker=dict(symbol="star-diamond", color="black", size=15)),
                        go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                                   marker=dict(symbol="star-diamond", color="black", size=15))],
                       rows=[1, 1], cols=[1, 2])
        for i in range(3):
            fig.add_traces([get_ellipse(naive.mu_[i], np.diag(naive.vars_[i]),), get_ellipse(lda.mu_[i], lda.cov_)],
                           rows=[1, 1], cols=[1, 2])

        fig.write_image(f"C:/Users/itani/BioInformatics/2-B/IML/IML_EX3/LDA_vs_naive_bayes_{f}.png")
        # Add traces for data-points setting symbols and colors
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(title_text=rf"$\text{{Comparing Gaussian Classifiers - {f[:-4]} dataset}}$",
                          width=800, height=400, showlegend=False)
        fig.write_image(f"lda.vs.naive.bayes.{f[:-4]}.png")

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
