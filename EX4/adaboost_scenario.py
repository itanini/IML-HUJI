import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.utils import utils_new
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, 0), generate_data(test_size, 0)
    (train_X_noised, train_y_noised), (test_X_noised, test_y_noised) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model, test_error = Q1(n_learners, noise, test_X_noised, test_y_noised, train_X_noised,
                           train_y_noised)

    # Question 2: Plotting decision surfaces
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    if noise ==0:

        Q2(test_X, test_y, train_X, model, lims)

    # Question 3: Decision surface of best performing ensemble
        Q3(lims, test_X, test_error, test_y,model)

    # Question 4: Decision surface with weighted samples

    Q4(lims, model, noise, train_X_noised, train_y_noised)


def Q4(lims, model, noise, train_X_noised, train_y_noised):
    D = 20 * model.D_ / model.D_.max()
    fig = go.Figure()
    fig.add_trace(
        utils_new.decision_surface(model.predict, lims[0], lims[1], density=60, showscale=False)
    )
    fig.add_trace(
        go.Scatter(
            x=train_X_noised[:, 0],
            y=train_X_noised[:, 1],
            mode="markers",
            showlegend=False,
            marker=dict(
                size=D,
                color=train_y_noised,
                symbol=np.where(train_y_noised == 1, "circle", "x")
            )
        )
    )
    fig.update_layout(
        width=500,
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title="Sample Distribution of Final AdaBoost"
    )
    fig.write_image(f"C:/Users/itani/BioInformatics/2-B/IML/IML- EX4/adaboost_{noise}_weighted_samples.png")


def Q3(lims, test_X, test_error, test_y, model):
    best_t = np.argmin(test_error) + 1
    fig = go.Figure()
    fig.add_trace(
        utils_new.decision_surface(lambda X: model.partial_predict(X, best_t), lims[0], lims[1], density=60,
                                   showscale=False)
    )
    fig.add_trace(
        go.Scatter(
            x=test_X[:, 0],
            y=test_X[:, 1],
            mode="markers",
            showlegend=False,
            marker=dict(color=test_y, symbol=np.where(test_y == 1, "circle", "x"))
        )
    )
    fig.update_layout(
        width=500,
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title=f"Minimal generalization error committee<br>Size: {best_t}, Accuracy: {1 - round(test_error[best_t - 1], 2)}"
    )
    fig.write_image \
        (f"C:/Users/itani/BioInformatics/2-B/IML/IML- EX4/"
         f"adaboost_best_generalization_error.png")


def Q2(test_X, test_y, train_X, model, lims):
    T = [5, 50, 100, 250]

    fig = make_subplots(rows=1, cols=4, subplot_titles=[
        f"Classification by a {T[0]}-sized committee",
        f"Classification by a {T[1]}-sized committee",
        f"Classification by a {T[2]}-sized committee",
        f"Classification by a {T[3]}-sized committee"
    ])
    for i, t in enumerate(T):
        fig.add_traces(
            [
                utils_new.decision_surface(lambda X: model.partial_predict(X, t), lims[0], lims[1], density=60,
                                           showscale=False),
                go.Scatter(
                    x=test_X[:, 0],
                    y=test_X[:, 1],
                    mode="markers",
                    showlegend=False,
                    marker=dict(
                        color=test_y,
                        symbol=np.where(test_y == 1, "circle", "x"),
                    )
                )
            ],
            rows=1,
            cols=i + 1
        )
    fig.update_layout(height=500, width=2000).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.write_image("C:/Users/itani/BioInformatics/2-B/IML/IML- EX4/adaboost_decision_boundaries.png")


def Q1(n_learners, noise, test_X_noised, test_y_noised, train_X_noised, train_y_noised):
    model = AdaBoost(DecisionStump, n_learners).fit(train_X_noised, train_y_noised)
    train_error = [model.partial_loss(train_X_noised, train_y_noised, t) for t in range(1, n_learners + 1)]
    test_error = [model.partial_loss(test_X_noised, test_y_noised, t) for t in range(1, n_learners + 1)]
    fig = go.Figure(
        data=[
            go.Scatter(x=list(range(1, n_learners + 1)), y=train_error, name="Train Error", mode="lines",
                       line=dict(color='yellow')),
            go.Scatter(x=list(range(1, n_learners + 1)), y=test_error, name="Test Error", mode="lines",
                       line=dict(color='pink'))
        ],
        layout=go.Layout(
            width=600, height=500,
            title={
                "text": "<b>AdaBoost Misclassification as func of the Number of Classifiers</b>",
                "font": {"size": 18, "color": 'rgb(255, 255, 255)'},
                "x": 0.5,
                "y": 0.95
            },
            xaxis_title="<b>Iteration</b>",
            yaxis_title="<b>Misclassification Error</b>",
            plot_bgcolor='rgb(0,0,0)',
            paper_bgcolor='rgb(0,0,0)',
            font=dict(family='Arial, sans-serif', color='rgb(255, 255, 255)', size=12),
            legend=dict(x=0.01, y=0.95)
        )
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='rgb(255, 255, 255)', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='rgb(255, 255, 255)', mirror=True)
    fig.write_image(f"C:/Users/itani/BioInformatics/2-B/IML/IML- EX4/noise_=_{noise}_adaboost_Q1.png")
    return model, test_error


if __name__ == '__main__':
    np.random.seed(30)
    for intercept in [0, .4]:
        fit_and_evaluate_adaboost(intercept)


