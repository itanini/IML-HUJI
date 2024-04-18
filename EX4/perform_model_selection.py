from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    #load data
    X, y = datasets.load_diabetes(return_X_y = True)

    train_x, train_y = X[:n_samples, :], y[:n_samples]
    test_x, test_y = X[n_samples:, :], y[n_samples:]




    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas_l = np.linspace(0.0001, 2, n_evaluations)
    lambdas_r = np.linspace(0.0001, 0.05, n_evaluations)

    ridge_train_errors = np.zeros(n_evaluations)
    ridge_validation_errors = np.zeros(n_evaluations)
    lasso_train_errors = np.zeros(n_evaluations)
    lasso_validation_errors = np.zeros(n_evaluations)

    for i in range(n_evaluations):
        ridge_model = RidgeRegression(lam=lambdas_r[i], include_intercept=True)
        lasso_model = Lasso(lambdas_l[i], max_iter=5000, fit_intercept=True)

        ridge_train_errors[i], ridge_validation_errors[i] = cross_validate(
            estimator=ridge_model, X=train_x, y=train_y, scoring=mean_square_error, cv=5)

        lasso_train_errors[i], lasso_validation_errors[i] = cross_validate(
            estimator=lasso_model, X=train_x, y=train_y, scoring=mean_square_error, cv=5)

    # Create subplots
    subplot_titles = ("Ridge Regulation", "Lasso Regulation")
    fig = make_subplots(cols=2, subplot_titles=subplot_titles)
    fig.update_layout(title="Error in different types of regulations as function of lambda parameter")

    fig.add_scatter(x=lambdas_r, y=ridge_train_errors, row=1, col=1, name="ridge train error")
    fig.add_scatter(x=lambdas_l, y=lasso_train_errors, row=1, col=2, name="lasso train error")

    ridge_test = go.Scatter(x=lambdas_r, y=ridge_validation_errors, name="ridge validation error")
    lasso_test = go.Scatter(x=lambdas_l, y=lasso_validation_errors, name="lasso validation error")
    fig.add_traces([ridge_test, lasso_test], rows=[1, 1], cols=[1, 2])

    fig.update_xaxes(title_text="Lambda", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=1, col=1)
    fig.update_xaxes(title_text="Lambda", row=1, col=2)
    fig.update_yaxes(title_text="Error", row=1, col=2)

    fig.write_image("C:/Users/itani/BioInformatics/2-B/IML/IML- EX4/intial/lasso_vs_ridge.png")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_arg_lam_r = np.argmin(ridge_validation_errors)
    min_arg_lam_l = np.argmin(lasso_validation_errors)

    best_ridge_lambda = lambdas_r[min_arg_lam_r]
    best_lasso_lambda = lambdas_l[min_arg_lam_l]

    ridge_model = RidgeRegression(lam=best_ridge_lambda, include_intercept=True)
    lasso_model = Lasso(best_lasso_lambda, max_iter=5000, fit_intercept=True)

    ridge_model.fit(train_x, train_y)
    lasso_model.fit(train_x, train_y)
    ls_model = LinearRegression().fit(train_x, train_y)

    print("The regularization parameter values that achieved the best validation errors are:")
    print("Ridge lambda:", best_ridge_lambda)
    print("Lasso lambda:", best_lasso_lambda)

    ridge_test_error = ridge_model.loss(test_x, test_y)
    lasso_test_error = mean_square_error(test_y, lasso_model.predict(test_x))
    ls_test_error = ls_model.loss(test_x, test_y)

    print("The test errors of each of the fitted models are:")
    print("Ridge model:", ridge_test_error)
    print("Lasso model:", lasso_test_error)
    print("Least squares model:", ls_test_error)


if __name__ == '__main__':
    np.random.seed(30)
    select_regularization_parameter()
