from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    samples = np.arange(X.shape[0])
    # sets sample indexes for folds at size cv
    folds = np.array_split(samples, cv)

    train_score, validation_score = .0, .0

    for fold in folds:
        in_train_ids = samples[~np.isin(samples,fold)]
        set_samples, set_responses = X[in_train_ids], y[in_train_ids]
        model = deepcopy(estimator).fit(set_samples,set_responses)

        train_score += scoring(set_responses, model.predict(set_samples))/cv
        validation_score += scoring(y[fold], model.predict(X[fold]))/cv

    return train_score, validation_score
