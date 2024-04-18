
from __future__ import annotations
from typing import Tuple, NoReturn

from numpy import ndarray

from IMLearn.base import BaseEstimator
import numpy as np
from IMLearn.metrics import *
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_loss = np.inf
        best_j = None
        best_threshold = None
        best_sign = None

        for j in range(X.shape[1]):  # Iterate over columns
            values = X[:, j]
            threshold, loss = self._find_threshold(values, y, 1)
            if loss < best_loss:
                best_loss = loss
                best_j = j
                best_threshold = threshold
                best_sign = 1

            if best_loss == 0:
                break

            threshold, loss = self._find_threshold(values, y, -1)
            if loss < best_loss:
                best_loss = loss
                best_j = j
                best_threshold = threshold
                best_sign = -1

            if best_loss == 0:
                break

        self.j_ = best_j
        self.threshold_ = best_threshold
        self.sign_ = best_sign
#
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        y_pred = np.sign((X[:, self.j_] - self.threshold_) * self.sign_)
        y_pred = np.where(y_pred == 0, self.sign_, y_pred)  # class all zeros as sign.
        return y_pred
#
    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        def cur_loss(sign: int, y: np.ndarray) -> ndarray:
            return np.sum(np.abs(y)[np.sign(y) == sign])

        sorted_values = np.argsort(values)
        values, labels = values[sorted_values], labels[sorted_values]

        loss = cur_loss(sign, labels)

        loss = np.append(loss, loss - np.cumsum(labels * sign))

        id = np.argmin(loss)
        return np.concatenate([[-np.inf], values[1:], [np.inf]])[id], loss[id]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X[:, self.j_])
        return misclassification_error(y_true=y, y_pred=y_pred, normalize=True)

