from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self.cov_inv_, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        self.classes_, self.pi_ = np.unique(y, return_counts=True)
        # the i indices fraction of elements of the self.classes_[i] in y
        self.pi_ = self.pi_ / len(y)
        # self.mu_ =np.array([[]])
        # for v in self.classes_:
        #     v_mu = np.mean([X[y == v]])
        #     self.mu_ = np.vstack([self.mu_, v_mu])
        self.mu_ = np.array([np.mean(X[y == X_e_v], axis=0) for X_e_v in self.classes_])
        X_e_v = X - self.mu_[y.astype(int)]
        self.cov_ = np.einsum("ki,kj->kij", X_e_v, X_e_v).sum(axis=0) / (len(X) - len(self.classes_))

        self.cov_inv_ = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # returns the value that has the maximum likelihood in relation to the X samples
        most_likely = np.argmax(self.likelihood(X), axis=1)
        return self.classes_[most_likely]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("The estimator must be fitted before calling the `likelihood` function.")

        normalization_factor = np.sqrt((2 * np.pi) ** X.shape[1] * np.linalg.det(self.cov_))
        difference = X[:, np.newaxis, :] - self.mu_
        exponent = -0.5 * np.sum(difference.dot(self.cov_inv_) * difference, axis=2)
        likelihoods = np.exp(exponent) / normalization_factor
        return likelihoods * self.pi_

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
