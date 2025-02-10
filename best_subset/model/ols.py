import numpy as np

class LinearRegression:
    """
    A version that returns the log-likelihood components (score & information),
    consistent with the logistic code's sign conventions.
    """

    @staticmethod
    def loglike(params: np.ndarray, X: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
        """
        (Up to a constant) the log-likelihood for linear regression with normal errors 
        (ignoring constant terms like -n/2 * log(2*pi*sigma^2)).
        """
        residuals = y - np.dot(X, params)
        # Typically: -0.5 * sum of weighted residual^2
        return -0.5 * np.dot(W * residuals, residuals)

    @staticmethod
    def score_function(params: np.ndarray, X: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Gradient of the log-likelihood w.r.t. params = X^T [ W * residuals ].
        (Positive sign—unlike the SSE gradient, which has a negative sign.)
        """
        residuals = y - np.dot(X, params)
        return - 2 * np.dot(X.T, W * residuals)

    @staticmethod
    def information_matrix(params: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        The *negative* Hessian of the log-likelihood (a.k.a. the 'information').
        This is X^T W X, which should be positive (semi-)definite.
        """
        return 2 * np.dot(X.T, W[:, np.newaxis] * X)