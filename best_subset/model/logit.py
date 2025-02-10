import numpy as np

class LogisticRegression:
    """
    Represents a logistic regression model and provides methods for its 
    core calculations (log-likelihood, CDF, information matrix, score function).
    """

    @staticmethod
    def loglike(params: np.ndarray, X: np.ndarray, y: np.ndarray, W: np.ndarray) -> float:
        """Calculates the log-likelihood."""
        q = 2 * y - 1
        return np.sum(W * np.log(LogisticRegression.cdf(q * np.dot(X, params))))

    @staticmethod
    def cdf(X: np.ndarray) -> np.ndarray:
        """Calculates the cumulative distribution function (CDF)."""
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def information_matrix(params: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Calculates the information matrix (negative Hessian)."""
        L = LogisticRegression.cdf(np.dot(X, params))
        return -np.dot(W * L * (1 - L) * X.T, X)

    @staticmethod
    def score_function(params: np.ndarray, X: np.ndarray, y: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Calculates the score function (gradient)."""
        L = LogisticRegression.cdf(np.dot(X, params))
        return np.dot(W * (y - L), X)
