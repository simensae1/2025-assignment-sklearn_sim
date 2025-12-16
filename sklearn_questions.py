"""Assignment - making a sklearn estimator and cv splitter.

The goal of this assignment is to implement by yourself:

- a scikit-learn estimator for the KNearestNeighbors for classification
  tasks and check that it is working properly.
- a scikit-learn CV splitter where the splits are based on a Pandas
  DateTimeIndex.

Detailed instructions for question 1:
The nearest neighbor classifier predicts for a point X_i the target y_k of
the training sample X_k which is the closest to X_i. We measure proximity with
the Euclidean distance. The model will be evaluated with the accuracy (average
number of samples corectly classified). You need to implement the `fit`,
`predict` and `score` methods for this class. The code you write should pass
the test we implemented. You can run the tests by calling at the root of the
repo `pytest test_sklearn_questions.py`. Note that to be fully valid, a
scikit-learn estimator needs to check that the input given to `fit` and
`predict` are correct using the `validate_data, check_is_fitted` functions
imported in this file.
You can find more information on how they should be used in the following doc:
https://scikit-learn.org/stable/developers/develop.html#rolling-your-own-estimator.
Make sure to use them to pass `test_nearest_neighbor_check_estimator`.


Detailed instructions for question 2:
The data to split should contain the index or one column in
datatime format. Then the aim is to split the data between train and test
sets when for each pair of successive months, we learn on the first and
predict of the following. For example if you have data distributed from
november 2020 to march 2021, you have have 4 splits. The first split
will allow to learn on november data and predict on december data, the
second split to learn december and predict on january etc.

We also ask you to respect the pep8 convention: https://pep8.org. This will be
enforced with `flake8`. You can check that there is no flake8 errors by
calling `flake8` at the root of the repo.

Finally, you need to write docstrings for the methods you code and for the
class. The docstring will be checked using `pydocstyle` that you can also
call at the root of the repo.

Hints
-----
- You can use the function:

from sklearn.metrics.pairwise import pairwise_distances

to compute distances between 2 sets of samples.
"""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score


import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin

from sklearn.model_selection import BaseCrossValidator

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score


class KNearestNeighbors(ClassifierMixin, BaseEstimator):
    """KNearestNeighbors classifier.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to use for prediction.
    """

    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fitting function.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        -------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        # Note: validate_data is called without the estimator parameter here 
        # to correctly validate the input arrays X and y.
        X, y = validate_data(X, y, ensure_min_samples=1, reset=True)
        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        """Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        -------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self)
        X = validate_data(X, reset=False, ensure_min_samples=1)

        # Compute Euclidean distance between test samples X and training samples self.X_
        distances = pairwise_distances(X, self.X_, metric='euclidean')

        # Get indices of n_neighbors closest training samples
        # argsort sorts by distance, we take the first n_neighbors indices
        k_nearest_indices = np.argsort(distances, axis=1)[:, :self.n_neighbors]

        # Get the labels of the n_neighbors
        k_nearest_labels = self.y_[k_nearest_indices]

        # Find the most common label among the k-neighbors (majority vote)
        y_pred = np.array([
            np.argmax(np.bincount(k_nearest_labels[i]))
            for i in range(k_nearest_labels.shape[0])
        ])

        return y_pred

    def score(self, X, y):
        """Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples,)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        y_pred = self.predict(X)
        # Using sklearn.metrics.accuracy_score for consistency
        return accuracy_score(y, y_pred)


class MonthlySplit(BaseCrossValidator):
    """CrossValidator based on monthly split.

    Split data based on the given `time_col` (or default to index). Each split
    corresponds to one month of data for the training and the next month of
    data for the test.

    Parameters
    ----------
    time_col : str, defaults to 'index'
        Column of the input DataFrame that will be used to split the data. This
        column should be of type datetime. If split is called with a DataFrame
        for which this column is not a datetime, it will raise a ValueError.
        To use the index as column just set `time_col` to `'index'`.
    """

    def __init__(self, time_col='index'):
        self.time_col = time_col

    def _get_time_data(self, X):
        """Helper to extract and validate time data."""
        if self.time_col == 'index':
            if not pd.api.types.is_datetime64_any_dtype(X.index):
                raise ValueError(
                    "The index must be of datetime type when time_col is 'index'."
                )
            time_data = X.index
        else:
            if not isinstance(X, pd.DataFrame):
                raise ValueError(
                    f"X must be a pandas DataFrame when time_col is "
                    f"'{self.time_col}', but got type {type(X)}"
                )
            if self.time_col not in X.columns:
                raise ValueError(
                    f"Column '{self.time_col}' not found in the input DataFrame."
                )
            time_data = X[self.time_col]
            if not pd.api.types.is_datetime64_any_dtype(time_data):
                raise ValueError(
                    f"The column '{self.time_col}' must be of datetime type, "
                    f"but got type {time_data.dtype}."
                )
        return time_data

    def get_n_splits(self, X, y=None, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        if not isinstance(X, (pd.DataFrame, pd.Series)):
             # We need a pandas object to access the index or a column
             raise ValueError("X must be a pandas DataFrame or Series.")

        time_data = self._get_time_data(X)
        months = time_data.to_period().dt.to_period('M').unique().sort_values()
        
        # Number of splits is the number of successive month pairs: len(months) - 1
        return max(0, len(months) - 1)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        idx_train : ndarray
            The training set indices for that split.
        idx_test : ndarray
            The testing set indices for that split.
        """
        if not isinstance(X, (pd.DataFrame, pd.Series)):
             raise ValueError("X must be a pandas DataFrame or Series.")

        time_data = self._get_time_data(X)
        # Convert datetime to Month-Period objects for accurate monthly grouping
        time_data_periods = time_data.to_period().dt.to_period('M')
        months = time_data_periods.unique().sort_values()

        for i in range(len(months) - 1):
            train_month = months[i]
            test_month = months[i+1]

            # Get the indices where the month matches the train_month
            # .index gives the DataFrame/Series index, which corresponds to the array index if X is a simple numpy array, 
            # but for a DataFrame/Series with a non-default index, we need the position in the array.
            # Since X might be a numpy array in the split/get_n_splits signature but the tests provide a DataFrame, 
            # we use np.where on the boolean masks for simplicity and robustness.
            
            is_train = time_data_periods == train_month
            is_test = time_data_periods == test_month

            idx_train = np.where(is_train)[0]
            idx_test = np.where(is_test)[0]

            if len(idx_train) > 0 and len(idx_test) > 0:
                yield (
                    idx_train, idx_test
                )
