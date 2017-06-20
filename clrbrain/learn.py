# SQLite database connection
# Author: David Young, 2017
"""Connects with a SQLite database for experiment and image
analysis storage.

Attributes:
    db_path: Path to the database.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

from clrbrain import cli
from clrbrain import config

class BlobEstimator(BaseEstimator, ClassifierMixin):

    def __init__(self, filename_base=None, thresholding_size=0):
        self.filename_base = filename_base
        self.thresholding_size = thresholding_size
        self.__name__ = "Blob Estimator"

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        
        settings = config.process_settings
        settings["thresholding_size"] = self.thresholding_size
        self.stat = self.predict(X)
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)
        
        stat = np.zeros(3)
        for i in range(len(X)):
            stat = np.add(stat, cli.process_file(self.filename_base, X[i, 0], X[i, 1]))
        return stat
    
    def score(self, X, y=None):
        sens = float(self.stat[1]) / (self.stat[1] + self.stat[2])
        ppv = float(self.stat[1]) / self.stat[0]
        #return self.y_[sens + ppv]
        return sens + ppv

if __name__ == "__main__":
    print("Starting learn module...")
    from sklearn.utils.estimator_checks import check_estimator
    
    est = BlobEstimator()
    check_estimator(est)  # passes
