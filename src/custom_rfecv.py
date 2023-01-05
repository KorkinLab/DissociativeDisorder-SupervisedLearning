"""
Implementation of RFE and CV
"""

from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import _safe_split
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.utils._joblib import Parallel, delayed, effective_n_jobs
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _score
from sklearn.metrics import check_scoring
from sklearn.base import MetaEstimatorMixin
from sklearn.feature_selection import RFE
import numpy as np

class custom_rfecv(RFE, MetaEstimatorMixin):
    # modified RFECV to add n_features_to_select parameter to the class.
    def __init__(self, estimator, step=1, min_features_to_select=1, n_features_to_select=0, cv='warn',
                 scoring='f1_weighted', verbose=0, n_jobs=None):
        self.estimator = estimator
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.min_features_to_select = min_features_to_select
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y, groups=None):

        X, y = check_X_y(X, y, "csr", ensure_min_features=2)

        # Initialization
        cv = check_cv(self.cv, y, is_classifier(self.estimator))
        scorer = check_scoring(self.estimator, scoring=self.scoring)
        n_features = X.shape[1]

        if 0.0 < self.step < 1.0:
            step = int(max(1, self.step * n_features))
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        # Build an RFE object, which will evaluate and score each possible
        # feature count, down to self.min_features_to_select
        rfe = RFE(estimator=self.estimator,
                  n_features_to_select=self.min_features_to_select,
                  step=self.step, verbose=self.verbose)

        # Determine the number of subsets of features by fitting across
        # the train folds and choosing the "features_to_select" parameter
        # that gives the least averaged error across all folds.

        # Note that joblib raises a non-picklable error for bound methods
        # even if n_jobs is set to 1 with the default multiprocessing
        # backend.
        # This branching is done so that to
        # make sure that user code that sets n_jobs to 1
        # and provides bound methods as scorers is not broken with the
        # addition of n_jobs parameter in version 0.18.

        if effective_n_jobs(self.n_jobs) == 1:
            parallel, func = list, _rfe_single_fit
        else:
            parallel = Parallel(n_jobs=self.n_jobs)
            func = delayed(_rfe_single_fit)

        # rfe_models
        scores = parallel(
            func(rfe, self.estimator, X, y, train, test, scorer)
            for train, test in cv.split(X, y, groups))

        # scores = []
        # rankings = []
        # for rfe_model in rfe_models:
        #     scores.append(rfe_model.scores_)
        #     rankings.append(rfe_model.ranking_.tolist())
        scores = np.sum(scores, axis=0)
        # self.ranking_jj = np.mean(rankings, axis=0)

        scores_rev = scores[::-1]
        argmax_idx = len(scores) - np.argmax(scores_rev) - 1
        if self.n_features_to_select > 0:
            print('n_features_to_select: ' + str(self.n_features_to_select))
        else:
            self.n_features_to_select = max(
            n_features - (argmax_idx * step),
            self.min_features_to_select)

        # Re-execute an elimination with best_k over the whole set
        rfe = RFE(estimator=self.estimator,
                  n_features_to_select=self.n_features_to_select, step=self.step,
                  verbose=self.verbose)

        rfe.fit(X, y)

        # Set final attributes
        self.support_ = rfe.support_
        self.n_features_ = rfe.n_features_
        self.ranking_ = rfe.ranking_
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self.transform(X), y)

        # Fixing a normalization error, n is equal to get_n_splits(X, y) - 1
        # here, the scores are normalized by get_n_splits(X, y)
        self.grid_scores_ = scores[::-1] / cv.get_n_splits(X, y, groups)
        return self


def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer):
    """
    Return the score for a fit across one fold.
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    rfe_model =rfe._fit(
        X_train, y_train, lambda estimator, features:
        _score(estimator, X_test[:, features], y_test, scorer))
    return rfe_model.scores_
