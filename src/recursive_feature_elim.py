"""
Recursive Feature Elimination
"""

import matplotlib.pyplot as plt
import numpy as np
from custom_rfecv import custom_rfecv
import constants


def recursive_feature_elim_cv(clf, X, y, cur_kf, n_features_to_select=None, title_prefix=''):
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    if n_features_to_select is None or n_features_to_select is 'all':
        title_prefix = title_prefix + '_all'
        rfecv = custom_rfecv(estimator=clf, step=1, cv=cur_kf, scoring='f1_weighted', n_jobs=1)
    else:
        print('else')
        title_prefix = title_prefix + '_' + str(n_features_to_select)
        rfecv = custom_rfecv(estimator=clf, step=1, cv=cur_kf, n_features_to_select=n_features_to_select,
                            scoring='f1_weighted', n_jobs=1)

    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    # print("feature importance:")
    # print(rfecv.estimator.feature_importances_)

    feature_ranking = rfecv.ranking_
    for i, val in enumerate(rfecv.grid_scores_):
        print(i+1, val)
    x_columns = np.asarray(X.columns.tolist())
    top_index_list = list(np.where(feature_ranking == 1)[0])
    top_feature_list = x_columns[top_index_list]
    np.savetxt(f'{constants.generated_path}{title_prefix}_top_features.csv', top_feature_list, delimiter=',', fmt="%s")


    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (f1-weighted)")
    plt.title(title_prefix + '_top_features')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.savefig(f'{constants.generated_path}{title_prefix}_top_features.png')

