"""
Random Forest implementation
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import constants


def get_random_forest_classifier():
    return RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=constants.random_state)


def rf_stat(x_df, y_df, cur_kf):
    x_resampled = x_df.to_numpy()
    y_resampled = y_df[constants.label_name].to_numpy()

    random_forest_default = get_random_forest_classifier()
    scores = cross_val_score(random_forest_default, x_resampled, y_resampled, cv=cur_kf, scoring='f1_weighted')
    return scores.mean(), scores.std()