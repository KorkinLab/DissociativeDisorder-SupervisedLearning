"""
Methods to sample data
"""

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import constants    # Begin custom imports
from utils import write_csv


def save_sampled_dataset(df, index_list, filename):
    """
    Method to save sampled data as CSV
    :param df:
    :param index_list:
    :param filename:
    :return:
    """
    # smote_index_total = len(df) - len(index_list)
    # for index in range(smote_index_total):
    #     index_list.append("SMOTE-%02d" % (index + 1))
    index_dict = dict(zip(df.index.tolist(), index_list))
    df = df.rename(index=index_dict)
    write_csv(df, filename, index=True)


def smote_feature_label(x, y, title_x, title_y):
    """
    Method to augment data with SMOTE
    :param x:
    :param y:
    :param title_x:
    :param title_y:
    :return:
    """
    x_resampled, y_resampled = SMOTE(sampling_strategy='not majority',
                                     random_state=constants.random_state).fit_resample(x, y)
    x_index_list = list(x.index)
    save_sampled_dataset(x_resampled, x_index_list, title_x)

    y_index_list = list(y.index)
    save_sampled_dataset(y_resampled, y_index_list, title_y)


def rus_feature_label(x_df, y_df, title_x, title_y):
    """
    Method to subset data with random under sampling
    :param x_df:
    :param y_df:
    :param title_x:
    :param title_y:
    :return:
    """
    rus = RandomUnderSampler(random_state=constants.random_state, replacement=False)
    x_resampled, y_resampled = rus.fit_resample(x_df, y_df)

    x_index_list = list(x_df.index)
    save_sampled_dataset(x_resampled, x_index_list, title_x)

    y_index_list = list(y_df.index)
    save_sampled_dataset(y_resampled, y_index_list, title_y)
