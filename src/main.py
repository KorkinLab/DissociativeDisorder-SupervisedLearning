"""
Execution begins through this script
"""

import pandas as pd
import timeit
from sklearn.model_selection import StratifiedKFold
from holdout_analysis import holdout_analysis
from sampling import rus_feature_label, smote_feature_label
import constants
from utils import feature_selector, pick_rows_with_categories


def run_holdout_analysis(x_df, y_df, original_y_df, label_name):
    kf10 = StratifiedKFold(n_splits=constants.splits, shuffle=False)
    holdout_analysis = holdout_analysis(kf10)
    n_features_to_select = constants.n_features  # change it to a number after running rfe and finding out which number is the best

    # --- run each of the functions below one by one to understand data flow
    if constants.sampling_type == 'rus':
        holdout_analysis.split_undersampled_data(x_df, y_df, patient_control_num=constants.test_size)
    elif constants.sampling_type == 'smote':
        holdout_analysis.split_smote_data(x_df, y_df, original_y_df, patient_control_num=constants.test_size)
    holdout_analysis.training(n_features_to_select, label_name, best_model=False)
    holdout_analysis.rfe(label_name, n_features_to_select=n_features_to_select)  # This line will take few mins!
    holdout_analysis.training(n_features_to_select, label_name, best_model=True)
    holdout_analysis.final(n_features_to_select, label_name)


def get_sampled_data(sampling_type):
    x_df = pd.read_csv(f'{constants.generated_path}x_{sampling_type}.csv', header=0, index_col=0)
    y_df = pd.read_csv(f'{constants.generated_path}y_{sampling_type}.csv', header=0, index_col=0)
    return x_df, y_df


if __name__ == "__main__":
    original_x_df = pd.read_csv(constants.numeric_file, header=0, index_col=0)
    original_y_df = pd.read_csv(constants.label_file, header=0, index_col=0)

    # pick two categories from multicateogry label, ex) group category label
    original_x_df, original_y_df = pick_rows_with_categories(original_x_df, original_y_df)

    # oversampling
    if constants.sampling_type == 'smote':
        smote_feature_label(original_x_df, original_y_df[constants.label_name],
                            title_x=f'{constants.generated_path}x_{constants.sampling_type}.csv',
                            title_y=f'{constants.generated_path}y_{constants.sampling_type}.csv')
    # undersampling
    elif constants.sampling_type == 'rus':
        rus_feature_label(original_x_df, original_y_df[constants.label_name],
                          title_x=f'{constants.generated_path}x_{constants.sampling_type}.csv',
                          title_y=f'{constants.generated_path}y_{constants.sampling_type}.csv')

    x_df, y_df = get_sampled_data(constants.sampling_type)
    assert x_df.shape[0] == y_df.shape[0]  # sanity check to make sure we have corresponding x and y data

    # Run analysis. Choose one of two below
    start_time = timeit.default_timer()

    run_holdout_analysis(x_df, y_df, original_y_df, constants.label_name)
    # run_nested_10fold_cv(x_df, y_df, constants.label_name)

    # feature_selector(original_x_df, f'{constants.asset_path}selected_feature_list.csv',
    #                  f'{constants.generated_path}/selected_features.csv')
    end_time = timeit.default_timer()
    training_time = (end_time - start_time)
    print('Total ML time - %.2fm' % (training_time / 60.0))

