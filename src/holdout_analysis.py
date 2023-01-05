"""
Implementation to split and holdout data
"""

import pickle
import pandas as pd
from sklearn.metrics import f1_score
from random_forest import rf_stat, get_random_forest_classifier
from recursive_feature_elim import recursive_feature_elim_cv
import constants
from utils import write_csv
from visualization import plot_importance_graph


class holdout_analysis(object):
    def __init__(self, cur_kf):
        self.cur_kf = cur_kf

    def split_smote_data(self, smote_x_df, smote_y_df, original_y_df, patient_control_num=10):
        # From the original samples, randomly select 10 control and 10 patient
        original_y_df = original_y_df.reset_index()
        smote_y_df = smote_y_df.reset_index()

        cat_y_series = original_y_df.groupby(constants.label_name).apply(
            lambda s: s.sample(patient_control_num, random_state=constants.random_state).index.tolist())

        test_index_list = []
        # if constants.label_name == 'suicide':
        #     test_index_list = cat_y_series.loc['1.No'] + cat_y_series.loc['2.Yes']
        # else:
        for label_i, label_category in enumerate(constants.label_categories):
            test_index_list = test_index_list + cat_y_series.loc[constants.label_categories[label_i]]

        all_index_list = smote_y_df.index.tolist()
        train_index_list = [train_index for train_index in all_index_list if train_index not in test_index_list]

        self.save_split_data(smote_x_df, smote_y_df, train_index_list, test_index_list)


    def split_undersampled_data(self, x_df, y_df, patient_control_num=5):
        y_df = y_df.reset_index()

        cat_y_series = y_df.groupby(constants.label_name).apply(
            lambda s: s.sample(patient_control_num, random_state=constants.random_state).index.tolist())

        test_index_list = cat_y_series.loc[constants.label_categories[0]] + cat_y_series.loc[constants.label_categories[1]]

        all_index_list = y_df.index.tolist()
        train_index_list = [train_index for train_index in all_index_list if train_index not in test_index_list]

        self.save_split_data(x_df, y_df, train_index_list, test_index_list)


    def save_split_data(self, smote_x_df, smote_y_df, train_index_list, test_index_list):
        test_x_df = smote_x_df.iloc[test_index_list]
        test_y_df = smote_y_df.iloc[test_index_list]
        train_x_df = smote_x_df.iloc[train_index_list]
        train_y_df = smote_y_df.iloc[train_index_list]
        assert test_x_df.shape[1] == train_x_df.shape[1]
        assert test_y_df.shape[1] == train_y_df.shape[1]
        assert test_x_df.shape[0] == test_y_df.shape[0]
        assert train_x_df.shape[0] == train_y_df.shape[0]

        # save the holdout data to a file
        write_csv(test_x_df, f'{constants.generated_path}/test_x.csv', index=True)
        write_csv(test_y_df, f'{constants.generated_path}/test_y.csv')

        # Save the remaining (subtracting the holdout) samples to a file
        write_csv(train_x_df, f'{constants.generated_path}/train_x.csv', index=True)
        write_csv(train_y_df, f'{constants.generated_path}/train_y.csv')


    def rfe(self, label_name, n_features_to_select='all'):
        # load train data
        train_x_df = pd.read_csv(f'{constants.generated_path}/train_x.csv', header=0, index_col=0)
        train_y_df = pd.read_csv(f'{constants.generated_path}/train_y.csv', header=0, index_col=0)
        assert train_x_df.index.equals(train_y_df.index)

        clf = get_random_forest_classifier()
        # do 10FCV RFE
        recursive_feature_elim_cv(clf, train_x_df, train_y_df[label_name], cur_kf=self.cur_kf,
                                  title_prefix=f'k10_{label_name}', n_features_to_select=n_features_to_select)

    def training(self, n_features_to_select, label_name, best_model=False):
        # load train data
        train_x_df = pd.read_csv(f'{constants.generated_path}/train_x.csv',
                                 header=0, index_col=0)
        train_y_df = pd.read_csv(f'{constants.generated_path}/train_y.csv',
                                 header=0, index_col=0)

        if not best_model:
            print('Initial training...')
            print(rf_stat(train_x_df, train_y_df, self.cur_kf))
        else:
            # Select X generalizable features
            temp_data = pd.read_csv(
                f'{constants.generated_path}k10_{label_name}_{str(n_features_to_select)}_top_features.csv',
                header=None)
            top_features = list(temp_data.iloc[:, 0])
            train_x_df = train_x_df[top_features]

            print('After RFE and retraining...')
            print(rf_stat(train_x_df, train_y_df, self.cur_kf))

            # Save the model to file
            clf = get_random_forest_classifier()
            clf.fit(train_x_df, train_y_df[label_name])
            filename = f'{constants.generated_path}k10_{str(n_features_to_select)}_{label_name}_rf.sav'
            pickle.dump(clf, open(filename, 'wb'))

    def final(self, n_features_to_select, label_name):
        # Finally use holdout data
        test_x_df = pd.read_csv(f'{constants.generated_path}/test_x.csv', header=0, index_col=0)
        test_y_df = pd.read_csv(f'{constants.generated_path}/test_y.csv', header=0, index_col=0)

        # Select same X features in holdout data
        temp_data = pd.read_csv(
            f'{constants.generated_path}k10_{label_name}_{str(n_features_to_select)}_top_features.csv',
            header=None)
        top_features = list(temp_data.iloc[:, 0])
        test_x_df = test_x_df[top_features]

        filename = f'{constants.generated_path}k10_{str(n_features_to_select)}_{label_name}_rf.sav'
        loaded_model = pickle.load(open(filename, 'rb'))

        # Pass data to saved model for prediction
        # print('')
        # print(test_x_df.columns.values.tolist())
        # print('')
        plot_importance_graph(loaded_model.feature_importances_, test_x_df.columns.values.tolist(), title=constants.label_name)
        prediced_test_y_df = loaded_model.predict(test_x_df)
        f1 = f1_score(test_y_df[label_name].tolist(), prediced_test_y_df, average='weighted')
        print('Test/holdout score')
        print(f1)