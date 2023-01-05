"""
Utility methods
"""

import pandas as pd
import re
import constants    # Begin custom imports


def remove_prefix_number(input_object):
    return re.sub(r"^([0-9].|[0-9])", '', str(input_object).strip())


def write_csv(output_df, output_file_name, index=False):
    output_df.to_csv(output_file_name, index=index)


def pick_rows_with_categories(original_x_df, original_y_df):
    for index, a_cat in enumerate(constants.label_categories):
        if index == 0:
            selected_rows = original_y_df[constants.label_name] == a_cat
        else:
            selected_rows = selected_rows.combine(original_y_df[constants.label_name] == a_cat, lambda a, b: a != b)

    return original_x_df.loc[selected_rows.tolist()], original_y_df.loc[selected_rows.tolist()]


def feature_selector(feature_df, selected_feature_filename, output_filename):
    temp_data = pd.read_csv(selected_feature_filename, header=None)
    top_features = list(temp_data.iloc[:, 0])
    top_features = [x.replace(u'\u200b', '') for x in top_features]
    feature_df = feature_df[top_features]
    write_csv(feature_df, output_filename, index=True)
