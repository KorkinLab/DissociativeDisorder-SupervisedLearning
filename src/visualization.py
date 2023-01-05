"""
Method to visualize the feature importance after training
"""

import collections
import matplotlib.pyplot as plt
import constants    # Begin custom imports


def plot_importance_graph(feature_importance, feature_column_names, title='', xerr=None):
    """
    This method plots the importance of top 20 features as a bar graph
    :param feature_importance:
    :param feature_column_names:
    :param title:
    :param xerr:
    :return:
    """
    main_fig, ax = plt.subplots(1, 1)
    feature_importance_dict = {}
    importance_feature_max = 20
    for index, feature_importance in enumerate(feature_importance):
        feature_importance_dict[index] = feature_importance
    sorted_feature_items = sorted(feature_importance_dict.items(), key=lambda kv: kv[1], reverse=True)
    sorted_feature_dict = collections.OrderedDict(sorted_feature_items[:importance_feature_max])
    sorted_feature_index = list(sorted_feature_dict.keys())
    sorted_feature_index.reverse()
    sorted_column_names = [feature_column_names[i] for i in sorted_feature_index]

    ax.barh(range(len(sorted_column_names)), feature_importance[sorted_feature_index], xerr=xerr,
            align='center', linewidth=3)

    plt.yticks(range(len(sorted_column_names)), sorted_column_names, fontsize='large')
    plt.xlabel('Feature Importance', fontsize='large')
    # plt.ylabel(f'Features', fontsize='large')
    plt.title(f'Feature Ranking for {title}', fontsize='x-large')
    plt.tight_layout()
    [i[1].set_linewidth(2) for i in ax.spines.items()]  # make the border box thicker

    plt.savefig(f'{constants.generated_path}{title}_feature_ranking.png')
    plt.close()
    # plt.show()
    return sorted_column_names
