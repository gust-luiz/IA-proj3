from numpy import array, mean
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from variables import TEST_SET_PERC, RANDOM_SEED

def train_test_sets(data_frame, result_label):
    result_label = array(data_frame.pop(result_label))

    return train_test_split(
        data_frame, result_label,
        stratify=result_label,
        test_size=TEST_SET_PERC,
        random_state=RANDOM_SEED
    )


def random_forest(trees=100, criterion='gini', max_depth=None, max_features='auto'):
    return RandomForestClassifier(
        n_estimators=trees,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
    )


def avg_model_shape(model):
    n_nodes = []
    max_depths = []

    for ind_tree in model.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    return [int(mean(v)) for v in [n_nodes, max_depths]]


def performance(model, datasets=[]):
    performance = []

    for dataset in datasets:
        performance.append((
            model.predict(dataset),
            model.predict_proba(dataset)[:, 1]
        ))

    return performance


def features_importance(model, features, top_n=10):
    fi_model = DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(
        'importance', ascending = False
    )

    return fi_model.head(top_n)
