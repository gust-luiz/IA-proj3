from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from variables import TEST_SET_PERC

def train_test_sets(data_frame, result_label):
    result_label = array(data_frame.pop(result_label))

    return train_test_split(
        data_frame, result_label,
        stratify=result_label,
        test_size=TEST_SET_PERC,
        random_state=42
    )


def get_random_forest(
    trees=100,
    criterion='gini',
    max_depth=None,
    max_features='auto',
):
    return RandomForestClassifier(
        n_estimators=trees,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
    )
