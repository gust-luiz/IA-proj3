from numpy import array
from sklearn.model_selection import train_test_split

from variables import TEST_SET_PERC

def train_test_sets(data_frame, result_label):
    result_label = array(data_frame.pop(result_label))

    return train_test_split(
        data_frame, result_label,
        stratify=result_label,
        test_size=TEST_SET_PERC,
        random_state=42
    )
