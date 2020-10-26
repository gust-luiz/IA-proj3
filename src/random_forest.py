from itertools import product

import matplotlib.pyplot as pyplot
from numpy import arange, array, mean
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from variables import RANDOM_SEED, TEST_SET_PERC


def train_test_sets(data_frame, result_label):
    '''Spliting a whole dataset into trainning and testing dataset

    Args:
    - `DataFrame:data_frame`: Original dataset
    - `str:result_label`: Column to be consider as goal

    Returns:
    - Four elements list as follow: trainning set, testing set, trainning labels and testing labels
    '''

    # Dropping 'Proteina C reativa mg/dL' if target is "has_covid19" column
    if result_label == 'has_covid19':
        data_frame = data_frame.drop([
            'Proteina C reativa mg/dL'
        ], axis=1, errors='ignore')

    # Dropping non laboratorial variables
    data_frame = data_frame.drop([
            'Patient addmited to regular ward (1=yes, 0=no)',
            'Patient addmited to semi-intensive unit (1=yes, 0=no)',
            'Patient addmited to intensive care unit (1=yes, 0=no)'
        ], axis=1, errors='ignore')

    result_label = data_frame.pop(result_label)

    return train_test_split(
        data_frame, result_label,
        stratify=result_label,
        test_size=TEST_SET_PERC,
        random_state=RANDOM_SEED
    )


def random_forest(trees=100, criterion='gini', max_depth=None, max_features='auto'):
    '''Initializing a Random Forest based on spefic controls

    Args:
    - `int:trees`: The number of trees in the forest.
    - `str:criterion`: The function to measure the quality of a split.
    - `int:max_depth`: The maximum depth of the tree or None.
    - `object:max_features`: The number of features to consider when looking for the best split,
        it could be an `int`, a `float`, a `str` or None

    Returns:
    - `RandomForestClassifier`
    '''
    return RandomForestClassifier(
        n_estimators=trees,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
        bootstrap=True
    )


def avg_model_shape(random_forest):
    '''Getting average #nodes and maximum depth from a random forest

    Args:
    - `RandomForestClassifier:random_forest`: RandomForest

    Returns:
    - `list:avg_shape` with #node and maximum depth
    '''
    n_nodes = []
    max_depths = []

    for ind_tree in random_forest.estimators_:
        n_nodes.append(ind_tree.tree_.node_count)
        max_depths.append(ind_tree.tree_.max_depth)

    return [int(mean(v)) for v in [n_nodes, max_depths]]


def performance(random_forest, datasets=[]):
    '''Evaluating Random Forest performance over datasets

    Args:
    - `RandomForestClassifier:random_forest`: Trained Random Forest
    - `list:datasets`: Dataset to check

    Returns:
    - `list:performance` with prediction and probability per dataset
    '''
    performance = []

    for dataset in datasets:
        performance.append((
            random_forest.predict(dataset),
            random_forest.predict_proba(dataset)[:, 1]
        ))

    return performance


def features_importance(random_forest, features, top_n=10):
    '''Listing the most important features on a Random Forest

    Args:
    - `RandomForestClassifier:random_forest`: Trained Random Forest
    - `list:features`: Features to consider on the list
    - `int:top_n`: List size

    Returns:
    - `DataFrame:fi_random_forest`
    '''
    return DataFrame({
        'feature': features,
        'importance': random_forest.feature_importances_
    }).sort_values(
        'importance', ascending = False
    ).head(top_n)


def evaluate(test_info, train_info):
    '''Compare machine learning model to baseline performance.

    Source: https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb
    '''
    baseline = {
        'recall': recall_score(test_info['labels'], [1 for _ in range(len(test_info['labels']))]),
        'precision': precision_score(test_info['labels'], [1 for _ in range(len(test_info['labels']))]),
        'roc': .5,
    }

    results = {
        'recall': recall_score(test_info['labels'], test_info['predictions']),
        'precision': precision_score(test_info['labels'], test_info['predictions']),
        'roc': roc_auc_score(test_info['labels'], test_info['probs']),
    }

    train_results = {
        'recall': recall_score(train_info['labels'], train_info['predictions']),
        'precision': precision_score(train_info['labels'], train_info['predictions']),
        'roc': roc_auc_score(train_info['labels'], train_info['probs']),
    }

    for metric in ['recall', 'precision', 'roc']:
        print(f'''{metric.capitalize()}
            Baseline: {round(baseline[metric], 2)}
            Test: {round(results[metric], 2)}
            Train: {round(train_results[metric], 2)}
        ''')

    # Calculate false positive rates and true positive rates
    base_false_pos, base_true_pos, _ = roc_curve(test_info['labels'], [1 for _ in range(len(test_info['labels']))])
    model_false_pos, model_true_pos, _ = roc_curve(test_info['labels'], test_info['probs'])

    return base_false_pos, base_true_pos, model_false_pos, model_true_pos


def plot_roc_curves(base_false_pos, base_true_pos, model_false_pos, model_true_pos):
    ''''Shows ROC curve.
    Source: https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb
    '''
    pyplot.figure(figsize = (8, 6))
    pyplot.rcParams['font.size'] = 16

    # Plot both curves
    pyplot.plot(base_false_pos, base_true_pos, 'b', label = 'baseline')
    pyplot.plot(model_false_pos, model_true_pos, 'r', label = 'model')
    pyplot.legend()
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC Curves')
    plt.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=pyplot.cm.Oranges):
    '''This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    pyplot.figure(figsize = (10, 10))
    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title, size = 24)
    pyplot.colorbar(aspect=4)
    tick_marks = arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=45, size = 14)
    pyplot.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    pyplot.grid(None)
    pyplot.tight_layout()
    pyplot.ylabel('True label', size = 18)
    pyplot.xlabel('Predicted label', size = 18)
