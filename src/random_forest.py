from itertools import product

import matplotlib.pyplot as pyplot
from numpy import arange, array, mean
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from variables import RANDOM_SEED, TEST_SET_PERC


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


# https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
def evaluate(test_info, train_info):
    """Compare machine learning model to baseline performance.
    Computes statistics and shows ROC curve."""
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


# https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
def plot_roc_curves(base_false_pos, base_true_pos, model_false_pos, model_true_pos):
    pyplot.figure(figsize = (8, 6))
    pyplot.rcParams['font.size'] = 16

    # Plot both curves
    pyplot.plot(base_false_pos, base_true_pos, 'b', label = 'baseline')
    pyplot.plot(model_false_pos, model_true_pos, 'r', label = 'model')
    pyplot.legend()
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('ROC Curves')


# https://towardsdatascience.com/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=pyplot.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
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
