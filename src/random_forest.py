from matplotlib.pyplot import axis
from numpy import arange, linspace, mean
from pandas import DataFrame, Series
from pandas.core.reshape.concat import concat
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from utils import plot_confusion_matrix, plot_roc_curves
from variables import RANDOM_SEED, TEST_SET_PERC


def train_test_sets(data_frame, result_label='', result_column=None):
    '''Spliting a whole dataset into trainning and testing dataset

    Args:
    - `DataFrame:data_frame`: Original dataset
    - `str:result_label`: Column to be consider as goal

    Returns:
    - Four elements list as follow: trainning set, testing set, trainning labels and testing labels
    '''
    # Dropping 'Proteina C reativa mg/dL' if target is "has_covid_19" column
    if result_column is None:
        if result_label == 'has_covid_19':
            data_frame = data_frame.drop([
                'proteina_c_reativa_mgdl'
            ], axis='columns', errors='ignore')

        result_column = data_frame.pop(result_label)

    train, test = DataFrame(), DataFrame()
    train_labels, test_labels = Series(), Series()

    for value_class in result_column.unique():
        to_drop = result_column.loc[result_column.values != value_class].index

        df = data_frame.copy().drop(index=to_drop)
        rc = result_column.copy().drop(index=to_drop)

        t_train, t_test, t_train_labels, t_test_labels = train_test_split(
            df, rc,
            stratify=rc,
            test_size=TEST_SET_PERC,
            random_state=RANDOM_SEED
        )

        train = concat([train, t_train], ignore_index=True)
        test = concat([test, t_test], ignore_index=True)
        train_labels = concat([train_labels, t_train_labels], ignore_index=True)
        test_labels = concat([test_labels, t_test_labels], ignore_index=True)

    return train, test, train_labels, test_labels


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
        bootstrap=True,
        random_state=RANDOM_SEED
    )


def best_random_forest():
    # Hyperparameter grid
    param_grid = {
        'n_estimators': linspace(10, 200).astype(int),
        'criterion': ['entropy', 'gini'],
        'max_depth': [None] + list(linspace(3, 20).astype(int)),
        'max_features': ['auto', 'sqrt', None] + list(arange(0.5, 1, 0.1)),
        'max_leaf_nodes': [None] + list(linspace(10, 50, 500).astype(int)),
        'min_samples_split': [2, 5, 10],
        'bootstrap': [True, False]
    }

    # Estimator for use in random search
    estimator = RandomForestClassifier()

    # Create the random search model
    return RandomizedSearchCV(
        estimator, param_grid,
        scoring = 'roc_auc', cv = 3,
        n_iter = 20, n_jobs = -1,
        random_state=RANDOM_SEED
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


def performance_comparison(random_forest, datasets=[]):
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


def evaluate(test_info, train_info, print_tab=0):
    '''Compare machine learning model to baseline performance.

    Source: https://github.com/WillKoehrsen/Machine-Learning-Projects/blob/master/Random%20Forest%20Tutorial.ipynb
    '''
    baseline = {
        'recall': recall_score(test_info['labels'], [1 for _ in range(len(test_info['labels']))]),
        'precision': precision_score(test_info['labels'], [1 for _ in range(len(test_info['labels']))]),
        'f1-score': f1_score(test_info['labels'], [1 for _ in range(len(test_info['labels']))]),
        'roc': .5,
    }

    results = {
        'recall': recall_score(test_info['labels'], test_info['predictions']),
        'precision': precision_score(test_info['labels'], test_info['predictions']),
        'f1-score': f1_score(test_info['labels'], test_info['predictions']),
        'roc': roc_auc_score(test_info['labels'], test_info['probs']),
    }

    train_results = {
        'recall': recall_score(train_info['labels'], train_info['predictions']),
        'precision': precision_score(train_info['labels'], train_info['predictions']),
        'f1-score': f1_score(train_info['labels'], train_info['predictions']),
        'roc': roc_auc_score(train_info['labels'], train_info['probs']),
    }

    for metric in ['recall', 'precision', 'f1-score', 'roc']:
        if print_tab > 0:
            print(print_tab * '\t', f'{metric.capitalize()}')
            print((print_tab + 1) * '\t', f'Baseline: {round(baseline[metric], 2)}')
            print((print_tab + 1) * '\t', f'Test: {round(results[metric], 2)}')
            print((print_tab + 1) * '\t', f'Train: {round(train_results[metric], 2)}')
        else:
            print(f'{round(baseline[metric], 2)};{round(results[metric], 2)};{round(train_results[metric], 2)}', end=';')

    # Calculate false positive rates and true positive rates
    base_false_pos, base_true_pos, _ = roc_curve(test_info['labels'], [1 for _ in range(len(test_info['labels']))])
    model_false_pos, model_true_pos, _ = roc_curve(test_info['labels'], test_info['probs'])

    return base_false_pos, base_true_pos, model_false_pos, model_true_pos


def stats_report(model, train, test, train_labels, test_labels, print_tab=0):
    if print_tab > 0:
        print('*' * 50)
    avg_n_nodes, avg_depth = avg_model_shape(model)

    if print_tab > 0:
        print('Forest Dimension:')
        print(f'\tAverage number of nodes {avg_n_nodes}')
        print(f'\tAverage maximum depth {avg_depth}')
    else:
        print(f"{avg_n_nodes};{avg_depth}", end=';')

    performance = performance_comparison(model, [train, test])

    if print_tab > 0:
        print('\nPerformance Comparison:')
        print(f'\tTrain ROC AUC Score: {roc_auc_score(train_labels, performance[0][1]):.5}')
        print(f'\tTest ROC AUC Score: {roc_auc_score(test_labels, performance[1][1]):.5}')
    else:
        print(f'{roc_auc_score(train_labels, performance[0][1]):.5};{roc_auc_score(test_labels, performance[1][1]):.5}', end=';')

    if print_tab > 0:
        print('\nFeature importances:')
        print(features_importance(model, list(train.columns)))

        print('\nModel Evaluation')

    evaluation = evaluate(
        {
            'labels': test_labels,
            'predictions': performance[1][0],
            'probs': performance[1][1]
        },
        {
            'labels': train_labels,
            'predictions': performance[0][0],
            'probs': performance[0][1]
        },
        print_tab=print_tab + 1
    )
    if print_tab > 0:
        plot_roc_curves(*evaluation)

    cm = confusion_matrix(test_labels, performance[1][0])
    # print(cm)
    if print_tab > 0:
        plot_confusion_matrix(
            cm,
            classes = ['Hospital', 'Em casa'],
            title = 'Matriz de Confusão Acompanhamento de casa'
        )

    if print_tab > 0:
        print(f'\nMean accuracy score:')
        print(f'{accuracy_score(test_labels, performance[1][0]):.3}')

        print('*' * 50)
    else:
        print(f'{accuracy_score(test_labels, performance[1][0]):.3}')
