from pandas import read_csv
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from data_cleaning import clear_dataset, fill_NAN_fields_mean, fill_NAN_fields_zero
from random_forest import (avg_model_shape, evaluate, features_importance, performance_comparison, random_forest,
                           train_test_sets)
from utils import path_relative_to, plot_confusion_matrix, plot_roc_curves, remove_non_laboratorial
from variables import RF_CRITERION, RF_MAX_DEPTH, RF_MAX_FEATURES, RF_TREES

# All available data minimally cleaned
data_frame = clear_dataset(read_csv(path_relative_to(__file__, '../ref/raw_covid19_dataset.csv')))

data_frame, non_laboratorial = remove_non_laboratorial(data_frame)

what_analyse = [
    ('has_covid_19', None),
    # ('', non_laboratorial['patient_addmited_to_regular_ward']),
    # ('', non_laboratorial['patient_addmited_to_semi_intensive_unit']),
    # ('', non_laboratorial['patient_addmited_to_intensive_care_unit']),
]

for label, dataset in what_analyse:
    train, test, train_labels, test_labels = train_test_sets(data_frame, result_label=label, result_column=dataset)

    train, test = fill_NAN_fields_zero(train), fill_NAN_fields_zero(test)

    model = random_forest(RF_TREES, RF_CRITERION, RF_MAX_DEPTH, RF_MAX_FEATURES)
    print(model)

    print('\nTrainning data:')
    model.fit(train, train_labels)

    avg_n_nodes, avg_depth = avg_model_shape(model)

    print(f'\tAverage number of nodes {avg_n_nodes}')
    print(f'\tAverage maximum depth {avg_depth}')

    performance = performance_comparison(model, [train, test])

    print(f'\tTrain ROC AUC Score: {roc_auc_score(train_labels, performance[0][1])}')
    print(f'\tTest ROC AUC Score: {roc_auc_score(test_labels, performance[1][1])}')

    print('\nFeature importances:')
    # Features for feature importances
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
    )

    plot_roc_curves(*evaluation)

    plot_confusion_matrix(
        confusion_matrix(test_labels, performance[1][0]),
        classes = ['Poor Health', 'Good Health'],
        title = 'Health Confusion Matrix'
    )

    accuracy = accuracy_score(test_labels, performance[1][0])
    print(f'\nMean accuracy score: {accuracy:.3}')
