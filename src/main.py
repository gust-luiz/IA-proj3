from pandas import read_csv
from sklearn.metrics import roc_auc_score

from data_cleaning import clear_dataset
from random_forest import avg_model_shape, features_importance, performance, random_forest, train_test_sets
from utils import path_relative_to
from variables import RF_CRITERION, RF_MAX_DEPTH, RF_MAX_FEATURES, RF_TREES

data_frame = read_csv(path_relative_to(__file__, '../ref/raw_covid19_dataset.csv'))
data_frame = clear_dataset(data_frame)
train, test, train_labels, test_labels = train_test_sets(data_frame, 'has_covid19')

print(data_frame.head())
print()
print(data_frame.shape)
print(train.shape)
print(test.shape)

model = random_forest(RF_TREES, RF_CRITERION, RF_MAX_DEPTH, RF_MAX_FEATURES)
print(model)

print('Trainning data:')
model.fit(train, train_labels)

avg_n_nodes, avg_depth = avg_model_shape(model)

print(f'\tAverage number of nodes {avg_n_nodes}')
print(f'\tAverage maximum depth {avg_depth}')

performance = performance(model, [train, test])

print(f'\tTrain ROC AUC Score: {roc_auc_score(train_labels, performance[0][1])}')
print(f'\tTest ROC AUC Score: {roc_auc_score(test_labels, performance[1][1])}')

print('Feature importances:')
# Features for feature importances
print(features_importance(model, list(train.columns)))
