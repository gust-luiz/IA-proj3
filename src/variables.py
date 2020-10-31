'''Should be between 0.0 and 1.0 and represent the proportion of NaN by columns
    on the dataset to it be dropped.'''
MISS_DATA_TO_DROP_PERC = .90


'''If float, should be between 0.0 and 1.0 and represent the proportion of the dataset
    to include in the test split. If int, represents the absolute number of test samples.'''
TEST_SET_PERC = 1/3


'''The number of trees in the forest.'''
RF_TREES = 200

'''The function to measure the quality of a split. Supported criteria are 'gini'
    for the Gini impurity and 'entropy' for the information gain.'''
RF_CRITERION = 'entropy'

'''The maximum depth of the tree. If None, then nodes are expanded until all leaves
    are pure or until all leaves contain less than min_samples_split samples.'''
RF_MAX_DEPTH = 9 #None

'''The number of features to consider when looking for the best split:
    - If int, then consider max_features features at each split.
    - If float, then max_features is a fraction and int(max_features * n_features) features are considered at each split.
    - If 'auto', then max_features=sqrt(n_features).
    - If 'sqrt', then max_features=sqrt(n_features) (same as 'auto').
    - If 'log2', then max_features=log2(n_features).
    - If None, then max_features=n_features.'''
RF_MAX_FEATURES = 'auto'


RANDOM_SEED = 42
