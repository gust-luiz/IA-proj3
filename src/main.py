from pandas import read_csv

from data_cleaning import clear_dataset
from random_forest import train_test_sets
from utils import path_relative_to

data_frame = read_csv(path_relative_to(__file__, '../ref/raw_covid19_dataset.csv'))
data_frame = clear_dataset(data_frame)
train, test, train_labels, test_labels = train_test_sets(data_frame, 'has_covid19')

print(data_frame.head())
print()
print(data_frame.shape)
print(train.shape)
print(test.shape)
