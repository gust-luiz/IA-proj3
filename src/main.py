from pandas import read_csv

from data_cleaning import drop_more_blank_columns
from utils import path_relative_to


data_frame = read_csv(path_relative_to(__file__, '../ref/raw_covid19_dataset.csv'))
data_frame = drop_more_blank_columns(data_frame)

print(data_frame.head())
