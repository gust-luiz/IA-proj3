from pandas import read_csv
from data_cleaning import clear_dataset, fill_NAN_fields_zero, drop_negative_excess_covid, drop_excess_data
from random_forest import best_random_forest, random_forest, stats_report, train_test_sets
from utils import path_relative_to, remove_non_laboratorial, create_went_home_column
from variables import RF_CRITERION, RF_MAX_DEPTH, RF_MAX_FEATURES, RF_TREES

# All available data minimally cleaned
data_frame = clear_dataset(read_csv(path_relative_to(__file__, '../ref/raw_covid19_dataset.csv')))

data_frame = create_went_home_column(data_frame)

data_frame, non_laboratorial = remove_non_laboratorial(data_frame)

what_analyse = [
    # ('has_covid_19', None, 'COVID-19'),
    # ('', non_laboratorial['patient_addmited_to_regular_ward'], 'Patient addmited to regular ward'),
    # ('', non_laboratorial['patient_addmited_to_semi_intensive_unit'], 'Patient addmited to semi-intensive unit'),
    # ('', non_laboratorial['patient_addmited_to_intensive_care_unit'], 'Patient addmited to intensive care unit'),
    ('', non_laboratorial['patient_went_to_home'], 'Patient went to home'),
]

for label, serie, title in what_analyse:
    if title == 'COVID-19':
        data_frame = drop_negative_excess_covid(data_frame)
    else:
        data_frame, serie = drop_excess_data(data_frame, serie)

    if serie is not None:
        print(serie.value_counts())

    print('Data shape', data_frame.shape)

    train, test, train_labels, test_labels = train_test_sets(data_frame, result_label=label, result_column=serie)
    train, test = fill_NAN_fields_zero(train), fill_NAN_fields_zero(test)

    print('+' * 20)
    print('Testing:', title)
    print('+' * 20)

    print('\t\tParameterized RandomForest')
    model = random_forest(RF_TREES, RF_CRITERION, RF_MAX_DEPTH, RF_MAX_FEATURES)
    print('\tModel Configuration:')
    print('\t', model)

    model.fit(train, train_labels)

    stats_report(model, train, test, train_labels, test_labels, 1)

    print('\n' * 2)

    print('Automatically searched Best Model')
    model = best_random_forest()
    model.fit(train, train_labels)

    print('\tBest params:')
    print('\t\t', model.best_params_)
    model = model.best_estimator_

    stats_report(model, train, test, train_labels, test_labels, 1)
