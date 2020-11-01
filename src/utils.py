from itertools import product
from os.path import dirname, realpath, sep
from re import sub

import matplotlib.pyplot as pyplot
from numpy import arange, newaxis


def path_relative_to(known_path, relative_path):
    '''Considering an known absolute path as start point and a relative path,
    a absolute path is put together

    Args:
    - `str:known_path`: Reference absolute path
    - `str:relative_path`: Relative path from `known_path`

    Returns:
    - Absolute path
    '''
    known_path = dirname(realpath(known_path)).split(sep)
    relative_path = relative_path.split(sep)

    while relative_path[0] == '..':
        relative_path.pop(0)
        known_path.pop()

    return sep.join(known_path + relative_path)


def to_snake_case(string):
    '''Converting a string to snake case by:

    - Removing '(explanation/description)'
    - Removing non-alphanumeric chars
    - Replacing multiples spaces to just one

    Source: https://www.kaggle.com/julianosilva23/covid-19-cases-random-forest-1-task#Used-Libs

    Args:
    - `srt:string`: String to convert

    Returns:
    - Converted string
    '''
    string = sub('\(.*\)', '', string)
    string = sub('-', ' ', string)
    string = sub('[^a-z A-z 0-9]', '', string)
    string = sub(' +', ' ', string)

    return string.strip().replace(' ', '_').lower() #apply snake_case pattern


def remove_non_laboratorial(data_frame):
    dropped_data = {}
    non_laboratorial_labels = [
        'age_group',
        'patient_addmited_to_regular_ward',
        'patient_addmited_to_semi_intensive_unit',
        'patient_addmited_to_intensive_care_unit',
    ]

    for label in non_laboratorial_labels:
        dropped_data[label] = data_frame.pop(label)

    return data_frame, dropped_data


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
    pyplot.show()


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=pyplot.cm.Oranges):
    '''This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis='columns')[:, newaxis]
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
    pyplot.show()
