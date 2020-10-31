from os.path import dirname, realpath, sep
from re import sub


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
    string = sub('[^a-z A-z 0-9]', '', string)
    string = sub(' +', ' ', string)

    return string.strip().replace(' ', '_').lower() #apply snake_case pattern
