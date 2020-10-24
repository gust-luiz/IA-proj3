from os.path import dirname, realpath, sep


def path_relative_to(known_path, relative_path):
    known_path = dirname(realpath(known_path)).split(sep)
    relative_path = relative_path.split(sep)

    while relative_path[0] == '..':
        relative_path.pop(0)
        known_path.pop()

    return sep.join(known_path + relative_path)
