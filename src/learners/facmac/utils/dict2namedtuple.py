# from https://github.com/oxwhirl/facmac

from collections import namedtuple


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)
