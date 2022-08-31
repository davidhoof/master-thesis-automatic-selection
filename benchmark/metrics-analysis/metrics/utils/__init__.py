from math import isclose
from numbers import Number

import torch


def get_children(model: torch.nn.Module):
    """
    Gets all children from given torch module
    :param model:
    :return:
    """
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children


def dicts_almost_equal(dict1, dict2, rel_tol=1e-8):
    """
    If dictionary value is a number, then check that the numbers are almost equal, otherwise check if values are exactly equal
    Note: does not currently try converting strings to digits and comparing them. Does not care about ordering of keys in dictionaries
    Just returns true or false.
    Source: https://stackoverflow.com/questions/23549419/assert-that-two-dictionaries-are-almost-equal
    :param dict1 first dict
    :param dict2 second dict
    :param rel_tol precision
    """
    if len(dict1) != len(dict2):
        return False
    # Loop through each item in the first dict and compare it to the second dict
    for key, item in dict1.items():
        # If it is a nested dictionary, need to call the function again
        if isinstance(item, dict):
            # If the nested dictionaries are not almost equal, return False
            if not dicts_almost_equal(dict1[key], dict2[key], rel_tol=rel_tol):
                return False
        # If it's not a dictionary, then continue comparing
        # Put in else statement or else the nested dictionary will get compared twice and
        # On the second time will check for exactly equal and will fail
        else:
            # If the value is a number, check if they are approximately equal
            if isinstance(item, Number):
                # if not abs(dict1[key] - dict2[key]) <= rel_tol:
                # https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
                if not isclose(dict1[key], dict2[key], rel_tol=rel_tol):
                    return False
            else:
                if not (dict1[key] == dict2[key]):
                    return False
    return True
