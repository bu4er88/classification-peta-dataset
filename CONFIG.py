# TO ADD A NEW HEAD JUST FOLLOW THIS TEMPLATE BELLOW:
#
# HEAD_N = {
#     'head_type': 'binary' | 'one-hot-multiple' | 'one-hot-single' | 'one-hot-single',
#      'columns': [list of columns]
# }

HEAD_1 = {
    'head_name': 'gender',
    'head_type': 'one-hot-single',
    'columns': ['personalFemale',
                'personalMale']
}

HEAD_2 = {
    'head_name': 'rest_labels',
    'head_type': 'one-hot-multiple',
    'columns': []
}

HEAD_3 = {
    'head_name': 'age',
    'head_type': 'one-hot-single',
    'columns': ['personalLess15',
                'personalLess30',
                'personalLess45',
                'personalLess60',
                'personalLarger60']
}

HEAD_4 = {
    'head_name': 'accessory',
    'head_type': 'one-hot-single',
    'columns': ['accessoryNothing',
                'accessorySunglasses',
                'accessoryHat',
                'accessoryHairBand',
                'accessoryMuffler',
                'accessoryHeadphone',
                'accessoryKerchief']
}

# HEAD_5 = {
#     'head_name': 'carrying',
#     'head_type': 'one-hot-single',
#     'columns': ['carryingBackpack',
#                 'carryingMessengerBag',
#                 'carryingOther',
#                 'carryingNothing',
#                 'carryingPlasticBags',
#                 'carryingFolder',
#                 'carryingSuitcase',
#                 'carryingLuggageCase',
#                 'carryingShoppingTro',
#                 'carryingBabyBuggy',
#                 'carryingUmbrella']
# }


#################################
# DON'T CHANGE THE CODE BELLOW! #
#################################

import pandas as pd
import numpy as np

# TODO:
#  DONE - 1. eliminate try except!
#  DONE - 2. use module "inspect", __dict__ (globals -> dict, instead of eval)
#  DONE - 3. assert only one head is empty
#  4. add head_names ('gender', 'age'...)

def label_constructor():
    globals_dict = globals()
    labels_set = set(pd.read_pickle('labels.pkl').iloc[:, 0].to_list())
    label_column_dict = {}
    head_type = {}
    head_name = {}
    # filling label_column_dict and head_type
    for glob in globals_dict.keys():
        if glob.startswith('HEAD'):
            label_column_dict[glob] = globals_dict[glob]['columns']
            head_type[glob] = globals_dict[glob]['head_type']
            head_name[glob] = globals_dict[glob]['head_name']
    # check only one head is empty
    empty_values = [value for value in label_column_dict.values() if value == []]
    assert len(empty_values) <= 1, "Only one HEAD can contain an empty list of columns: HEAD = {'columns'=[]}!"
    # fill column list for HEAD with empty list if exists
    for k in label_column_dict.keys():
        if len(label_column_dict[k]) == 0:
            label_column_dict.pop(k)
            label_column_dict[k] = list(labels_set - set(np.sum(np.asarray([*label_column_dict.values()], dtype='object'))))
            break
    # TODO:
    #  1. add assert to check if sets of columns are uncrossed
    #  2. add assert to check if head names are different
    return label_column_dict, head_type, head_name


if __name__ == '__main__':
    label_column_dict, head_type, head_name = label_constructor()
    print('head_name:\n', head_name)
    print('label_column_dict:\n', label_column_dict)
    print('head_type:\n', head_type)


