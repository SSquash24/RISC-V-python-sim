"""
read_yaml.py
Contains a single function, read_yaml, which decodes opcodes from a YAML file and returns them as 2 dictionaries.
One dictionary to go from name to opcode, and one for opcode to name
"""

import yaml


def read_yaml(filename : str, ret_CSR = False):
    """
    Reads and decodes file
    :param filename: YAML file name
    :return: (from_opcode, to_opcode) - both dicts
    """
    with open(f"yamls/{filename}", 'r') as file:
        data = yaml.safe_load(file)
        opcodes = data['opcode']
        csrs = {}
        if 'csr' in data.keys():
            csrs = data['csr']
            # change values from int to 32bits
            csrs = {k : bin(v)[2:].rjust(32, '0') for k, v in csrs.items()}

    # calculate inv_opcodes
    inv_opcodes = {}

    def rec_search(tree, pre, dic):
        if type(tree) == list:
            for key, val in tree[-1].items():
                new_dict = dict(dic)
                new_dict[tuple(tree[-2])] = key
                rec_search(val, pre, new_dict)
        else:
            inv_opcodes[tree] = pre + [dic]

    for opcode, info in opcodes.items():


        if type(info[-1]) == dict:
            rec_search(info, [opcode, info[1]], {})
        else:
            inv_opcodes[info[0]] = [opcode, info[1], {}]

    if ret_CSR:
        return opcodes, inv_opcodes, csrs
    return opcodes, inv_opcodes