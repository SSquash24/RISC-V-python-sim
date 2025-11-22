"""
read_yaml.py
Contains a single function, read_yaml, which decodes opcodes from a YAML file and returns them as 2 dictionaries.
One dictionary to go from name to opcode, and one for opcode to name
"""

import yaml

"""
Explanation of this project's YAML structure:
There is 1 YAML file per module, which tells the module the presense, and encoding of each instruction
    This is found under the 'opcode' name in the file, which returns a dictionary, keys of which are opcode values
    At the top level it is a list. The 1st element is the opcode name, the 2nd is the instruction type.
    The 3rd and 4th (or more consistently -2 and -1th) elements, if they exist, further subdivide the opcode
    The -1th element is another dictionary split by values, and the -2nd element describes what bits in the instruction
    to use for the dictionary key.
    This continues recursively until the leaf, which is the instruction name
    
The YAML file may also include a 'csr' section, which is a dict mapping CSR register names to their address

Finally, there may be a 'pseudo' section, which translates pseudoinstructions into their underlying instruction, e.g.:
    PSEUDO (No args): [(REAL_INSTR), arg1, arg2, ...]
if an arg is of value 'INP:{number}' then this should be replaced with the corresponding argument given to the pseudo
aka 'INP:1' is replaced with the 1st argument, 'INP:2' the 2nd etc.
"""


def read_yaml(filename : str, ret_CSR = False, ret_Pseudos = False):
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
            csrs = {k : bin(v)[2:].rjust(12, '0') for k, v in csrs.items()}

        pseudos = {}
        if 'pseudos' in data.keys():
            pseudos = data['pseudos']

    # calculate inv_opcodes
    inv_opcodes = {}

    def rec_search(tree, pre, dic):
        if type(tree) == list and isinstance(tree[-1], dict):
            if len(tree) == 3:
                pre = [pre[0],tree[0]]
            for key, val in tree[-1].items():
                new_dict = dict(dic)
                new_dict[tuple(tree[-2])] = key
                rec_search(val, pre, new_dict)
        elif type(tree) == str:
            inv_opcodes[tree] = pre + [dic]
        elif type(tree) == list:
            inv_opcodes[tree[-1]] = [pre[0], tree[0]] + [dic]
        else:
            raise ValueError(f"Cannot parse yaml file: {filename}")

    for opcode, info in opcodes.items():


        if type(info[-1]) == dict:
            rec_search(info, [opcode, info[1]], {})
        else:
            inv_opcodes[info[0]] = [opcode, info[1], {}]

    if ret_CSR:
        if ret_Pseudos:
            return opcodes, inv_opcodes, csrs, pseudos
        else:
            return opcodes, inv_opcodes, csrs
    elif ret_Pseudos:
        return opcodes, inv_opcodes, pseudos
    return opcodes, inv_opcodes