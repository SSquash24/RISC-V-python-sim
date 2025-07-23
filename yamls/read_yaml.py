"""
read_yaml.py
Contains a single function, read_yaml, which decodes opcodes from a YAML file and returns them as 2 dictionaries.
One dictionary to go from name to opcode, and one for opcode to name
"""

import yaml


def read_yaml(filename : str):
    """
    Reads and decodes file
    :param filename: YAML file name
    :return: (from_opcode, to_opcode) - both dicts
    """
    with open(f"yamls/{filename}", 'r') as file:
        data = yaml.safe_load(file)
        opcodes = data['opcode']

    # calculate inv_opcodes
    inv_opcodes = {}
    for opcode, info in opcodes.items():
        match info[1]:  # instr_type
            case 'R':
                # dependent on funct3 and funct7
                for funct3, data3 in info[2].items():
                    for funct7, instr in data3.items():
                        inv_opcodes[instr] = (opcode, info[1], funct3, funct7)
            case 'I' | 'S' | 'B':
                # dependent on funct3
                for funct3, data3 in info[2].items():
                    if type(data3) is str:
                        inv_opcodes[data3] = (opcode, info[1], funct3)
                    else:
                        # a few instrs have limits on imm
                        for imm, instr in data3[1].items():
                            inv_opcodes[instr] = (opcode, info[1], funct3, data3[0], imm)

            case 'U' | 'J':
                inv_opcodes[info[0]] = (opcode, info[1])
    return opcodes, inv_opcodes