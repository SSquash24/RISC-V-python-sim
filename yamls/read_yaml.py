# read_yaml.py
# Contains data read from yaml files:
#   opcode: nested dicts
#   inv_opcode: dict

import yaml


# opcode & inv_opcode:
with open("yamls/opcodes.yaml", 'r') as file:
    data = yaml.safe_load(file)
    opcodes = data['opcode']
    base_instr_types = data['base_instr_types']

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
