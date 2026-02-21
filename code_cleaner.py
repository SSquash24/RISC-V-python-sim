from modules.RISC_modules import Module
import struct
from math import ceil

reg_pseudos = {
    'ra': 'x1',
    'sp': 'x2',
    'gp': 'x3',
    'tp': 'x4',
    't0': 'x5',
    't1': 'x6',
    't2': 'x7',
    's0': 'x8',
    'fp': 'x8',
    's1': 'x9',
    'a0': 'x10',
    'a1': 'x11',
    'a2': 'x12',
    'a3': 'x13',
    'a4': 'x14',
    'a5': 'x15',
    'a6': 'x16',
    'a7': 'x17',
    's2': 'x18',
    's3': 'x19',
    's4': 'x20',
    's5': 'x21',
    's6': 'x22',
    's7': 'x23',
    's8': 'x24',
    's9': 'x25',
    's10': 'x26',
    's11': 'x27',
    't3': 'x28',
    't4': 'x29',
    't5': 'x30',
    't6': 'x31',
}


def clean_instr(instr, pseudos):
    instr = instr.strip().replace(',', '').split('#', 1)[0].split()
    if not instr:
        return []

    instr = [reg_pseudos[i] if i in reg_pseudos.keys() else i for i in instr]

    if instr[0] in pseudos.keys():
        args = instr[1:]
        instr = pseudos[instr[0]].copy()
        for i, arg in enumerate(instr):
            if arg.split(':')[0] == 'INP':
                index = int(arg.split(':')[1]) - 1
                instr[i] = args[index]
    return instr

def clean_code(code, pseudos, word_size):
    res = []
    addresses = {}

    clean_code = []
    data_section = False
    var = None
    data_vals = []

    for i in code:
        i = clean_instr(i, pseudos)
        if i == ['.data']:
            data_section = True
        elif i == ['.code']:
            data_section = False
        elif i:
            if data_section:
                if i[0][0] == '.':
                    data_vals.append((var, i))
                else:
                    var = i[0][:-1]
            else:
                clean_code.append(i)

    code = clean_code

    clean_data = {} # name: addr
    addr = 0
    data_dump = []
    for var, data in data_vals:
        to_add = 0
        match data[0][1:]:

            case 'int':
                # align prev stuff
                if len(data_dump) > 0:
                    data_dump[-1] = data_dump[-1].ljust(word_size * 8, '0')
                    addr = ceil(addr / word_size) * word_size

                val = int(data[1])
                to_add = 4
                data_dump.append(Module.twos_comp_bin(val, word_size * 8))

            case 'bool':
                val = int(data[1])
                if addr % word_size == 0:
                    data_dump.append(Module.twos_comp_bin(val, 8).rjust(word_size * 8, '0'))
                else:
                    data_dump[-1] += Module.twos_comp_bin(val, 8).rjust(word_size * 8, '0')
                to_add = 1
            case 'float':
                # align prev stuff
                if len(data_dump) > 0:
                    data_dump[-1] = data_dump[-1].ljust(word_size * 8, '0')
                    addr = ceil(addr / word_size) * word_size

                val = float(data[1])
                bits, = struct.unpack('!I', struct.pack('!f', val))
                data_dump.append("{:032b}".format(bits))
                to_add = 4

        clean_data[var] = addr
        addr += to_add

    for line in code:
        if line[-1][-1] == ':':
            # this is a named address
            addresses[line[-1][:-1].strip()] = len(res)*word_size
        else:
            res.append(line)

    # addresses from name to line
    loc = 0
    res2 = []
    data_start_addr = len(res) * word_size
    for line in res:
        line = [str(addresses[i]-loc) if i in addresses.keys() else i for i in line]
        line = [['x0', str(clean_data[i]+data_start_addr)] if i in clean_data.keys() else [i] for i in line]
        line = [i for x in line for i in x]
        res2.append(line)
        loc += 4

    return res2, data_dump

