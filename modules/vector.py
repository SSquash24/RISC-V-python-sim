from RISC_modules import *
from yamls.read_yaml import read_yaml


class VECTOR(Module):

    opcodes, inv_opcodes = read_yaml('vector.yaml')


    def __init__(self, state, vlen=128, vtype=None):
        super().__init__()
        self.state['vregs'] = [0 for _ in range(32)]
        if 'CSR' in state['modules']:
            # TODO set csr values
            self.state['csrs'] += {'008': (0, lambda a, b: print(f"VSTART read={a} write={b}")),
                                   '009': (0, lambda a, b: print(f"VXSAT read={a} write={b}" )),
                                   '00a': (0, lambda a, b: print(f"VXRM read={a} write={b}")),
                                   '00f': (0, lambda a, b: print(f"VCSR read={a} write={b}")),
                                   'c20': (vlen, lambda a, b: print(f"VL read={a} write={b}")),
                                   'c21': (0, lambda a, b: print(f"VTYPE read={a} write={b}")),
                                   'c22': (vlen/8, lambda a, b: print(f"VLENB read={a} write={b}"))
                                   }

    def assemble(self, *instr):
        pass # TODO