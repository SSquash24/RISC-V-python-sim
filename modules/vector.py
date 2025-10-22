from modules.Float import RV_Float
from modules.RISC_modules import *
from yamls.read_yaml import read_yaml
from math import ceil

vxrm_to_val = {'rnu': 0, 'rne': 1, 'rdn': 2, 'rod': 3}
val_to_vxrm = {v:k for k,v in vxrm_to_val.items()}

vsew = {'e8': '000', 'e16': '001', 'e32': '010', 'e64': '011'}
vlmul = {'m1': '000', 'm2': '001', 'm4': '010', 'm8': '011', 'mf2': '111', 'mf4': '110', 'mf8': '101'}
vta = {'ta': '0', 't0': '1'}
vma = {'ma': '0', 'm0': '1'}


class Vector(Module):

    opcodes, inv_opcodes, csrs = read_yaml('vector.yaml', ret_CSR=True)


    def _read_vgroup(self, reg, type=int):
        """
        Returns list of elements from vector group selected as reg
        """
        bits = ''.join(self.state['vregs'][reg:ceil(reg+self.lmul)])
        if self.lmul < 1:
            bits = bits[:32//self.lmul]
        split_bits = [bits[i:i+self.sew] for i in range(0, len(bits), self.sew)]
        if type is int:
            return [self.twos_to_int(i) for i in split_bits]
        elif type is RV_Float:
            return [RV_Float.from_bits(i) for i in split_bits]
        else:
            raise RVError(f"Unknown type to read vector group: {type}")

    def _write_vgroup(self, reg, values, type=int):
        if type is int:
            values = [self.twos_comp_bin(i) for i in values]
        elif type is RV_Float:
            values = [i.bits() for i in values]
        else:
            raise RVError(f"Unknown type to store vector group: {type}")
        if self.lmul < 1:
            bits = ''.join(values)
            if self.ta:
                self.state['vregs'][reg] = bits.rjust(self.vlen, '1')
            else:
                self.state['vregs'][reg] = bits + self.state['vregs'][reg][len(bits):]
        else:
            jump = self.vlen // self.sew
            for i in range(self.lmul):
                self.state['vregs'][reg+i] = ''.join(values[i*jump:(i+1)*jump])



    def __init__(self, state, vlen=128, vtype=None):
        super().__init__(state)
        self.state['vregs'] = ['0'*vlen for _ in range(32)]
        self.vstart = 0
        self.vxsat = 0
        self.vxrm = 0
        self.vlen = vlen
        self.lmul = 1
        self.sew = 32
        self.ta = True
        self.ma = True
        if 'CSR' in state['modules']:
            # TODO set csr values

            # TODO check range on csrs
            self.state['csrs'][Vector.csrs['vstart']] = (lambda _: self.vstart, lambda x: setattr(self, 'vstart', x)) #min bits that can hold VLMAX-1
            self.state['csrs'][Vector.csrs['vxsat']] = (lambda _: self.vxsat, lambda x: setattr(self, 'vxsat', x)) #0 or 1
            self.state['csrs'][Vector.csrs['vxrm']] = (lambda _: self.vxrm, lambda x: setattr(self, 'vxrm', x))  # 2 bits
            self.state['csrs'][Vector.csrs['vcsr']] = (lambda _: (self.vxrm << 1) + self.vxsat, lambda x: x) # TODO write function, 3 bits
            self.state['csrs'][Vector.csrs['vl']] = (lambda _: vlen, lambda _: RVError("Cannot write to vl"))
            # TODO self.state['csrs'][Vector.csrs['vtype']]
            self.state['csrs'][Vector.csrs['vlenb']] = (lambda _: vlen//8, lambda _: RVError("Cannot write to vlenb"))

            self.state['csr_dict'].update(Vector.csrs)

    def to_vtype(self, args, len=11):
        # return (vma[ma] + vta[ta] + vsew[sew] + vlmul[lmul]).rjust(len, '0')
        print(args)

        pass # TODO

    def assemble(self, *instr):
        try:
            opcode, instr_type, args = Vector.inv_opcodes[instr[0]]
            args = args.copy()
        except KeyError:
            return False, None
        print(opcode, instr)
        match instr[0]: # there's a lot of unique cases on assembly, so we match instr rather than type
            case 'vsetvli':
                args[(30,20)] = self.to_vtype(instr[3:])
                return True, self.assemble_32(opcode, rd=self.reg(args[1]), rs1=self.reg(args[2]), flags=args)
            case 'vsetivli':
                args[(29,20)] = self.to_vtype(instr[3:], len=10)
                args[(19,15)] = bin(int(args[2]))[2:].rjust(5, '0')
                return True, self.assemble_32(opcode, rd=self.reg(args[1]), flags=args)
            case 'vsetvl':
                return True, self.assemble_32(opcode, rd=self.reg(args[1]), rs1=self.reg(args[2]), rs2=self.reg(args[3]), flags=args)

    def unassemble(self, *instr):
        return False, None