from modules.Float import RV_Float
from modules.RISC_modules import *
from yamls.read_yaml import read_yaml
from math import ceil

vxrm_to_val = {'rnu': 0, 'rne': 1, 'rdn': 2, 'rod': 3}
val_to_vxrm = {v:k for k,v in vxrm_to_val.items()}

vsew = {'e8': '000', 'e16': '001', 'e32': '010', 'e64': '011'}#
inv_vsew = {v: k for k, v in vsew.items()}
vlmul = {'m1': '000', 'm2': '001', 'm4': '010', 'm8': '011', 'mf2': '111', 'mf4': '110', 'mf8': '101'}
inv_vlmul = {v: k for k, v in vlmul.items()}


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
        sew = vsew[args[0]]
        lmul = vlmul[args[1]]
        ta = '1' if 'ta' in args else '0'
        ma = '1' if 'ma' in args else '0'
        return (ma + ta + lmul + sew).rjust(len, '0')

    def assemble(self, *instr):
        try:
            opcode, instr_type, args = Vector.inv_opcodes[instr[0]]
            args = args.copy()
        except KeyError:
            return False, None
        match instr_type:
            case 'M':
                args[(25, 25)] = 0 if instr[-1] == 'v0.t' else 1
                return True, self.assemble_32(opcode, rd=self.reg(instr[1]), flags=args)

            case 'M1':
                args[(25,25)] = 0 if instr[-1] == 'v0.t' else 1
                return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs1=self.reg(instr[2]), flags=args)

            case '1':
                return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs1=self.reg(instr[2]), flags=args)

            case 'M2' | 'OPIVV' | 'OPFVV' | 'OPMVV' | 'OPIVX' | 'OPFVF' | 'OPMVX':
                args[(25, 25)] = 0 if instr[-1] == 'v0.t' else 1
                return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs1=self.reg(instr[2]), rs2=self.reg(instr[3]), flags=args)

            case '2':
                return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs1=self.reg(instr[2]), rs2=self.reg(instr[3]),
                                      flags=args)

            case 'OPIVI':
                args[(25, 25)] = 0 if instr[-1] == 'v0.t' else 1
                args[(19,15)] = self.twos_comp_bin(instr[3], 5)
                return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs1=self.reg(instr[2]), flags=args)

            case 'VMVR':
                return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs2=self.reg(instr[2]), flags=args)

            case 'VZEXT':
                args[(25, 25)] = 0 if instr[-1] == 'v0.t' else 1
                return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs2=self.reg(instr[2]), flags=args)

            case 'V':
                # special case, match instr
                match instr[0]:
                    case 'vsetvli':
                        args[(30, 20)] = self.to_vtype(instr[3:])
                        return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs1=self.reg(instr[2]), flags=args)
                    case 'vsetivli':
                        args[(29, 20)] = self.to_vtype(instr[3:], len=10)
                        return True, self.assemble_32(opcode, rd=self.reg(instr[1]), rs1=self.reg(instr[2]), flags=args)
                    case _:
                        return False, None
            case _:
                return False, None
    def unassemble(self, binary):
        assert len(binary) == 32

        opcode, rd, rs1, rs2, funct3, funct7 = super()._decompile_commons(binary)
        vm = binary[6] == '0'
        try:
            args1 = Vector.opcodes[opcode]
        except KeyError:
            return False, None

        instr_type = args1[1]
        to_match = args1[-2]
        args1 = args1[-1][int(binary[31 - to_match[0]: 32 - to_match[1]], 2)]
        while type(args1) is list:
            if len(args1) == 3:
                instr_type = args1[0]
            to_match = args1[-2]
            try:
                args1 = args1[-1][int(binary[31-to_match[0]: 32-to_match[1]], 2)]
            except KeyError:
                return False, None

        match instr_type:
            case 'M':
                res = (args1, rd, 'v0.t') if vm else (args1, rd)
                return True, res
            case 'M1':
                res = (args1, rd, rs1, 'v0.t') if vm else (args1, rd, rs1)
                return True, res
            case '1':
                return True, (args1, rd, rs1)
            case 'M2' | 'OPIVV' | 'OPFVV' | 'OPMVV' | 'OPIVX' | 'OPFVF' | 'OPMVX':
                res = (args1, rd, rs1, rs2, 'v0.t') if vm else (args1, rd, rs1, rs2)
                return True, res
            case '2':
                return True, (args1, rd, rs1, rs2)
            case 'OPIVI':
                imm = self.twos_to_int(binary[12:17])
                res = (args1, rd, rs1, imm, 'v0.t') if vm else (args1, rd, rs1, imm)
                return True, res
            case 'VMVR':
                return True, (args1, rd, rs2)
            case 'VZEXT':
                res = (args1, rd, rs2, 'v0.t') if vm else (args1, rd, rs2)
                return True, res
            case 'V':
                # special case, match instr
                match args1:
                    case 'vsetvli':
                        sew = inv_vsew[binary[9:12]]
                        lmul = inv_vlmul[binary[6:9]]
                        tama = ('ta') if binary[5] == '1' else ()
                        tama += ('ma') if binary[4] == '1' else ()
                        return True, (args1, rd, rs1, sew, lmul) + tama
                    case 'vsetivli':
                        sew = inv_vsew[binary[9:12]]
                        lmul = inv_vlmul[binary[6:9]]
                        tama = ('ta') if binary[5] == '1' else ()
                        tama += ('ma') if binary[4] == '1' else ()
                        return True, (args1, rd, rs1, sew, lmul) + tama
                    case _:
                        return False, None
            case _:
                return False, None