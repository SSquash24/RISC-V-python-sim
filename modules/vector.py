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

    def _read_vgroup(self, reg, sew=None, lmul=None, type=None):
        """
        Returns list of elements from vector group selected as reg
        """
        if sew is None:
            sew = self.sew
        if lmul is None:
            lmul = self.lmul
        bits = ''.join(self.state['vregs'][reg:ceil(reg+lmul)])
        if lmul < 1:
            bits = bits[:int(lmul*self.vlen)]
        split_bits = [bits[i:i+sew] for i in range(0, len(bits), sew)]
        match type:
            case 'int':
                res = [self.twos_to_int(i) for i in split_bits]
            case 'unsigned':
                res = [int(i, 2) for i in split_bits]
            case _:
                res = split_bits
        return res

    def _write_vgroup(self, reg, values, mask=None, sew=None, lmul=None, type=None):
        # TODO mask

        match type:
            case 'int':
                values = [self.twos_comp_bin(i, self.sew) for i in values]
            case 'unsigned':
                values = [bin(i)[2:].rjust(self.sew, '0') for i in values]

        if sew is None:
            sew = self.sew
        if lmul is None:
            lmul = self.lmul

        if lmul < 1:
            bits = ''.join(values)
            if self.ta:
                self.state['vregs'][reg] = bits.rjust(self.vlen, '1')
            else:
                self.state['vregs'][reg] = bits + self.state['vregs'][reg][len(bits):]
        else:
            jump = self.vlen // sew
            for i in range(self.lmul):
                self.state['vregs'][reg+i] = ''.join(values[i*jump:(i+1)*jump])

    def vbits_to_int(self, bits):
        return [self.twos_to_int(i) for i in bits]

    def vint_to_bits(self, lst):
        return [self.twos_comp_bin(i, self.sew) for i in lst]

    def vbits_to_float(self, bits):
        assert all([len(i) == 32 for i in bits])
        return [RV_Float.from_bits(i) for i in bits]

    def vfloat_to_bits(self, lst):
        return [i.bits() for i in lst]

    def __init__(self, state, vlen=128, vtype=None):
        super().__init__(state)
        self.state['vregs'] = ['0'*vlen for _ in range(32)]
        self.vstart = 0
        self.vxsat = 0
        self.vxrm = 0
        self.vlen = vlen
        self.lmul = 1
        self.sew = 32
        self.vl = vlen // self.sew
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

    def _get_mask(self, instr):
        return self.state['vreg'][0] if instr[-1] == 'v0.t' else None

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

    def store(self, binary, addr):
        """Stores arbitrary length binary string at mem addr(bites)"""
        addr, mod = divmod(addr, 4)
        mod *= 8
        len1 = 32 - mod
        data = self.state['mem'][addr]
        if len(binary) <= len1:
            self.state['mem'][addr] = data[:mod] + binary + data[mod+len(binary):]
        else:
            self.state['mem'][addr] = data[:mod] + binary[:len1]
            binary = binary[len1:]
            addr += 4
            while len(binary) >= 32:
                self.state['mem'][addr] = binary[:32]
                binary = binary[32:]
                addr += 4
            if len(binary) != 0:
                self.state['mem'][addr] = binary + self.state['mem'][addr][len(binary):]


    def map(self, vd, vs1, vs2, func, data_type, vm):
        data1 = self._read_vgroup(vs1, type=data_type)
        data2 = self._read_vgroup(vs2, type=data_type)
        data = [func(data1[i], data2[i]) for i in range(len(data1))]
        self._write_vgroup(vd, data, type=data_type, mask=vm)

    def run_instr(self, instr):
        match instr[0]:
            # TODO use vstart on loads and stores
            case 'vle32.v' | 'vlseg2e32.v' | 'vlseg3e32.v' | 'vlseg4e32.v' | 'vlseg5e32.v' | 'vlseg6e32.v' | 'vlseg7e32.v' | 'vlseg8e32.v':
                eew = 32
                emul = (eew / self.sew) * self.lmul
                num_groups = 1 if instr[0] == 'vle32.v' else int(instr[0][5])
                addr = self._read_reg(instr[2], signed=False)

                num_mem_reads = ceil((emul * self.vlen * num_groups) / 32)
                data = ''.join([self.state['mem'][addr+(4*a)] for a in range(num_mem_reads)])
                vect = [data[i:i+eew] for i in range(0, self.vlen * emul, eew)]

                # split data
                vect = [vect[i:i+num_groups] for i in range(0, len(vect), num_groups)]
                for r, data in enumerate(vect):
                    self._write_vgroup(instr[1]+(r*emul), data, self._get_mask(instr), lmul=emul, sew=eew)

            # load FFs TODO

            case 'vl1re32.v' | 'vl2re32.v' | 'vl4re32.v' | 'vl8re32.v':
                eew = 32 # TODO: why is eew needed at all?
                num_regs = int(instr[0][2])
                addr = self._read_reg(instr[2], signed=False)
                data = ''.join([self.state['mem'][addr+i] for i in range(0, ceil((self.vlen*num_regs)/(8*4)), 4)])
                vect = [data[i:i+eew] for i in range(0, self.vlen*num_regs, eew)]
                self._write_vgroup(instr[1], vect, self._get_mask(instr), lmul=num_regs, sew=eew)


            case 'vluxei32.v' | 'vluxseg2ei32.v' | 'vluxseg3ei32.v' | 'vluxseg4ei32.v' | 'vluxseg5ei32.v' | 'vluxseg6ei32.v' | 'vluxseg7ei32.v' | 'vluxseg8ei32.v'\
                | 'vloxei32.v' | 'vloxseg2ei32.v' | 'vloxseg3ei32.v' | 'vloxseg4ei32.v' | 'vloxseg5ei32.v' | 'vloxseg6ei32.v' | 'vloxseg7ei32.v' | 'vloxseg8ei32.v':
                # for simplicity we will access both in order, so implementation is the same between vlux... and vlox...
                # FOR NOW ONLY WORKS ON eew=32, TODO generalise
                eew = 32
                emul = (eew / self.sew) * self.lmul # eew, emul is used for index values
                num_groups = 1 if instr[0] == 'vluxei32.v' else int(instr[0][7])
                addr = self._read_reg(instr[2], signed=False)
                offsets = self._read_vgroup(instr[3], sew=eew, lmul=emul, type='int')

                vect = []
                for i in offsets:
                    loc = addr + i
                    assert i % 4 == 0
                    vect.append(self.state['mem'][loc])

                vect = [vect[i:i + num_groups] for i in range(0, len(vect), num_groups)]
                for r, data in enumerate(vect):
                    self._write_vgroup(instr[1] + (r*self.lmul), data, self._get_mask(instr))

            case 'vlse32.v' | 'vlsseg2e32.v' | 'vlsseg3e32.v' | 'vlsseg4e32.v' | 'vlsseg5e32.v' | 'vlsseg6e32.v' | 'vlsseg7e32.v' | 'vlsseg8e32.v':
                eew = 32
                emul = (eew / self.sew) * self.lmul
                num_groups = 1 if instr[0] == 'vlse32.v' else int(instr[0][6])
                stride = self._read_reg(instr[3])
                assert stride % 4 == 0 # TODO check if need to change?
                addr = self._read_reg(instr[2], signed=False)

                vect = []
                for i in range(int((emul*self.vlen*num_groups)/eew)):
                    data = ''.join([self.state['mem'][addr+i] for i in range(0, eew//8, 4)])
                    vect.append(data[:eew])
                    addr += stride

                # split data
                vect = [vect[i:i + num_groups] for i in range(0, len(vect), num_groups)]
                for r, data in enumerate(vect):
                    self._write_vgroup(instr[1] + (r * emul), data, self._get_mask(instr), lmul=emul, sew=eew)


            case 'vse32.v' | 'vsseg2e32.v' | 'vsseg3e32.v' | 'vsseg4e32.v' | 'vsseg5e32.v' | 'vsseg6e32.v' | 'vsseg7e32.v' | 'vsseg8e32.v':
                eew = 32
                emul = (eew / self.sew) * self.lmul
                num_groups = 1 if instr[0] == 'vle32.v' else int(instr[0][5])
                addr = self._read_reg(instr[2], signed=False)

                data = [self._read_vgroup(instr[1]+i, lmul=emul, sew=eew) for i in range(num_groups)]
                data = ''.join([data[i % num_groups][i//num_groups] for i in range(num_groups*len(data[0]))])
                self.store(data, addr)

            case 'vs1r.v' | 'vs2r.v' | 'vs4r.v' | 'vs8r.v':
                num_regs = int(instr[0][2])
                addr = self._read_reg(instr[2], signed=False)
                data = ''.join(self._read_vgroup(instr[1], sew=32, lmul=num_regs))
                self.store(data, addr)

            case 'vsuxei32.v' | 'vsuxseg2ei32.v' | 'vsuxseg3ei32.v' | 'vsuxseg4ei32.v' | 'vsuxseg5ei32.v' | 'vsuxseg6ei32.v' | 'vsuxseg7ei32.v' | 'vsuxseg8ei32.v' \
                'vsoxei32.v' | 'vsoxseg2ei32.v' | 'vsoxseg3ei32.v' | 'vsoxseg4ei32.v' | 'vsoxseg5ei32.v' | 'vsoxseg6ei32.v' | 'vsoxseg7ei32.v' | 'vsoxseg8ei32.v':
                eew = 32
                emul = (eew / self.sew) * self.lmul  # eew, emul is used for index values
                num_groups = 1 if instr[0] == 'vluxei32.v' else int(instr[0][7])
                addr = self._read_reg(instr[2], signed=False)
                offsets = self._read_vgroup(instr[3], sew=eew, lmul=emul, type='int')

                data = [self._read_vgroup(instr[1] + i, lmul=emul, sew=eew) for i in range(num_groups)]
                for i in range(len(data[0])):
                    for j in range(num_groups):
                        self.store(data[j][i], addr+offsets[i])

            case 'vsse32.v' | 'vsseg2e32.v' | 'vsseg3e32.v' | 'vsseg4e32.v' | 'vsseg5e32.v' | 'vsseg6e32.v' | 'vsseg7e32.v' | 'vsseg8e32.v':
                eew = 32
                emul = (eew / self.sew) * self.lmul  # eew, emul is used for index values
                num_groups = 1 if instr[0] == 'vluxei32.v' else int(instr[0][7])
                addr = self._read_reg(instr[2], signed=False)
                stride = self._read_reg(instr[3])

                data = [self._read_vgroup(instr[1] + i, lmul=emul, sew=eew) for i in range(num_groups)]
                for i in range(len(data[0])):
                    for j in range(num_groups):
                        self.store(data[j][i], addr)
                    addr += stride

            case 'vadd.vv':
                self.map(instr[1], instr[2], instr[3], lambda i,j: i+j, 'int', self._get_mask(instr))

            case 'vsub.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i - j, 'int', self._get_mask(instr))

            case 'vminu.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: min(i,j), 'unsigned', self._get_mask(instr))

            case 'vmin.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: min(i, j), 'int', self._get_mask(instr))
            case 'vmaxu.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: max(i, j), 'unsigned', self._get_mask(instr))
            case 'vmax.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: max(i, j), 'int', self._get_mask(instr))
            case 'vand.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i & j, 'unsigned', self._get_mask(instr))
            case 'vor.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i | j, 'unsigned', self._get_mask(instr))
            case 'vxor.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i ^ j, 'unsigned', self._get_mask(instr))
            case 'vrgather.vv':
                vs2 = self._read_vgroup(instr[2])
                vs1 = self._read_vgroup(instr[3], type='unsigned')
                data = [vs2[i] if i < self.vl else 0 for i in vs1]
                self._write_vgroup(instr[1], data, mask=self._get_mask(instr))
            case 'vrgatherei16.vv':
                vs2 = self._read_vgroup(instr[2])
                vs1 = self._read_vgroup(instr[3], type='unsigned', sew=16, lmul=(16/self.sew)*self.lmul)
                data = [vs2[i] if i < self.vl else 0 for i in vs1]
                self._write_vgroup(instr[1], data, mask=self._get_mask(instr))
            case 'vadc.vv':
                
            case _:
                    raise RVError(f"Vector module cannot run command {instr}")