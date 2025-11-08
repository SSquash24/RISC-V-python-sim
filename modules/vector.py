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

    def _read_vgroup(self, reg, sew=None, type=None):
        """
        Returns list of elements from vector group selected as reg
        """
        if sew is None:
            sew = self.sew
        n_regs = ceil(self.vl * sew / self.vlen)
        bits = ''.join(self.state['vregs'][reg:reg+n_regs])[:n_regs * self.vlen]
        split_bits = [bits[i:i+sew] for i in range(0, len(bits), sew)]
        match type:
            case 'int':
                res = [self.twos_to_int(i) for i in split_bits]
            case 'unsigned':
                res = [int(i, 2) for i in split_bits]
            case _:
                res = split_bits
        return res

    def _write_vgroup(self, reg, values, mask=None, sew=None, type=None):
        # TODO mask

        if sew is None:
            sew = self.sew

        match type:
            case 'int':
                values = [self.twos_comp_bin(i, self.sew) for i in values]
            case 'unsigned':
                values = [bin(i)[2:].rjust(self.sew, '0') for i in values]

        bits = ''.join(values)
        i = 0
        while len(bits) >= self.vlen:
            self.state['vregs'][reg+i] = bits[:self.vlen]
            bits = bits[self.vlen:]
            i += 1
        if len(bits) != 0:
            if self.ta:
                self.state['vregs'][reg+i] = bits.ljust(self.vlen, '1')
            else:
                self.state['vregs'][reg+i] = bits + self.state['vregs'][reg+i][len(bits):]

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
        self.ta = 1
        self.ma = 1
        self.vill = 0
        if 'CSR' in state['modules']:
            # TODO set csr values

            # TODO check range on csrs
            self.state['csrs'][Vector.csrs['vstart']] = (lambda _: self.vstart, lambda x: setattr(self, 'vstart', x)) #min bits that can hold VLMAX-1
            self.state['csrs'][Vector.csrs['vxsat']] = (lambda _: self.vxsat, lambda x: setattr(self, 'vxsat', x)) #0 or 1
            self.state['csrs'][Vector.csrs['vxrm']] = (lambda _: self.vxrm, lambda x: setattr(self, 'vxrm', x))  # 2 bits
            self.state['csrs'][Vector.csrs['vcsr']] = (lambda _: (self.vxrm << 1) + self.vxsat, lambda x: x) # TODO write function, 3 bits
            self.state['csrs'][Vector.csrs['vl']] = (lambda _: vlen, lambda _: RVError("Cannot write to vl"))
            self.state['csrs'][Vector.csrs['vtype']] = (lambda _: (self.vill << 31) + (self.ma << 7) + (self.ta << 6) + (int(vsew[f'e{self.sew}'], 2) << 3) + (self.lmul - 1 if self.lmul >= 1 else int(vlmul[f'mf{int(1/self.lmul)}'], 2)), lambda _: RVError("To write to vtype use vsetvli"))
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


    def map(self, vd, vs2, vs1, func, data_type, vm=None, sewd=None, sew2=None, sew1=None, vs1type='v'):
        datai = self._read_vgroup(vs2, type=data_type, sew=self.sew if sew2 is None else sew2)
        match vs1type:
            case 'v':
                dataj = self._read_vgroup(vs1, type=data_type, sew=self.sew if sew1 is None else sew1)
            case 'x':
                match data_type:
                    case 'int':
                        dataj = self._read_reg(vs1, True)
                    case 'unsigned':
                        dataj = self._read_reg(vs1, False)
                    case None:
                        dataj = bin(self._read_reg(vs1, False))[2:].rjust(sew1, '0')
                dataj = [dataj for _ in range(self.vl)]
            case 'i':
                match data_type:
                    case 'int':
                        dataj = vs1 if vs1 > 0 else vs1+(2**5) # convert to unsigned
                    case 'unsigned':
                        pass
                    case None:
                        dataj = self.twos_comp_bin(vs1, 5)
                dataj = [dataj for _ in range(self.vl)]

        data = [func(datai[i], dataj[i]) for i in range(self.vl)]
        self._write_vgroup(vd, data, type=data_type, sew=self.sew if sewd is None else sewd, mask=vm)

    def mmap(self, md, m1, m2, func):
        """Reads masks m1 and m2 as an unsigned int, operates func on them, and stores in md"""
        m1 = int(self.state['vregs'][m1][:self.vl], 2)
        m2 = int(self.state['vregs'][m2][:self.vl], 2)
        bits = func(m1, m2)
        if self.ta == 1:
            self.state['vregs'][md] = bits.ljust(self.vlen, '0')
        else:
            bits = bits + self.state['vregs'][md][len(bits):]
            self.state['vregs'][md] = bits

    def redmap(self, vd, vs2, vs1, func, data_type, vm):
        data2 = self._read_vgroup(vs2, type=data_type)
        if vm is not None:
            data2 = [data2[i] for i in range(self.vl) if vm[i] == '1']
        val1 = self._read_vgroup(vs1, type=data_type)[0]
        res = func(val1, data2)
        self._write_vgroup(vd, [res], type=data_type, sew=self.sew)


    def cutdown(self, val, bits=None):
        """Treats the integer as bits, cuts off high bits to reach an int that can be stored in the given bits
        if bits=None, use self.sew"""
        if bits is None:
            bits = self.sew
        return val & (int('1' * bits, 2))

    def round(self, flt):
        return int(flt) # TODO

    def run_instr(self, instr):
        match instr[0]:
            # TODO use vstart on loads and stores
            case 'vle32.v' | 'vlseg2e32.v' | 'vlseg3e32.v' | 'vlseg4e32.v' | 'vlseg5e32.v' | 'vlseg6e32.v' | 'vlseg7e32.v' | 'vlseg8e32.v':
                eew = 32
                emul = (eew / self.sew) * self.lmul
                reg_jump = int(max(1, emul))
                num_groups = 1 if instr[0] == 'vle32.v' else int(instr[0][5])
                addr = self._read_reg(instr[2], signed=False)
                assert addr % 4 == 0
                addr = addr // 4

                num_mem_reads = ceil(self.vl * self.sew / 32)
                data = ''.join([self.state['mem'][addr+a] for a in range(num_mem_reads)])
                vect = [data[i:i+eew] for i in range(0, int(self.vl * eew), eew)]

                # split data
                vect = [[vect[(i*num_groups)+g] for i in range(self.vl)] for g in range(num_groups)]
                for r, v in enumerate(vect):
                    self._write_vgroup(instr[1]+(r*reg_jump), v, self._get_mask(instr), sew=eew)

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

            case 'vadd.vv' | 'vadd.vx' | 'vadd.vi':
                self.map(instr[1], instr[2], instr[3], lambda i,j: i+j, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])

            case 'vsub.vv' | 'vsub.vx':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i - j, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            # TODO vsrub.vx/vi
            case 'vminu.vv' | 'vminu.vx':
                self.map(instr[1], instr[2], instr[3], lambda i, j: min(i,j), 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vmin.vv' | 'vmin.vx':
                self.map(instr[1], instr[2], instr[3], lambda i, j: min(i, j), 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vmaxu.vv' | 'vmaxu.vx':
                self.map(instr[1], instr[2], instr[3], lambda i, j: max(i, j), 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vmax.vv' | 'vmax.vx':
                self.map(instr[1], instr[2], instr[3], lambda i, j: max(i, j), 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vand.vv' | 'vand.vx' | 'vand.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i & j, 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vor.vv' | 'vor.vx' | 'vor.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i | j, 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vxor.vv' | 'vxor.vx' | 'vxor.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i ^ j, 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vrgather.vv' | 'vrgather.vx' | 'vrgather.vi':
                vs2 = self._read_vgroup(instr[2])
                match instr[0][-1]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='unsigned')
                    case 'x':
                        vs1 = [self._read_reg(instr[3], False) for _ in range(self.vl)]

                data = [vs2[i] if i < self.vl else 0 for i in vs1]
                self._write_vgroup(instr[1], data, mask=self._get_mask(instr))
            case 'vrgatherei16.vv':
                vs2 = self._read_vgroup(instr[2])
                vs1 = self._read_vgroup(instr[3], type='unsigned', sew=16, lmul=(16/self.sew)*self.lmul)
                data = [vs2[i] if i < self.vl else 0 for i in vs1]
                self._write_vgroup(instr[1], data, mask=self._get_mask(instr))
            # TODO vslideup.vx/vi, vslidedown.vx/vi
            case 'vadc.vvm' | 'vadc.vxm' | 'vadc.vim':
                vs2 = self._read_vgroup(instr[2], type='unsigned')
                match instr[0][-2]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='unsigned')
                    case 'x':
                        vs1 = [self._read_reg(instr[3], False) for _ in range(self.vl)]
                    case 'i':
                        vs1 = instr[3] if instr[3] >= 0 else instr[3]+(2**5)
                mask = self._get_mask('v0.t')
                data = [vs1[i] + vs2[i] + int(mask[i]) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='unsigned')
            case 'vmadc.vvm' | 'vmadc.vxm':
                vs2 = self._read_vgroup(instr[2], type='unsigned')
                match instr[0][-2]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='unsigned')
                    case 'x':
                        vs1 = [self._read_reg(instr[3], False) for _ in range(self.vl)]
                mask = self._get_mask(instr)
                if mask is None:
                    mask = '0' * self.vl
                data = ''.join(['0' if vs1[i] + vs2[i] + int(mask[i]) < (1 << self.sew) else '1' for i in range(self.vl)])
                self.state['vregs'][0] = data.ljust(self.vlen, '0')
            case 'vsbc.vvm' | 'vsbc.vxm':
                vs2 = self._read_vgroup(instr[2], type='unsigned')
                match instr[0][-2]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='unsigned')
                    case 'x':
                        vs1 = [self._read_reg(instr[3], False) for _ in range(self.vl)]
                mask = self._get_mask('v0.t')
                data = [vs2[i] - vs1[i] - int(mask[i]) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='unsigned')

            case 'vmsbc.vvm' | 'vmsbc.vxm':
                vs2 = self._read_vgroup(instr[2], type='unsigned')
                match instr[0][-2]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='unsigned')
                    case 'x':
                        vs1 = [self._read_reg(instr[3], False) for _ in range(self.vl)]
                mask = self._get_mask(instr)
                if mask is None:
                    mask = '0' * self.vl
                data = ''.join(['0' if vs2[i] < vs1[i] + int(mask[i]) else '1' for i in range(self.vl)])
                self.state['vregs'][0] = data.ljust(self.vlen, '0')

            case 'vmerge.vvm' | 'vmerge.vxm' | 'vmerge.vim':
                vs2 = self._read_vgroup(instr[2], type='int')
                match instr[0][-2]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='unsigned')
                    case 'x':
                        vs1 = [self._read_reg(instr[3], False) for _ in range(self.vl)]
                    case 'i':
                        vs1 = instr[3] if instr[3] >= 0 else instr[3] + (2 ** 5)
                mask = self._get_mask('v0.t')
                data = [vs1[i] if mask[i] else vs2[i] for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int')

            case 'vmv.v.v' | 'vmv.v.x' | 'vmv.v.i':
                match instr[0][-2]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='int')
                    case 'x':
                        vs1 = [bin(self._read_reg(instr[3], True)) for _ in range(self.vl)]
                    case 'i':
                        vs1 = instr[3]
                self._write_vgroup(instr[1], vs1, type='int')

            case 'vmseq.vv' | 'vmseq.vx' | 'vmseq.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: 1 if i == j else 0, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vmsne.vv' | 'vmsne.vx' | 'vmsne.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: 1 if i != j else 0, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vmsltu.vv' | 'vmsltu.vx':
                self.map(instr[1], instr[2], instr[3], lambda i, j: 1 if i < j else 0, 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vmslt.vv' | 'vmslt.vx':
                self.map(instr[1], instr[2], instr[3], lambda i, j: 1 if i < j else 0, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vmsleu.vv' | 'vmsleu.vx' | 'vmsleu.vi' | 'vmsle.vv' | 'vmsle.vx ' | 'vmsle.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: 1 if i <= j else 0, 'unsigned' if instr[0][-4] == 'u' else 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])

            # TODO vmsgtu.vi, vmsgt.vi
            case 'vsaddu.vv' | 'vsaddu.vx' | 'vsaddu.vi':
                def vsaddu(i,j):
                    x = i + j
                    if x >= (1 << self.sew):
                        x = (1 << self.sew) - 1
                        self.vxsat = 1
                    return x
                self.map(instr[1], instr[2], instr[3], vsaddu, 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vsadd.vv' | 'vsadd.vx' | 'vsadd.vi':
                def vsadd(i,j):
                    x = i + j
                    if x >= 1 << (self.sew-1):
                        x = (1 << (self.sew-1)) - 1
                        self.vxsat = 1
                    elif x < -(1<<(self.sew-1)):
                        x = -(1 << (self.sew-1))
                        self.vssat = 1
                    return x
                self.map(instr[1], instr[2], instr[3], vsadd, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            # TODO vmv[size]r.v
            case 'vssubu.vv' | 'vssubu.vx':
                def vssubu(i, j):
                    x = i - j
                    if x < 0:
                        x = 0
                        self.vxsat = 1
                    return x

                self.map(instr[1], instr[2], instr[3], vssubu, 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vssub.vv' | 'vssub.vx':
                def vssub(i, j):
                    x = i - j
                    if x >= 1 << (self.sew - 1):
                        x = (1 << (self.sew - 1)) - 1
                        self.vxsat = 1
                    elif x < -(1 << (self.sew - 1)):
                        x = -(1 << (self.sew - 1))
                        self.vssat = 1
                    return x

                self.map(instr[1], instr[2], instr[3], vssub, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vsll.vv' | 'vssl.vx':
                self.map(instr[1], instr[2], instr[3], lambda i,j: i << j, 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vsmul.vv' | 'vsmul.vx':

                def vsmul(i, j):
                    x = self.round(i * j / (2**(self.sew-1)))
                    if x >= 1 << (self.sew - 1):
                        x = (1 << (self.sew - 1)) - 1
                        self.vxsat = 1
                    elif x < -(1 << (self.sew - 1)):
                        x = -(1 << (self.sew - 1))
                        self.vssat = 1
                    return x

                self.map(instr[1], instr[2], instr[3], vsmul, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vsrl.vv' | 'vsrl.vx' | 'vsrl.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i >> j, 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vsra.vv' | 'vsra.vx' | 'vsra.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i >> j, 'int',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vssrl.vv' | 'vssrl.vx' | 'vssrl.vi':
                self.map(instr[1], instr[2], instr[3], lambda i, j: self.round(i / (2**(j % self.sew))),
                         'unsigned', self._get_mask(instr), vs1type=instr[0][-1])
            case 'vssra.vv' | 'vssra.vx' | 'vssra.vi':
                vs2 = self._read_vgroup(instr[2], type='int')
                match instr[0][-1]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='unsigned')
                    case 'x':
                        vs1 = [self._read_reg(instr[3], False) for _ in range(self.vl)]
                    case 'i':
                        vs1 = instr[3] if instr[3] >= 0 else instr[3] + (2 ** 5)
                data = [self.round(vs2[i] / (2**(vs1[i] % self.sew))) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='unsigned', mask=self._get_mask(instr))
            case 'vnsrl.wv' | 'vnsrl.wx' | 'vnsrl.wi':
                self.map(instr[1], instr[2], instr[3], lambda i,j: i >> (j % self.sew), 'unsigned',
                         self._get_mask(instr), vs1type=instr[0][-1])
            case 'vnsra.wv' | 'vnsra.wx' | 'vnsra.wi':
                vs2 = self._read_vgroup(instr[2], type='int', sew=2 * self.sew)
                match instr[0][-1]:
                    case 'v':
                        vs1 = self._read_vgroup(instr[3], type='unsigned')
                    case 'x':
                        vs1 = [self._read_reg(instr[3], False) for _ in range(self.vl)]
                    case 'i':
                        vs1 = instr[3] if instr[3] >= 0 else instr[3] + (2 ** 5)
                data = [vs2[i] >> (vs1[i] % self.sew) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int', mask=self._get_mask(instr))

            # TODO vnclip, vnclipu, vwredsum, vwredsumu

            # OPMVV
            case 'vredsum.vs':
                self.redmap(instr[1], instr[2], instr[3], lambda i, j: i+sum(j), 'int', self._get_mask(instr))
            case 'vredand.vs':
                self.redmap(instr[1], instr[2], instr[3], lambda i, j: [i := i & x for x in j][-1], 'unsigned',
                            self._get_mask(instr))
            case 'vredor.vs':
                self.redmap(instr[1], instr[2], instr[3], lambda i, j: [i := i | x for x in j][-1], 'unsigned',
                    self._get_mask(instr))
            case 'vredor.vs':
                self.redmap(instr[1], instr[2], instr[3], lambda i, j: [i := i ^ x for x in j][-1], 'unsigned',
                        self._get_mask(instr))
            case 'vredminu.vs' | 'vredmin.vs':
                self.redmap(instr[1], instr[2], instr[3], lambda i, j: min([i]+j),
                            'unsigned' if instr[0][-4] == 'u' else 'int', self._get_mask(instr))
            case 'vredmaxu.vs' | 'vredmax.vs':
                self.redmap(instr[1], instr[2], instr[3], lambda i, j: max([i] + j),
                            'unsigned' if instr[0][-4] == 'u' else 'int', self._get_mask(instr))
            case 'vaaddu.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: self.round((i+j)/2), 'unsigned', self._get_mask(instr))
            case 'vaaddu.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: self.round((i + j) / 2), 'int', self._get_mask(instr))
            case 'vasubu.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: self.round((i - j) / 2), 'unsigned', self._get_mask(instr))
            case 'vasub.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: self.round((i - j) / 2), 'int', self._get_mask(instr))
            case 'vmv.x.s':
                self._set_reg(instr[1], self._read_vgroup(instr[2], type='int')[0])
            case 'vzext.vf8' | 'vzext.vf4' | 'vzext.vf2' | 'vsext.vf8' | 'vsext.vf4' | 'vsext.vf2':
                sew = self.sew // int(instr[0][-1])
                elem_type = 'unsigned' if instr[0][1] == 'z' else 'int'
                assert sew >= 8
                data = self._read_vgroup(instr[2], sew=sew, type=elem_type)
                self._write_vgroup(instr[1], data, type=elem_type, mask=self._get_mask(instr))

            # TODO vm___.m

            case 'vcompress.vm':
                assert self.vstart == 0
                data = self._read_vgroup(instr[2])
                mask = self.state['vregs'][instr[3]]
                data = [data[i] for i in range(self.vl) if mask[i] == '1']
                self._write_vgroup(instr[1], data)

            case 'vmandn.mm':
                self.mmap(instr[1], instr[2], instr[3], lambda m1, m2: m1 & (m2 ^ ((2**self.vl)-1)))
            case 'vmand.mm':
                self.mmap(instr[1], instr[2], instr[3], lambda m1, m2: m1 & m2)
            case 'vmor.mm':
                self.mmap(instr[1], instr[2], instr[3], lambda m1, m2: m1 | m2)
            case 'vmxor.mm':
                self.mmap(instr[1], instr[2], instr[3], lambda m1, m2: m1 ^ m2)
            case 'vmorn.mm':
                self.mmap(instr[1], instr[2], instr[3], lambda m1, m2: m1 | (m2 ^ ((2**self.vl)-1)))
            case 'vmnand.mm':
                self.mmap(instr[1], instr[2], instr[3], lambda m1, m2: (m1 & m2) ^ ((2**self.vl)-1))
            case 'vmnor.mm':
                self.mmap(instr[1], instr[2], instr[3], lambda m1, m2: (m1 | m2) ^ ((2**self.vl)-1))
            case 'vmxnor.mm':
                self.mmap(instr[1], instr[2], instr[3], lambda m1, m2: (m1 ^ m2) ^ ((2 ** self.vl) - 1))

            case 'vdivu.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i // j, 'unsigned', self._get_mask(instr))
            case 'vdiv.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i // j, 'int', self._get_mask(instr))
            case 'vremu.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i % j, 'unsigned', self._get_mask(instr))
            case 'vrem.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i % j, 'int', self._get_mask(instr))

            case 'vmulhu.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: (i*j)>>self.sew, 'unsigned', self._get_mask(instr))
            case 'vmul.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: self.cutdown(i*j), 'int', self._get_mask(instr))
            case 'vmulhsu':
                data1 = self._read_vgroup(instr[2], type='int')
                data2 = self._read_vgroup(instr[3], type='unsigned')
                data = [(data1[i]*data2[i]) >> self.sew for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int', mask=self._get_mask(instr))
            case 'vmulh.vv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: (i * j) >> self.sew, 'int', self._get_mask(instr))

            case 'vmadd.vv':
                datad = self._read_vgroup(instr[1], type='int')
                data1 = self._read_vgroup(instr[2], type='int')
                data2 = self._read_vgroup(instr[3], type='int')
                data = [self.cutdown((data1[i]*datad[i])+data2[i]) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int', mask=self._get_mask(instr))
            case 'vnmsub.vv':
                datad = self._read_vgroup(instr[1], type='int')
                data1 = self._read_vgroup(instr[2], type='int')
                data2 = self._read_vgroup(instr[3], type='int')
                data = [self.cutdown(-(data1[i] * datad[i]) + data2[i]) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int', mask=self._get_mask(instr))
            case 'vmacc.vv' | 'vwmacc.vv':
                datad = self._read_vgroup(instr[1], type='int')
                data1 = self._read_vgroup(instr[2], type='int')
                data2 = self._read_vgroup(instr[3], type='int')
                data = [self.cutdown((data1[i] * data2[i]) + datad[i]) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int', sew=2*self.sew if instr[0][1] == 'w' else self.sew,
                                   mask=self._get_mask(instr))
            case 'vnmasc.vv':
                datad = self._read_vgroup(instr[1], type='int')
                data1 = self._read_vgroup(instr[2], type='int')
                data2 = self._read_vgroup(instr[3], type='int')
                data = [self.cutdown(-(data1[i] * data2[i]) + datad[i]) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int', mask=self._get_mask(instr))
            case 'vwaddu.vv' | 'vwaddu.wv' | 'vwadd.vv' | 'vwadd.wv':
                self.map(instr[1], instr[2], instr[3], lambda i,j:i+j, 'unsigned' if instr[0][-4] == 'u' else 'int',
                         self._get_mask(instr), sewd=2*self.sew, sew2=self.sew if instr[0][-2] == 'v' else 2*self.sew)
            case 'vwsubu.vv' | 'vwsubu.wv' | 'vwsub.vv' | 'vwsub.wv':
                self.map(instr[1], instr[2], instr[3], lambda i, j: i - j, 'unsigned' if instr[0][-4] == 'u' else 'int',
                         self._get_mask(instr), sewd=2*self.sew, sew2=self.sew if instr[0][-2] == 'v' else 2 * self.sew)
            case 'vwmulu.vv' | 'vwmul.vv':
                self.map(instr[1], instr[2], instr[3], lambda i,j:i*j, 'unsigned' if instr[0][-4] == 'u' else 'int',
                         self._get_mask(instr), sewd=2*self.sew)
            case 'vwmulsu':
                data1 = self._read_vgroup(instr[2], type='int')
                data2 = self._read_vgroup(instr[3], type='unsigned')
                data = [data1[i] + data2[i] for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int', sew=2*self.sew, mask=self._get_mask(instr))

            case 'vwmaccu.vv':
                datad = self._read_vgroup(instr[1], type='unsigned')
                data1 = self._read_vgroup(instr[2], type='unsigned')
                data2 = self._read_vgroup(instr[3], type='unsigned')
                data = [self.cutdown((data1[i] * data2[i]) + datad[i]) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='unsigned', sew=2 * self.sew,
                                   mask=self._get_mask(instr))
            case 'vwmaccsu.vv':
                datad = self._read_vgroup(instr[1], type='int')
                data1 = self._read_vgroup(instr[2], type='int')
                data2 = self._read_vgroup(instr[3], type='unsigned')
                data = [self.cutdown((data1[i] * data2[i]) + datad[i]) for i in range(self.vl)]
                self._write_vgroup(instr[1], data, type='int', sew=2 * self.sew,
                                   mask=self._get_mask(instr))

            # TODO non-OPIVV/OPMVV
            case 'vsetvli' | 'vsetivli' | 'vsetvl':
                avl = instr[2] if instr[0][4] == 'i' else self._read_reg(instr[2])
                if instr[0][-1] == 'i':
                    self.vill = 0
                    assert instr[3] in vsew.keys()
                    self.sew = int(instr[3][1:])
                    assert instr[4] in vlmul.keys()
                    if instr[4][1] == 'f':
                        self.lmul = 1 / int(instr[4][2])
                    else:
                        self.lmul = int(instr[4][1])
                    self.ta = int('ta' in instr)
                    self.ma = int('ma' in instr)
                else:
                    settings = self._read_reg(instr[3])
                    self.vill = settings >> 31
                    self.sew = (settings >> 3) % 8
                    self.lmul = settings % 8
                    self.ta = (settings >> 6) % 2
                    self.ta = (settings >> 7) % 2

                vlmax = (self.vlen * self.lmul) // self.sew
                self.vl = min(avl, vlmax)
                self._set_reg(instr[1], self.vl)

            case _:
                raise RVError(f"Vector module cannot run command {instr}")