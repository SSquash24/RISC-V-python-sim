import math

from modules.RISC_modules import *
from yamls.read_yaml import read_yaml
import struct

rm = {0: 'RNE', 1: 'RTZ', 2: 'RDN', 3: 'RUP', 4: 'RMM', 7: 'DYN'}
inv_rm = {v: bin(k)[2:].rjust(3, '0') for k, v in rm.items()}

class RV_Float:
    """Float stored in IEEE 754 format"""

    @staticmethod
    def _convert(num: float) -> (int, int, int):
        bits, = struct.unpack('!I', struct.pack('!f', num))
        bits = "{:032b}".format(bits)
        res = RV_Float.from_bits(bits)
        return res.sign, res.exponent, res.mantissa


    def _normalise(self, mantissa, exponent, sign):
        """
        :param mantissa: In explicit form, with the 1
        :return: rounded mantissa (without the 1st 1), exponent
        """
        cutoff = ''
        while mantissa >= (1 << 24):
            exponent += 1
            cutoff = str(mantissa % 2) + cutoff
            mantissa >>= 1
        while mantissa < (1 << 23) and exponent > -127:
            exponent -= 1
            mantissa <<= 1
        if exponent > 127:
            self.inexact = True
            return 0, 128 # infinity
        self.inexact = '1' in cutoff
        return RV_Float.round(mantissa, cutoff, sign, rm=self.rm), exponent


    def __init__(self, value=None, rm='RNE'):
        self.rm = rm
        self.inexact = False
        if value is not None:
            self.sign, self.exponent, self.mantissa = RV_Float._convert(value)
        else:
            self.sign = 0
            self.exponent = -127
            self.mantissa = 0

    @staticmethod
    def from_bits(bits):
        res = RV_Float()
        res.sign = int(bits[0])
        res.exponent = int(bits[1:9], 2) - 127
        if res.exponent == -127:
            res.mantissa = int(bits[9:], 2)
        else:
            res.mantissa = int('1' + bits[9:], 2)
        return res

    def bits(self):
        """Return the float as 32 bits"""
        return str(self.sign) + bin(self.exponent+127)[2:].rjust(8, '0') + bin(self.mantissa)[3:].rjust(23, '0')

    def value(self) -> float:
        """Return the float in python float format"""
        # if '1' not in self.bits()[1:]:
        #     return 0.0
        return pow(-1, self.sign) * pow(2, self.exponent-23) * self.mantissa

    def makeNaN(self):
        self.sign = 0
        self.mantissa = 3 << 22
        self.exponent = 128

    def isNaN(self):
        return self.exponent == 128

    def set_rm(self, rm):
        self.rm = rm


    @staticmethod
    def round(val, cutoff, sign, rm='RNE'):
        if '1' not in cutoff:
            return val
        match rm:
            case 'RNE':
                if cutoff[0] == '1' and (len(cutoff) > 1 and '1' not in cutoff[1:]):
                    # round to even
                    return val + (val % 2)
                elif cutoff[0] == '0':
                    return val
                else:
                    return val+1
            case 'RTZ':
                return val
            case 'RDN':
                return val if sign == '0' else val+1
            case 'RUP':
                return val+1 if sign == '0' else val
            case 'RMM':
                if cutoff[0] == '0':
                    return val
                else:
                    return val + 1
            case '_':
                raise NotImplemented


    def __add__(self, other):
        assert(isinstance(other, RV_Float))
        assert(self.rm == other.rm)
        if self.isNaN() or other.isNaN():
            res = RV_Float(rm=self.rm)
            res.makeNaN()
            return res
        elif (self.exponent < other.exponent):
            return other.__add__(self)
        else:
            exp_diff = self.exponent - other.exponent
            x = self.mantissa << exp_diff
            if self.sign == 1:
                x = -x
            x += other.mantissa if other.sign == 0 else -other.mantissa
            new_sign = int(x < 0)
            if new_sign:
                x = -x

            x, new_exp = self._normalise(x, other.exponent, new_sign)
            res = RV_Float(rm=self.rm)
            res.sign = new_sign
            res.exponent = new_exp
            res.mantissa = x
            return res



    def __sub__(self, other):
        assert(isinstance(other, RV_Float))
        if self.isNaN() or other.isNaN():
            res = RV_Float(rm=self.rm)
            res.makeNaN()
            return res
        other.sign = (other.sign + 1) % 2
        res = self + other
        other.sign = (other.sign + 1) % 2
        return res

    def __mul__(self, other):
        assert(isinstance(other, RV_Float))
        if self.isNaN() or other.isNaN():
            res = RV_Float(rm=self.rm)
            res.makeNaN()
            return res
        res = RV_Float()
        res.sign = 1 if self.sign + other.sign == 1 else 0
        res.exponent = self.exponent + other.exponent
        res.mantissa = self.mantissa * other.mantissa
        res.mantissa, res.exponent = self._normalise(res.mantissa, res.exponent-23, res.sign)
        return res

    def __truediv__(self, other):
        assert(isinstance(other, RV_Float))
        if self.isNaN() or other.isNaN():
            res = RV_Float(rm=self.rm)
            res.makeNaN()
            return res
        res = RV_Float()
        res.sign = 1 if self.sign + other.sign == 1 else 0
        res.exponent = self.exponent - other.exponent
        res.mantissa = self.mantissa // other.mantissa # TODO inexact
        res.mantissa, res.exponent = self._normalise(res.mantissa, res.exponent, res.sign)
        return res

    def __neg__(self):
        res = RV_Float()
        res.sign = (self.sign+1) % 2
        res.exponent = self.exponent
        res.mantissa = self.mantissa
        return res

    def sqrt(self):
        if self.isNaN() or self.sign == 1:
            res = RV_Float(rm=self.rm)
            res.makeNaN()
            return res
        # Optional TODO: Replace this with IEEE sqrt method and add inexact
        res = RV_Float()
        res.sign, res.exponent, res.mantissa = RV_Float._convert(math.sqrt(self.value()))
        return res


    def __lt__(self, other):
        assert(isinstance(other, RV_Float))
        if self.isNaN() or other.isNaN():
            return False
        elif self.sign < other.sign:
            return False
        elif self.sign > other.sign:
            return True
        else:
            if self.exponent < other.exponent:
                return self.sign == 0
            elif self.exponent > other.exponent:
                return self.sign != 0
            else:
                if self.mantissa < other.mantissa:
                    return self.sign == 0
                else:
                    return self.sign != 0

    def __le__(self, other):
        assert(isinstance(other, RV_Float))
        if self.isNaN() or other.isNaN():
            return False
        if self == other:
            return True
        else:
            return self < other

    def __eq__(self, other):
        assert(isinstance(other, RV_Float))
        if self.isNaN() or other.isNaN():
            return False
        return self.bits == other.bits

    def __int__(self):
        val = self.value()
        ival = int(val)
        if ival == val:
            self.inexact = False
            return ival
        match self.rm:
            case 'RNE':
                if val % 1 == 0.5:
                    return ival + (ival % 2)
                elif val % 1 < 0.5:
                    return ival
                else:
                    return ival
            case 'RTZ':
                return ival
            case 'RDN':
                return ival - int(ival < 0)
            case 'RUP':
                return ival + int(ival > 0)
            case 'RMM':
                return ival + (1 if ival > 0 else -1)



NaN = RV_Float()
NaN.makeNaN()

# TODO: Make Signalling NaNs work
sNaN = RV_Float()
sNaN.makeNaN()
sNaN.mantissa = 5 << 21

class Float(Module):

    opcodes, inv_opcodes, csrs, pseudos = read_yaml('float.yaml', ret_CSR=True, ret_Pseudos=True)

    def _cur_rm(self, instr_rm=None):
        if instr_rm is None or instr_rm == 'DYN':
            print(self.fcsr)
            print(self.fcsr & 224)
            print((self.fcsr & 224) >> 5)
            return rm[(self.fcsr & 224) >> 5]
        else:
            return instr_rm
    def _cur_NV(self):
        return self.fcsr & 16 != 0
    def _cur_DZ(self):
        return self.fcsr & 8 != 0
    def _cur_OF(self):
        return self.fcsr & 4 != 0
    def _cur_UF(self):
        return self.fcsr & 2 != 0
    def _cur_NX(self):
        return self.fcsr & 1 != 0

    def set_NX(self, value):
        if value:
            self.fcsr |= 1
        elif self.fcsr & 1 == 1:
            self.fcsr -= 1

    def _set_rm(self, value):
        rm_bin = inv_rm[value] << 5
        self.fcsr -= (self.fcsr & 224)
        self.fcsr |= rm_bin

    def _set_freg(self, freg: int, val: RV_Float):
        assert(0 <= freg < self.state['FLEN'])
        # possible improvement: convert to IEEE 754 using rounding mode
        self.state['fregs'][freg] = val

    def _read_freg(self, freg: int, rm=None) -> RV_Float:
        assert(0 <= freg < self.state['FLEN'])
        self.state['fregs'][freg].rm = self._cur_rm(rm)
        return self.state['fregs'][freg]


    def __init__(self, state):
        super().__init__(state)

        state['fregs'] = [RV_Float() for _ in range(32)]
        state['FLEN'] = 32
        state['pseudos'] |= self.pseudos

        self.fcsr = 0
        if 'CSR' in state['modules']:
            state['csrs'][Float.csrs['fcsr']] = (lambda _: self.fcsr,
                                            lambda x: setattr(self, 'fcsr', x % 256))

            state['csrs'][Float.csrs['fflags']] = (lambda _: self.fcsr % 32,
                                            lambda x: setattr(self, 'fcsr', (self.fcsr // 32)*32 + x%32))

            state['csrs'][Float.csrs['ffrm']] = (lambda _: (self.fcsr // 32) % 8,
                                            lambda x: setattr(self, 'fcsr', (self.fcsr // 256)*256 + x*32 + (self.fcsr % 32)))
            print(state['csrs'])

            if state['debug']:
                print("Using module Float, which has detected module CSR.")

        elif state['debug']:
            print("Using module Float")
            print("Warning: Float module cannot use its CSR, as there is no CSR module.")

    def assemble(self, *instr):
        try:
            opcode, instr_type, args = Float.inv_opcodes[instr[0]]
            args = args.copy()
        except KeyError:
            return False, None
        match instr_type:
            case 'R':
                if (14, 12) in args.keys():
                    if len(instr) == 3:
                        return True, self.assemble_32(opcode, rd=Module.reg(instr[1]), rs1=Module.reg(instr[2]), flags=args)
                    elif len(instr) == 4:
                        return True, self.assemble_32(opcode, rd=Module.reg(instr[1]), rs1=Module.reg(instr[2]), rs2=Module.reg(instr[3]), flags=args)
                    else:
                        return False, None

                else:
                    # rounding mode needs spec
                    args[(14,12)] = inv_rm[instr[-1]]
                    if len(instr) == 4:
                        return True, self.assemble_32(opcode, rd=Module.reg(instr[1]), rs1=Module.reg(instr[2]), flags=args)
                    elif len(instr) == 5:
                        return True, self.assemble_32(opcode, rd=Module.reg(instr[1]), rs1=Module.reg(instr[2]),
                                               rs2=Module.reg(instr[3]), flags=args)
                    else:
                        return False, None
            case 'I':
                return True, Module.assemble_I(opcode, instr[1], instr[2], int(instr[3]), flags=args)
            case 'S':
                return True, Module.assemble_S(opcode, instr[2], instr[1], int(instr[3]), flags=args)
            case 'R4':
                args[(14, 12)] = inv_rm[instr[-1]]
                return True, Module.assemble_32(opcode, rd=Module.reg(instr[1]), rs1=Module.reg(instr[2]),
                                                rs2=Module.reg(instr[3]), rs3=Module.reg(instr[4]), flags=args)
            case _:
                return False, None

    def unassemble(self, binary):
        assert len(binary) == 32
        opcode, rd, rs1, rs2, funct3, funct7 = super()._decompile_commons(binary)
        funct5, width = divmod(funct7, 4)
        try:
            args1 = Float.opcodes[opcode]
        except KeyError:
            return False, None

        instr_type = args1[1]

        try:
            match instr_type:
                case 'R': # OP-FP
                    assert args1[-2] == [26, 25]
                    assert args1[-1][width][-2] == [31,27]
                    postcheck5 = args1[-1][width][-1][funct5]
                    if type(postcheck5) is str:
                        return True, (postcheck5, rd, rs1, rs2, rm[funct3])
                    else:
                        match postcheck5[-2]:
                            case [14, 12]:
                                return True, (postcheck5[-1][funct3], rd, rs1, rs2)
                            case [24, 20]:
                                postcheckr2 = postcheck5[-1][rs2]
                                if type(postcheckr2) is str:
                                    return True, (postcheckr2, rd, rs1, rm[funct3])
                                else:
                                    assert postcheckr2[-2] == [14, 12]
                                    return True, (postcheckr2[-1][funct3], rd, rs1)

                case 'I': # loads
                    assert args1[-2] == [14, 12]
                    imm = Module.twos_to_int(binary[0:12])
                    return True, (args1[-1][funct3], rd, rs1, imm)
                case 'S': # stores
                    assert args1[-2] == [14, 12]
                    imm = Module.twos_to_int(binary[0:7] + binary[20:25])
                    return True, (args1[-1][funct3], rs2, rs1, imm)
                case 'R4': # FMADD.S and similar
                    assert args1[-2] == [26, 25]
                    return True, (args1[-1][width], rd, rs1, rs2, int(binary[:5], 2), rm[funct3])
        except KeyError:
            pass
        return False, None

    def run_instr(self, instr):

        instr, *args = instr
        match instr:
            # OP-FP
            case 'fadd.s':
                r = args[3]
                self._set_freg(args[0], self._read_freg(args[1], r) + self._read_freg(args[2], r))
                if self._read_freg(args[0]).inexact:
                    self.set_NX(1)
            case 'fsub.s':
                r = args[3]
                self._set_freg(args[0], self._read_freg(args[1], r) - self._read_freg(args[2], r))
                if self._read_freg(args[0]).inexact:
                    self.set_NX(1)
            case 'fmul.s':
                r = args[3]
                self._set_freg(args[0], self._read_freg(args[1], r) * self._read_freg(args[2], r))
                if self._read_freg(args[0]).inexact:
                    self.set_NX(1)
            case 'fdiv.s':
                r = args[3]
                f2 = self._read_freg(args[2], r)
                if f2.value() == 0.0:
                    # set NaN bit
                    self._set_freg(args[0], NaN)
                    pass
                else:
                    self._set_freg(args[0], self._read_freg(args[1], r) / f2)
            case 'fsgnj.s':
                new_val = self._read_freg(args[0])
                new_val.sign = self._read_freg(args[1]).sign
                self._set_freg(args[0], new_val)
            case 'fsgnjn.s':
                new_val = self._read_freg(args[0])
                new_val.sign = (self._read_freg(args[1]).sign + 1) % 2
                self._set_freg(args[0], new_val)
            case 'fsgnjx.s':
                new_val = self._read_freg(args[0])
                new_val.sign = int(self._read_freg(args[1]).sign != new_val.sign)
                self._set_freg(args[0], new_val)
            case 'fsqrt.s':
                r = args[2]
                self._set_freg(args[0], self._read_freg(args[1], r).sqrt())
                if self._read_freg(args[0], r).inexact:
                    self.set_NX(1)
            case 'fmin.s':
                self._set_freg(args[0], min(self._read_freg(args[1]), self._read_freg(args[2])))
            case 'fmax.s':
                self._set_freg(args[0], max(self._read_freg(args[1]), self._read_freg(args[2])))
            case 'fle.s':
                self._set_reg(args[0], 1.0 if self._read_freg(args[1]) < self._read_freg(args[2]) else 0.0)
            case 'flt.s':
                self._set_reg(args[0], 1.0 if self._read_freg(args[1]) <= self._read_freg(args[2]) else 0.0)
            case 'feq.s':
                self._set_reg(args[0], 1.0 if self._read_freg(args[1]) == self._read_freg(args[2]) else 0.0)
            case 'fcvt.s.w':
                r = args[2]
                self._set_freg(args[0], RV_Float(self._read_reg(args[1]), rm=r))
            case 'fcvt.s.wu':
                r = args[2]
                self._set_freg(args[0], RV_Float(self._read_reg(args[1], signed=False), rm=r))
            case 'fcvt.w.s':
                x = self._read_freg(args[1], args[2])
                ix = int(x)
                if x.inexact:
                    self.set_NX(1)
                if ix >= (1 << 30):
                    ix = (1 << 30) - 1
                    self.set_NX(1)
                elif ix < (-1 << 30):
                    ix = -1 << 30
                    self.set_NX(1)
                self._set_reg(args[0], ix)
            case 'fcvt.wu.s':
                x = self._read_freg(args[1], args[2])
                ix = int(x)
                if x.inexact:
                    self.set_NX(1)
                if ix >= (1 << 31):
                    ix = (1 << 31) - 1
                    self.set_NX(1)
                elif ix < 0:
                    ix = 0
                    self.set_NX(1)
                self._set_reg(args[0], ix)
            case 'fmv.x.w':
                self._set_reg(args[0], Module.twos_to_int(self._read_freg(args[1]).bits()))
            case 'fclass.s':
                x = self._read_freg(args[1])
                res = 1
                if x.isNaN():
                    # quiet, signalling, or infinity
                    match (x.mantissa - (1 << 23)):
                        case 0:
                            # infinity
                            if x.sign:
                                pass # res already correct
                            else:
                                res <<= 7
                        case 4194304: # 1<<22
                            # quiet
                            res <<= 9
                        case _:
                            res <<= 8
                else:
                    if x.value() == 0.0:
                        if x.sign:
                            res <<= 3
                        else:
                            res <<= 4
                    elif x.exponent == -127 and x.mantissa < (1<<23):
                        #subnormal number
                        if x.sign:
                            res <<= 2
                        else:
                            res <<= 5
                    else:
                        # normal number
                        if x.sign:
                            res <<= 1
                        else:
                            res <<= 6
                self._set_reg(args[0], res)

            case 'fmadd.s':
                r = args[4]
                self._set_freg(args[0], (self._read_freg(args[1], r) * self._read_freg(args[2], r)) + self._read_freg(args[3], r))
                if self._read_freg(args[0]).inexact:
                    self.set_NX(1)
            case 'fmsub.s':
                r = args[4]
                self._set_freg(args[0], (self._read_freg(args[1], r) * self._read_freg(args[2]), r) - self._read_freg(args[3]), r)
                if self._read_freg(args[0]).inexact:
                    self.set_NX(1)
            case 'fnmsub.s':
                r = args[4]
                self._set_freg(args[0], -(self._read_freg(args[1], r) * self._read_freg(args[2], r)) + self._read_freg(args[3], r))
                if self._read_freg(args[0]).inexact:
                    self.set_NX(1)
            case 'fnmadd.s':
                r = args[4]
                self._set_freg(args[0], -(self._read_freg(args[1], r) * self._read_freg(args[2], r)) - self._read_freg(args[3], r))
                if self._read_freg(args[0]).inexact:
                    self.set_NX(1)
            case 'flw':
                loc, offset = divmod(self._read_reg(args[1]) + args[2], 4)
                assert offset == 0
                self._set_freg(args[0], RV_Float.from_bits(self.state['mem'][loc]))
            case 'fsw':
                loc, offset = divmod(self._read_reg(args[0]) + args[2], 4)
                assert offset == 0
                self.state['mem'][loc] = self._read_freg(args[1]).bits()


    def reset_module(self):
        self.state['fregs'] = [RV_Float() for _ in range(32)]



if __name__ == '__main__':
    x1 = float(input("float 1: "))
    x2 = float(input("float 2: "))
    res = RV_Float(x1) + RV_Float(x2)
    print("final res: ", res.bits(), res.value())
    print("should be: ", RV_Float(x1+x2).bits())
