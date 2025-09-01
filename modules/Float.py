import math

from modules.RISC_modules import *
from yamls.read_yaml import read_yaml
import struct

rm = {'000': 'RNE', '001': 'RTZ', '010': 'RDN', '011': 'RUP', '100': 'RMM', '111': 'DYN'}
inv_rm = {v: k for k, v in rm.items()}

class RV_Float:
    """Float stored in IEEE 754 format"""

    @staticmethod
    def _convert(num: float) -> (int, int, int):
        bits, = struct.unpack('!I', struct.pack('!f', num))
        bits = "{:032b}".format(bits)
        sign = int(bits[0])
        exponent = int(bits[1:9], 2)-127
        mantissa = int('1'+bits[9:], 2)
        return sign, exponent, mantissa

    @staticmethod
    def _normalise(mantissa, exponent, sign, rm='RNE'):
        """
        :param mantissa: In explicit form, with the 1
        :return: rounded mantissa (without the 1st 1), exponent
        """
        cutoff = ''
        while mantissa >= (1 << 24):
            exponent += 1
            cutoff = str(mantissa % 2) + cutoff
            mantissa >>= 1
        while mantissa < (1 << 23):
            exponent -= 1
            mantissa <<= 1
        return RV_Float.round(mantissa, cutoff, sign, rm=rm), exponent


    def __init__(self, value=0.0, rm='RNE'):
        self.rm = rm
        self.sign, self.exponent, self.mantissa = RV_Float._convert(value)
        print(self.bits(), self.value())

    def bits(self):
        """Return the float as 32 bits"""
        return str(self.sign) + bin(self.exponent+127)[2:].rjust(8, '0') + bin(self.mantissa)[3:].rjust(23, '0')

    def value(self) -> float:
        """Return the float in python float format"""
        if '1' not in self.bits()[1:]:
            return 0.0
        return pow(-1, self.sign) * pow(2, self.exponent-23) * self.mantissa

    def makeNaN(self):
        self.sign = 0
        self.mantissa = 1 << 23
        self.exponent = 128

    def isNaN(self):
        return self.mantissa == 1<<23 and self.exponent == 128

    def set_rm(self, rm):
        self.rm = rm

    @staticmethod
    def round(val, cutoff, sign, rm='RNE'):
        if cutoff == '':
            return val
        match rm:
            case 'RNE':
                if cutoff[0] == '0' or (len(cutoff) > 1 and '1' in cutoff[1:]):
                    return val
                return val+1
            case 'RTZ':
                return val
            case 'RDN':
                return val if sign == '0' else val+1
            case 'RUP':
                return val+1 if sign == '0' else val
            case '_':
                raise NotImplemented


    def __add__(self, other):
        assert(isinstance(other, RV_Float))
        assert(self.rm == other.rm)
        if (self.exponent < other.exponent):
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

            x, new_exp = self._normalise(x, other.exponent, new_sign, rm=self.rm)
            res = RV_Float(rm=self.rm)
            res.sign = new_sign
            res.exponent = new_exp
            res.mantissa = x
            return res



    def __sub__(self, other):
        assert(isinstance(other, RV_Float))
        other.sign = (other.sign + 1) % 2
        res = self + other
        other.sign = (other.sign + 1) % 2
        return res

    def __mul__(self, other):
        assert(isinstance(other, RV_Float))
        res = RV_Float()
        res.sign = 1 if self.sign + other.sign == 1 else 0
        res.exponent = self.exponent + other.exponent
        res.mantissa = self.mantissa * other.mantissa
        res.mantissa, res.exponent = self._normalise(res.mantissa, res.exponent-23, res.sign, rm=self.rm)
        return res

    def __truediv__(self, other):
        assert(isinstance(other, RV_Float))
        res = RV_Float()
        res.sign = 1 if self.sign + other.sign == 1 else 0
        res.exponent = self.exponent - other.exponent
        res.mantissa = self.mantissa // other.mantissa
        return res

    def sqrt(self):
        # Optional TODO: Replace this with IEEE sqrt method
        res = RV_Float()
        res.sign, res.exponent, res.mantissa = RV_Float._convert(math.sqrt(self.value()))
        return res


    def __lt__(self, other):
        assert(isinstance(other, RV_Float))
        if self.sign < other.sign:
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
        if self == other:
            return True
        else:
            return self < other

    def __eq__(self, other):
        assert(isinstance(other, RV_Float))
        return self.bits == other.bits



NaN = RV_Float()
NaN.makeNaN()

class Float(Module):

    opcodes, inv_opcodes, csrs = read_yaml('float.yaml', ret_CSR=True)
    NaN = 'Canonical NaN'

    def _cur_rm(self):
        return rm[bin(self.fcsr & 224, 2)[2:].rjust(3, '0')]
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

    def _set_freg(self, freg: int, val: RV_Float):
        assert(0 <= freg < self.state['FLEN'])
        # possible improvement: convert to IEEE 754 using rounding mode
        self.state['fregs'][freg] = val

    def _read_freg(self, freg: int) -> RV_Float:
        assert(0 <= freg < self.state['FLEN'])
        return self.state['fregs'][freg]


    def __init__(self, state):
        super().__init__(state)

        state['fregs'] = [RV_Float() for _ in range(32)]
        state['FLEN'] = 32

        self.fcsr = 0
        if 'CSR' in state['modules']:
            state['csrs'][Float.csrs['fcsr']] = (lambda _: self.fcsr,
                                            lambda x: setattr(self, 'fcsr', x % 256))

            state['csrs'][Float.csrs['fflags']] = (lambda _: self.fcsr % 32,
                                            lambda x: setattr(self, 'fcsr', (self.fcsr // 32)*32 + x%32))

            state['csrs'][Float.csrs['ffrm']] = (lambda _: (self.fcsr // 32) % 8,
                                            lambda x: setattr(self, 'fcsr', (self.fcsr // 256)*256 + x*32 + (self.fcsr % 32)))

        elif state['debug']:
            print("Warning: Float module cannot use its CSR, as there is no CSR module.")

    def assemble(self, *instr):
        try:
            opcode, instr_type, args = Float.inv_opcodes[instr[0]]
        except KeyError:
            return False, None
        match instr_type:
            case 'R':
                if (14, 12) in args.keys():
                    if len(instr) != 4:
                        return False, None
                else:
                    # rounding mode needs spec
                    if len(instr) != 5:
                        return False, None
                    args[(14,12)] = inv_rm[instr[-1]]
                return True, self.assemble_32(opcode, rd=Module.reg(instr[1]), rs1=Module.reg(instr[2]),
                                       rs2=Module.reg(instr[3]), flags=args)
            case 'I':
                return True, Module.assemble_I(opcode, instr[1], instr[2], int(instr[3]), flags=args)
            case 'R4':
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
                        # funct3 is rounding mode
                        return True, (postcheck5, rd, rs1, rs2, rm[funct3])
                    else:
                        # funct3 is further deliminator
                        assert postcheck5[-2] == [14,12]
                        return True, (postcheck5[-1][funct3], rd, rs1, rs2)
                case 'I': # loads and stores
                    assert args1[-2] == [14, 12]
                    imm = Module.twos_to_int(binary[0:12])
                    return True, (args1[-1][funct3], rd, rs1, imm)
                case 'R4': # FMADD.S and similar
                    return True, (args1[0], rd, rs1, rs2, int(binary[:5], 2))
        except KeyError:
            pass
        return False, None

    def run_instr(self, instr):

        # TODO rounding, NaN
        instr, *args = instr
        match instr:
            # OP-FP
            case 'fadd.s':
                self._set_freg(args[0], self._read_freg(args[1]) + self._read_freg(args[2]))
            case 'fsub.s':
                self._set_freg(args[0], self._read_freg(args[1]) - self._read_freg(args[2]))
            case 'fmul.s':
                self._set_freg(args[0], self._read_freg(args[1]) * self._read_freg(args[2]))
            case 'fdiv.s':
                if float(args[2]) == 0.0:
                    # set NaN bit
                        # TODO
                    pass
                else:
                    self._set_freg(args[0], self._read_freg(args[1]) / self._read_freg(args[2]))
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
                self._set_freg(args[0], self._read_freg(args[1]).sqrt())
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
                self._set_freg(args[0], RV_Float(self._read_reg(args[1])))
            case 'fcvt.s.wu':
                self._set_freg(args[0], RV_Float(self._read_reg(args[1], signed=False)))
            case 'fcvt.w.s':
                x = self._read_freg(args[1])
                if x >= (1 << 30):
                    # set inexact TODO
                    self._set_reg(args[0], (1 << 30)-1)
                elif x < (-1 << 30):
                    # set inexact TODO
                    self._set_reg(args[0], (-1 << 30))
                elif int(x.value()) == x.value():
                    self._set_reg(args[0], int(x.value()))
                else:
                    match self._cur_rm():
                        case 'RNE':
                            if x % 1 == 0.5:
                                # nearest even
                                x = math.floor(x) if math.floor(x) % 2 == 0 else math.ceil(x)
                            else:
                                x = round(x)
                        case 'RTZ':
                            x = int(x)
                        case 'RDN':
                            x = math.floor(x)
                        case 'RUP':
                            x = math.ceil(x)
                        case 'RMM':
                            if x % 1 == 0.5:
                                # nearest max mag
                                x = math.ceil(x) if x > 0 else math.floor(x)
                            else:
                                x = round(x)
                        case _:
                            raise RVError(f"Float module: Unknown/Reserved RM {self._cur_rm()}")
                    # set inexact TODO
                    self._set_reg(args[0], x)
            case 'fcvt.wu.s':
                x = self._read_freg(args[1])
                if x >= (1 << 31):
                    # set inexact TODO
                    self._set_reg(args[0], (1 << 31) - 1)
                elif x < 0:
                    # set inexact TODO
                    self._set_reg(args[0], 0)
                elif int(x) == x:
                    self._set_reg(args[0], int(x))
                else:
                    match self._cur_rm():
                        case 'RNE':
                            if x % 1 == 0.5:
                                # nearest even
                                x = math.floor(x) if math.floor(x) % 2 == 0 else math.ceil(x)
                            else:
                                x = round(x)
                        case 'RTZ' | 'RDN':
                            x = int(x)
                        case 'RUP' | 'RMM':
                            x = math.ceil(x)
                        case _:
                            raise RVError(f"Float module: Unknown/Reserved RM {self._cur_rm()}")
                    # set inexact TODO
                    self._set_reg(args[0], x)
            case 'fmv.x.w':
                self._set_reg(args[0], struct.unpack('i', struct.pack('!f', self._read_freg(args[1])))[0])
            case 'fclass':
                x = self._read_freg(args[1])
                # cases TODO


            # TODO fclass, other opcodes




if __name__ == '__main__':
    x1 = float(input("float 1: "))
    x2 = float(input("float 2: "))
    res = RV_Float(x1) * RV_Float(x2)
    print("final res: ", res.bits(), res.value())
    print("should be: ", RV_Float(x1*x2).bits())
