"""
RISC_modules.py
provides abstract Module class with useful static methods, and defines exceptions
"""


from abc import abstractmethod
from yamls.read_yaml import read_yaml

class AssemblerError(Exception):
    """
    Exception raised when the compiler fails
    """
    
class RVError(Exception):
    """
    Exception raised when the RISC-V simulation hits an error,
    e.g.: an invalid opcode
    """
    pass


class Module:

    order = 1

    def __init__(self, state):
        self.state = state

    @abstractmethod
    def unassemble(self, binary):
        """
        turn a binary instruction into decoded assembly version
        :param binary: string of 1s and 0s, length of an instruction
        :return: (success, decoded) - (Boolean, String)
        """
        pass

    @abstractmethod
    def assemble(self, *instr):
        """
        turn assembly instruction into a binary string
        :param instr: first word of the instruction e.g. addi
        :param args: remaining args, e.g. x12, <imm_value>
        :return: (success, binary string) - (Boolean, String)
        """
        pass
    
    @abstractmethod
    def run_instr(self, instr):
        """
        Run a single instruction of this module
        :param instr: instructions in assembly list form
        """
        pass
    
    # useful methods

    @staticmethod
    def reg(x):
        if type(x) is int:
            val = x
        else:
            assert x[0] == 'x'
            val = int(x[1:])
        if not 0 <= val <= 31:
            raise AssemblerError(f"Invalid reg value: {x}")
        return val

    @staticmethod
    def twos_comp_bin(val: int, bits):
        """compute the 2's complement of int value val"""
        if val < 0:
            return bin((-val - 1) ^ ((2 << (bits - 1)) - 1))[2:]

        return bin(val)[2:].rjust(bits, '0')

    @staticmethod
    def twos_to_int(val: str):
        if val[0] == '0':
            return int(val, 2)
        else:
            return int(val, 2) - (1 << len(val))

    @staticmethod
    def _decompile_commons(binary: str):
        return int(binary[25:], 2), int(binary[20:25], 2), int(binary[12:17], 2), int(binary[7:12], 2), int(
            binary[17:20], 2), int(binary[:7], 2)

    # # assemble methods
    # @staticmethod
    # def assemble_R(opcode, rd, funct3, rs1, rs2, funct7):
    #

    @staticmethod
    def assemble_I(opcode, rd, rs1, imm, flags):
        if not -2048 <= imm <= 2047:
            raise AssemblerError(f"Invalid immediate value for I-type instr: {imm}")
        flags[(31,20)] = Module.twos_comp_bin(imm, bits=12)
        return Module.assemble_32(opcode, Module.reg(rd), Module.reg(rs1), flags=flags)

    @staticmethod
    def assemble_S(opcode, rs1, rs2, imm, flags):
        if not -2048 <= imm <= 2047:
            raise AssemblerError(f"Invalid immediate value for S-type instr: {imm}")
        imm_bin = Module.twos_comp_bin(imm, bits=12)
        flags[(31,25)] = imm_bin[0:7]
        flags[(11,7)] = imm_bin[7:]
        return Module.assemble_32(opcode, rs1=Module.reg(rs1), rs2=Module.reg(rs2), flags=flags)

    @staticmethod
    def assemble_B(opcode, rs1, rs2, imm, flags):
        if not imm % 2 == 0:
            raise AssemblerError(
                f"Invalid immediate value for B-type instr: {imm}\nValue should be an even number.")
        imm = imm >> 1
        if not -2048 <= imm <= 2047:
            raise AssemblerError(f"Invalid immediate value for B-type instr: {imm}")
        imm_bin = Module.twos_comp_bin(imm, bits=12)
        flags[(31,31)] = imm_bin[0]
        flags[(30,25)] = imm_bin[2:8]
        flags[(11,8)] = imm_bin[8:]
        flags[(7,7)] = imm_bin[1]
        return Module.assemble_32(opcode, rs1=Module.reg(rs1), rs2=Module.reg(rs2), flags=flags)

    @staticmethod
    def assemble_U(opcode, rd, imm):
        if not (imm >> 12) << 12 == imm:
            raise AssemblerError(
                f"Invalid immediate value for U-type instr: {imm}\nValue should be a multiple of 2^12")
        imm = imm >> 12
        if not -524288 <= imm <= 524287:
            raise AssemblerError(f"Invalid immediate value for U-type instr: {imm}")
        return Module.assemble_32(opcode, rd=Module.reg(rd), flags={(31,12): Module.twos_comp_bin(imm, bits=20)})

    @staticmethod
    def assemble_J(opcode, rd, imm):
        if not imm % 2 == 0:
            raise AssemblerError(
                f"Invalid immediate value for J-type instr: {imm}\nValue should be an even number.")
        imm = imm >> 1
        if not -524288 <= imm <= 524287:
            raise AssemblerError(f"Invalid immediate value for J-type instr: {imm}")
        imm = Module.twos_comp_bin(imm, bits=20)
        return Module.assemble_32(opcode, rd=Module.reg(rd),
                                        flags={(31, 31): imm[0], (30, 21): imm[10:], (20, 20): imm[9],
                                               (19, 12): imm[1:9]})

    @staticmethod
    def assemble_32(opcode, rd=None, rs1=None, rs2=None, flags=None):
        res = ['u' for _ in range(32)]
        if type(opcode) is int:
            opcode = bin(opcode)[2:].zfill(7)
        res[-7:] = opcode
        if rd is not None:
            if type(rd) is int:
                rd = bin(rd)[2:].zfill(5)
            assert len(rd) == 5
            res[20:25] = rd
        if rs1 is not None:
            if type(rs1) is int:
                rs1 = bin(rs1)[2:].zfill(5)
            assert len(rs1) == 5
            res[12:17] = rs1
        if rs2 is not None:
            if type(rs2) is int:
                rs2 = bin(rs2)[2:].zfill(5)
            assert len(rs2) == 5
            res[7:12] = rs2
        if flags is not None:
            for index, data in flags.items():
                if type(data) is int:
                    data = bin(data)[2:].zfill(1+index[0]-index[1])

                res[31-index[0]:32-index[1]] = data
        assert 'u' not in res
        return ''.join(res)


    def _set_reg(self, reg, val):
        """
        Protected method
        Use this instead of manually writing to the self.reg list to neatly catch errors and deal with special registers (e.g. 0)
        :param reg: Integer val of register
        :param val: Integer value to write
        :return: None if successful, otherwise RVError exception
        """
        if not (0 <= reg < self.state['reg_count']):
            raise RVError(f"Invalid register write: x{reg}")
        if reg != 0:  # reg 0 never changes
            self.state['regs'][reg] = val

    def _read_reg(self, reg, signed=True):
        """
        Protected method
        Use instead of manually reading the self.reg list to neatly catch errors
        :param reg: Integer val of register
        :param signed: Read as signed value (Default) as opposed to unsigned
        :return: reg value as Integer if successful, otherwise RVError exception
        """
        if not (0 <= reg < self.state['reg_count']):
            raise RVError(f"Invalid register read: x{reg}")
        if signed:
            return self.state['regs'][reg]
        else:
            return self.state['regs'][reg] % (1 << ((self.state['word_size'] * 8) - 1))