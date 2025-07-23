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

    # assemble methods
    @staticmethod
    def assemble_R(opcode, rd, funct3, rs1, rs2, funct7):
        return ''.join(
            [Module.twos_comp_bin(*i) for i in
             [(funct7, 7), (rs2, 5), (rs1, 5), (funct3, 3), (rd, 5), (opcode, 7)]])

    @staticmethod
    def assemble_I(opcode, rd, funct3, rs1, imm):
        if not -2048 <= imm <= 2047:
            raise AssemblerError(f"Invalid immediate value for I-type instr: {imm}")
        return ''.join(
            [Module.twos_comp_bin(*i) for i in [(imm, 12), (rs1, 5), (funct3, 3), (rd, 5), (opcode, 7)]])

    @staticmethod
    def assemble_S(opcode, funct3, rs1, rs2, imm):
        if not -2048 <= imm <= 2047:
            raise AssemblerError(f"Invalid immediate value for S-type instr: {imm}")
        imm_bin = Module.twos_comp_bin(imm, bits=12)
        return (imm_bin[:7] + ''.join([Module.twos_comp_bin(*i) for i in [(rs2, 5), (rs1, 5), (funct3, 3)]])
                + imm_bin[7:] + Module.twos_comp_bin(opcode, 7))

    @staticmethod
    def assemble_B(opcode, funct3, rs1, rs2, imm):
        if not imm % 2 == 0:
            raise AssemblerError(
                f"Invalid immediate value for B-type instr: {imm}\nValue should be an even number.")
        imm = imm // 2
        if not -2048 <= imm <= 2047:
            raise AssemblerError(f"Invalid immediate value for B-type instr: {imm}")
        imm_bin = Module.twos_comp_bin(imm, bits=12)
        return imm_bin[0] + imm_bin[2:8] + ''.join(
            [Module.twos_comp_bin(*i) for i in [(rs2, 5), (rs1, 5), (funct3, 3)]]) + imm_bin[8:] + imm_bin[
            1] + Module.twos_comp_bin(opcode, 7)

    @staticmethod
    def assemble_U(opcode, rd, imm):
        if not (imm >> 12) << 12 == imm:
            raise AssemblerError(
                f"Invalid immediate value for U-type instr: {imm}\nValue should be a multiple of 2^12")
        imm = imm >> 12
        if not -524288 <= imm <= 524287:
            raise AssemblerError(f"Invalid immediate value for U-type instr: {imm}")
        return ''.join([Module.twos_comp_bin(*i) for i in [(imm, 20), (rd, 5), (opcode, 7)]])

    @staticmethod
    def assemble_J(opcode, rd, imm):
        if not imm % 2 == 0:
            raise AssemblerError(
                f"Invalid immediate value for J-type instr: {imm}\nValue should be an even number.")
        imm = imm >> 1
        if not -524288 <= imm <= 524287:
            raise AssemblerError(f"Invalid immediate value for J-type instr: {imm}")
        imm_bin = Module.twos_comp_bin(imm, bits=20)
        return imm_bin[0] + imm_bin[10:] + imm_bin[9] + imm_bin[1:9] + ''.join(
            [Module.twos_comp_bin(*i) for i in [(rd, 5), (opcode, 7)]])

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