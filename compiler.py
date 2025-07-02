# compiler.py: translate between Assembly shorthands and binary


import yamls.read_yaml as read_yaml

print(read_yaml.opcodes)

class Compiler:

    @staticmethod
    def twos_comp_bin(val : int, bits):
        """compute the 2's complement of int value val"""
        if val < 0:
            return bin( ~val )[2:].rjust(bits, '1')

        return bin(val)[2:].rjust(bits, '0')

    @staticmethod
    def twos_to_int(val: str):
        if val[0] == '0':
            return int(val, 2)
        else:
            return int(val, 2) - (1 << len(val))



    def compile_file(self, filename):
        code = open(filename, mode='r').read().splitlines()
        return [self.compile_instr(line) for line in code]


    def compile_instr(self, instr):
        pass

    def decompile_instr(self, binary):
        pass


#implementations of specific RISCV architectures
class Comp_RV32I(Compiler):

    @staticmethod
    def decompile_from_base_type(bin: str, instr_type: str):
        # assert instr_type in Comp_RV32I.base_instr_types
        assert len(bin) == 32

        opcode = Compiler.twos_to_int(bin[25:])
        rd = int(bin[20:25], 2)
        funct3 = Compiler.twos_to_int(bin[17:20])
        rs1 = int(bin[12:17], 2)
        rs2 = int(bin[7:12], 2)
        funct7 = Compiler.twos_to_int(bin[:7])

        match instr_type:
            case 'R':
                return opcode, rd, funct3, rs1, rs2, funct7
            case 'I':
                return opcode, rd, funct3, rs1, Compiler.twos_to_int(bin[7:12])
            case 'S':
                imm = Compiler.twos_to_int(bin[:7] + bin[20:25])
                return opcode, funct3, rs1, rs2, imm
            case 'B':
                imm = Compiler.twos_to_int(bin[0] + bin[24] + bin[1:7] + bin[20:24])
                return opcode, funct3, rs1, rs2, imm
            case 'U':
                return opcode, rd, Compiler.twos_to_int(bin[:20])
            case 'J':
                imm = Compiler.twos_to_int(bin[0] + bin[12:20] + bin[11] + bin[1:11])
                return opcode, rd, imm
        return None

    @staticmethod
    def compile_R(opcode, rd, funct3, rs1, rs2, funct7):
        return ''.join(
            [Compiler.twos_comp_bin(*i) for i in [(funct7, 7), (rs2, 5), (rs1, 5), (funct3, 3), (rd, 5), (opcode, 7)]])

    @staticmethod
    def compile_I(opcode, rd, funct3, rs1, imm):
        assert -2048 <= imm <= 2047
        return ''.join([Compiler.twos_comp_bin(*i) for i in [(imm, 12), (rs1, 5), (funct3, 3), (rd, 5), (opcode, 7)]])

    @staticmethod
    def compile_S(opcode, funct3, rs1, rs2, imm):
        assert -2048 <= imm <= 2047
        imm_bin = Compiler.twos_comp_bin(imm, bits=12)
        return imm_bin[:8] + ''.join([Compiler.twos_comp_bin(*i) for i in [(rs2, 5), (rs1, 5), (funct3, 3)]]) + imm_bin[
                                                                                                                8:] + Compiler.twos_comp_bin(
            opcode, 7)

    @staticmethod
    def compile_B(opcode, funct3, rs1, rs2, imm):
        assert imm % 2 == 0
        imm = int(imm / 2)
        assert -2048 <= imm <= 2047
        imm_bin = Compiler.twos_comp_bin(imm, bits=12)
        return imm_bin[0] + imm_bin[2:9] + ''.join(
            [Compiler.twos_comp_bin(*i) for i in [(rs2, 5), (rs1, 5), (funct3, 3)]]) + imm_bin[9:] + imm_bin[
            1] + Compiler.twos_comp_bin(opcode, 7)

    @staticmethod
    def compile_U(opcode, rd, funct3, rs1, imm):
        assert (imm >> 12) << 12 == imm
        imm = imm >> 12
        assert -524288 <= imm <= 524287
        return ''.join([Compiler.twos_comp_bin(*i) for i in [(imm, 20), (rd, 5), (opcode, 7)]])

    @staticmethod
    def compile_J(opcode, rd, funct3, rs1, imm):
        assert imm % 2 == 0
        imm = imm >> 1
        assert -524288 <= imm <= 524287
        imm_bin = Compiler.twos_comp_bin(imm, bits=20)
        return imm_bin[0] + imm_bin[10:] + imm_bin[9] + imm_bin[1:9] + ''.join(
            [Compiler.twos_comp_bin(*i) for i in [(rd, 5), (opcode, 7)]])

