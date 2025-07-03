# compiler.py: translate between Assembly shorthands and binary


import yamls.read_yaml as read_yaml

print(read_yaml.opcodes)

class Compiler:

    @staticmethod
    def twos_comp_bin(val : int, bits):
        """compute the 2's complement of int value val"""
        if val < 0:
            return bin((-val-1) ^ ((2 << (bits-1)) - 1))[2:]

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

    @staticmethod
    def compile_instr(instr):
        pass

    @staticmethod
    def decompile_instr(binary: str):
        pass


#implementations of specific RISCV architectures
class Comp_RV32I(Compiler):

    @staticmethod
    def decompile_instr(binary: str):
        # assert instr_type in Comp_RV32I.base_instr_types
        assert len(binary) == 32

        opcode = int(binary[25:], 2)
        args1 = read_yaml.opcodes[opcode]
        instr_type = args1[1]
        rd = int(binary[20:25], 2)
        funct3 = int(binary[17:20], 2)
        rs1 = int(binary[12:17], 2)
        rs2 = int(binary[7:12], 2)
        funct7 = int(binary[:7], 2)

        match instr_type:
            case 'R':
                return args1[2][funct3][funct7], rd, rs1, rs2
            case 'I':
                instr = args1[2][funct3]
                imm = Compiler.twos_to_int(binary[0:12])
                if type(instr) != str:
                    instr = instr[imm // 16]  # imm[11:5]
                    imm = imm % 16
                return instr, rd, rs1, imm
            case 'S':
                imm = Compiler.twos_to_int(binary[:7] + binary[20:25])
                return args1[2][funct3], rs1, rs2, imm
            case 'B':
                imm = Compiler.twos_to_int(binary[0] + binary[24] + binary[1:7] + binary[20:24] + '0')
                return args1[2][funct3], rs1, rs2, imm
            case 'U':
                return args1[0], rd, (Compiler.twos_to_int(binary[:20]) << 12)
            case 'J':
                imm = Compiler.twos_to_int(binary[0] + binary[12:20] + binary[11] + binary[1:11] + '0')
                return args1[0], rd, imm
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
        return (imm_bin[:7] + ''.join([Compiler.twos_comp_bin(*i) for i in [(rs2, 5), (rs1, 5), (funct3, 3)]])
                + imm_bin[7:] + Compiler.twos_comp_bin(opcode, 7))

    @staticmethod
    def compile_B(opcode, funct3, rs1, rs2, imm):
        assert imm % 2 == 0
        imm = imm // 2
        assert -2048 <= imm <= 2047
        imm_bin = Compiler.twos_comp_bin(imm, bits=12)
        print(imm_bin)
        return imm_bin[0] + imm_bin[2:8] + ''.join(
            [Compiler.twos_comp_bin(*i) for i in [(rs2, 5), (rs1, 5), (funct3, 3)]]) + imm_bin[8:] + imm_bin[
            1] + Compiler.twos_comp_bin(opcode, 7)

    @staticmethod
    def compile_U(opcode, rd, imm):
        assert (imm >> 12) << 12 == imm
        imm = imm >> 12
        assert -524288 <= imm <= 524287
        return ''.join([Compiler.twos_comp_bin(*i) for i in [(imm, 20), (rd, 5), (opcode, 7)]])

    @staticmethod
    def compile_J(opcode, rd, imm):
        assert imm % 2 == 0
        imm = imm >> 1
        assert -524288 <= imm <= 524287
        imm_bin = Compiler.twos_comp_bin(imm, bits=20)
        return imm_bin[0] + imm_bin[10:] + imm_bin[9] + imm_bin[1:9] + ''.join(
            [Compiler.twos_comp_bin(*i) for i in [(rd, 5), (opcode, 7)]])

    @staticmethod
    def compile_instr(instr):

        def reg(x):
            if type(x) is int:
                return x
            else:
                assert x[0] == 'x'
                return int(x[1:])

        opcode, instr_type, *args = read_yaml.inv_opcodes[instr[0]]
        match instr_type:
            case 'R':
                return Comp_RV32I.compile_R(opcode, reg(instr[1]), args[0], reg(instr[2]), reg(instr[3]), args[1])
            case 'I':
                imm = int(instr[3])
                if len(args) > 1:
                    imm += int(args[1]) << 4
                return Comp_RV32I.compile_I(opcode, reg(instr[1]), args[0], reg(instr[2]), imm)
            case 'S':
                return Comp_RV32I.compile_S(opcode, args[0], reg(instr[1]), reg(instr[2]), int(instr[3]))
            case 'B':
                return Comp_RV32I.compile_B(opcode, args[0], reg(instr[1]), reg(instr[2]), int(instr[3]))
            case 'U':
                return Comp_RV32I.compile_U(opcode, reg(instr[1]), int(instr[2]))
            case 'J':
                return Comp_RV32I.compile_J(opcode, reg(instr[1]), int(instr[2]))
        return None
