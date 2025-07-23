# # assembler.py: translate between Assembly shorthands and binary
#
# import yamls.read_yaml as read_yaml
#
# class RVError(Exception):
#     """
#     Exception raised when the RISC-V simulation hits an error,
#     e.g.: an invalid opcode
#     """
#     pass
#
#
#
# class BaseAssembler:
#     """
#     Base abstract class for RISC-V compilers
#     """
#     def __init__(self):
#         self.bits = 32
#
#     @staticmethod
#     def twos_comp_bin(val : int, bits):
#         """compute the 2's complement of int value val"""
#         if val < 0:
#             return bin((-val-1) ^ ((2 << (bits-1)) - 1))[2:]
#
#         return bin(val)[2:].rjust(bits, '0')
#
#     @staticmethod
#     def twos_to_int(val: str):
#         if val[0] == '0':
#             return int(val, 2)
#         else:
#             return int(val, 2) - (1 << len(val))
#
#     def _decompile_commons(self, binary: str):
#         return int(binary[25:], 2), int(binary[20:25], 2), int(binary[12:17], 2), int(binary[7:12], 2), int(binary[17:20], 2), int(binary[:7], 2)
#
#
#     def compile_file(self, filename):
#         code = open(filename, mode='r').read().splitlines()
#         code = [i.strip().replace(',', '').split() for i in code]
#         res = []
#         for line in code:
#             if not line:
#                 continue
#             if line[0][0] == '.':
#                 match line[0][1:]:
#                     case 'int':
#                         val = int(line[1])
#                         res.append(self.twos_comp_bin(val, self.bits))
#                     case 'bool':
#                         val = int(line[1])
#                         res.append(self.twos_comp_bin(val, 8).rjust(self.bits, '0'))
#             elif line[0][0] != '#':
#                 res.append(self.compile_instr(line))
#
#         return res
#
#     @staticmethod
#     def compile_instr(instr): # should be implemented by inheriting class
#         pass
#
#     @staticmethod
#     def decompile_instr(binary: str): # should be implemented by inheriting class
#         pass
#
#
# #implementations of specific RISCV architectures
# class Assembler_RV32I(BaseAssembler):
#     """
#     RV32I implementation of compiler, used both for assembly and by the simulator for understanding opcodes
#     """
#
#     @staticmethod
#     def decompile_instr(binary: str): # override
#         # assert instr_type in Comp_RV32I.base_instr_types
#         assert len(binary) == 32
#
#         opcode, rd, rs1, rs2, funct3, funct7 = super()._decompile_commons(binary)
#
#         try:
#             args1 = read_yaml.opcodes[opcode]
#         except KeyError:
#             raise RVError("unknown opcode")
#         instr_type = args1[1]
#
#
#         match instr_type:
#             case 'R':
#                 return args1[2][funct3][funct7], rd, rs1, rs2
#             case 'I':
#                 instr = args1[2][funct3]
#                 imm = BaseAssembler.twos_to_int(binary[0:12])
#                 if type(instr) != str:
#                     if instr[0] == 0:
#                         return instr[1][imm], rd, rs1
#                     start = 1 << (instr[0]-1)
#
#                     instr = instr[1][imm // start]
#                     imm = imm % start
#                 return instr, rd, rs1, imm
#             case 'S':
#                 imm = BaseAssembler.twos_to_int(binary[:7] + binary[20:25])
#                 return args1[2][funct3], rs1, rs2, imm
#             case 'B':
#                 imm = BaseAssembler.twos_to_int(binary[0] + binary[24] + binary[1:7] + binary[20:24] + '0')
#                 return args1[2][funct3], rs1, rs2, imm
#             case 'U':
#                 return args1[0], rd, (BaseAssembler.twos_to_int(binary[:20]) << 12)
#             case 'J':
#                 imm = BaseAssembler.twos_to_int(binary[0] + binary[12:20] + binary[11] + binary[1:11] + '0')
#                 return args1[0], rd, imm
#         return None
#
#     @staticmethod
#     def compile_R(opcode, rd, funct3, rs1, rs2, funct7):
#         return ''.join(
#             [BaseAssembler.twos_comp_bin(*i) for i in [(funct7, 7), (rs2, 5), (rs1, 5), (funct3, 3), (rd, 5), (opcode, 7)]])
#
#     @staticmethod
#     def compile_I(opcode, rd, funct3, rs1, imm):
#         if not -2048 <= imm <= 2047:
#             raise AssemblerError(f"Invalid immediate value for I-type instr: {imm}")
#         return ''.join([BaseAssembler.twos_comp_bin(*i) for i in [(imm, 12), (rs1, 5), (funct3, 3), (rd, 5), (opcode, 7)]])
#
#     @staticmethod
#     def compile_S(opcode, funct3, rs1, rs2, imm):
#         if not -2048 <= imm <= 2047:
#             raise AssemblerError(f"Invalid immediate value for S-type instr: {imm}")
#         imm_bin = BaseAssembler.twos_comp_bin(imm, bits=12)
#         return (imm_bin[:7] + ''.join([BaseAssembler.twos_comp_bin(*i) for i in [(rs2, 5), (rs1, 5), (funct3, 3)]])
#                 + imm_bin[7:] + BaseAssembler.twos_comp_bin(opcode, 7))
#
#     @staticmethod
#     def compile_B(opcode, funct3, rs1, rs2, imm):
#         if not imm % 2 == 0:
#             raise AssemblerError(f"Invalid immediate value for B-type instr: {imm}\nValue should be an even number.")
#         imm = imm // 2
#         if not -2048 <= imm <= 2047:
#             raise AssemblerError(f"Invalid immediate value for B-type instr: {imm}")
#         imm_bin = BaseAssembler.twos_comp_bin(imm, bits=12)
#         return imm_bin[0] + imm_bin[2:8] + ''.join(
#             [BaseAssembler.twos_comp_bin(*i) for i in [(rs2, 5), (rs1, 5), (funct3, 3)]]) + imm_bin[8:] + imm_bin[
#             1] + BaseAssembler.twos_comp_bin(opcode, 7)
#
#     @staticmethod
#     def compile_U(opcode, rd, imm):
#         if not (imm >> 12) << 12 == imm:
#             raise AssemblerError(f"Invalid immediate value for U-type instr: {imm}\nValue should be a multiple of 2^12")
#         imm = imm >> 12
#         if not -524288 <= imm <= 524287:
#             raise AssemblerError(f"Invalid immediate value for U-type instr: {imm}")
#         return ''.join([BaseAssembler.twos_comp_bin(*i) for i in [(imm, 20), (rd, 5), (opcode, 7)]])
#
#     @staticmethod
#     def compile_J(opcode, rd, imm):
#         if not imm % 2 == 0:
#             raise AssemblerError(f"Invalid immediate value for J-type instr: {imm}\nValue should be an even number.")
#         imm = imm >> 1
#         if not -524288 <= imm <= 524287:
#             raise AssemblerError(f"Invalid immediate value for J-type instr: {imm}")
#         imm_bin = BaseAssembler.twos_comp_bin(imm, bits=20)
#         return imm_bin[0] + imm_bin[10:] + imm_bin[9] + imm_bin[1:9] + ''.join(
#             [BaseAssembler.twos_comp_bin(*i) for i in [(rd, 5), (opcode, 7)]])
#
#     @staticmethod
#     def compile_instr(instr): # override
#
#         # helper function that finds register int value
#         def reg(x):
#             if type(x) is int:
#                 val = x
#             else:
#                 assert x[0] == 'x'
#                 val = int(x[1:])
#             if not 0 <= val <= 31:
#                 raise AssemblerError(f"Invalid reg value: {x}")
#             return val
#
#         opcode, instr_type, *args = read_yaml.inv_opcodes[instr[0]]
#         match instr_type:
#             case 'R':
#                 return Assembler_RV32I.compile_R(opcode, reg(instr[1]), args[0], reg(instr[2]), reg(instr[3]), args[1])
#             case 'I':
#                 if len(args) > 1:
#                     if args[1] == 0:
#                         imm = args[2]
#                     else:
#                         imm = int(instr[3]) + (int(args[2]) << (args[1]-1))
#                 else:
#                     imm = int(instr[3])
#                 return Assembler_RV32I.compile_I(opcode, reg(instr[1]), args[0], reg(instr[2]), imm)
#             case 'S':
#                 return Assembler_RV32I.compile_S(opcode, args[0], reg(instr[1]), reg(instr[2]), int(instr[3]))
#             case 'B':
#                 return Assembler_RV32I.compile_B(opcode, args[0], reg(instr[1]), reg(instr[2]), int(instr[3]))
#             case 'U':
#                 return Assembler_RV32I.compile_U(opcode, reg(instr[1]), int(instr[2]))
#             case 'J':
#                 return Assembler_RV32I.compile_J(opcode, reg(instr[1]), int(instr[2]))
#             case _:
#                 raise AssemblerError(f"Invalid instruction type: {instr_type}")
#
#
#
# class Assembler_RV32I_CSR(Assembler_RV32I):
#     @staticmethod
#     def decompile_instr(binary: str):
#         base = super().decompile_instr(binary)
#         if base is not None:
#             return base
#
#         opcode, _, rs1, _, funct3, _ = super()._decompile_commons(binary)
#         csr = int(binary[:12], 2)
#         if opcode == read_yaml.inv_opcodes['SYSTEM'][0]:
#
