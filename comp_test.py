# import unittest
#
# from assembler import Assembler_RV32I
#
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         test_data = {   "00000000001000001100111110110011": ('xor', 31, 1, 2),
#                         "11111111110110000000011110010011": ('addi', 15, 16, -3),
#                         "00100000110110000101011110010011": ('srai', 15, 16, 13),
#                         "00000000110110000101011110010011": ('srli', 15, 16, 13),
#                         "00010000010101001010000110100011": ('sw', 9, 5, 259),
#                         "10010000110000111110000111100011": ('bltu', 7, 12, -1790),
#                         "00000000001010000000000001101111": ('jal', 0, 524290),
#                         "00000000001010000000001010010111": ('auipc', 5, 2621440),
#                         "00000000000110001000001001110011": ("ebreak", 4, 17),
#                         "00000000000010001000001001110011": ("ecall", 4, 17)
#                         }
#
#         for instr, res in test_data.items():
#             with self.subTest(res=res):
#                 self.assertEqual(Assembler_RV32I.decompile_instr(instr), res, f"decomp RV32I instr: {instr}")
#                 self.assertEqual(Assembler_RV32I.compile_instr(res), instr, f"compile RV32I instr: {res}")
#
#
#
#
# if __name__ == '__main__':
#     unittest.main()
