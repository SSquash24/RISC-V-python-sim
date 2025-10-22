import pytest
from modules.Float import Float
from riscv import RISCV # used for clean_instr command
from modules.RV32I import RV32I
from tests.test_commons import *

tests = [
    ('fadd.s f22, f12, f2, RNE', '00000000001001100000101101010011'),
    ('fsub.s f3, f13, f23, RDN', '00001001011101101010000111010011'),
    ('fmul.s f14, f4, f24, RMM', '00010001100000100100011101010011'),
    ('fdiv.s f1, f1, f1, RTZ', '00011000000100001001000011010011'),
    ('fsgnj.s f2, f0, f3', '00100000001100000000000101010011'),
    ('fsgnjn.s f17, f29, f31', '00100001111111101001100011010011'),
    ('fsgnjx.s f30, f28, f26', '00100001101011100010111101010011'),
    ('fsqrt.s f5, f25, RUP', '01011000000011001011001011010011'),
    ('fmin.s f4, f6, f11', '00101000101100110000001001010011'),
    ('fmax.s f7, f8, f9', '00101000100101000001001111010011'),
    ('fle.s x10, f20, f30', '10100001111010100000010101010011'),
    ('flt.s x11, f11, f11', '10100000101101011001010111010011'),
    ('feq.s x16, f8, f4', '10100000010001000010100001010011'),
    ('fcvt.s.w f2, x1, DYN', '11010000000000001111000101010011'),
    ('fcvt.s.wu f3, x2, RNE', '11010000000100010000000111010011'),
    ('fcvt.w.s x4, f31, RMM', '11000000000011111100001001010011'),
    ('fcvt.wu.s x31, f30, RTZ', '11000000000111110001111111010011'),
    ('fmv.x.w x30, f29', '11100000000011101000111101010011'),
    ('fclass.s x15, f15', '11100000000001111001011111010011'),
    ('fmadd.s f0, f1, f2, f3, RUP', '00011000001000001011000001000011'),
    ('fmsub.s f25, f24, f23, f22, RDN', '10110001011111000010110011000111'),
    ('fnmsub.s f31, f30, f1, f0, DYN', '00000000000111110111111111001011'),
    ('fnmadd.s f0, f10, f20, f30, RTZ', '11110001010001010001000001001111'),
    ('flw f12, x7, 32', '00000010000000111010011000000111'),
    ('fsw f17, x0, -96', '11111011000100000010000000100111')
]

pseudos = [
    ('frcsr x4', ['csrrs', 'x4', 'x0', '3']),
    ('fscsr x30', ['csrrw', 'x0', 'x30', '3'])
]

# sim = Float({'mem_size': 16, 'modules':[Float.__name__], 'pc': 0, 'mem': [], 'debug': True, 'status': 'RUNNING'})
cleaner = RISCV(modules=[Float, RV32I], mem_size=16, code=[], debug=True)
mod = cleaner.modules['Float']

@pytest.mark.parametrize('instr, binary', tests)
def test_assemble(instr, binary):
    tcom_assemble(cleaner, mod, instr, binary)

@pytest.mark.parametrize('instr, binary', tests)
def test_unassemble(instr, binary):
    tcom_unassemble(mod, instr, binary)

@pytest.mark.parametrize('pseudo, translated', pseudos)
def test_pseudos(pseudo, translated):
    print(cleaner.state['pseudos'])
    assert cleaner.clean_instr(pseudo) == translated