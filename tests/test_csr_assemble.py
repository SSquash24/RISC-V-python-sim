import pytest

from modules.csr import CSR
from modules.RV32I import RV32I
from riscv import RISCV
from test_commons import *

tests = [
    ('csrrw x5, x10, 3', '00000000001101010001001011110011'),
    ('csrrs x3, x13, 27', '00000001101101101010000111110011'),
    ('csrrc x29, x9, 911', '00111000111101001011111011110011'),
    ('csrrwi x14, 31, 101', '00000110010111111101011101110011'),
    ('csrrsi x17, 1, 441', '00011011100100001110100011110011'),
    ('csrrci x0, 12, 7', '00000000011101100111000001110011')
]

sim = RISCV([RV32I, CSR], 16, [], True)
module = sim.modules['CSR']

@pytest.mark.parametrize('instr, binary', tests)
def test_csr_assemble(instr, binary):
    tcom_assemble(sim, module, instr, binary)

@pytest.mark.parametrize('instr, binary', tests)
def test_csr_unassemble(instr, binary):
    tcom_unassemble(module, instr, binary)